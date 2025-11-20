import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime

from database import create_document, get_documents, db

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Brand Guardian AI backend is running"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


# -------- Brand Guardian AI: Simple Heuristic Analyzer --------

POSITIVE_WORDS = set(
    "amazing awesome love great good excellent fast secure reliable innovative happy satisfied best winning improved".split()
)
NEGATIVE_WORDS = set(
    "bad slow hate terrible worst insecure buggy crash issue problem disappointing expensive confusing lost broken".split()
)

class AnalyzeRequest(BaseModel):
    brand: str = Field(..., description="Brand name to analyze")
    keywords: Optional[List[str]] = Field(default=None, description="Additional search keywords")
    samples: Optional[List[str]] = Field(
        default=None,
        description="Optional sample social posts/comments to analyze. If omitted, a generic analysis is generated.",
    )
    max_samples: int = Field(50, ge=1, le=500, description="Max texts to analyze")

class AnalyzeResponse(BaseModel):
    brand: str
    keywords: List[str]
    overall_sentiment: str
    sentiment_score: float
    pros: List[str]
    cons: List[str]
    recommendations: List[str]
    sample_posts: List[Dict[str, Any]]
    created_at: datetime


def score_text(text: str) -> int:
    score = 0
    words = [w.strip(".,!?:;()[]\"' ").lower() for w in text.split()]
    for w in words:
        if w in POSITIVE_WORDS:
            score += 1
        if w in NEGATIVE_WORDS:
            score -= 1
    return score


def summarize(samples: List[str]) -> Dict[str, Any]:
    if not samples:
        return {
            "score": 0.0,
            "overall": "neutral",
            "pros": [
                "Strong brand recognition in tech communities",
                "Perceived as secure and reliable",
            ],
            "cons": [
                "Limited presence on certain social platforms",
                "Mixed feedback on pricing and support response times",
            ],
            "top_positive": [],
            "top_negative": [],
        }

    scored = [(s, score_text(s)) for s in samples]
    total = sum(s for _, s in scored)
    norm = max(len(samples), 1)
    avg = total / norm
    overall = "positive" if avg > 0.2 else ("negative" if avg < -0.2 else "neutral")

    positives = sorted([t for t in scored if t[1] > 0], key=lambda x: -x[1])[:5]
    negatives = sorted([t for t in scored if t[1] < 0], key=lambda x: x[1])[:5]

    def extract_themes(items: List[str]) -> List[str]:
        themes: Dict[str, int] = {}
        for text in items:
            for token in [w.lower().strip(".,!?:;()[]\"'") for w in text.split()]:
                if len(token) < 4:
                    continue
                if token in POSITIVE_WORDS or token in NEGATIVE_WORDS:
                    continue
                themes[token] = themes.get(token, 0) + 1
        return [k for k, _ in sorted(themes.items(), key=lambda kv: -kv[1])[:6]]

    pros_themes = extract_themes([t for t, s in positives])
    cons_themes = extract_themes([t for t, s in negatives])

    pros = [f"Users mention {k}" for k in pros_themes] or ["Positive feedback present"]
    cons = [f"Concerns about {k}" for k in cons_themes] or ["Some negative feedback present"]

    return {
        "score": avg,
        "overall": overall,
        "pros": pros,
        "cons": cons,
        "top_positive": [t for t, _ in positives],
        "top_negative": [t for t, _ in negatives],
    }


def make_recommendations(overall: str, pros: List[str], cons: List[str]) -> List[str]:
    recs: List[str] = []
    if overall != "positive":
        recs.append("Launch a transparent communication thread addressing top concerns within 24-48 hours.")
    if cons:
        recs.append("Create bite-sized social content that directly counters the top 2-3 pain points.")
        recs.append("Empower support team with templated responses and track resolution SLAs publicly.")
    if pros:
        recs.append("Amplify positive testimonials via featured posts and case studies.")
    recs.append("Set up weekly sentiment tracking with alerts for sudden drops.")
    recs.append("Engage brand advocates to co-create content and host AMAs.")
    return recs[:6]


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_brand(req: AnalyzeRequest):
    brand = req.brand.strip()
    if not brand:
        raise HTTPException(status_code=400, detail="Brand is required")

    # In a production version, we'd fetch posts from APIs. For MVP, use provided samples.
    samples: List[str] = []
    if req.samples:
        samples = [s for s in req.samples if isinstance(s, str) and s.strip()][: req.max_samples]

    summary = summarize(samples)

    record = {
        "brand": brand,
        "keywords": req.keywords or [brand],
        "overall_sentiment": summary["overall"],
        "sentiment_score": round(float(summary["score"]), 3),
        "pros": summary["pros"],
        "cons": summary["cons"],
        "recommendations": make_recommendations(summary["overall"], summary["pros"], summary["cons"]),
        "sample_posts": [
            {"text": t, "source": "sample", "sentiment": "positive"} for t in summary.get("top_positive", [])
        ]
        + [
            {"text": t, "source": "sample", "sentiment": "negative"} for t in summary.get("top_negative", [])
        ],
        "created_at": datetime.utcnow(),
    }

    try:
        create_document("analysis", record)
    except Exception:
        # If DB is not configured, continue without storing
        pass

    return AnalyzeResponse(**record)


@app.get("/analyses")
def list_analyses(brand: Optional[str] = None, limit: int = 10):
    try:
        filt = {"brand": brand} if brand else {}
        docs = get_documents("analysis", filt, limit)
        # Convert ObjectId and datetime
        out = []
        for d in docs:
            d.pop("_id", None)
            if isinstance(d.get("created_at"), datetime):
                d["created_at"] = d["created_at"].isoformat()
            out.append(d)
        return {"items": out}
    except Exception as e:
        # If DB not available, return empty
        return {"items": []}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
