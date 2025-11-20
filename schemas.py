"""
Database Schemas for Brand Guardian AI

Each Pydantic model represents a collection in MongoDB. The collection name is the
lowercased class name. Example: class Analysis -> collection "analysis".
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Analysis(BaseModel):
    """
    Brand analysis results
    Collection: "analysis"
    """
    brand: str = Field(..., description="Brand name analyzed")
    keywords: List[str] = Field(default_factory=list, description="Search keywords used")
    overall_sentiment: str = Field(..., description="overall sentiment label: positive | neutral | negative")
    sentiment_score: float = Field(..., description="normalized score -1..1")
    pros: List[str] = Field(default_factory=list, description="Key positive themes")
    cons: List[str] = Field(default_factory=list, description="Key negative themes")
    recommendations: List[str] = Field(default_factory=list, description="Actionable suggestions")
    sample_posts: List[Dict[str, Any]] = Field(default_factory=list, description="Small sample of analyzed posts")

class User(BaseModel):
    name: str
    email: str
    address: str
    age: Optional[int] = Field(None, ge=0, le=120)
    is_active: bool = True

class Product(BaseModel):
    title: str
    description: Optional[str] = None
    price: float = Field(..., ge=0)
    category: str
    in_stock: bool = True
