from pydantic import BaseModel

class Review(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    is_positive: bool 