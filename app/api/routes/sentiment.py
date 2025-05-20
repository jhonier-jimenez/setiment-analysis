from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.schemas.sentiment import Review, SentimentResponse
from app.api.services.sentiment_service import SentimentService
from app.config import settings

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.post("/analyze", response_model=SentimentResponse)
@limiter.limit(f"{settings.RATE_LIMIT_REQUESTS}/minute")
async def analyze_sentiment(request: Request, review: Review):
    try:
        # Check text length
        if len(review.text) > settings.MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Text exceeds maximum length of {settings.MAX_TEXT_LENGTH} characters"
            )
        
        sentiment_service = SentimentService(request.app.state.model, request.app.state.tokenizer)
        sentiment, confidence, is_positive = sentiment_service.analyze_sentiment(review.text)
        
        return SentimentResponse(
            sentiment=sentiment,
            confidence=confidence,
            is_positive=is_positive
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
@limiter.limit(f"{settings.RATE_LIMIT_REQUESTS}/minute")
async def health_check(request: Request):
    return {
        "status": "healthy",
        "model_loaded": request.app.state.model is not None,
        "tokenizer_loaded": request.app.state.tokenizer is not None,
        "settings": {
            "max_text_length": settings.MAX_TEXT_LENGTH,
            "model_path": str(settings.MODEL_PATH),
            "tokenizer_path": str(settings.TOKENIZER_PATH),
            "rate_limit": f"{settings.RATE_LIMIT_REQUESTS} requests per minute"
        }
    } 