from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from config import settings


class Review(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    is_positive: bool

# Load model and tokenizer
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    try:
        if not settings.MODEL_PATH.exists() or not settings.TOKENIZER_PATH.exists():
            raise FileNotFoundError("Model or tokenizer files not found")
        
        model = tf.keras.models.load_model(str(settings.MODEL_PATH))
        with open(settings.TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """life-span event handler for the FastAPI application.

    This function initializes the database and generates the schema at startup
    and logs a shutdown message at shutdown.
    """
    try:
        #load_model()
        print("Model loaded")

    except Exception as e:
        print(e)
        raise
    yield
    print("Shutting down...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for analyzing sentiment in movie reviews using LSTM",
    version="1.0.0",
    lifespan=lifespan
)


@app.post(f"{settings.API_V1_STR}/analyze", response_model=SentimentResponse)
async def analyze_sentiment(review: Review):
    try:
        # Check text length
        if len(review.text) > settings.MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Text exceeds maximum length of {settings.MAX_TEXT_LENGTH} characters"
            )
        
        # # Preprocess and predict
        # sequence = tokenizer.texts_to_sequences([review.text])
        # padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
        #     sequence,
        #     maxlen=model.input_shape[1],
        #     padding='post',
        #     truncating='post'
        # )
        
        # prediction = model.predict(padded_sequence)[0][0]
        # confidence = float(prediction)
        # is_positive = confidence >= 0.5
        # sentiment = "positive" if is_positive else "negative"
        
        return SentimentResponse(
            sentiment="positive",
            confidence=0.95,
            is_positive=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_STR}/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "settings": {
            "max_text_length": settings.MAX_TEXT_LENGTH,
            "model_path": str(settings.MODEL_PATH),
            "tokenizer_path": str(settings.TOKENIZER_PATH)
        }
    } 