# from fastapi import FastAPI
# from contextlib import asynccontextmanager
# from typing import AsyncGenerator
# import tensorflow as tf
# import pickle
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.util import get_remote_address
# from slowapi.errors import RateLimitExceeded

# from app.config import settings
# from app.api.routes.sentiment import router as sentiment_router

# # Initialize rate limiter
# limiter = Limiter(key_func=get_remote_address)

# @asynccontextmanager
# async def lifespan(app: FastAPI) -> AsyncGenerator:
#     """life-span event handler for the FastAPI application.
    
#     This function initializes the model and tokenizer at startup
#     and logs a shutdown message at shutdown.
#     """
#     try:
#         # Load the trained model
#         model = tf.keras.models.load_model(str(settings.MODEL_PATH))
        
#         # Load tokenizer
#         with open(settings.TOKENIZER_PATH, 'rb') as handle:
#             tokenizer = pickle.load(handle)
            
#         # Store model and tokenizer in app state
#         app.state.model = model
#         app.state.tokenizer = tokenizer

#     except Exception as e:
#         print(f"Error during startup: {str(e)}")
#         raise
#     yield
#     print("Shutting down...")

# app = FastAPI(
#     title=settings.PROJECT_NAME,
#     description="API for analyzing sentiment in movie reviews using LSTM",
#     version="1.0.0",
#     lifespan=lifespan
# )

# # Setup rate limiter
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# # Include routers
# app.include_router(sentiment_router, prefix=settings.API_V1_STR) 

import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os # To construct file paths reliably

# --- Configuration & Preprocessing Setup (Must match training script) ---

MODEL_PATH = 'sentiment_model.h5'
TOKENIZER_PATH = 'tokenizer_config.json'
MAX_LENGTH = 200  # This MUST match the max_length used during training/saving
PADDING_TYPE = 'post' # This MUST match the padding type used during training/saving
OOV_TOKEN = '' # This MUST match the oov_tok used during training/saving (use None if Keras default was used)

# Ensure NLTK data is available (best practice: ensure it's downloaded beforehand)
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except Exception:
    print("NLTK data not found. Please run nltk.download('wordnet') and nltk.download('stopwords') first.")
    # Optionally, attempt download here, but it's better done outside the app startup
    nltk.download('wordnet')
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
w_tokenizer = WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()

def remove_tags(string):
    """Cleans text by removing URLs and non-alphanumeric characters."""
    if not isinstance(string, str): # Handle potential non-string input
        return ""
    result = re.sub('https://.*','',string)  # remove URLs
    result = re.sub( r'[^a-zA-Z0-9\s]', '',result) # remove non-alphanumeric, keep spaces
    result = result.lower()
    return result

def preprocess_text(text):
    """Applies all preprocessing steps used during training."""
    text = remove_tags(text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    lemmatized_text = ""
    for w in w_tokenizer.tokenize(text):
        lemmatized_text += lemmatizer.lemmatize(w) + " "
    return lemmatized_text.strip() # Remove trailing space

# --- FastAPI App Setup ---

app = FastAPI(title="Sentiment Analysis API")

# Global variables to hold the loaded model and tokenizer
model = None
tokenizer = None

@app.on_event("startup")
async def load_resources():
    """Load the model and tokenizer when the app starts."""
    global model, tokenizer
    print("Loading Keras model...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        # Depending on deployment, you might want to raise an error or exit
        return
    model = load_model(MODEL_PATH)
    print("Model loaded.")

    print("Loading tokenizer...")
    if not os.path.exists(TOKENIZER_PATH):
         print(f"Error: Tokenizer file not found at {TOKENIZER_PATH}")
         # Depending on deployment, you might want to raise an error or exit
         return
    try:
        with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
            tokenizer_json = json.load(f) # Load the outer JSON string first
            tokenizer = tokenizer_from_json(tokenizer_json) # Then parse the inner JSON string
        print("Tokenizer loaded.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        tokenizer = None # Ensure tokenizer is None if loading failed

# --- Pydantic Models for Request and Response ---

class TextInput(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    probability: float # Include the raw probability for more info

# --- API Endpoint ---

@app.post("/predict_sentiment", response_model=SentimentResponse)
async def predict_sentiment(data: TextInput):
    """
    Predicts the sentiment (Positive/Negative) of a given text.
    """
    global model, tokenizer

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or Tokenizer not loaded. Server might be starting or encountered an error.")

    if not data.text or not data.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        # 1. Preprocess the input text
        processed_text = preprocess_text(data.text)

        # 2. Convert text to sequence using the loaded tokenizer
        sequence = tokenizer.texts_to_sequences([processed_text]) # Needs to be a list

        # 3. Pad the sequence
        padded_sequence = pad_sequences(sequence, padding=PADDING_TYPE, maxlen=MAX_LENGTH)

        # 4. Make prediction
        prediction = model.predict(padded_sequence, verbose=0) # verbose=0 suppresses progress bar
        probability = float(prediction[0][0]) # Get the scalar probability

        # 5. Determine sentiment label
        sentiment = "Positive" if probability >= 0.5 else "Negative"

        return SentimentResponse(
            text=data.text,
            sentiment=sentiment,
            probability=probability
        )
    except Exception as e:
        # Log the error for debugging
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")

# Example of how to run (save this code as e.g., main.py):
# uvicorn main:app --reload