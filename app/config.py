from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Sentiment Analysis API"
    
    # Model Settings
    MAX_TEXT_LENGTH: int = Field(default=500, description="Maximum length of text input")
    MODEL_PATH: Path = Field(default="models/sentiment_model.h5", description="Path to the trained LSTM model")
    TOKENIZER_PATH: Path = Field(default="models/tokenizer.pkl", description="Path to the tokenizer file")
    
    # Server Settings
    HOST: str = Field(default="0.0.0.0", description="Host to run the server on")
    PORT: int = Field(default=8000, description="Port to run the server on")
    RELOAD: bool = Field(default=True, description="Enable auto-reload")
    
    # Rate Limiter Settings
    RATE_LIMIT_REQUESTS: int = Field(default=10, description="Number of requests allowed per minute")
    RATE_LIMIT_MINUTES: int = Field(default=1, description="Time window for rate limiting in minutes")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 