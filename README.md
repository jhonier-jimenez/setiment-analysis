# Sentiment Analysis Project

This repository contains a sentiment analysis project with both the analysis notebooks and a FastAPI-based API for analyzing sentiment in movie reviews using LSTM.

## Project Structure

```
.
├── 01 - exploración de datos.ipynb
├── 02 - preprocesado.ipynb
├── 03 - arquitectura de linea de base.ipynb
├── app/
│   ├── main.py
│   ├── config.py
│   ├── requirements.txt
│   ├── .env
│   ├── docker/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── .dockerignore
│   └── models/
│       ├── sentiment_model.h5
│       └── tokenizer.pkl
└── README.md
```

## Requirements

- Python 3.11 or higher
- Docker and Docker Compose (for containerized deployment)

## FastAPI Setup

The FastAPI application is located in the `app` directory.

### Local Development Setup

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment and install dependencies:
```bash
cd app
uv venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
uv pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn main:app --reload
```

### Docker Setup

1. Build and run using Docker Compose:
```bash
cd app
docker-compose -f docker/docker-compose.yml up --build
```

2. Or build and run using Docker directly:
```bash
cd app
docker build -t sentiment-analysis-api -f docker/Dockerfile .
docker run -p 8000:8000 -v $(pwd)/models:/app/models sentiment-analysis-api
```

The API will be available at http://localhost:8000

## API Endpoints

### Analyze Sentiment
- **POST** `/api/v1/analyze`
- Analyzes the sentiment of a movie review text
- Request body:
```json
{
    "text": "Your movie review text here"
}
```
- Response:
```json
{
    "sentiment": "positive|negative",
    "confidence": 0.95,
    "is_positive": true
}
```

### Health Check
- **GET** `/api/v1/health`
- Checks the health status of the API and model loading
- Response:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "tokenizer_loaded": true,
    "settings": {
        "max_text_length": 500,
        "model_path": "models/sentiment_model.h5",
        "tokenizer_path": "models/tokenizer.pkl"
    }
}
```

## API Documentation

You can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Configuration

The API uses Pydantic settings for configuration. You can modify the settings in `app/.env`:

```env
MAX_TEXT_LENGTH=500
MODEL_PATH=models/sentiment_model.h5
TOKENIZER_PATH=models/tokenizer.pkl
HOST=0.0.0.0
PORT=8000
RELOAD=true
```

All settings have default values and can be overridden through environment variables. The settings are automatically loaded from the `.env` file.

## Docker Configuration

The application can be run in a Docker container. The Docker configuration is located in the `app/docker` directory and includes:

- A multi-stage build to minimize the final image size
- Volume mounting for the models directory
- Environment variable configuration
- Automatic restart policy
- Python 3.11 runtime

To customize the Docker setup, you can modify:
- `app/docker/Dockerfile` for build configuration
- `app/docker/docker-compose.yml` for runtime configuration
- Environment variables in the docker-compose file
