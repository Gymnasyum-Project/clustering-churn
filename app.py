"""FastAPI Prediction Service for Gymnasium Customer Retention.

Implements the inference API contract for Express backend integration:
- Primary Endpoints:
    POST /api/predict
    GET  /api/health
- Compatibility Endpoints (for backend-project):
    POST /predict
    GET  /health
- Validates 8 numerical inputs via Pydantic (rejects missing fields with HTTP 422).
- Returns segment label, code, probabilities, and domain-grounded top factors.
"""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from predict import FEATURE_COLUMNS, load_model, predict

# Load environment configuration from .env if present
load_dotenv()

HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", "8000"))
MODEL_PATH: str = os.getenv("MODEL_PATH", "models/random_forest_8fitur_v3.joblib")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gymnasium-prediction-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preloads the machine learning model on application startup."""
    resolved_model_path = Path(MODEL_PATH)
    if not resolved_model_path.is_absolute():
        resolved_model_path = Path(__file__).resolve().parent / resolved_model_path

    logger.info(f"Preloading model artifact from: {resolved_model_path}")
    try:
        model, metadata = load_model(model_path=resolved_model_path)
        app.state.model = model
        app.state.metadata = metadata
        app.state.model_loaded = True
        logger.info("Model and metadata successfully preloaded into application state.")
    except Exception as exc:
        logger.error(f"Failed to preload model at startup: {exc}")
        app.state.model = None
        app.state.metadata = None
        app.state.model_loaded = False
    yield


app = FastAPI(
    title="Gymnasium Customer Retention Risk Inference API",
    description="Machine Learning service predicting customer retention segment (Active, At Risk, Need Attention)",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local testing and cross-service communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    """Request payload schema requiring exactly the 8 backend features."""

    customer_id: Optional[str] = Field(
        default=None,
        description="Optional customer identifier for tracking and correlation",
        examples=["CUST00001"],
    )
    attendance_frequency_7d: float = Field(
        ...,
        description="Number of attended sessions in the last 7 days",
        examples=[0.0],
    )
    missed_session_count_7d: float = Field(
        ...,
        description="Number of missed booked sessions in the last 7 days",
        examples=[0.0],
    )
    reschedule_frequency_7d: float = Field(
        ...,
        description="Number of rescheduled sessions in the last 7 days",
        examples=[0.0],
    )
    remaining_package_sessions: float = Field(
        ...,
        description="Remaining sessions available in active membership package",
        examples=[10.0],
    )
    days_until_package_expiry: float = Field(
        ...,
        description="Days remaining before package expiration",
        examples=[30.0],
    )
    days_since_last_visit: float = Field(
        ...,
        description="Number of days elapsed since customer's last gym visit",
        examples=[11.0],
    )
    reminder_count_7d: float = Field(
        ...,
        description="Number of notification reminders sent in the last 7 days",
        examples=[0.0],
    )
    reminder_response_count_7d: float = Field(
        ...,
        description="Number of reminder notifications responded to in the last 7 days",
        examples=[0.0],
    )

    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {
                "customer_id": "CUST00001",
                "attendance_frequency_7d": 0.0,
                "missed_session_count_7d": 0.0,
                "reschedule_frequency_7d": 0.0,
                "remaining_package_sessions": 10.0,
                "days_until_package_expiry": 30.0,
                "days_since_last_visit": 11.0,
                "reminder_count_7d": 0.0,
                "reminder_response_count_7d": 0.0,
            }
        },
    }


class PredictResponse(BaseModel):
    """Response payload schema returning retention prediction and insights."""

    customer_id: Optional[str] = Field(default=None, description="Echoed customer ID")
    label: str = Field(description="Segment label: Active, At Risk, or Need Attention")
    segmen: str = Field(description="Alias for label for Express backend compatibility")
    segmen_kode: int = Field(description="Segment integer code: 0=Active, 1=At Risk, 2=Need Attention")
    probabilities: Dict[str, float] = Field(description="Prediction probability per class")
    top_factors: List[str] = Field(description="Key domain factors explaining the prediction")


class HealthResponse(BaseModel):
    """Health check response payload schema."""

    status: str = Field(default="ok", description="Service health status", examples=["ok"])
    model_version: str = Field(
        default="random-forest-8fitur-v1",
        description="Version identifier of the deployed prediction model",
        examples=["random-forest-8fitur-v1"],
    )


def execute_prediction(payload: PredictRequest) -> PredictResponse:
    """Core prediction handler called by both API and compatibility routes."""
    feature_dict = {
        "attendance_frequency_7d": payload.attendance_frequency_7d,
        "missed_session_count_7d": payload.missed_session_count_7d,
        "reschedule_frequency_7d": payload.reschedule_frequency_7d,
        "remaining_package_sessions": payload.remaining_package_sessions,
        "days_until_package_expiry": payload.days_until_package_expiry,
        "days_since_last_visit": payload.days_since_last_visit,
        "reminder_count_7d": payload.reminder_count_7d,
        "reminder_response_count_7d": payload.reminder_response_count_7d,
    }

    try:
        resolved_model_path = Path(MODEL_PATH)
        if not resolved_model_path.is_absolute():
            resolved_model_path = Path(__file__).resolve().parent / resolved_model_path

        prediction_result = predict(feature_dict, model_path=resolved_model_path)
    except Exception as exc:
        logger.error(f"Inference error during predict(): {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        )

    return PredictResponse(
        customer_id=payload.customer_id,
        label=prediction_result["label"],
        segmen=prediction_result["segmen"],
        segmen_kode=prediction_result["segmen_kode"],
        probabilities=prediction_result["probabilities"],
        top_factors=prediction_result["top_factors"],
    )


def execute_health() -> HealthResponse:
    """Core health handler returning service readiness and model version."""
    return HealthResponse(
        status="ok",
        model_version="random-forest-8fitur-v1",
    )


# Primary Endpoints
@app.post(
    "/api/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Predict customer retention risk segment",
)
def api_predict(payload: PredictRequest) -> PredictResponse:
    """Primary inference route accepting 8 customer features."""
    return execute_prediction(payload)


@app.get(
    "/api/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["System"],
    summary="Inference service health status",
)
def api_health() -> HealthResponse:
    """Primary health check route."""
    return execute_health()


# Express Backend Compatibility Aliases
@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Compatibility"],
    summary="Express backend alias for /api/predict",
)
def compat_predict(payload: PredictRequest) -> PredictResponse:
    """Compatibility route directly bound to Express backend INFERENCE_URL/predict."""
    return execute_prediction(payload)


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Compatibility"],
    summary="Express backend alias for /api/health",
)
def compat_health() -> HealthResponse:
    """Compatibility health route for Express backend."""
    return execute_health()


if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, log_level=LOG_LEVEL.lower())
