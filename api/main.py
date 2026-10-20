"""
Heart disease risk API.

Endpoints:
  GET  /health   is the service up and is the model loaded?
  POST /predict  one patient's data in (JSON) -> prediction + confidence out
  GET  /metrics  monitoring numbers for Prometheus (requests, latency, predictions)
  GET  /docs     interactive documentation (built in by FastAPI)

Run locally:  uvicorn api.main:app --reload --port 8000
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("heart-api")

# Monitoring: Prometheus reads these from GET /metrics every few seconds.
REQUESTS = Counter("http_requests", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("http_request_duration_seconds", "Time to answer a request", ["path"])
PREDICTIONS = Counter("predictions", "Predictions made", ["result"])
RISK = Histogram("disease_probability", "Predicted probability of heart disease",
                 buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

EXAMPLE_PATIENT = {
    "age": 63, "sex": 1, "chest_pain_type": 1, "resting_bp": 145, "cholesterol": 233,
    "fasting_blood_sugar": 1, "resting_ecg": 2, "max_heart_rate": 150,
    "exercise_angina": 0, "st_depression": 2.3, "st_slope": 3,
    "num_major_vessels": 0, "thalassemia": 6,
}


class Patient(BaseModel):
    """One patient's measurements. Ranges and codes match the data validation rules."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_PATIENT]})

    age: int = Field(ge=18, le=100, description="Age in years")
    sex: Literal[0, 1] = Field(description="0 = female, 1 = male")
    chest_pain_type: Literal[1, 2, 3, 4] = Field(
        description="1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic")
    resting_bp: float = Field(ge=60, le=250, description="Resting blood pressure (mm Hg)")
    cholesterol: float = Field(ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fasting_blood_sugar: Literal[0, 1] = Field(description="1 if fasting blood sugar > 120 mg/dl")
    resting_ecg: Literal[0, 1, 2] = Field(
        description="0 normal, 1 ST-T wave abnormality, 2 left ventricular hypertrophy")
    max_heart_rate: float = Field(ge=50, le=250, description="Maximum heart rate achieved")
    exercise_angina: Literal[0, 1] = Field(description="1 if exercise caused chest pain")
    st_depression: float = Field(ge=0, le=10, description="ST depression from exercise vs rest")
    st_slope: Literal[1, 2, 3] = Field(description="1 upsloping, 2 flat, 3 downsloping")
    num_major_vessels: Optional[Literal[0, 1, 2, 3]] = Field(
        default=None, description="Vessels coloured by fluoroscopy (0-3); leave out if unknown")
    thalassemia: Optional[Literal[3, 6, 7]] = Field(
        default=None, description="3 normal, 6 fixed defect, 7 reversible defect; leave out if unknown")


class PredictionResponse(BaseModel):
    prediction: int = Field(description="1 = heart disease likely, 0 = unlikely")
    label: str
    disease_probability: float = Field(description="Model's probability of heart disease")
    confidence: float = Field(description="Probability of the predicted class")
    decision_threshold: float = Field(description="Probability at which 'disease' is predicted")
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, not on every request."""
    app.state.model = joblib.load(MODEL_DIR / "model.joblib")
    app.state.metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text())
    logger.info("Loaded model %s (%s, threshold %.3f)",
                app.state.metadata["model_version"], app.state.metadata["model_type"],
                app.state.metadata["decision_threshold"])
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log and count every request: method, path, status code and how long it took."""
    start = time.perf_counter()
    response = await call_next(request)
    seconds = time.perf_counter() - start
    route = request.scope.get("route")
    path = route.path if route else "unmatched"  # avoids one label per random URL
    REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    LATENCY.labels(path).observe(seconds)
    logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path,
                response.status_code, seconds * 1000)
    return response


@app.get("/health")
def health(request: Request) -> dict:
    meta = request.app.state.metadata
    return {"status": "ok", "model_version": meta["model_version"],
            "model_type": meta["model_type"]}


@app.get("/metrics")
def metrics() -> Response:
    """Expose the monitoring numbers in Prometheus' text format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: Patient, request: Request) -> PredictionResponse:
    model, meta = request.app.state.model, request.app.state.metadata
    # One-row table; astype(float) turns a left-out field (None) into NaN,
    # which the pipeline's imputer then fills, exactly as in training.
    row = pd.DataFrame([patient.model_dump()])[meta["features"]].astype(float)

    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(model.predict(row)[0])  # uses the tuned threshold
    confidence = probability if prediction == 1 else 1 - probability

    # Log and count the outcome only, never the patient's health data.
    logger.info("prediction=%d probability=%.3f", prediction, probability)
    PREDICTIONS.labels("disease" if prediction == 1 else "no_disease").inc()
    RISK.observe(probability)
    return PredictionResponse(
        prediction=prediction,
        label="heart disease likely" if prediction == 1 else "heart disease unlikely",
        disease_probability=round(probability, 4),
        confidence=round(confidence, 4),
        decision_threshold=meta["decision_threshold"],
        model_version=meta["model_version"],
    )
