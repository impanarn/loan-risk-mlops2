from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import json
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from typing import Dict, Any

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT   = Counter("api_requests_total", "Total API requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])

app = FastAPI(title="Loan Risk Prediction API", version="1.0.0")

# Load model and feature names at startup
model = joblib.load("models/model.pkl")
with open("data/processed/feature_names.json") as f:
    FEATURE_NAMES = json.load(f)

logger.info(f"Model loaded. Expecting {len(FEATURE_NAMES)} features.")

class LoanApplication(BaseModel):
    features: Dict[str, Any]  # flexible: accepts any feature dict

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "AMT_INCOME_TOTAL": 202500.0,
                    "AMT_CREDIT": 406597.5,
                    "AMT_ANNUITY": 24700.5,
                    "DAYS_BIRTH": -9461,
                    "DAYS_EMPLOYED": -637,
                    "CODE_GENDER": 1,
                    "NAME_EDUCATION_TYPE": 4
                }
            }
        }

class PredictionResponse(BaseModel):
    default_probability: float
    risk_label: str
    threshold_used: float
    features_received: int

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "LightGBM Loan Risk",
        "n_features": len(FEATURE_NAMES)
    }

@app.get("/features")
def get_features():
    """Returns the list of expected feature names."""
    return {"features": FEATURE_NAMES, "count": len(FEATURE_NAMES)}

@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    start = time.time()
    try:
        # Build a row with all features, defaulting missing ones to 0
        row = {feat: 0 for feat in FEATURE_NAMES}
        row.update(application.features)

        # Validate no extra unknown features
        unknown = set(application.features.keys()) - set(FEATURE_NAMES)
        if unknown:
            logger.warning(f"Unknown features ignored: {unknown}")

        df = pd.DataFrame([row])[FEATURE_NAMES]  # enforce column order
        proba = model.predict_proba(df)[0][1]
        label = "HIGH RISK" if proba >= 0.5 else "LOW RISK"

        logger.info(f"Prediction: prob={proba:.4f}, label={label}, features={len(application.features)}")
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()

        return PredictionResponse(
            default_probability=round(float(proba), 4),
            risk_label=label,
            threshold_used=0.5,
            features_received=len(application.features)
        )
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)