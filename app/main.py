"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app import model as model_service
from app.config import FEATURE_ORDER, MODEL_TYPE, MODEL_VERSION
from app.schemas import PredictRequest, PredictResponse

app = FastAPI(title="ADNI Risk Scoring API", version=MODEL_VERSION)


def risk_label(score: float) -> str:
    if score < 0.33:
        return "low"
    if score <= 0.66:
        return "medium"
    return "high"


@app.get("/health")
def health():
    try:
        model_service.get_model()
        model_loaded = True
    except Exception:
        model_loaded = False

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        score = model_service.predict_risk(request.model_dump())
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train the toy model first.",
        )

    return PredictResponse(
        risk_score=score,
        risk_label=risk_label(score),
        model_version=MODEL_VERSION,
        model_type=MODEL_TYPE,
        features_used=FEATURE_ORDER,
    )
