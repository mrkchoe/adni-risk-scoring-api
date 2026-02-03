from __future__ import annotations

import httpx

from app.config import MODEL_PATH
from app.main import app
from scripts.train_synthetic_model import train_and_save


def get_client() -> httpx.Client:
    transport = httpx.ASGITransport(app=app)
    return httpx.Client(transport=transport, base_url="http://test")


def ensure_model() -> None:
    if not MODEL_PATH.exists():
        train_and_save()


def test_predict_valid() -> None:
    ensure_model()

    payload = {
        "age": 72,
        "education_years": 16,
        "mmse_score": 26.5,
        "cdr_global": 0.5,
        "adas13": 12.3,
        "hippocampal_volume": 5200.0,
    }

    with get_client() as client:
        response = client.post("/predict", json=payload)

    data = response.json()
    assert response.status_code == 200
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_label"] in {"low", "medium", "high"}
    assert isinstance(data["features_used"], list)


def test_predict_invalid() -> None:
    ensure_model()

    payload = {
        "age": 10,
        "education_years": 16,
        "mmse_score": 26.5,
        "cdr_global": 0.5,
        "adas13": 12.3,
        "hippocampal_volume": 5200.0,
    }

    with get_client() as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
