from __future__ import annotations

import httpx

from app.config import MODEL_PATH
from app.main import app
from scripts.train_synthetic_model import train_and_save


def get_client() -> httpx.Client:
    transport = httpx.ASGITransport(app=app)
    return httpx.Client(transport=transport, base_url="http://test")


def test_health() -> None:
    if not MODEL_PATH.exists():
        train_and_save()

    with get_client() as client:
        response = client.get("/health")

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
