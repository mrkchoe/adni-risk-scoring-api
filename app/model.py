"""Model loading and prediction utilities."""

from __future__ import annotations

from typing import Dict, List

import joblib
import numpy as np

from app.config import FEATURE_ORDER, MODEL_PATH

_MODEL = None


def load_model():
    """Load the model from disk."""
    return joblib.load(MODEL_PATH)


def get_model():
    """Return cached model, loading if needed."""
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


def build_feature_vector(payload: Dict[str, float], feature_order: List[str]) -> np.ndarray:
    """Build a single-row feature vector in the correct order."""
    return np.array([[payload[name] for name in feature_order]], dtype=float)


def predict_risk(payload: Dict[str, float]) -> float:
    """Predict risk score using the cached model."""
    model = get_model()
    features = build_feature_vector(payload, FEATURE_ORDER)
    proba = model.predict_proba(features)[0, 1]
    return float(proba)
