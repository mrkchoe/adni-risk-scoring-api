"""Configuration constants for model inference."""

from __future__ import annotations

from pathlib import Path

MODEL_VERSION = "toy-1.0.0"
MODEL_TYPE = "sklearn-logistic-regression"

FEATURE_ORDER = [
    "age",
    "education_years",
    "mmse_score",
    "cdr_global",
    "adas13",
    "hippocampal_volume",
]

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "toy" / "model.joblib"
