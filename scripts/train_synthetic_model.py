"""Train and persist a synthetic logistic regression model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from app.config import FEATURE_ORDER, MODEL_PATH


@dataclass(frozen=True)
class SyntheticDataConfig:
    n_samples: int = 1200
    seed: int = 42


def generate_synthetic_data(
    config: SyntheticDataConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)

    age = rng.integers(40, 101, size=config.n_samples)
    education_years = rng.integers(0, 31, size=config.n_samples)
    mmse_score = rng.uniform(0, 30, size=config.n_samples)
    cdr_global = rng.uniform(0, 3, size=config.n_samples)
    adas13 = rng.uniform(0, 85, size=config.n_samples)
    hippocampal_volume = rng.uniform(2500, 9500, size=config.n_samples)

    X = np.column_stack(
        [
            age,
            education_years,
            mmse_score,
            cdr_global,
            adas13,
            hippocampal_volume,
        ]
    ).astype(float)

    linear = (
        0.04 * (age - 70)
        - 0.03 * (education_years - 12)
        - 0.08 * (mmse_score - 24)
        + 0.6 * cdr_global
        + 0.02 * adas13
        - 0.0004 * (hippocampal_volume - 6000)
        + rng.normal(0, 0.5, size=config.n_samples)
    )
    probabilities = 1 / (1 + np.exp(-linear))
    y = rng.binomial(1, probabilities, size=config.n_samples)

    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=500, random_state=7)
    model.fit(X, y)
    return model


def save_model(model: LogisticRegression, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def train_and_save(
    config: SyntheticDataConfig = SyntheticDataConfig(),
    model_path: Path = MODEL_PATH,
) -> Path:
    X, y = generate_synthetic_data(config)
    model = train_model(X, y)
    save_model(model, model_path)
    return model_path


def main() -> None:
    model_path = train_and_save()
    print(f"Saved synthetic model to {model_path}")
    print(f"Feature order: {FEATURE_ORDER}")


if __name__ == "__main__":
    main()
