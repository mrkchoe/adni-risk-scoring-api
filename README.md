# adni-risk-scoring-api

Research-grade ML inference API for synthetic Alzheimer’s risk scoring.
No ADNI data is included. The model artifacts are trained only on synthetic data.

## Overview
- Minimal, production-style FastAPI service for risk scoring.
- Toy logistic regression model trained on synthetic clinical features.
- Pydantic v2 validation with strict schemas.
- Tests using pytest + httpx.

## Quickstart

Create and activate a virtual environment:
```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

Train the synthetic model:
```
python scripts/train_synthetic_model.py
```

Start the API server:
```
uvicorn app.main:app --reload
```

## API Usage

Health check:
```
curl http://127.0.0.1:8000/health
```

Predict example:
```
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 72,
    "education_years": 16,
    "mmse_score": 26.5,
    "cdr_global": 0.5,
    "adas13": 12.3,
    "hippocampal_volume": 5200.0
  }'
```

Example response:
```
{
  "risk_score": 0.42,
  "risk_label": "medium",
  "model_version": "toy-1.0.0",
  "model_type": "sklearn-logistic-regression",
  "features_used": [
    "age",
    "education_years",
    "mmse_score",
    "cdr_global",
    "adas13",
    "hippocampal_volume"
  ]
}
```

## Tests
```
pytest
```

## Compliance Note
This repository is safe for public release. It does not include any ADNI data
or ADNI-trained weights. The toy model is trained exclusively on synthetic data.

## Regenerating the Toy Model
```
python scripts/train_synthetic_model.py
```
