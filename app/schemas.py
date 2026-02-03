"""Pydantic schemas for API input/output."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=40, le=100, description="Age in years (40-100).")
    education_years: int = Field(
        ..., ge=0, le=30, description="Years of education completed (0-30)."
    )
    mmse_score: float = Field(
        ..., ge=0.0, le=30.0, description="Mini-Mental State Exam score (0-30)."
    )
    cdr_global: float = Field(
        ..., ge=0.0, le=3.0, description="Clinical Dementia Rating global score (0-3)."
    )
    adas13: float = Field(
        ..., ge=0.0, le=85.0, description="ADAS-Cog 13 total score (0-85)."
    )
    hippocampal_volume: float = Field(
        ..., gt=0.0, description="Hippocampal volume (>0), pre-extracted feature."
    )


class PredictResponse(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_label: Literal["low", "medium", "high"]
    model_version: str
    model_type: str
    features_used: List[str]
