from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    text: str | None = None
    model_input: str | None = None
    artifact_dir: str | None = None


class PredictResponse(BaseModel):
    prediction: str
    label_id: int
    scores: dict[str, float]
    score_type: str
    label_order: list[str]
    model_type: str


class BatchPredictItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    text: str | None = None
    model_input: str | None = None


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[BatchPredictItem] = Field(default_factory=list)
    artifact_dir: str | None = None


class BatchPredictResponse(BaseModel):
    items: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    default_artifact_dir: str