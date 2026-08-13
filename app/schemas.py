from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str
    model_loaded: bool
    model_dir: str
    model_error: str | None = None


class RunSummary(BaseModel):
    id: str
    name: str
    source_path: str
    validation_f1: float | None = None
    validation_loss: float | None = None
    test_f1: float | None = None
    test_loss: float | None = None


class RunsResponse(BaseModel):
    runs: list[RunSummary]


class RunMetricsResponse(BaseModel):
    run: RunSummary
    validation: dict[str, Any]
    test: dict[str, Any]
    config: dict[str, Any]


class HistoryPoint(BaseModel):
    x: float
    y: float


class RunHistoryResponse(BaseModel):
    run_id: str
    training_loss: list[HistoryPoint]
    validation_loss: list[HistoryPoint]
    validation_f1: list[HistoryPoint]


class RunArtifactsResponse(BaseModel):
    run_id: str
    artifacts: list[str]


class PredictRequest(BaseModel):
    question: str = Field(min_length=1, max_length=3000)
    context: str = Field(min_length=1, max_length=50000)


class PredictStats(BaseModel):
    input_tokens: int
    kept_tokens: int
    compression_ratio: float


class PredictResponse(BaseModel):
    compressed_text: str
    kept_tokens: list[str]
    stats: PredictStats
