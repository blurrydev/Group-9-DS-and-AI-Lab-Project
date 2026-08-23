from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import artifact_store
from app.config import ROOT_DIR, settings
from app.inference import InferenceService
from app.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RunArtifactsResponse,
    RunHistoryResponse,
    RunMetricsResponse,
    RunsResponse,
)


app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_service = InferenceService(settings)


@app.on_event("startup")
def load_model_on_startup() -> None:
    inference_service.load()


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        environment=settings.app_env,
        model_loaded=inference_service.state.ready,
        model_dir=str(settings.model_dir),
        model_error=inference_service.state.error,
    )


@app.get("/api/runs", response_model=RunsResponse)
def runs() -> RunsResponse:
    return RunsResponse(runs=artifact_store.list_runs())


@app.get("/api/runs/{run_id}/metrics", response_model=RunMetricsResponse)
def run_metrics(run_id: str) -> RunMetricsResponse:
    try:
        data = artifact_store.get_run_metrics(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RunMetricsResponse(**data)


@app.get("/api/runs/{run_id}/history", response_model=RunHistoryResponse)
def run_history(run_id: str) -> RunHistoryResponse:
    try:
        data = artifact_store.get_run_history(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RunHistoryResponse(**data)


@app.get("/api/runs/{run_id}/artifacts", response_model=RunArtifactsResponse)
def run_artifacts(run_id: str) -> RunArtifactsResponse:
    try:
        artifacts = artifact_store.list_artifacts(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RunArtifactsResponse(run_id=run_id, artifacts=artifacts)


@app.get("/api/runs/{run_id}/artifacts/{artifact_name}")
def artifact_file(run_id: str, artifact_name: str) -> FileResponse:
    try:
        file_path = artifact_store.get_artifact_path(run_id, artifact_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(path=file_path)


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        prediction = inference_service.predict(
            question=payload.question,
            context=payload.context,
            generate_answer=payload.generate_answer,
            retrieve_rag=payload.retrieve_rag,
            top_k=payload.top_k,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400 if isinstance(exc, ValueError) else 503, detail=str(exc)) from exc
    return PredictResponse(**prediction)



frontend_dir = ROOT_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
def index() -> FileResponse:
    page = frontend_dir / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(path=page)


@app.get("/{path_name:path}")
def static_fallback(path_name: str) -> FileResponse:
    candidate = (frontend_dir / path_name).resolve()
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if frontend_dir.resolve() not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid file path")
    return FileResponse(path=candidate)
