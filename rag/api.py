"""FastAPI service for the local Hindi RAG retrieval-and-compression pipeline."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.compressor import QueryAwareCompressor
from rag.prompts import answer_prompt
from rag.vector_store import DEFAULT_EMBEDDING_MODEL, FAISSRetriever


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1_000, description="Hindi or multilingual user question.")
    top_k: int = Field(default=3, ge=1, le=10)
    minimum_retained: int = Field(default=8, ge=0, le=100)


class CompressionResponse(BaseModel):
    text: str
    retained_tokens: int
    input_tokens: int
    retention_ratio: float
    used_fallback: bool


class ContextResponse(BaseModel):
    chunk_id: str
    title: str
    source_url: str
    retrieval_score: float
    compression: CompressionResponse


class QueryResponse(BaseModel):
    question: str
    contexts: list[ContextResponse]
    prompt: str


def settings() -> tuple[Path, Path, Path, str, str | None]:
    """Read deployment paths from the environment, with project-local defaults."""
    return (
        Path(os.getenv("RAG_CHUNKS_PATH", "corpus/processed/chunks.jsonl")),
        Path(os.getenv("RAG_CHECKPOINT_PATH", "checkpoints/final-compressor")),
        Path(os.getenv("RAG_FAISS_INDEX_PATH", "corpus/index")),
        os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        os.getenv("RAG_DEVICE"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    chunks_path, checkpoint_path, index_path, embedding_model, device = settings()
    if not checkpoint_path.is_dir():
        raise RuntimeError(f"RAG model checkpoint not found: {checkpoint_path}")
    app.state.retriever = FAISSRetriever.load(index_path, embedding_model)
    app.state.compressor = QueryAwareCompressor(checkpoint_path, device=device)
    yield


app = FastAPI(
    title="Indic-LLMLingua Hindi RAG API",
    version="0.1.0",
    description="Retrieves and query-aware compresses Hindi RAG context.",
    lifespan=lifespan,
)

allowed_origins = [origin.strip() for origin in os.getenv("RAG_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    return {"status": "ok", "retriever": "faiss", "chunks_loaded": len(request.app.state.retriever.chunks)}


@app.post("/v1/rag/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest, request: Request) -> QueryResponse:
    """Retrieve documents and compress them for a frontend or generator service."""
    results = request.app.state.retriever.search(payload.question, payload.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No relevant context was found in the knowledge base.")

    contexts: list[ContextResponse] = []
    generator_contexts: list[dict[str, str]] = []
    for result in results:
        record = result.record
        compressed = request.app.state.compressor.compress(
            payload.question, str(record["text"]), payload.minimum_retained
        )
        context = ContextResponse(
            chunk_id=str(record.get("chunk_id", record.get("document_id", "unknown"))),
            title=str(record.get("title", "Untitled source")),
            source_url=str(record.get("source_url", "Unknown source")),
            retrieval_score=round(result.score, 4),
            compression=CompressionResponse(**asdict(compressed)),
        )
        contexts.append(context)
        generator_contexts.append({"title": context.title, "source_url": context.source_url, "text": context.compression.text})
    return QueryResponse(question=payload.question, contexts=contexts, prompt=answer_prompt(payload.question, generator_contexts))
