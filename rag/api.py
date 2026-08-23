"""FastAPI service for the local Hindi RAG retrieval-and-compression pipeline."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Query, Request

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.compressor import BaseCompressor, CompressionResult, create_compressor
from rag.generator import BaseGenerator, create_generator
from rag.prompts import answer_prompt
from rag.vector_store import DEFAULT_EMBEDDING_MODEL, FAISSRetriever


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1_000, description="Hindi or multilingual user question.")
    top_k: int = Field(default=3, ge=1, le=10)
    minimum_retained: int = Field(default=8, ge=0, le=100)
    generate: bool = Field(default=True, description="Whether to generate the final LLM answer using the compressed prompt.")


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
    answer: str | None = Field(default=None, description="Generated natural language Hindi answer.")


def settings() -> dict[str, object]:
    """Read deployment paths and model settings from the environment."""
    return {
        "chunks_path": Path(os.getenv("RAG_CHUNKS_PATH", "corpus/processed/chunks.jsonl")),
        "checkpoint_path": Path(os.getenv("RAG_CHECKPOINT_PATH", "checkpoints/final-compressor")),
        "index_path": Path(os.getenv("RAG_FAISS_INDEX_PATH", "corpus/index")),
        "embedding_model": os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "device": os.getenv("RAG_DEVICE"),
        "compressor_type": os.getenv("RAG_COMPRESSOR_TYPE", "remote"),
        "compressor_endpoint": os.getenv("RAG_COMPRESSOR_ENDPOINT") or os.getenv("RAG_HF_SPACE") or "nnnhitesh/TokenCompressor",
        "hf_token": os.getenv("HF_TOKEN") or os.getenv("RAG_HF_TOKEN"),
        "generator_provider": os.getenv("RAG_GENERATOR_PROVIDER"),
        "generator_model": os.getenv("RAG_GENERATOR_MODEL"),
        "generator_api_key": os.getenv("RAG_GENERATOR_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"),
        "generator_base_url": os.getenv("RAG_GENERATOR_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL"),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = settings()
    index_path = cfg["index_path"]
    embedding_model = str(cfg["embedding_model"])

    app.state.retriever = FAISSRetriever.load(index_path, embedding_model)
    app.state.compressor = create_compressor(
        compressor_type=cfg["compressor_type"],
        checkpoint=cfg["checkpoint_path"],
        endpoint=cfg["compressor_endpoint"],
        hf_token=cfg["hf_token"],
        device=cfg["device"],
    )
    app.state.generator = create_generator(
        api_key=cfg["generator_api_key"],
        base_url=cfg["generator_base_url"],
        model=cfg["generator_model"],
        provider=cfg["generator_provider"],
        allow_mock_fallback=False,
    )
    yield


app = FastAPI(
    title="Indic-LLMLingua Hindi RAG API",
    version="0.1.0",
    description="Retrieves, query-aware compresses, and answers Hindi RAG questions.",
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
    compressor = getattr(request.app.state, "compressor", None)
    compressor_type = type(compressor).__name__ if compressor else "unknown"
    generator = getattr(request.app.state, "generator", None)
    generator_type = type(generator).__name__ if generator else "none"
    return {
        "status": "ok",
        "retriever": "faiss",
        "chunks_loaded": len(request.app.state.retriever.chunks),
        "compressor": compressor_type,
        "generator": generator_type,
    }


@app.post("/v1/rag/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest, request: Request) -> QueryResponse:
    """Retrieve documents, compress them, and optionally generate an answer."""
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

    prompt = answer_prompt(payload.question, generator_contexts)
    answer = None

    if payload.generate and getattr(request.app.state, "generator", None) is not None:
        try:
            answer = request.app.state.generator.generate(prompt)
        except Exception as exc:
            answer = f"Error generating answer: {exc}"

    return QueryResponse(
        question=payload.question,
        contexts=contexts,
        prompt=prompt,
        answer=answer,
    )
