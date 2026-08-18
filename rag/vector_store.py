"""FAISS-backed multilingual semantic retrieval for the Hindi RAG corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.retriever import RetrievedChunk


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _dependencies():
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "FAISS retrieval requires faiss-cpu and sentence-transformers. "
            "Install project dependencies with `uv sync`."
        ) from error
    return faiss, SentenceTransformer


@dataclass(frozen=True)
class FaissIndexPaths:
    index: Path
    metadata: Path


class FAISSRetriever:
    """Cosine-similarity retrieval over normalized multilingual embeddings."""

    def __init__(self, index: object, chunks: list[dict[str, object]], model: object) -> None:
        if not chunks:
            raise ValueError("The FAISS metadata is empty.")
        self.index = index
        self.chunks = chunks
        self.model = model

    @staticmethod
    def paths(directory: str | Path) -> FaissIndexPaths:
        root = Path(directory)
        return FaissIndexPaths(index=root / "chunks.faiss", metadata=root / "chunks.jsonl")

    @classmethod
    def build(
        cls,
        chunks_path: str | Path,
        index_directory: str | Path,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 32,
    ) -> int:
        faiss, SentenceTransformer = _dependencies()
        with Path(chunks_path).open(encoding="utf-8") as handle:
            chunks = [json.loads(line) for line in handle if line.strip()]
        if not chunks:
            raise ValueError(f"Chunk corpus is empty: {chunks_path}")
        model = SentenceTransformer(model_name)
        vectors = model.encode(
            [str(chunk["text"]) for chunk in chunks],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        paths = cls.paths(index_directory)
        paths.index.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(paths.index))
        with paths.metadata.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        return len(chunks)

    @classmethod
    def load(cls, index_directory: str | Path, model_name: str = DEFAULT_EMBEDDING_MODEL) -> "FAISSRetriever":
        faiss, SentenceTransformer = _dependencies()
        paths = cls.paths(index_directory)
        if not paths.index.is_file() or not paths.metadata.is_file():
            raise FileNotFoundError(
                f"FAISS index is missing in {paths.index.parent}. Build it with "
                "`python build_index.py --chunks corpus/processed/chunks.jsonl`."
            )
        with paths.metadata.open(encoding="utf-8") as handle:
            chunks = [json.loads(line) for line in handle if line.strip()]
        index = faiss.read_index(str(paths.index))
        if index.ntotal != len(chunks):
            raise ValueError("FAISS index and metadata have different chunk counts. Rebuild the index.")
        return cls(index=index, chunks=chunks, model=SentenceTransformer(model_name))

    def search(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        vector = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, indexes = self.index.search(vector, min(top_k, len(self.chunks)))
        return [
            RetrievedChunk(record=self.chunks[index], score=float(score))
            for score, index in zip(scores[0], indexes[0])
            if index >= 0
        ]
