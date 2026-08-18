"""A dependency-free BM25 retriever for an attributed JSONL chunk corpus."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)


def terms(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class RetrievedChunk:
    record: dict[str, object]
    score: float


class BM25Retriever:
    """In-memory BM25 suitable for a small project corpus (thousands of chunks)."""

    def __init__(self, chunks: list[dict[str, object]], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("The chunk corpus is empty. Build or provide chunks.jsonl first.")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(terms(str(chunk["text"]))) for chunk in chunks]
        self.lengths = [sum(frequencies.values()) for frequencies in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths)
        self.document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            self.document_frequency.update(frequencies.keys())

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "BM25Retriever":
        corpus_path = Path(path)
        with corpus_path.open(encoding="utf-8") as handle:
            chunks = [json.loads(line) for line in handle if line.strip()]
        return cls(chunks)

    def search(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_terms = set(terms(question))
        count = len(self.chunks)
        scored: list[RetrievedChunk] = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                inverse_frequency = math.log(1 + (count - self.document_frequency[term] + 0.5) / (self.document_frequency[term] + 0.5))
                denominator = term_frequency + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                score += inverse_frequency * (term_frequency * (self.k1 + 1) / denominator)
            if score > 0:
                scored.append(RetrievedChunk(self.chunks[index], score))
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]
