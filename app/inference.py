from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.config import ROOT_DIR, Settings
from rag.compressor import BaseCompressor, create_compressor
from rag.generator import BaseGenerator, create_generator
from rag.prompts import answer_prompt
from rag.vector_store import FAISSRetriever


@dataclass
class InferenceState:
    ready: bool
    error: str | None


class InferenceService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.compressor: BaseCompressor | None = None
        self.generator: BaseGenerator | None = None
        self.retriever: FAISSRetriever | None = None
        self.state = InferenceState(ready=False, error=None)

    def load(self) -> None:
        try:
            self.compressor = create_compressor()
            self.generator = create_generator(allow_mock_fallback=True)
            index_path = Path(os.getenv("RAG_FAISS_INDEX_PATH", str(ROOT_DIR / "corpus" / "index")))
            if index_path.exists() and (index_path / "chunks.faiss").exists():
                try:
                    self.retriever = FAISSRetriever.load(index_path)
                except Exception:
                    self.retriever = None
            self.state = InferenceState(ready=True, error=None)
        except Exception as exc:
            self.state = InferenceState(ready=False, error=str(exc))

    def predict(
        self,
        question: str,
        context: str | None = None,
        generate_answer: bool = True,
        retrieve_rag: bool = False,
        top_k: int = 3,
    ) -> dict:
        if not self.state.ready or self.compressor is None:
            raise RuntimeError(self.state.error or "Compressor is not available")

        sources: list[dict] = []
        generator_contexts: list[dict[str, str]] = []
        combined_compressed: list[str] = []
        total_input_tokens = 0
        total_kept_tokens = 0

        # Case 1: Retrieve context automatically from RAG knowledge base
        if (retrieve_rag or not context or not context.strip()) and self.retriever is not None:
            retrieved = self.retriever.search(question, top_k=top_k)
            for idx, r in enumerate(retrieved, start=1):
                rec = r.record
                title = str(rec.get("title", f"Source {idx}"))
                url = str(rec.get("source_url", ""))
                raw_text = str(rec.get("text", ""))

                comp = self.compressor.compress(question, raw_text)
                combined_compressed.append(comp.text)
                total_input_tokens += comp.input_tokens
                total_kept_tokens += comp.retained_tokens

                sources.append({
                    "title": title,
                    "source_url": url,
                    "score": round(r.score, 4),
                    "compressed_text": comp.text,
                })
                generator_contexts.append({"title": title, "source_url": url, "text": comp.text})

            compressed_text = "\n\n".join(combined_compressed)

        # Case 2: Use user-provided context
        else:
            raw_text = (context or "").strip()
            if not raw_text:
                raise ValueError("Context cannot be empty unless RAG auto-retrieval is available.")

            comp = self.compressor.compress(question, raw_text)
            compressed_text = comp.text
            total_input_tokens = comp.input_tokens
            total_kept_tokens = comp.retained_tokens
            generator_contexts.append({"title": "User Context", "source_url": "Direct Input", "text": comp.text})

        compression_ratio = 0.0 if total_input_tokens == 0 else max(0.0, 1.0 - (total_kept_tokens / total_input_tokens))
        prompt = answer_prompt(question, generator_contexts)
        answer = None

        if generate_answer and self.generator is not None:
            try:
                answer = self.generator.generate(prompt)
            except Exception as exc:
                answer = f"Error generating answer: {exc}"

        return {
            "compressed_text": compressed_text,
            "kept_tokens": compressed_text.split(),
            "stats": {
                "input_tokens": total_input_tokens,
                "kept_tokens": total_kept_tokens,
                "compression_ratio": compression_ratio,
            },
            "answer": answer,
            "prompt": prompt,
            "sources": sources if sources else None,
        }

