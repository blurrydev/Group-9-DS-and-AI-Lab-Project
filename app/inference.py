from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from rag.compressor import BaseCompressor, create_compressor


@dataclass
class InferenceState:
    ready: bool
    error: str | None


class InferenceService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.compressor: BaseCompressor | None = None
        self.state = InferenceState(ready=False, error=None)

    def load(self) -> None:
        try:
            self.compressor = create_compressor()
            self.state = InferenceState(ready=True, error=None)
        except Exception as exc:
            self.state = InferenceState(ready=False, error=str(exc))

    def predict(self, question: str, context: str) -> dict:
        if not self.state.ready or self.compressor is None:
            raise RuntimeError(self.state.error or "Compressor is not available")

        result = self.compressor.compress(question, context)
        context_tokens = context.split()
        retained_tokens = result.text.split() if result.text else []
        total = len(context_tokens)
        kept = result.retained_tokens if result.retained_tokens > 0 else len(retained_tokens)
        compression_ratio = 0.0 if total == 0 else max(0.0, 1.0 - (kept / total))

        return {
            "compressed_text": result.text,
            "kept_tokens": retained_tokens,
            "stats": {
                "input_tokens": total,
                "kept_tokens": kept,
                "compression_ratio": compression_ratio,
            },
        }
