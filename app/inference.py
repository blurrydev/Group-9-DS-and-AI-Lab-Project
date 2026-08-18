from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from app.config import Settings


@dataclass
class InferenceState:
    ready: bool
    error: str | None


class InferenceService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tokenizer = None
        self.model = None
        self.device = torch.device(settings.device)
        self.state = InferenceState(ready=False, error=None)

    def load(self) -> None:
        try:
            model_dir = self.settings.model_dir
            if not model_dir.exists():
                self.state = InferenceState(
                    ready=False,
                    error=(
                        f"MODEL_DIR does not exist: {model_dir}. "
                        "Add a fine-tuned model artifact or change MODEL_DIR."
                    ),
                )
                return

            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            self.model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
            self.model.to(self.device)
            self.model.eval()
            self.state = InferenceState(ready=True, error=None)
        except Exception as exc:  # pragma: no cover - defensive branch for runtime env issues
            self.state = InferenceState(ready=False, error=str(exc))

    def predict(self, question: str, context: str) -> dict:
        if not self.state.ready or self.model is None or self.tokenizer is None:
            raise RuntimeError(self.state.error or "Model is not available")

        context_tokens = context.split()
        encoding = self.tokenizer(
            question.split(),
            context_tokens,
            is_split_into_words=True,
            truncation="only_second",
            max_length=self.settings.max_length,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        sequence_ids = encoding.sequence_ids(batch_index=0)
        encoding = {key: value.to(self.device) for key, value in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)

        pred_labels = outputs.logits.argmax(dim=-1).squeeze().cpu().tolist()

        kept_tokens: list[str] = []
        previous_word = None
        for pred, word_id, seq_id in zip(pred_labels, word_ids, sequence_ids):
            if word_id is None or seq_id != 1:
                continue
            if word_id == previous_word:
                continue
            if pred == 1 and word_id < len(context_tokens):
                kept_tokens.append(context_tokens[word_id])
            previous_word = word_id

        compressed_text = " ".join(kept_tokens)
        total = len(context_tokens)
        kept = len(kept_tokens)
        compression_ratio = 0.0 if total == 0 else (1.0 - kept / total)

        return {
            "compressed_text": compressed_text,
            "kept_tokens": kept_tokens,
            "stats": {
                "input_tokens": total,
                "kept_tokens": kept,
                "compression_ratio": compression_ratio,
            },
        }
