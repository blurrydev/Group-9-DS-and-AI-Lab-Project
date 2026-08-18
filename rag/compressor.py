"""Inference wrapper for the fine-tuned query-aware XLM-R token classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


@dataclass(frozen=True)
class CompressionResult:
    text: str
    retained_tokens: int
    input_tokens: int
    retention_ratio: float
    used_fallback: bool


class QueryAwareCompressor:
    """Keep tokens predicted as label 1, using the training-time pair encoding."""

    def __init__(self, checkpoint: str | Path, max_length: int = 512, device: str | None = None) -> None:
        self.checkpoint = str(checkpoint)
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(self.checkpoint).to(self.device)
        self.model.eval()
        if self.model.config.num_labels != 2:
            raise ValueError(f"Expected a binary token classifier, found {self.model.config.num_labels} labels")

    @torch.inference_mode()
    def compress(self, question: str, context: str, minimum_retained: int = 8) -> CompressionResult:
        context_words = context.split()
        if not context_words:
            return CompressionResult("", 0, 0, 0.0, False)
        encoding = self.tokenizer(
            question.split(),
            context_words,
            is_split_into_words=True,
            truncation="only_second",
            max_length=self.max_length,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)
        sequence_ids = encoding.sequence_ids(batch_index=0)
        model_inputs = {name: value.to(self.device) for name, value in encoding.items()}
        probabilities = self.model(**model_inputs).logits.softmax(dim=-1)[0, :, 1].cpu().tolist()

        selected: list[str] = []
        visited_word_ids: set[int] = set()
        for probability, word_id, sequence_id in zip(probabilities, word_ids, sequence_ids):
            if sequence_id != 1 or word_id is None or word_id in visited_word_ids:
                continue
            visited_word_ids.add(word_id)
            if probability >= 0.5:
                selected.append(context_words[word_id])

        # A highly sparse/noisy prediction is worse than passing a short retrieved
        # passage unchanged. The fallback also avoids empty LLM prompts.
        fallback = len(selected) < minimum_retained
        text = context if fallback else " ".join(selected)
        retained = len(context_words) if fallback else len(selected)
        return CompressionResult(
            text=text,
            retained_tokens=retained,
            input_tokens=len(context_words),
            retention_ratio=retained / len(context_words),
            used_fallback=fallback,
        )
