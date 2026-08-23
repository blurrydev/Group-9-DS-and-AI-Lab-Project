"""Inference wrappers for query-aware Hindi context compression.

Supports both local token classification models and remote deployed endpoints
(e.g., Hugging Face Spaces via Gradio Client, generic HTTP REST endpoints).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompressionResult:
    text: str
    retained_tokens: int
    input_tokens: int
    retention_ratio: float
    used_fallback: bool


class BaseCompressor(Protocol):
    """Protocol defining the interface for context compressors."""

    def compress(self, question: str, context: str, minimum_retained: int = 8) -> CompressionResult:
        ...


class QueryAwareCompressor:
    """Local inference using fine-tuned query-aware XLM-R token classification."""

    def __init__(self, checkpoint: str | Path, max_length: int = 512, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.checkpoint = str(checkpoint)
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(self.checkpoint).to(self.device)
        self.model.eval()
        if self.model.config.num_labels != 2:
            raise ValueError(f"Expected a binary token classifier, found {self.model.config.num_labels} labels")

    def compress(self, question: str, context: str, minimum_retained: int = 8) -> CompressionResult:
        import torch

        context_words = context.split()
        if not context_words:
            return CompressionResult("", 0, 0, 0.0, False)

        with torch.inference_mode():
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

        fallback = len(selected) < minimum_retained
        text = context if fallback else " ".join(selected)
        retained = len(context_words) if fallback else len(selected)
        return CompressionResult(
            text=text,
            retained_tokens=retained,
            input_tokens=len(context_words),
            retention_ratio=retained / max(len(context_words), 1),
            used_fallback=fallback,
        )


class HFSpaceCompressor:
    """Compressor that queries a deployed Hugging Face Space endpoint via Gradio Client."""

    def __init__(
        self,
        space_id_or_url: str,
        hf_token: str | None = None,
        api_name: str = "/compress",
        timeout: float = 30.0,
    ) -> None:
        from gradio_client import Client

        self.space_id_or_url = space_id_or_url
        self.hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        self.api_name = api_name
        self.timeout = timeout
        logger.info("Initializing Gradio Client for Hugging Face Space: %s", space_id_or_url)
        self.client = Client(self.space_id_or_url, token=self.hf_token)

    def compress(self, question: str, context: str, minimum_retained: int = 8) -> CompressionResult:
        context_words = context.split()
        if not context_words:
            return CompressionResult("", 0, 0, 0.0, False)

        try:
            # Gradio Client supports positional and keyword arguments
            try:
                result = self.client.predict(
                    question=question,
                    context=context,
                    api_name=self.api_name,
                )
            except Exception:
                # Fallback to positional invocation
                result = self.client.predict(
                    question,
                    context,
                    api_name=self.api_name,
                )

            if isinstance(result, (list, tuple)):
                compressed_text = str(result[0]).strip()
            elif isinstance(result, dict):
                compressed_text = str(result.get("compressed_context") or result.get("compressed_text", "")).strip()
            else:
                compressed_text = str(result).strip()

            retained_words = compressed_text.split()
            fallback = len(retained_words) < minimum_retained
            text = context if fallback else compressed_text
            retained = len(context_words) if fallback else len(retained_words)

            return CompressionResult(
                text=text,
                retained_tokens=retained,
                input_tokens=len(context_words),
                retention_ratio=retained / max(len(context_words), 1),
                used_fallback=fallback,
            )
        except Exception as exc:
            logger.warning("Remote compression failed (%s). Falling back to uncompressed context.", exc)
            return CompressionResult(
                text=context,
                retained_tokens=len(context_words),
                input_tokens=len(context_words),
                retention_ratio=1.0,
                used_fallback=True,
            )


class HTTPCompressor:
    """Compressor that sends requests to a generic REST endpoint."""

    def __init__(self, endpoint_url: str, headers: dict[str, str] | None = None, timeout: float = 30.0) -> None:
        import httpx

        self.endpoint_url = endpoint_url
        self.headers = headers or {}
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def compress(self, question: str, context: str, minimum_retained: int = 8) -> CompressionResult:
        context_words = context.split()
        if not context_words:
            return CompressionResult("", 0, 0, 0.0, False)

        try:
            response = self.client.post(
                self.endpoint_url,
                json={"question": question, "context": context},
                headers=self.headers,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                compressed_text = str(data.get("compressed_text") or data.get("compressed_context") or data.get("text", "")).strip()
            else:
                compressed_text = str(data).strip()

            retained_words = compressed_text.split()
            fallback = len(retained_words) < minimum_retained
            text = context if fallback else compressed_text
            retained = len(context_words) if fallback else len(retained_words)

            return CompressionResult(
                text=text,
                retained_tokens=retained,
                input_tokens=len(context_words),
                retention_ratio=retained / max(len(context_words), 1),
                used_fallback=fallback,
            )
        except Exception as exc:
            logger.warning("HTTP compression request failed (%s). Falling back to uncompressed context.", exc)
            return CompressionResult(
                text=context,
                retained_tokens=len(context_words),
                input_tokens=len(context_words),
                retention_ratio=1.0,
                used_fallback=True,
            )


DEFAULT_COMPRESSOR_ENDPOINT = "nnnhitesh/TokenCompressor"


def create_compressor(
    compressor_type: str | None = None,
    checkpoint: str | Path | None = None,
    endpoint: str | None = None,
    hf_token: str | None = None,
    device: str | None = None,
    max_length: int = 512,
) -> BaseCompressor:
    """Factory function to build a local or remote compressor based on config or env."""
    kind = (compressor_type or os.getenv("RAG_COMPRESSOR_TYPE", "")).strip().lower()
    endpoint_target = endpoint or os.getenv("RAG_COMPRESSOR_ENDPOINT") or os.getenv("RAG_HF_SPACE")
    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("RAG_HF_TOKEN")

    if kind == "local":
        local_checkpoint = checkpoint or os.getenv("RAG_CHECKPOINT_PATH")
        if not local_checkpoint or (isinstance(local_checkpoint, (str, Path)) and not Path(local_checkpoint).exists() and not str(local_checkpoint).startswith("nnnhitesh/")):
            local_checkpoint = "nnnhitesh/xlm-roberta-prompt-compressor"
        return QueryAwareCompressor(local_checkpoint, max_length=max_length, device=device)

    endpoint_target = endpoint_target or DEFAULT_COMPRESSOR_ENDPOINT
    if endpoint_target.startswith("http://") or endpoint_target.startswith("https://"):
        if ".hf.space" in endpoint_target or "huggingface.co/spaces/" in endpoint_target:
            return HFSpaceCompressor(endpoint_target, hf_token=token)
        return HTTPCompressor(endpoint_target)
    return HFSpaceCompressor(endpoint_target, hf_token=token)


