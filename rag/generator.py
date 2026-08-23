"""LLM answer generation for the Hindi RAG pipeline.

Supports any OpenAI-compatible API (e.g., OpenAI, Groq, Gemini via OpenAI endpoint,
Qwen/custom endpoints, local Ollama, vLLM) with automatic fallback and error handling.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)



class BaseGenerator(Protocol):
    """Protocol for downstream answer generators."""

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        ...


class OpenAIGenerator:
    """Answer generator using an OpenAI-compatible chat completion endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        from openai import OpenAI

        self.api_key = api_key or os.getenv("RAG_GENERATOR_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("RAG_GENERATOR_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL")
        self.model = model or os.getenv("RAG_GENERATOR_MODEL") or "qwen/qwen3.5-397b-a17b"
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "Missing API key for generator. Provide `api_key` or set `OPENAI_API_KEY`, `API_KEY`, "
                "or `RAG_GENERATOR_API_KEY` in environment variables."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "आप एक सटीक और सहायक हिंदी सहायक हैं। केवल दिए गए संदर्भ के आधार पर उत्तर दें।",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Generator returned an empty response (possibly blocked by safety filters).")
        return content.strip()


class MockGenerator:
    """Mock generator for testing and offline development."""

    def __init__(self, response_prefix: str = "[उत्तर]: ") -> None:
        self.response_prefix = response_prefix

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        return f"{self.response_prefix}यह संदर्भ के आधार पर उत्पन्न किया गया एक स्वचालित उत्तर है।"


def create_generator(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    allow_mock_fallback: bool = False,
) -> BaseGenerator | None:
    """Factory to create an appropriate generator from arguments or environment variables.

    Supports automatic detection of GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY, etc.
    Returns None if no API key or provider is configured and allow_mock_fallback is False.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    key = (
        api_key
        or os.getenv("RAG_GENERATOR_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or gemini_key
        or groq_key
        or os.getenv("API_KEY")
    )
    url = base_url or os.getenv("RAG_GENERATOR_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL")
    mod = model or os.getenv("RAG_GENERATOR_MODEL")

    selected_provider = (provider or os.getenv("RAG_GENERATOR_PROVIDER", "")).strip().lower()

    if selected_provider == "mock":
        return MockGenerator()

    # Auto-configure for Google Gemini OpenAI-compatible endpoint if Gemini key is present
    if (selected_provider == "gemini" or (key == gemini_key and gemini_key)) and not url:
        url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        mod = mod or "gemini-3.6-flash"



    # Auto-configure for Groq if Groq key is present
    elif (selected_provider == "groq" or (key == groq_key and groq_key)) and not url:
        url = "https://api.groq.com/openai/v1"
        mod = mod or "llama-3.3-70b-versatile"

    if not key:
        if allow_mock_fallback:
            logger.warning("No API key found for generator. Falling back to MockGenerator.")
            return MockGenerator()
        return None

    return OpenAIGenerator(api_key=key, base_url=url, model=mod)

