"""Command-line entry point for query-aware Hindi RAG retrieval, compression, and answering."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag.compressor import DEFAULT_COMPRESSOR_ENDPOINT, create_compressor
from rag.generator import create_generator

from rag.prompts import answer_prompt
from rag.retriever import BM25Retriever
from rag.vector_store import DEFAULT_EMBEDDING_MODEL, FAISSRetriever


def ask(args: argparse.Namespace) -> None:
    if args.retriever == "faiss":
        retriever = FAISSRetriever.load(args.index_dir, args.embedding_model)
    else:
        retriever = BM25Retriever.from_jsonl(args.chunks)
    retrieved = retriever.search(args.question, args.top_k)
    if not retrieved:
        raise SystemExit("No matching chunks were retrieved. Add relevant source documents to the corpus.")

    compressor = create_compressor(
        compressor_type=args.compressor_type,
        checkpoint=args.checkpoint,
        endpoint=args.endpoint,
        hf_token=args.hf_token,
        max_length=args.max_length,
        device=args.device,
    )
    contexts = []
    for result in retrieved:
        record = result.record
        compressed = compressor.compress(args.question, str(record["text"]), args.minimum_retained)
        contexts.append(
            {
                "chunk_id": str(record.get("chunk_id", record.get("document_id", "unknown"))),
                "title": str(record.get("title", "Untitled source")),
                "source_url": str(record.get("source_url", "Unknown source")),
                "retrieval_score": round(result.score, 4),
                "compression": {
                    "text": compressed.text,
                    "retained_tokens": compressed.retained_tokens,
                    "input_tokens": compressed.input_tokens,
                    "retention_ratio": round(compressed.retention_ratio, 4),
                    "used_fallback": compressed.used_fallback,
                },
            }
        )
    generator_contexts = [{**item, "text": item["compression"]["text"]} for item in contexts]
    prompt = answer_prompt(args.question, generator_contexts)

    output: dict[str, object] = {
        "question": args.question,
        "contexts": contexts,
        "prompt": prompt,
    }

    if args.generate:
        generator = create_generator(
            api_key=args.generator_api_key,
            base_url=args.generator_base_url,
            model=args.generator_model,
            provider=args.generator_provider,
            allow_mock_fallback=args.allow_mock_generator,
        )
        if generator is None:
            output["answer"] = (
                "Error: No generator configured or missing API key. Set OPENAI_API_KEY, API_KEY, "
                "or RAG_GENERATOR_API_KEY, or pass --generator-api-key."
            )
        else:
            try:
                output["answer"] = generator.generate(prompt)
            except Exception as exc:
                output["answer"] = f"Error generating answer: {exc}"

    print(json.dumps(output, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    ask_parser = subparsers.add_parser("ask", help="Retrieve, compress, and optionally answer a Hindi question.")
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--chunks", type=Path, default=Path("corpus/processed/chunks.jsonl"))
    ask_parser.add_argument("--retriever", choices=("faiss", "bm25"), default="faiss")
    ask_parser.add_argument("--index-dir", type=Path, default=Path("corpus/index"))
    ask_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ask_parser.add_argument("--compressor-type", choices=("local", "remote", "hf_space"), default=os.getenv("RAG_COMPRESSOR_TYPE", "remote"))
    ask_parser.add_argument("--endpoint", default=DEFAULT_COMPRESSOR_ENDPOINT, help="Remote endpoint URL or Hugging Face Space ID.")
    ask_parser.add_argument("--hf-token", default=None, help="Hugging Face API token for authenticated Spaces.")
    ask_parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/final-compressor"))
    ask_parser.add_argument("--top-k", type=int, default=3)
    ask_parser.add_argument("--max-length", type=int, default=512)
    ask_parser.add_argument("--minimum-retained", type=int, default=8)
    ask_parser.add_argument("--device", choices=("cpu", "cuda"))
    ask_parser.add_argument("--generate", action="store_true", help="Generate final natural language answer using an LLM.")
    ask_parser.add_argument("--generator-provider", default=os.getenv("RAG_GENERATOR_PROVIDER"), help="Generator provider (openai, gemini, groq, mock).")
    ask_parser.add_argument("--generator-model", default=os.getenv("RAG_GENERATOR_MODEL"), help="Model name for generator LLM.")
    ask_parser.add_argument("--generator-api-key", default=None, help="API key for the generator LLM.")
    ask_parser.add_argument("--generator-base-url", default=None, help="Base URL for OpenAI-compatible endpoint.")
    ask_parser.add_argument("--allow-mock-generator", action="store_true", help="Fallback to mock generator if no API key is provided.")

    ask_parser.set_defaults(func=ask)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

