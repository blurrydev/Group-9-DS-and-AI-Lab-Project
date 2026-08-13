"""Command-line entry point for query-aware Hindi RAG retrieval and compression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.compressor import QueryAwareCompressor
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
    compressor = QueryAwareCompressor(args.checkpoint, max_length=args.max_length, device=args.device)
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
    print(json.dumps({"question": args.question, "contexts": contexts, "prompt": answer_prompt(args.question, generator_contexts)}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    ask_parser = subparsers.add_parser("ask", help="Retrieve and compress contexts for a Hindi question.")
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--chunks", type=Path, default=Path("corpus/processed/chunks.jsonl"))
    ask_parser.add_argument("--retriever", choices=("faiss", "bm25"), default="faiss")
    ask_parser.add_argument("--index-dir", type=Path, default=Path("corpus/index"))
    ask_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ask_parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/final-compressor"))
    ask_parser.add_argument("--top-k", type=int, default=3)
    ask_parser.add_argument("--max-length", type=int, default=512)
    ask_parser.add_argument("--minimum-retained", type=int, default=8)
    ask_parser.add_argument("--device", choices=("cpu", "cuda"))
    ask_parser.set_defaults(func=ask)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
