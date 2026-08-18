"""Create a local FAISS vector index from the source-attributed RAG chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.vector_store import DEFAULT_EMBEDDING_MODEL, FAISSRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=Path("corpus/processed/chunks.jsonl"))
    parser.add_argument("--index-dir", type=Path, default=Path("corpus/index"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    count = FAISSRetriever.build(args.chunks, args.index_dir, args.embedding_model, args.batch_size)
    print(f"Built FAISS index for {count} chunks in {args.index_dir}")


if __name__ == "__main__":
    main()
