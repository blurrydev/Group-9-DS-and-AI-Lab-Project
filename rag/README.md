# Local Hindi RAG Prototype

The first RAG implementation is intentionally local and inspectable:

```text
chunks.jsonl → BM25 retrieval → XLM-R query-aware compression → generator-ready prompt
```

Run it after building a substantive corpus:

```bash
python main.py ask \
  --question "प्रधानमंत्री जन धन योजना में बीमा लाभ क्या है?" \
  --chunks corpus/processed/chunks.jsonl \
  --checkpoint checkpoints/final-compressor
```

The command emits JSON containing source metadata, retrieval scores, compression
statistics, compressed contexts, and a Hindi prompt. Feed `prompt` into any
chosen instruction-following generator. The generation provider is deliberately
not hard-coded because its model, credentials, and hosting choice must be made
separately.

This is a lexical BM25 baseline. Replace it with multilingual embedding retrieval
and a vector index once the source corpus is sufficiently large and evaluated.

## Frontend API

Run the local API from the repository root:

```bash
uv run uvicorn rag.api:app --host 0.0.0.0 --port 8000 --reload
```

The interactive API documentation is available at `http://localhost:8000/docs`.
The frontend queries `POST /v1/rag/query`:

```json
{
  "question": "प्रधानमंत्री जन धन योजना में बीमा लाभ क्या है?",
  "top_k": 3,
  "minimum_retained": 8
}
```

It returns source metadata, compressed context, and a generator-ready `prompt`.
Configure frontend origins with `RAG_CORS_ORIGINS`, as a comma-separated list;
the local defaults are `http://localhost:3000` and `http://localhost:5173`.
For deployment, `RAG_CHUNKS_PATH`, `RAG_CHECKPOINT_PATH`, and `RAG_DEVICE`
(`cpu` or `cuda`) can override the local defaults.
