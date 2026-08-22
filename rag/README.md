# Local Hindi RAG Prototype

The first RAG implementation is intentionally local and inspectable:

```text
chunks.jsonl → multilingual embeddings → FAISS retrieval → XLM-R query-aware compression → generator-ready prompt
```

Build a FAISS index after building or changing the corpus. The first run downloads
the multilingual embedding model:

```bash
uv sync
uv run python build_index.py --chunks corpus/processed/chunks.jsonl
```

Then run a query:

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

FAISS uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` with
normalized embeddings and inner-product (cosine) similarity. Index files remain
local in `corpus/index/` and are reproducible from `chunks.jsonl`.

The old BM25 implementation remains available for comparison only:

```bash
uv run python main.py ask --retriever bm25 --question "..."
```

## Running Context Compression

The RAG pipeline supports both **deployed remote model endpoints** (such as Hugging Face Spaces or REST APIs) and **local PyTorch checkpoints**.

### 1. Using a Deployed Hugging Face Model Endpoint (Remote)

Query using a remote Hugging Face Space ID or URL via CLI:

```bash
uv run python main.py ask \
  --question "प्रधानमंत्री जन धन योजना में बीमा लाभ क्या है?" \
  --compressor-type remote \
  --endpoint "username/space-name" \
  --hf-token "$HF_TOKEN"
```

Or run the FastAPI server with environment variables:

```bash
export RAG_COMPRESSOR_TYPE="remote"
export RAG_COMPRESSOR_ENDPOINT="username/space-name"   # or https://your-space.hf.space
export HF_TOKEN="hf_..."                              # optional for private spaces

uv run uvicorn rag.api:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Using a Local Model Checkpoint

```bash
uv run python main.py ask \
  --question "प्रधानमंत्री जन धन योजना में बीमा लाभ क्या है?" \
  --compressor-type local \
  --checkpoint checkpoints/final-compressor
```

---

## Frontend API

Run the API service from the repository root:

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

### Environment Configuration

| Variable | Description | Default |
| :--- | :--- | :--- |
| `RAG_COMPRESSOR_TYPE` | Compressor backend (`remote`, `hf_space`, `local`) | `remote` |
| `RAG_COMPRESSOR_ENDPOINT` | HF Space ID (e.g. `user/space`) or URL (`https://...hf.space`) | `nnnhitesh/TokenCompressor` |
| `HF_TOKEN` / `RAG_HF_TOKEN` | Hugging Face Token for private or gated spaces | None |
| `RAG_CHECKPOINT_PATH` | Path to local model checkpoint | `checkpoints/final-compressor` |
| `RAG_FAISS_INDEX_PATH` | Path to directory containing `chunks.faiss` and `chunks.jsonl` | `corpus/index` |
| `RAG_EMBEDDING_MODEL` | Sentence-transformers embedding model name | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `RAG_DEVICE` | Compute device (`cpu` or `cuda`) | Auto-detect |
| `RAG_CORS_ORIGINS` | Comma-separated allowed frontend origins | `http://localhost:3000,http://localhost:5173` |

