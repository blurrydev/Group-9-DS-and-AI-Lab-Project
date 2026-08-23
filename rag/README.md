# Local Hindi RAG Prototype

The first RAG implementation is intentionally local and inspectable:

```text
chunks.jsonl → multilingual embeddings → FAISS retrieval → XLM-R query-aware compression → prompt assembly → LLM answer generation
```

Build a FAISS index after building or changing the corpus. The first run downloads
the multilingual embedding model:

```bash
uv sync
uv run python build_index.py --chunks corpus/processed/chunks.jsonl
```

Then run a query with direct answer generation:

```bash
uv run python main.py ask \
  --question "प्रधानमंत्री जन धन योजना में बीमा लाभ क्या है?" \
  --generate
```

The command emits JSON containing source metadata, retrieval scores, compression
statistics, compressed contexts, the formatted Hindi prompt, and the final generated `answer`.

FAISS uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` with
normalized embeddings and inner-product (cosine) similarity. Index files remain
local in `corpus/index/` and are reproducible from `chunks.jsonl`.

The old BM25 implementation remains available for comparison only:

```bash
uv run python main.py ask --retriever bm25 --question "..." --generate
```

## Running Context Compression & Answer Generation

The RAG pipeline supports both **deployed remote model endpoints** (such as Hugging Face Spaces or REST APIs) and **local PyTorch checkpoints**, as well as any OpenAI-compatible LLM generator (OpenAI, Groq, Ollama, Qwen, Gemini, etc.).

### 1. Generating Answers via CLI

```bash
# Using environment variables for generator API
export API_KEY="your-api-key"
export BASE_URL="https://api.openai.com/v1" # or custom endpoint / Groq / Ollama

uv run python main.py ask \
  --question "प्रधानमंत्री जन धन योजना में बीमा लाभ क्या है?" \
  --generate \
  --generator-model "gpt-4o-mini"
```

### 2. Using a Deployed Hugging Face Model Endpoint (Remote Compressor)

```bash
uv run python main.py ask \
  --question "प्रधानमंत्री जन धन योजना में बीमा लाभ क्या है?" \
  --compressor-type remote \
  --endpoint "username/space-name" \
  --hf-token "$HF_TOKEN" \
  --generate
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
  "minimum_retained": 8,
  "generate": true
}
```

It returns source metadata, compressed context, prompt, and the generated `answer`.

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
| `RAG_GENERATOR_MODEL` | Generator model name | `qwen/qwen3.5-397b-a17b` |
| `OPENAI_API_KEY` / `API_KEY` | API Key for LLM answer generator | None |
| `OPENAI_BASE_URL` / `BASE_URL` | Base URL for OpenAI-compatible generator API | None |
| `RAG_CORS_ORIGINS` | Comma-separated allowed frontend origins | `http://localhost:3000,http://localhost:5173` |


