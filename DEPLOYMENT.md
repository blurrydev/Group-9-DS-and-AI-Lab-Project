# Deployment Guide

This project can be deployed as a single FastAPI web service (dashboard + API) with no frontend build step.

## Option 1: Render (Free Tier)

The repository already includes [render.yaml](render.yaml). It is configured for a free-tier service and starts with:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Steps

1. Push this repository to GitHub.
2. In Render, create a new Blueprint deployment.
3. Select your repository.
4. Render will detect `render.yaml` and create the `indic-llmlingua-dashboard` service.
5. Wait for build and deploy to complete.

### Included behavior on free tier

- `GET /` serves the dashboard from `frontend/`.
- `GET /api/health` returns app and model readiness.
- `POST /api/predict` works without external LLM keys because `RAG_GENERATOR_PROVIDER=mock` is set in `render.yaml`.
- If `corpus/index/chunks.faiss` is not present, retrieval mode (`retrieve_rag=true`) is unavailable, but direct context compression still works.

### Required files for best results

- `submission_artifacts/model/` should include model weights (`model.safetensors` or `pytorch_model.bin`) for local compressor usage if configured.
- `corpus/index/chunks.faiss` should exist for retrieval mode.

## Option 2: Docker (Any Host)

```bash
docker build -t indic-llmlingua-dashboard .
docker run --rm -p 8000:8000 \
  -e APP_ENV=production \
  -e RAG_GENERATOR_PROVIDER=mock \
  indic-llmlingua-dashboard
```

Open `http://localhost:8000`.

## Production env reference

Use `.env.deploy.example` as a starting point when you need to override defaults.

## Post-deploy checks

1. Open `/` and confirm dashboard loads.
2. Open `/api/health` and confirm `status` is `ok`.
3. Open `/api/runs` and confirm experiment runs are listed.
4. Send a `POST /api/predict` request with `question` and `context`.
5. If using retrieval, send `retrieve_rag=true` and verify `sources` are present.
