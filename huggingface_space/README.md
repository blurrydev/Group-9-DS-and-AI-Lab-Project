---
title: XLM-R Prompt Compressor
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.20.1
app_file: app.py
pinned: false
license: mit
short_description: Query-aware Hindi context compression using fine-tuned XLM-RoBERTa
---

# XLM-RoBERTa Query-Aware Context Compressor

This Space hosts the fine-tuned XLM-RoBERTa token classification model for query-aware Hindi context compression.

- **Model**: [`nnnhitesh/xlm-roberta-prompt-compressor`](https://huggingface.co/nnnhitesh/xlm-roberta-prompt-compressor)
- **API Endpoint**: `/compress`
- **Inputs**: `question` (string), `context` (string)
- **Output**: `compressed_context` (string)
