# Indic-LLMLingua: Hindi prompt Compression Dataset Card

This directory contains the training and validation dataset splits for the **Indic-LLMLingua** query-aware prompt compression project, distilled from AI4Bharat's IndicQA Hindi subset.

---

## 📊 Hugging Face Dataset Details (Hindi Subset)

*   **Original Dataset:** AI4Bharat IndicQA (Hindi subset `indicqa-hi`)
*   **Hugging Face Repository:** [AI4Bharat/IndicQA](https://huggingface.co/datasets/ai4bharat/IndicQA)
*   **Raw Data Source Ingested:** `indicqa-hi/validation-00000-of-00001.parquet`
*   **Total Raw Rows Ingested:** **1,052 rows**
*   **Distillation Success Rate:** **93.82%** ($987$ processed rows successfully, $65$ rows skipped due to teacher model API safety-filter blocks returning `None`).

---

## 📑 Dataset Schema

### 1. Raw Ingested Schema (from HF Parquet)
Each raw row in the parquet contains:
*   `context` (string): The background passage extracted from Hindi Wikipedia.
*   `question` (string): The Hindi reading comprehension question created by native speakers.
*   `answers` (struct): Contains:
    *   `text` (list of strings): Ground truth answer text spans (e.g., `["दौलतखाना"]`).
    *   `answer_start` (list of integers): Character offset of the answer start in the context.

### 2. Distilled Split Schema (`train_indicqa.jsonl` & `val_indicqa.jsonl`)
The processed JSON Lines files contain the following key-value pairs per line:
*   `question` (string): The Hindi question.
*   `original_answer` (string): Extracted first element of the ground truth answer text.
*   `tokens` (list of strings): The original context split into individual words via whitespace tokenization.
*   `labels` (list of integers): Binary labels of matching length to the `tokens` array (where `1` indicates a preserved token and `0` indicates a discarded token).

---

## 📈 Dataset Split & Compression Metrics

The 987 processed rows are partitioned using an 80/20 division split at the *Wikipedia article level* to prevent context leakage.

| Metric | Training Split (`train_indicqa.jsonl`) | Validation Split (`val_indicqa.jsonl`) | Combined Dataset |
| :--- | :---: | :---: | :---: |
| **Total Rows** | 789 | 198 | 987 |
| **Avg. Context Tokens** | 498 | 489 | 496 |
| **Avg. Compressed Tokens** | 19 | 20 | 19 |
| **Prompt Size Reduction (%)** | **96.1%** | **96.0%** | **96.1%** |
| **Answer Retention Accuracy (%)** | **90.0%** | **91.9%** | **90.4%** |
| **Class Distribution (1s vs 0s)** | 3.82% / 96.18% | 4.09% / 95.91% | 3.87% / 96.13% |

---

## 📂 Storage & Contribution Conventions

To maintain a clean and lightweight git history:

1. **Tracked Distilled JSONL Datasets:**
   * Only the processed split datasets (`train_indicqa.jsonl` and `val_indicqa.jsonl`) are tracked in this directory.
   * Do not remove or modify these splits without consulting the evaluation team.

2. **Ignored Raw & Temporary Files:**
   * Raw downloaded data (`data/raw_hindi_qa.json`) and master combined datasets (`data/master_processed_data.jsonl`) are ignored via `.gitignore` to avoid pushing heavy redundant files.
   * If you run the pipelines (`fetch_data.py` and `distill.py`), these files will be generated locally but will not be committed to Git.
