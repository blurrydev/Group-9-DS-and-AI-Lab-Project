# Indic-LLMLingua: Query-Aware Hindi Prompt Compression in RAG
## Milestone 2 Report: Dataset Preparation, Resilient Distillation, and Token Alignment

**Course:** Data Science and AI Labs (IIT Madras)  
**Group:** 9  
**Date:** July 8, 2026  
**Deadline:** July 9, 2026  

---

## 1. Executive Summary & Transition from Milestone 1

### 1.1 Milestone 1 Summary
In Milestone 1, Group 9 defined the problem statement for **Indic-LLMLingua**, addressing two critical points of failure in existing prompt compression models (like LLMLingua-2) when applied to regional RAG pipelines:
1. **Morphological Destruction:** The degradation and arbitrary fragmentation of morphologically rich Indic text (such as Hindi) under English-centric models.
2. **Task-Agnostic Information Loss:** The loss of query-critical facts in RAG pipelines due to compression mechanisms operating independently of the user's question.

Milestone 1 also mapped out the project scope (focusing on the Hindi subset of the IndicQA benchmark), established baselines (standard uncompressed RAG and zero-shot LLMLingua-2), defined evaluation metrics (Exact Match, F1, Compression Ratio, and End-to-End Latency), and compiled core literature references.

### 1.2 Scope of Milestone 2
Milestone 2 shifts from theoretical definition to practical **dataset preparation and preprocessing**. This document outlines the ingestion pipeline, dataset curation, quality controls, synthetic distillation mechanism, generative-to-classification token alignment strategy, and train/validation partitioning. Our objective is to establish a high-fidelity, query-aware sequence labeling dataset to train the student token-classification model in the subsequent phase.

---

## 2. Data Sources, Ownership, and Constraints

### 2.1 Primary Data Source
The primary source of raw data for the project is the Hindi subset of the **IndicQA** benchmark, created by **AI4Bharat**. 
* **Repository and Format:** Hosted on Hugging Face as Apache Arrow Parquet partitions. The pipeline pulls the validation partition of the Hindi QA subset:
  `https://huggingface.co/datasets/ai4bharat/IndicQA/resolve/main/indicqa-hi/validation-00000-of-00001.parquet`
* **Dataset Characteristics:** The dataset is composed of structural triples: context paragraphs, question strings, and ground-truth answer spans.

### 2.2 Ownership, Licensing, and Usage Constraints
* **Dataset Creator:** AI4Bharat Research Lab, Wadhwani School of Data Science and AI, IIT Madras.
* **License:** The dataset is released under the **Creative Commons Attribution 4.0 International (CC-BY-4.0)** license.
  * *Usage Permissions:* Allowed to share, copy, adapt, and build upon the dataset for both commercial and non-commercial purposes.
  * *Obligations:* Attribution must be provided by citing the IndicQA research paper.
* **Upstream Constraints:** The context paragraphs in IndicQA are derived from Hindi Wikipedia, which is licensed under the **Creative Commons Attribution-ShareAlike (CC-BY-SA)**. Any distribution of modified contexts must respect these copyleft terms.

### 2.3 Volume Constraints
The raw IndicQA Hindi subset contains exactly **1,052 records**. As an extractive question-answering dataset manually translated and validated by native speakers, its size is modest. This limitation requires a highly efficient training strategy (e.g., fine-tuning encoder architectures like XLM-RoBERTa using transfer learning) to prevent overfitting.

---

## 3. Synthetic Generation & LLM Distillation Process

To convert the extractive QA dataset into a token classification dataset for prompt compression, we employ **Dataset Distillation**. A powerful, frontier "Teacher" LLM is prompted to compress the context relative to the question, and its output is used to label the original context.

### 3.1 Teacher Model Selection
We select **Qwen-3.5-397B** (`qwen/qwen3.5-397b-a17b`) as the teacher model. 
* *Justification:* Qwen-3.5 is a Mixture-of-Experts (MoE) LLM with state-of-the-art multilingual capabilities. It possesses a deep comprehension of Hindi morphology, syntax, and grammatical endings, which avoids the morphological destruction typical of English-centric models.

### 3.2 Hyperparameter Configuration
* **Temperature:** Set to `0.1`. In text compression, creativity is undesirable. A low temperature minimizes stochastic sampling, forcing the model to make deterministic pruning decisions. This ensures that the generated compressed text remains a literal subset of the original context, preventing the introduction of synonyms, paraphrasing, or hallucinations.
* **Top-P:** Set to `0.95` to allow reasonable coverage of vocabulary probabilities while cutting off the tail of highly improbable words.

### 3.3 Prompt Design
The teacher model is invoked with a system prompt and a user prompt structured as follows:

```text
[System Prompt]
You are a highly efficient text compressor for Hindi text. 
Given a [Question] and a [Context], aggressively remove all words from the [Context] 
that are irrelevant to the [Question]. 
CRITICAL RULES:
1. Output ONLY the compressed Hindi text. No introductions, no explanations.
2. Do NOT change the grammar, do NOT translate to English, and do NOT add new words.
3. Ensure the exact information needed to answer the question remains intact in the output.

[User Prompt]
[Question]: {question}

[Context]: {context}
```

### 3.4 API Security Policy Filtering & Data Attrition
Of the **1,052 raw rows** ingested from the IndicQA dataset:
* **Successfully Processed:** **987 rows**.
* **Skipped Rows:** **65 rows** ($6.18\%$ of the dataset).
* **Reason for Skipping:** The API returned `None` values for these 65 records. This occurred due to the host API provider's internal security policies, safety filters, or content moderation triggers. If a keyword in the Hindi Wikipedia context or the question flagged the safety filter, the model generation was blocked, returning `None`. Rather than failing the execution, these cases were caught, logged, and skipped, resulting in a finalized master corpus of 987 high-quality distilled rows.

---

## 4. Alignment Between Tasks (Generative to Classification)

The teacher LLM performs a **generative task** (generating compressed text), while the student model must be trained for a **token classification task** (predicting whether to keep or discard each token in the original context). To bridge this gap, we implement a token alignment module in [distill.py](../../distill.py).

### 4.1 Token Alignment Algorithm
The function `align_tokens(original_text, compressed_text)` performs the alignment:
1. **Whitespace Tokenization:** The original context is split by whitespace into a list of base tokens:
   $$T_{\text{orig}} = [t_1, t_2, \dots, t_N]$$
2. **Compressed Vocabulary Construction:** The compressed text output from the teacher LLM is split by whitespace to build a set of preserved tokens:
   $$S_{\text{comp}} = \{c_1, c_2, \dots, c_M\}$$
3. **Binary Label Assignment:** A label array $Y = [y_1, y_2, \dots, y_N]$ is constructed. For each token $t_i$:
   $$y_i = \begin{cases} 1 & \text{if } t_i \in S_{\text{comp}} \\ 0 & \text{otherwise} \end{cases}$$
4. **Record Compilation:** The tokens and labels are structured alongside the metadata:
   ```json
   {
     "tokens": ["महात्मा", "गांधी", "का", "जन्म", "..."],
     "labels": [1, 1, 0, 1, 0],
     "question": "गांधी जी का जन्म कब हुआ था?",
     "original_answer": "2 अक्टूबर 1869"
   }
   ```

### 4.2 Handling Alignment Limitations
* **Punctuation Attachment:** In simple whitespace splits, punctuation marks remain attached to adjacent words (e.g., `"नगर,"` vs `"नगर"`). If the teacher model strips the comma, the token `"नगर,"` might be labeled $0$. To minimize this noise, the student tokenizer's subword mapping will handle boundary alignments during fine-tuning.
* **Morphological Preservation:** The system prompt explicitly restricts the teacher from modifying word endings or suffixes. This ensures that the root words do not shift, allowing whitespace matching to achieve high reliability.

### 4.3 Concrete Alignment and Distillation Examples
To illustrate how raw query-context pairs are transformed into query-aware token classification targets, consider these two actual samples from the validation split (`data/val_indicqa.jsonl`):

#### Example 1:
*   **Question:** `मुग़ल काल में आवासीय और प्रशासनिक भवन को क्या कहा जाता था?` *(English translation: "What were residential and administrative buildings called in the Mughal period?")*
*   **Ground-Truth Answer:** `दौलतखाना`
*   **Original Context (Sample Segment):** 
    > "... भारत की सबसे बड़ी सामूहिक मस्जिद है, साथ ही आवासीय तथा प्रशासकीय इमारते हैं जिसे दौलतखाना कहते हैं। ..."
*   **Generative compressed output (Qwen-3.5-397B):** 
    > "शाही जिसमें भारत की सबसे बड़ी सामूहिक मस्जिद है, साथ ही आवासीय तथा प्रशासकीय इमारते हैं जिसे दौलतखाना कहते हैं।"
*   **Aligned Tokens & Binary Labels:**
    *   `"आवासीय"` $\rightarrow$ **1** (Preserved)
    *   `"तथा"` $\rightarrow$ **1** (Preserved)
    *   `"प्रशासकीय"` $\rightarrow$ **1** (Preserved)
    *   `"इमारते"` $\rightarrow$ **1** (Preserved)
    *   `"हैं"` $\rightarrow$ **1** (Preserved)
    *   `"जिसे"` $\rightarrow$ **1** (Preserved)
    *   `"दौलतखाना"` $\rightarrow$ **1** (Preserved) *(Contains answer)*
    *   `"कहते"` $\rightarrow$ **1** (Preserved)
    *   `"हैं।"` $\rightarrow$ **1** (Preserved)
    *   All surrounding context describing the construction years, gardens, and Akbar's move to Fatehpur Sikri is labeled **0** (Discarded).

#### Example 2:
*   **Question:** `नाना साहब के पिता कौन थे ?` *(English translation: "Who was Nana Saheb's father?")*
*   **Ground-Truth Answer:** `माधवनारायण राव`
*   **Original Context (Sample Segment):**
    > "(धोंडू पन्त) नाना साहब ने सन् 1824 में वेणुग्राम निवासी माधवनारायण राव के घर जन्म लिया था। इनके पिता पेशवा बाजीराव द्वितीय के सगोत्र भाई थे।"
*   **Generative compressed output (Qwen-3.5-397B):**
    > "नाना साहब के पिता पेशवा बाजीराव द्वितीय के सगोत्र भाई थे।"
*   **Aligned Tokens & Binary Labels:**
    *   `"नाना"` $\rightarrow$ **1** (Preserved)
    *   `"साहब"` $\rightarrow$ **1** (Preserved)
    *   `"के"` $\rightarrow$ **1** (Preserved)
    *   `"पिता"` $\rightarrow$ **1** (Preserved)
    *   `"पेशवा"` $\rightarrow$ **1** (Preserved)
    *   `"बाजीराव"` $\rightarrow$ **1** (Preserved)
    *   `"द्वितीय"` $\rightarrow$ **1** (Preserved)
    *   `"के"` $\rightarrow$ **1** (Preserved)
    *   `"सगोत्र"` $\rightarrow$ **1** (Preserved)
    *   `"भाई"` $\rightarrow$ **1** (Preserved)
    *   `"थे।"` $\rightarrow$ **1** (Preserved)
    *   The middle tokens describing the birth date `1824` and location `वेणुग्राम निवासी` are labeled **0** (Discarded).

---

## 5. Dataset Description & Feature Distribution

### 5.1 Dataset Size and Split Metrics
The $987$ successfully distilled rows were divided into training and validation sets using an 80/20 split:
* **Training Dataset (`train_indicqa.jsonl`):** **789 rows**.
* **Validation Dataset (`val_indicqa.jsonl`):** **198 rows**.

### 5.2 Token Lengths and Compression Ratios
An analysis of the dataset splits reveals a significant reduction in context size:

| Dataset Metric | Training Split | Validation Split | Combined Dataset |
| :--- | :---: | :---: | :---: |
| **Total Processed Rows** | 789 | 198 | 987 |
| **Avg. Context Length** | 498 tokens | 489 tokens | 496 tokens |
| **Avg. Compressed Length** | 19 tokens | 20 tokens | 19 tokens |
| **Compression Ratio (Reduction %)**| **96.1%** | **96.0%** | **96.1%** |
| **Answer Retention Accuracy** | **90.0%** | **91.9%** | **90.4%** |

### 5.3 Class Feature Distribution (Imbalance Analysis)
The binary targets exhibit an extreme class imbalance, which is characteristic of high-ratio prompt compression:
* **Training Set:** $3.82\%$ Class 1 (Preserve), $96.18\%$ Class 0 (Discard).
* **Validation Set:** $4.09\%$ Class 1 (Preserve), $95.91\%$ Class 0 (Discard).

```
Training Class Distribution:
[Class 1: Preserve]  ██░░░░░░░░░░░░░░░░░░ (3.82%)
[Class 0: Discard]   ████████████████████ (96.18%)
```

*Implication for Model Training:* This $24:1$ class imbalance means standard accuracy cannot be used as a loss metric. Student training must employ a class-weighted loss (e.g., Weighted Cross Entropy) or Focal Loss to penalize misclassifications of the minority class.

---

## 6. Dataset Quality and Pipeline Resiliency

### 6.1 Assessment of Dataset Quality & Anomalies
* **Missing Values:** In the raw HF Parquet, a small subset of records contained missing or empty lists for the `answers` field. The ingestion script checks for this (`answers and len(answers.get("text", [])) > 0`), extracts the first element if available, and assigns an empty string fallback `""` otherwise. This preserves the sample while preventing runtime parsing crashes.
* **Duplicate Entries:** Context paragraphs corresponding to multiple questions are preserved as distinct rows, because a query-aware model must learn different compression masks for the same context based on the user's question. Duplicate *question-context* combinations were filtered out.
* **Noise Mitigation:** Low-temperature generation prevented the teacher model from outputting conversational filler (e.g., "Here is the compressed text..."), resulting in clean target strings.

### 6.2 Engineering Resiliency Controls
* **Exponential Backoff:** The distillation pipeline handles rate limits (HTTP 429) and timeouts dynamically:
  $$\text{Wait Time} = \text{base\_wait\_time} \times 2^{\text{attempt}} \quad (\text{base} = 10\text{s})$$
* **Safety protection Catch:** Rather than terminating when a safety filter returns `None`, the script logs the index and skips to the next row.
* **Resumability:** The pipeline checks the current line count of `data/master_processed_data.jsonl` on startup and skips previously processed rows, allowing it to recover from power or network interruptions without wasting API credits.

---

## 7. Dataset Adequacy & Augmentation Strategy

### 7.1 Evaluation of Adequacy
While $987$ rows is sufficient for fine-tuning small encoder models (like `mBERT` or `xlm-roberta-large` which contain pre-trained multilingual language models), it is relatively small for training larger sequence taggers from scratch. The dataset is adequate for a baseline but could benefit from expansion.

### 7.2 Augmentation and Expansion Options
To improve generalization in later phases of the project, we have identified the following expansion avenues:
1. **Multilingual Expansion:** Ingesting other IndicQA subsets (e.g., Bengali, Tamil, Telugu) and running them through the same distillation pipeline to train a unified multilingual model.
2. **Hindi SQuAD:** Utilizing the machine-translated Hindi SQuAD dataset as an additional corpus for data distillation.
3. **Data Augmentation:** Applying back-translation (Hindi $\rightarrow$ English $\rightarrow$ Hindi) on the questions or performing random synonym replacement (using Hindi WordNet) to augment the existing 987 samples.

---

## 8. Split Strategy and Leakage Prevention

### 8.1 Partitioning Logic
The dataset is split into an $80/20$ ratio. A random shuffle is applied using a fixed random seed (`seed = 42`) to guarantee experimental reproducibility.

### 8.2 Document-Level Data Leakage Prevention
In IndicQA, multiple questions are associated with a single Wikipedia article context.
* **The Overlap Risk:** If the train-test split is performed randomly at the question-answer level, questions sharing the exact same context paragraph will be split across the training and validation sets. The model would evaluate on contexts it has already seen during training, leading to data leakage and inflated validation metrics.
* **Mitigation:** The split is grouped by **distinct Wikipedia articles**. The unique context paragraphs are partitioned into train and validation sets first, ensuring that all questions belonging to a specific context document are kept in the same split. The validation set contains completely unseen contexts, forcing the model to generalize.

---

## 9. RAG-Specific Preparations, Chunking, & Vector Databases

Because the downstream application of our token classifier is a **Hindi RAG Pipeline**, the dataset preparation matches several production constraints:

### 9.1 Document Chunking Strategy
* **Context Paragraph Integrity:** The IndicQA context paragraphs average 496 tokens. This aligns with standard RAG chunk sizes ($512$ tokens).
* **Chunking Configuration:** In production, documents are split using recursive character chunking with a chunk size of $512$ characters/tokens and a $10\%$ overlap ($50$ tokens) to prevent factual truncation at boundaries.

### 9.2 Embedding Models & Vector Database Options
* **Embedding Model:** We evaluate **LaBSE** (Language-Agnostic BERT Sentence Embedding) and **mUSE** (Multilingual Universal Sentence Encoder) for generating vector representations of Hindi chunks, alongside commercial options like `text-embedding-3-large`.
* **Vector Database:** For storage and similarity search, we utilize **ChromaDB** or **FAISS** for local development, and explore **Milvus** for scalable vector search.

---

## 10. Student Model Training & Prompt Formatting

The generated dataset is structured specifically for training a student sequence labeling model:

### 10.1 Input Representation
The student model (e.g., `xlm-roberta-large`) will ingest the question and context as a single concatenated sequence:
$$\text{Input} = \texttt{[CLS]} \text{ Question } \texttt{[SEP]} \text{ Context } \texttt{[SEP]}$$

### 10.2 Label Alignment
* **Question Tokens:** All tokens in the question segment are labeled with a dummy tag (e.g., $-100$ in PyTorch CrossEntropyLoss) to ensure the model does not calculate loss on the question itself.
* **Context Tokens:** Context tokens are aligned with the binary label sequence $Y \in \{0, 1\}$.
* **Token Length Considerations:** XLM-RoBERTa supports sequences up to 512 tokens. Since the combined average token length of the question and context is around 520 tokens, we truncate or apply a sliding window approach with a stride of 128 tokens for samples exceeding the sequence limit.

---

## 11. Reproducibility & Preprocessing Guide

To reproduce the dataset extraction, distillation, and validation pipeline, follow the step-by-step instructions below.

### 11.1 Directory Layout
Ensure your workspace matches the following structure:
```text
├── data/
│   ├── raw_hindi_qa.json      # Generated by fetch_data.py
│   ├── train_indicqa.jsonl    # Pre-split training data
│   └── val_indicqa.jsonl      # Pre-split validation data
├── Milestone Files/
│   └── Milestone 2/
│       └── Milestone_2_Report.md
├── fetch_data.py
├── distill.py
└── validate_data.py
```

### 11.2 Setup Environment
1. **Virtual Environment Setup:**
   Using `uv`:
   ```bash
   uv sync
   ```
2. **Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   API_KEY="your_api_key"
   BASE_URL="api_base_url"
   ```

### 11.3 Execution Sequence
Execute the pipeline components in order:

```bash
# Step 1: Fetch raw parquet data
uv run fetch_data.py

# Step 2: Distill data using Qwen-3.5-397B & align tokens
uv run distill.py

# Step 3: Run validation metrics
uv run validate_data.py
```
*Expected Outputs:* Running `validate_data.py` will print the average original length, average compressed length, total compression reduction ($96.1\%$), and answer retention rate ($90.4\%$) to the console.
