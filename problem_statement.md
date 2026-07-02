## Problem Statement

The widespread adoption of Large Language Models (LLMs) in Retrieval-Augmented Generation (RAG) pipelines relies on supplying models with extensive context. However, processing lengthy prompts significantly increases computational overhead, inflates inference costs, and frequently degrades the LLM's ability to accurately extract targeted information. 

While state-of-the-art prompt compressors like LLMLingua-2 have successfully reduced latency by formulating compression as a token classification problem (labeling individual tokens to "preserve" or "discard"), they exhibit two critical points of failure when applied to regional NLP applications:
1. **Morphological Destruction:** Existing compressors are predominantly trained on English corpora. They fail to map the syntactic redundancy and grammatical structures of morphologically rich Indic languages (such as Hindi) and code-switched text. This results in arbitrary word fragmentation and the loss of native grammatical integrity.
2. **Task-Agnostic Information Loss:** Current token classifiers calculate preservation probabilities independently of the user's explicit query. In RAG scenarios, this task-agnostic trimming frequently results in the catastrophic dropping of specific facts required to answer the prompt.

**Scope and Boundaries**
* **In-Scope:** The project will focus specifically on the Hindi subset of the IndicQA benchmark for the initial implementation. It encompasses the data distillation of an extractive token-classification dataset, the fine-tuning of cross-lingual feature encoders (e.g., `xlm-roberta-large` or `multilingual-BERT`) for query-aware token classification, and the integration of this compressor into a LangChain-based RAG pipeline.
* **Out-of-Scope:** The initial phase will not cover the other 10 languages in the IndicQA dataset, which are reserved for future project expansion. We will not explore abstractive/generative prompt compression (which introduces hallucination risks) or multimodal (image/video) prompt compression.

**Relevant Stakeholders**
* **End-Users:** Non-English speakers relying on localized AI systems, search engines, and automated regional support bots.
* **AI Developers & Engineers:** Teams building RAG pipelines for regional demographics who need to optimize latency and API costs.
* **Academic/Project Evaluators:** Course instructors, teaching assistants, and peers evaluating the technical rigor of the submission.
* **Data Annotators & Linguists:** Experts who evaluate linguistic integrity and verify that compressed text preserves correct grammatical syntax.

**Objectives**
1. **Compression Efficiency:** Achieve an average token compression ratio of 2x to 5x on Hindi RAG contexts without corrupting the native script.
2. **Retrieval & Generation Accuracy:** Maintain or improve Exact Match (EM) and F1 scores on the Hindi subset of the IndicQA benchmark when comparing the compressed prompt to the full, uncompressed baseline.
3. **Latency Optimization:** Deliver an end-to-end RAG inference speedup of at least 1.5x, demonstrating that the generation time saved offsets the token classifier's computational overhead.

---

## Literature Review & Existing Solutions

**Current Solutions and Academic Work**
* **LLMLingua & LongLLMLingua (Jiang et al., 2023):** These approaches utilize causal small language models (SLMs) to compress prompts based on information entropy and perplexity. While LongLLMLingua is task-aware, its autoregressive generation process is computationally heavy and slow.
* **LLMLingua-2 (Pan et al., 2024):** This framework treats prompt compression as a token classification task (preserve/discard) using a bidirectional Transformer encoder. It captures full context and is highly efficient (achieving 1.6x-2.9x end-to-end speedups), but the model is entirely task-agnostic and trained exclusively on English meeting transcripts (MeetingBank).
* **IndicQA Benchmark (2024):** A recent comprehensive benchmark for non-factoid Question Answering in 11 Indic languages. It highlights the complexities of retrieving and reasoning over regional texts but does not inherently provide a latency-optimized prompt compression framework.

**Strengths & Weaknesses vs. Our Approach**
* *Weaknesses of Current Approaches:* LLMLingua-2 is exceptionally fast but suffers from semantic information loss in QA tasks because it ignores the query. Furthermore, its English-centric training causes severe degradation when applied zero-shot to Indic scripts.
* *Strengths & Key Differences of Our Approach:* We introduce explicit **Query-Awareness** into the latency-efficient token classification architecture. By restructuring the input to ingest `[CLS] Question [SEP] Context [SEP]`, our bidirectional self-attention mechanism weights the preservation of context tokens directly against the provided query. Furthermore, fine-tuning on distilled Hindi IndicQA data ensures the model learns regional syntactic redundancy, preventing word fragmentation in Hindi text.

**Performance Benchmarks, Baselines, and Metrics**
* **Baseline Models for Comparison:** 
  1. Standard Uncompressed RAG Pipeline (using a frontier LLM for generation).
  2. LLMLingua-2 (evaluating its zero-shot transfer capabilities on Hindi text).
* **Evaluation Metrics:**
  * **Answer Quality:** Exact Match (EM) and ROUGE/F1 Scores against the IndicQA ground truth.
  * **Efficiency:** Compression Ratio (Original Context Tokens / Compressed Context Tokens).
  * **Speed:** End-to-End Latency (measured in seconds per query).

**References**
1. Pan, Z., et al. (2024). *LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression*. arXiv:2403.12968v2 [cs.CL].
2. Jiang, H., et al. (2023). *LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression*. arXiv:2310.06839 [cs.CL].
3. Singh, A. K., et al. (2024). *IndicQA Benchmark: A Multilingual Benchmark to Evaluate Question Answering capability of LLMs for Indic Languages*. arXiv:2407.13522 [cs.CL].

---
