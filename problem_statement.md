# Problem Statement

### The Core Challenge
The widespread adoption of Large Language Models (LLMs) has been heavily driven by techniques like Retrieval-Augmented Generation (RAG) and In-Context Learning (ICL). These architectures rely on supplying models with extensive context to ground their responses. However, processing these lengthy prompts introduces severe bottlenecks: it significantly increases computational overhead, inflates inference costs, and frequently degrades the LLM's ability to accurately perceive and extract targeted information from the middle of the context window. 

To mitigate this, prompt compression has emerged as a crucial optimization strategy. State-of-the-art methods, such as LLMLingua-2, have successfully formulated context compression as a token classification problem (labeling tokens to "preserve" or "discard") using bidirectional Transformer encoders[cite: 1]. This ensures low latency and high faithfulness to the original text.

### Limitations of Current State-of-the-Art
Despite these architectural advancements, deploying current prompt compression models in regional and localized NLP pipelines exposes two critical points of failure:

1. **English-Centric Bias and Morphological Destruction:** Existing token classifiers are overwhelmingly trained on English corpora, such as the MeetingBank dataset[cite: 1]. Syntactic redundancy, filler words, and grammatical structures behave fundamentally differently in morphologically rich Indic languages (such as Hindi and Bengali) and in code-switched text (such as Hinglish). When applied to these regional languages, English-trained compressors fail to map localized redundancy patterns. This results in arbitrary word fragmentation and the destruction of native grammatical integrity, rendering the compressed text incomprehensible to the downstream LLM.
2. **Task-Agnostic Information Loss in RAG Pipelines:** Current token classification models determine the preservation probability of a token based entirely on its general information entropy, remaining entirely agnostic to the user's specific downstream task. In search and RAG scenarios, where a document may contain multiple distinct facts, task-agnostic trimming often inadvertently discards the exact localized information required to answer the user's specific query. 

### The Solution: What We Are Building
This project aims to bridge the gap between high-speed prompt compression and multilingual RAG accuracy by developing a **Query-Aware Multilingual Token Classifier**. 

Instead of relying on general redundancy, our model will be trained to dynamically classify and compress document context strictly conditioned on a specific user query. By leveraging the IndicQA benchmark, we will transform abstractive regional datasets into extractive, token-level classification data. We will then fine-tune cross-lingual feature encoders (e.g., `xlm-roberta-large` or `multilingual-BERT`[cite: 1]) to simultaneously ingest a `[Question]` and a `[Context]`, calculating preservation probabilities that protect both semantic relevance and native structural integrity.
