# Group 9 DS & AI Lab Project - Worklog & Acknowledgment

This folder contains the official record of work, task allocation, and peer acknowledgment for **Group 9**'s project: **Query-Aware Hindi Prompt Compression in Retrieval-Augmented Generation (RAG)**.

---

## 👥 Group Members

| Name | Email ID | Role / Specialization |
| :--- | :--- | :--- |
| **ANURAG MONDAL** | [21f1002807@ds.study.iitm.ac.in](mailto:21f1002807@ds.study.iitm.ac.in) | Cross-lingual Encoder Engineering |
| **BHAVYA JAIN** | [21f1003868@ds.study.iitm.ac.in](mailto:21f1003868@ds.study.iitm.ac.in) | LangChain RAG Integration |
| **D CHIRAG RAO** | [21f1002300@ds.study.iitm.ac.in](mailto:21f1002300@ds.study.iitm.ac.in) | Pipeline Integration / Evaluation |
| **HITESH** | [22f2001256@ds.study.iitm.ac.in](mailto:22f2001256@ds.study.iitm.ac.in) | Benchmarking & Evaluation Metrics |
| **HITESH BINJRAWAT** | [22f2001255@ds.study.iitm.ac.in](mailto:22f2001255@ds.study.iitm.ac.in) | IndicQA Data Processing |
| **SOUMYABRATA MAHAPATRA** | [21f1003070@ds.study.iitm.ac.in](mailto:21f1003070@ds.study.iitm.ac.in) | Data Distillation / Model Training |

---

## 📅 Milestone 1: Problem Definition & Literature Review

### Requirements
*   Define the problem statement clearly.
*   Identify scope and boundaries of the project.
*   Identify relevant stakeholders.
*   State measurable objectives.
*   Conduct literature review of current solutions, analyzing strengths and weaknesses.
*   Include performance benchmarks, evaluation metrics, and credible references.
*   Prepare presentation slides and submit markdown formatting.

### Task Allocation & Work Done
| Member | Specific Tasks Completed | Deliverable |
| :--- | :--- | :--- |
| **ANURAG MONDAL** | Literature review of token classification vs. perplexity-based prompt compression (LLMLingua vs LLMLingua-2). | [Milestone 1 Report](../Milestone%20Files/Milestone%201/Milestone%201%20Report.md) |
| **BHAVYA JAIN** | Prepared presentation slides for Milestone 1 Review meeting detailing project pipeline. | [Milestone 1 Presentation PPTX](../Milestone%20Files/Milestone%201/Milestone%201%20Presentation.pptx) |
| **D CHIRAG RAO** | Defined project scope & boundaries, identified stakeholders, and structured repository. | [Milestone 1 Report](../Milestone%20Files/Milestone%201/Milestone%201%20Report.md) |
| **HITESH** | Defined measurable project objectives, evaluation metrics (Exact Match, F1, Latency, and Compression Ratio) & baselines. | [Milestone 1 Report](../Milestone%20Files/Milestone%201/Milestone%201%20Report.md) |
| **HITESH BINJRAWAT** | Researched IndicQA datasets, analyzed morphological challenges in Indic QA, and compiled credible references. | [Milestone 1 Report](../Milestone%20Files/Milestone%201/Milestone%201%20Report.md) |
| **SOUMYABRATA MAHAPATRA** | Authored the core problem statement emphasizing morphological destruction & task-agnostic information loss. | [Milestone 1 Report](../Milestone%20Files/Milestone%201/Milestone%201%20Report.md) |

### Peer Acknowledgment Matrix
| Task Owner | Anurag M. | Bhavya J. | Chirag R. | Hitesh | Hitesh B. | Soumyabrata M. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ANURAG MONDAL** | — | [x] | [x] | [x] | [x] | [x] |
| **BHAVYA JAIN** | [x] | — | [x] | [x] | [x] | [x] |
| **D CHIRAG RAO** | [x] | [x] | — | [x] | [x] | [x] |
| **HITESH** | [x] | [x] | [x] | — | [x] | [x] |
| **HITESH BINJRAWAT** | [x] | [x] | [x] | [x] | — | [x] |
| **SOUMYABRATA MAHAPATRA** | [x] | [x] | [x] | [x] | [x] | — |

---

## 📅 Milestone 2: Dataset Preparation & Distillation

### Requirements
* Identify and verify data sources, ownership, and usage constraints.
* Describe dataset feature distribution, class balance, and size.
* Handle missing values, inconsistencies, and noise through automated pipeline error handling.
* Generate a synthetic distillation dataset using an LLM teacher and justify prompt design choices.
* Establish precise token alignment connections between the generative output and token classification labels.
* Define the strategy for training/validation splits (80/20) and prevent data leakage.
* Document all preprocessing and environment setups to ensure reproducibility.

### Task Allocation & Work Done
| Member | Specific Tasks Completed | Deliverable |
| :--- | :--- | :--- |
| **ANURAG MONDAL** | Engineered the fault-tolerant LLM distillation pipeline framework, managed API orchestration, state management, and exponential backoff strategies (`distill.py`). | [distill.py](../distill.py) |
| **BHAVYA JAIN** | Handled raw data extraction, Parquet parsing, and ingestion preprocessing from Hugging Face (`fetch_data.py`). | [fetch_data.py](../fetch_data.py) |
| **D CHIRAG RAO** | Developed the word-to-token text alignment algorithm and binary mapping logic for token classification. | [Milestone 2 Report](../Milestone%20Files/Milestone%202/Milestone_2_Report.md) |
| **HITESH** | Engineered the exploratory data analysis (EDA) suite, calculating validation metrics, compression ratios (96%), and retention accuracy (`validate_data.py`). | [validate_data.py](../validate_data.py) |
| **HITESH BINJRAWAT** | Managed environment isolation, `.env` configuration security, and repository structure best practices. | [README.md](../README.md) |
| **SOUMYABRATA MAHAPATRA** | Authored the Milestone 2 technical report, compiled dataset metrics, and formatted presentation deliverables. | [Milestone 2 Report](../Milestone%20Files/Milestone%202/Milestone_2_Report.md) |

### Peer Acknowledgment Matrix
| Task Owner | Anurag M. | Bhavya J. | Chirag R. | Hitesh | Hitesh B. | Soumyabrata M. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ANURAG MONDAL** | — | [x] | [x] | [ ] | [x] | [x] |
| **BHAVYA JAIN** | [x] | — | [x] | [ ] | [x] | [x] |
| **D CHIRAG RAO** | [x] | [x] | — | [ ] | [x] | [x] |
| **HITESH** | [x] | [x] | [x] | — | [x] | [x] |
| **HITESH BINJRAWAT** | [x] | [x] | [x] | [ ] | — | [x] |
| **SOUMYABRATA MAHAPATRA** | [x] | [x] | [x] | [ ] | [x] | — |
