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
| **ANURAG MONDAL** | — | [x] | [x] |x] | [x] | [x] |
| **BHAVYA JAIN** | [x] | — | [x] | [x] | [x] | [x] |
| **D CHIRAG RAO** | [x] | [x] | — | [x] | [x] | [x] |
| **HITESH** | [x] | [x] | [x] | — | [x] | [x] |
| **HITESH BINJRAWAT** | [x] | [x] | [x] | [x] | — | [x] |
| **SOUMYABRATA MAHAPATRA** | [x] | [x] | [x] | [x] | [x] | — |

---

## 📅 Milestone 3: Model Architecture & Pipeline Verification


### Requirements
* **Milestone Lead:** **HITESH BINJRAWAT**
* Document the dataset directory structure, including raw and processed data along with training, validation, and test splits.
* Describe all preprocessing steps applied to the dataset prior to training, including task-specific transformations.
* Explain the model architecture by highlighting its major components and their interactions.
* Create a diagram illustrating the complete data flow from raw input to the final model output.
* Specify how the processed data conforms to the model's expected input format, including dimensions, tensor structures, and other required specifications.
* Justify the choice of the model architecture by discussing its strengths, limitations, and comparison with alternative approaches.
* Implement and validate a small-scale end-to-end pipeline using a subset of the dataset to verify workflow integration.
* Present sample model outputs and document the loss functions and evaluation metrics used to assess model performance. 

### Task Allocation & Work Done
| Member | Specific Tasks Completed | Deliverable |
| :--- | :--- | :--- |
| **ANURAG MONDAL** | Documented the dataset folder structure (raw, processed, train/validation/test splits) and explained all preprocessing steps applied before training. | [Milestone_3_Report.md](../Milestone%20Files/Milestone%203/Milestone_3_Report.md) |
| **BHAVYA JAIN** | Justified the choice of model architecture by discussing its strengths, limitations, and comparison with alternative approaches. | [Milestone_3_Report.md](../Milestone%20Files/Milestone%203/Milestone_3_Report.md) |
| **D CHIRAG RAO** | Presented sample model outputs and documented the loss functions and evaluation metrics used to assess model performance. | [Milestone_3_Report.md](../Milestone%20Files/Milestone%203/Milestone_3_Report.md) |
| **HITESH** | Described the processed data format expected by the model, including input shapes, tensor dimensions, and embedding/token specifications. | [Milestone_3_Report.md](../Milestone%20Files/Milestone%203/Milestone_3_Report.md) |
| **HITESH BINJRAWAT** (Lead)| Explained the model architecture and created a data-flow diagram showing how inputs are transformed into predictions. | [Milestone_3_Report.md](../Milestone%20Files/Milestone%203/Milestone_3_Report.md) |
| **SOUMYABRATA MAHAPATRA** | Implemented and validated a small-scale end-to-end pipeline to ensure all workflow components function correctly. | [pipeline_test.py](../pipeline_test.py) |
  
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

## 📅 Milestone 4: Model Training, Fine-Tuning & Empirical Evaluation

### Requirements & Lead
* **Milestone Lead:** **ANURAG MONDAL**
* **Deadline:** 30 July 2026
* Briefly describe the datasets used and the necessary preprocessing (including generation of additional Indic supervision via 3-stage XL-Sum distillation).
* Provide a concise recap of the model architecture with key components and layers.
* Specify full training configuration (loss functions, evaluation metrics, optimizers, learning rate, batch size, number of epochs, hardware requirements, training strategies).
* Describe hyperparameter experiments with performance comparison tables across settings.
* Describe generalization and training stability techniques and their impact.
* Provide quantitative (Accuracy: 76.03%, Recall: 80.08%, F1: 62.50%) and qualitative results with sample compressed output.
* List generated artifacts (checkpoints, metrics, logs, notebooks, plots).
* Discuss key findings from training (what worked, what did not perform as expected, bottlenecks, plans for improvement).

### Task Allocation & Work Done
| Member | Specific Tasks Completed | Deliverable |
| :--- | :--- | :--- |
| **ANURAG MONDAL** (Lead) | Led the team for Milestone 4; configured full training pipeline, hyperparameter setup, AdamW optimizer, FP16 mixed precision, and fine-tuned XLM-RoBERTa model on dual Tesla T4 GPUs (`train-xlmr-llmlingua-indic.ipynb`). | [train-xlmr-llmlingua-indic.ipynb](../notebooks/train-xlmr-llmlingua-indic.ipynb) / [Milestone_4_Report.md](../Milestone%20Files/Milestone%204/Milestone_4_Report.md) |
| **BHAVYA JAIN** | Co-managed model training and fine-tuning execution, gradient checkpointing configuration, dynamic collator setup, and model checkpoint saving strategies. | [train-xlmr-llmlingua-indic.ipynb](../notebooks/train-xlmr-llmlingua-indic.ipynb) / [Milestone_4_Report.md](../Milestone%20Files/Milestone%204/Milestone_4_Report.md) |
| **D CHIRAG RAO** | Co-led additional data collection & distillation; processed multi-stage XL-Sum Indic QA data distillation and subword alignment splitting (`PreProcessing.ipynb`). | [PreProcessing.ipynb](../notebooks/PreProcessing.ipynb) / [Milestone_4_Report.md](../Milestone%20Files/Milestone%204/Milestone_4_Report.md) |
| **HITESH** | Co-led data collection & distillation pipeline; verified dataset split balance (1609 train / 201 val / 202 test) and aligned token-level labels with sentencepiece BPE tokens. | [PreProcessing.ipynb](../notebooks/PreProcessing.ipynb) / [Milestone_4_Report.md](../Milestone%20Files/Milestone%204/Milestone_4_Report.md) |
| **HITESH BINJRAWAT** | Co-analyzed key findings and qualitative output; evaluated confusion matrix, precision-recall trade-offs, and sample compressed text quality. | [Milestone_4_Report.md](../Milestone%20Files/Milestone%204/Milestone_4_Report.md) |
| **SOUMYABRATA MAHAPATRA** | Co-conducted quantitative analysis of training results across 10 epochs; analyzed training loss vs eval F1 trajectory, metrics logging, and artifact generation. | [Milestone_4_Report.md](../Milestone%20Files/Milestone%204/Milestone_4_Report.md) |

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

## 📅 Milestone 5: Model Evaluation & Analysis
### Requirements & Lead
**Milestone Lead:** **HITESH**  
**Deadline:** 6 August 2026

* Briefly restate the trained model and evaluation pipeline from the previous milestone.
* Describe the evaluation dataset, including dataset size, composition, and preprocessing performed during evaluation.
* Specify the evaluation environment, including hardware, software frameworks, library versions, and runtime configuration to ensure reproducibility.
* Define the evaluation metrics used (e.g., Accuracy, Precision, Recall, F1-score, ROC-AUC, etc.) and justify why they are appropriate for the problem.
* Present quantitative evaluation results using appropriate tables, comparisons across different models, configurations, and hyperparameter settings.
* Include evaluation visualizations such as confusion matrices, ROC curves, Precision-Recall curves, and other task-specific evaluation plots.
* Provide qualitative evaluation results by showcasing representative successful predictions and failure cases.
* Perform error analysis by identifying common error patterns and discussing possible reasons behind incorrect predictions.
* Discuss key observations, limitations, and any notable anomalies observed during model evaluation.

### Task Allocation & Work Done
| Member                    | Specific Tasks Completed                                                                                                                                                                                      
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
| **HITESH** (Lead)         | Led the Milestone 5 activities; coordinated the evaluation process, co-developed evaluation visualizations, performed error analysis, summarized observations and limitations, and compiled the final report. 
| **ANURAG MONDAL**         | Restated the trained model architecture and evaluation pipeline; documented the evaluation dataset, preprocessing workflow, evaluation environment, and reproducibility details.                             
| **D CHIRAG RAO**          | Co-documented the evaluation setup and generated evaluation visualizations, including confusion matrix, ROC curve, Precision-Recall curve, and other supporting plots.                                        
| **BHAVYA JAIN**           | Defined evaluation metrics, justified their selection, and documented metric calculations and interpretations.                                                                                                
| **SOUMYABRATA MAHAPATRA** | Conducted quantitative evaluation, prepared performance comparison tables, analyzed results across different configurations, and co-performed error analysis and observations.                                
| **HITESH BINJRAWAT**      | Performed qualitative analysis by reviewing representative model predictions, highlighting successful predictions and failure cases with supporting explanations.                                             


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

## 📅 Milestone 6: Deployment & Documentation
### Requirements & Lead
**Milestone Lead:** ****  
**Deadline:** 13 August 2026

* Create a clear, audience-friendly presentation covering your project’s objective, methodology, development stages, results, and key learnings, and submit it by next week.
* Write a detailed technical report documenting the project architecture, implementation, tools, pipeline, results, challenges, and solutions from M1 to M6, and submit it by next week.
* Prepare a simple, easy-to-understand report explaining the project’s purpose, impact, and functionality for a general audience.
* Develop a step-by-step user guide with usage instructions, example workflows, and screenshots to help users operate the application easily.
* Compile a developer guide with setup instructions, dependencies, configurations, implementation notes, and complete code to allow others to reproduce the project.
* Deploy the project with a stable, user-friendly interface that includes an instruction page, preset examples, and support for custom user uploads.

### Task Allocation & Work Done
| Member                    | Specific Tasks Completed                                                                                                                                                                                      
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
| **HITESH**                | Worked on creating the Backend, tested the Deployment part and Co-documented the Developer Guide. Working on the Final Technical Report.
| **ANURAG MONDAL**         | Worked on setting up the RAG pipeline and Co-documented the Non-Technical report. Working on the Final Technical Report.                          
| **D CHIRAG RAO**          | Worked on creating the Frontend, Tested the Deployment part and Co-documented the User Guide. Working on the Final Technical Report.                                        
| **BHAVYA JAIN**           | Worked on setting up the RAG pipeline and Co-documented the Non-Technical report. Working on the Final Project Presentation.
| **SOUMYABRATA MAHAPATRA** | Worked on creating the Frontend, tested the Deployment part and Co-documented the User Guide. Working on the Final Project Presentation.                                
| **HITESH BINJRAWAT**      | Worked on creating the Backend, tested the Deployment part and Co-documented the Developer Guide. Working on the Final Project Presentation.                                             


### Peer Acknowledgment Matrix
| Task Owner | Anurag M. | Bhavya J. | Chirag R. | Hitesh | Hitesh B. | Soumyabrata M. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ANURAG MONDAL** | — | [x] | [x] | [x] | [x] | [x] |
| **BHAVYA JAIN** | [x] | — | [x] | [x] | [x] | [x] |
| **D CHIRAG RAO** | [x] | [x] | — | [x] | [x] | [x] |
| **HITESH** | [x] | [x] | [x] | — | [x] | [x] |
| **HITESH BINJRAWAT** | [x] | [x] | [x] | [x] | — | [x] |
| **SOUMYABRATA MAHAPATRA** | [x] | [x] | [x] | [x] | [x] | — |
