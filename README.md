# Group 9: Data Science and AI Lab Project

A Python-based Deep Learning research project codebase for the **Data Science & AI Labs** course at **IIT Madras**.

---

## 📖 Project Overview: Multilingual and Code-Mixed Text Compression

This project explores the adaptation of token-classification-based text compression (similar to LLMLingua-2) for multilingual and code-mixed conversational transcripts. 

While state-of-the-art token classification compressors utilize multilingual feature encoders (such as `xlm-roberta-large` and `multilingual-BERT`), they are generally trained on monolingual datasets. We pivot to train a token classifier specifically on code-switched text (blending English, Hindi, and Bengali) to evaluate whether token classification compression holds up when syntactic redundancy is not bound by a single language's grammatical rules.

Since syntactic redundancy, filler words, and structural repetitions behave differently in code-mixed languages compared to standard monolingual texts, demonstrating that a small token classifier can learn these patterns without losing essential semantic meaning is highly valuable for localization pipelines and regional NLP applications.

For a detailed breakdown of the challenges, limitations of current state-of-the-art methods, and our proposed query-aware solution, please refer to the full [problem_statement.md](problem_statement.md).

---

## 🚀 Getting Started

This project is configured to use [uv](https://github.com/astral-sh/uv), an extremely fast Python package and project manager written in Rust. However, you can also set up and run this project using standard `pip` and `venv`.

### Prerequisites
- **Python >= 3.13** (specified in `pyproject.toml`)

---

### Option A: Setup using `uv` (Recommended)

`uv` automatically manages Python versions, virtual environments, and dependencies.

1. **Install `uv`** (if you don't have it installed):
   - **macOS/Linux**:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   - **Windows** (PowerShell):
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/blurrydev/Group-9-DS-and-AI-Lab-Project.git
   cd Group-9-DS-and-AI-Lab-Project
   ```

3. **Install dependencies & set up the environment**:
   ```bash
   uv sync
   ```
   *This command automatically creates a `.venv` directory, installs the project in editable mode, and syncs all dependencies listed in `pyproject.toml`.*

4. **Run the project**:
   ```bash
   uv run main.py
   ```

5. **Adding new dependencies**:
   To add packages to the project (which automatically updates `pyproject.toml`):
   ```bash
   uv add <package_name>
   # Example: uv add torch torchvision
   ```

---

### Option B: Setup using standard `pip` and `venv`

If you prefer not to use `uv`, you can use standard Python toolchains.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/blurrydev/Group-9-DS-and-AI-Lab-Project.git
   cd Group-9-DS-and-AI-Lab-Project
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - **macOS/Linux (Bash/Zsh)**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install the package in editable mode with dependencies**:
   ```bash
   pip install -e .
   ```
   *This reads the configuration from `pyproject.toml` and installs the project dependencies.*

5. **Run the project**:
   ```bash
   python main.py
   ```

---

## 📁 Project Structure

```text
├── data/                 # Project datasets (shared distilled text datasets go here)
├── .gitignore            # Tailored gitignore for Python, IDEs, and Deep Learning (weights/data/logs)
├── pyproject.toml        # Project metadata and dependencies (PEP 621 compliant)
├── README.md             # Setup and developer guide (this file)
├── problem_statement.md  # Detailed project problem statement
├── main.py               # Entrypoint script
└── .python-version       # Python version lock file
```

---

## 🤝 Collaboration & Contribution Guidelines

This repository has a pre-configured `.gitignore` tailored specifically for deep learning research. To maintain a clean and lightweight repository, please adhere to the following rules:

### 1. Data and Datasets
* **Shared Data**: Processed text datasets (e.g., distilled code-mixed JSON/CSV/JSONL files) should be stored in the `data/` directory. These are tracked by Git to allow easy sharing and evaluation among group members.
* **Large/Raw Data**: Do NOT commit heavy raw datasets or binary archives (e.g., large `.zip`, `.tar.gz`, `.h5`, or binary database dumps). Keep raw data in directories like `raw_data/` or `dataset/`, which are completely ignored by Git.

### 2. Model Checkpoints & Weights
* **Do NOT commit model weights or checkpoints** (e.g., `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.onnx`).
* Save your checkpoints in `checkpoints/`, `weights/`, or `outputs/`. These folders are configured to be ignored by Git.

### 3. Logs & Experiment Tracking
* Keep local training logs (like TensorBoard `runs/`, PyTorch Lightning logs, or Weights & Biases `wandb/` outputs) out of Git. They are automatically ignored.

### 4. Jupyter Notebooks
* Notebook checkpoints (`.ipynb_checkpoints/`) are ignored.
* Notebook files (`.ipynb`) are **not** ignored by default, so you can share experiment runs. If you commit notebooks, consider clearing their outputs before committing to keep the file sizes small.
