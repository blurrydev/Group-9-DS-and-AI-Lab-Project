# Group 9: Data Science and AI Lab Project

A Python-based Deep Learning research project template and codebase.

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
├── .gitignore          # Tailored gitignore for Python, IDEs, and Deep Learning (weights/data/logs)
├── pyproject.toml      # Project metadata and dependencies (PEP 621 compliant)
├── README.md           # Setup and developer guide (this file)
├── main.py             # Entrypoint script
└── .python-version     # Python version lock file
```

---

## 🤝 Collaboration & Contribution Guidelines

This repository has a pre-configured `.gitignore` tailored specifically for deep learning research. To maintain a clean and lightweight repository, please adhere to the following rules:

### 1. Data and Datasets
* **Do NOT commit raw data or datasets** to Git.
* Store all data in a local folder named `data/` or `datasets/` at the root of the project. These folders are automatically ignored.

### 2. Model Checkpoints & Weights
* **Do NOT commit model weights or checkpoints** (e.g., `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.onnx`).
* Save your checkpoints in `checkpoints/`, `weights/`, or `outputs/`. These folders are configured to be ignored by Git.

### 3. Logs & Experiment Tracking
* Keep local training logs (like TensorBoard `runs/`, PyTorch Lightning logs, or Weights & Biases `wandb/` outputs) out of Git. They are automatically ignored.

### 4. Jupyter Notebooks
* Notebook checkpoints (`.ipynb_checkpoints/`) are ignored.
* Notebook files (`.ipynb`) are **not** ignored by default, so you can share experiment runs. If you commit notebooks, consider clearing their outputs before committing to keep the file sizes small.
