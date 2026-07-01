# Project Data Directory

This directory stores the datasets used for the **Multilingual and Code-Mixed Compression** project.

### 📋 Storage Conventions

1. **Shared Distilled/Processed Datasets**:
   - Small text-based datasets (distilled code-switched transcripts combining English, Hindi, and Bengali, e.g., in `.json`, `.csv`, or `.jsonl` formats) should be placed here.
   - These files are tracked by Git to ensure all team members have access to the same ground-truth datasets for training and evaluation.

2. **Large/Raw Datasets**:
   - Do **NOT** place raw dataset dumps (e.g. raw HingCorp, SAIL, or massive binary archives) directly in this directory if they exceed 50 MB.
   - For heavy raw files, use the `raw_data/` or `dataset/` directories at the project root. These are listed in `.gitignore` and will not be pushed to GitHub.
