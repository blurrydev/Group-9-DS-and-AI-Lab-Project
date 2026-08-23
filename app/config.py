from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]



@dataclass
class Settings:
    app_name: str = os.getenv("APP_NAME", "Query-Aware Prompt Compressor")
    app_env: str = os.getenv("APP_ENV", "development")
    model_dir: Path = Path(os.getenv("MODEL_DIR", str(ROOT_DIR / "submission_artifacts" / "model")))
    device: str = os.getenv("DEVICE", "cpu")
    max_length: int = int(os.getenv("MAX_LENGTH", "512"))
    cors_origins_raw: str = os.getenv("CORS_ORIGINS", "*")

    @property
    def cors_origins(self) -> list[str]:
        raw = self.cors_origins_raw.strip()
        if raw == "*":
            return ["*"]
        return [item.strip() for item in raw.split(",") if item.strip()]


settings = Settings()
