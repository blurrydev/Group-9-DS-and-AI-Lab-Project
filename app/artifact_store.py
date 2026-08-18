from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import ROOT_DIR


@dataclass
class RunDefinition:
    id: str
    name: str
    folder: Path
    metrics_file: str | None = "validation_metrics.json"
    test_file: str | None = "test_metrics.json"
    history_file: str | None = "training_history.csv"
    config_file: str | None = "config.json"
    artifact_prefix: str | None = None
    validation_override: dict[str, Any] | None = None
    test_override: dict[str, Any] | None = None
    config_override: dict[str, Any] | None = None


RUNS: dict[str, RunDefinition] = {
    "baseline": RunDefinition(
        id="baseline",
        name="Baseline",
        folder=ROOT_DIR / "Experiment" / "baseline",
    ),
    "hyperparameter_tuning": RunDefinition(
        id="hyperparameter_tuning",
        name="Hyperparameter Tuning (Best)",
        folder=ROOT_DIR / "Experiment" / "hyperparameter_tunning_results",
        config_file="best_config.json",
    ),
    "hyperparameter_tuning_exp_0": RunDefinition(
        id="hyperparameter_tuning_exp_0",
        name="Hyperparameter Exp 0",
        folder=ROOT_DIR / "Experiment" / "hyperparameter_tunning_results",
        metrics_file=None,
        test_file=None,
        history_file=None,
        config_file="best_config.json",
        artifact_prefix="hp_experiment_0",
        validation_override={
            "eval_loss": 0.933803,
            "eval_accuracy": 0.738928,
            "eval_precision": 0.546830,
            "eval_recall": 0.463606,
            "eval_f1": 0.501791,
            "epoch": 3.0,
        },
        config_override={
            "learning_rate": 1e-5,
            "per_device_train_batch_size": 8,
            "weight_decay": 0.01,
            "warmup_steps": 100,
        },
    ),
    "hyperparameter_tuning_exp_1": RunDefinition(
        id="hyperparameter_tuning_exp_1",
        name="Hyperparameter Exp 1",
        folder=ROOT_DIR / "Experiment" / "hyperparameter_tunning_results",
        metrics_file=None,
        test_file=None,
        history_file=None,
        config_file="best_config.json",
        artifact_prefix="hp_experiment_1",
        validation_override={
            "eval_loss": 0.912988,
            "eval_accuracy": 0.744509,
            "eval_precision": 0.551171,
            "eval_recall": 0.533633,
            "eval_f1": 0.542260,
            "epoch": 3.0,
        },
        config_override={
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 8,
            "weight_decay": 0.01,
            "warmup_steps": 100,
        },
    ),
    "hyperparameter_tuning_exp_2": RunDefinition(
        id="hyperparameter_tuning_exp_2",
        name="Hyperparameter Exp 2",
        folder=ROOT_DIR / "Experiment" / "hyperparameter_tunning_results",
        metrics_file=None,
        test_file=None,
        history_file=None,
        config_file="best_config.json",
        artifact_prefix="hp_experiment_2",
        validation_override={
            "eval_loss": 0.932724,
            "eval_accuracy": 0.723932,
            "eval_precision": 0.518929,
            "eval_recall": 0.363600,
            "eval_f1": 0.427595,
            "epoch": 3.0,
        },
        config_override={
            "learning_rate": 3e-5,
            "per_device_train_batch_size": 16,
            "weight_decay": 0.01,
            "warmup_steps": 200,
        },
    ),
    "hyperparameter_tuning_exp_3": RunDefinition(
        id="hyperparameter_tuning_exp_3",
        name="Hyperparameter Exp 3",
        folder=ROOT_DIR / "Experiment" / "hyperparameter_tunning_results",
        metrics_file=None,
        test_file=None,
        history_file=None,
        config_file="best_config.json",
        artifact_prefix="hp_experiment_3",
        validation_override={
            "eval_loss": 0.920724,
            "eval_accuracy": 0.748148,
            "eval_precision": 0.575971,
            "eval_recall": 0.424248,
            "eval_f1": 0.488602,
            "epoch": 3.0,
        },
        config_override={
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 8,
            "weight_decay": 0.1,
            "warmup_steps": 100,
        },
    ),
    "submission": RunDefinition(
        id="submission",
        name="Submission Artifacts",
        folder=ROOT_DIR / "submission_artifacts",
    ),
}


def _read_run_json(run: RunDefinition, file_name: str | None) -> dict[str, Any]:
    if not file_name:
        return {}
    return _read_json(run.folder / file_name)


def _run_validation(run: RunDefinition) -> dict[str, Any]:
    if run.validation_override is not None:
        return dict(run.validation_override)
    return _read_run_json(run, run.metrics_file)


def _run_test(run: RunDefinition) -> dict[str, Any]:
    if run.test_override is not None:
        return dict(run.test_override)
    return _read_run_json(run, run.test_file)


def _run_config(run: RunDefinition) -> dict[str, Any]:
    if run.config_override is not None:
        return dict(run.config_override)
    return _read_run_json(run, run.config_file)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def get_run(run_id: str) -> RunDefinition:
    run = RUNS.get(run_id)
    if run is None:
        raise KeyError(f"Unknown run id: {run_id}")
    return run


def list_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in RUNS.values():
        validation = _run_validation(run)
        test = _run_test(run)
        rows.append(
            {
                "id": run.id,
                "name": run.name,
                "source_path": str(run.folder.relative_to(ROOT_DIR)).replace("\\", "/"),
                "validation_f1": _safe_float(validation.get("eval_f1")),
                "validation_loss": _safe_float(validation.get("eval_loss")),
                "test_f1": _safe_float(test.get("eval_f1")),
                "test_loss": _safe_float(test.get("eval_loss")),
            }
        )
    return rows


def get_run_metrics(run_id: str) -> dict[str, Any]:
    run = get_run(run_id)
    return {
        "run": {
            "id": run.id,
            "name": run.name,
            "source_path": str(run.folder.relative_to(ROOT_DIR)).replace("\\", "/"),
        },
        "validation": _run_validation(run),
        "test": _run_test(run),
        "config": _run_config(run),
    }


def get_run_history(run_id: str) -> dict[str, Any]:
    run = get_run(run_id)
    if not run.history_file:
        return {
            "run_id": run_id,
            "training_loss": [],
            "validation_loss": [],
            "validation_f1": [],
        }

    history_path = run.folder / run.history_file

    if not history_path.exists():
        return {
            "run_id": run_id,
            "training_loss": [],
            "validation_loss": [],
            "validation_f1": [],
        }

    training_loss: list[dict[str, float]] = []
    validation_loss: list[dict[str, float]] = []
    validation_f1: list[dict[str, float]] = []

    with history_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = _safe_float(row.get("step"))
            epoch = _safe_float(row.get("epoch"))
            loss = _safe_float(row.get("loss"))
            eval_loss = _safe_float(row.get("eval_loss"))
            eval_f1 = _safe_float(row.get("eval_f1"))

            if step is not None and loss is not None:
                training_loss.append({"x": step, "y": loss})

            if epoch is not None and eval_loss is not None:
                validation_loss.append({"x": epoch, "y": eval_loss})

            if epoch is not None and eval_f1 is not None:
                validation_f1.append({"x": epoch, "y": eval_f1})

    validation_loss = _deduplicate_points(validation_loss)
    validation_f1 = _deduplicate_points(validation_f1)

    return {
        "run_id": run_id,
        "training_loss": training_loss,
        "validation_loss": validation_loss,
        "validation_f1": validation_f1,
    }


def _deduplicate_points(points: list[dict[str, float]]) -> list[dict[str, float]]:
    # Keep latest value for each x coordinate while preserving sorted output.
    dedup: dict[float, float] = {}
    for point in points:
        dedup[point["x"]] = point["y"]
    return [{"x": x, "y": dedup[x]} for x in sorted(dedup)]


def list_artifacts(run_id: str) -> list[str]:
    run = get_run(run_id)
    if not run.folder.exists():
        return []
    artifacts: list[str] = []
    for candidate in sorted(run.folder.iterdir()):
        if not candidate.is_file():
            continue
        if run.artifact_prefix and not candidate.name.startswith(run.artifact_prefix):
            continue
        if candidate.suffix.lower() in {".json", ".csv", ".png"}:
            artifacts.append(candidate.name)
    return artifacts


def get_artifact_path(run_id: str, artifact_name: str) -> Path:
    run = get_run(run_id)
    candidate = (run.folder / artifact_name).resolve()
    if run.folder.resolve() not in candidate.parents and candidate != run.folder.resolve():
        raise ValueError("Invalid artifact path")
    return candidate
