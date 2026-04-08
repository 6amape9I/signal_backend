from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import torch

from signal_backend.models.transformer_classifier import load_saved_transformer


BASELINE_MODEL_TYPES = {"tfidf_logreg", "tfidf_linear_svm"}
TRANSFORMER_MODEL_TYPE = "transformer_classifier"


@dataclass(slots=True)
class BaselineRuntime:
    model: Any
    vectorizer: Any


@dataclass(slots=True)
class TransformerRuntime:
    model: Any
    tokenizer: Any
    device: torch.device


@dataclass(slots=True)
class LoadedArtifact:
    artifact_dir: Path
    model_type: str
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]
    runtime: BaselineRuntime | TransformerRuntime
    run_summary: dict[str, Any]


_ARTIFACT_CACHE: dict[Path, LoadedArtifact] = {}


def _require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _load_mapping(mapping_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = json.loads(_require_path(mapping_path, "label mapping").read_text(encoding="utf-8"))
    return (
        {str(key): int(value) for key, value in payload["label_to_id"].items()},
        {int(key): value for key, value in payload["id_to_label"].items()},
    )


def _load_run_summary(artifact_dir: Path) -> dict[str, Any]:
    summary_path = _require_path(artifact_dir / "run_summary.json", "run summary")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if "model_type" not in payload:
        raise ValueError(f"'model_type' is missing in run summary: {summary_path}")
    return payload


def _load_baseline_artifact(artifact_dir: Path, model_type: str) -> LoadedArtifact:
    label_to_id, id_to_label = _load_mapping(artifact_dir / "label_mapping.json")
    model_path = _require_path(artifact_dir / "model.joblib", "baseline model artifact")
    vectorizer_path = _require_path(artifact_dir / "vectorizer.joblib", "baseline vectorizer artifact")
    run_summary = _load_run_summary(artifact_dir)
    return LoadedArtifact(
        artifact_dir=artifact_dir,
        model_type=model_type,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        runtime=BaselineRuntime(
            model=joblib.load(model_path),
            vectorizer=joblib.load(vectorizer_path),
        ),
        run_summary=run_summary,
    )


def _load_transformer_artifact(artifact_dir: Path, model_type: str) -> LoadedArtifact:
    label_to_id, id_to_label = _load_mapping(artifact_dir / "label_mapping.json")
    _require_path(artifact_dir / "best_model", "transformer best_model directory")
    _require_path(artifact_dir / "tokenizer", "transformer tokenizer directory")
    tokenizer, model = load_saved_transformer(artifact_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return LoadedArtifact(
        artifact_dir=artifact_dir,
        model_type=model_type,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        runtime=TransformerRuntime(
            model=model,
            tokenizer=tokenizer,
            device=device,
        ),
        run_summary=_load_run_summary(artifact_dir),
    )


def load_artifact(artifact_dir: Path) -> LoadedArtifact:
    resolved_dir = Path(artifact_dir).resolve()
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Artifact directory does not exist: {resolved_dir}")
    if not resolved_dir.is_dir():
        raise ValueError(f"Artifact path is not a directory: {resolved_dir}")

    run_summary = _load_run_summary(resolved_dir)
    model_type = str(run_summary["model_type"])
    if model_type in BASELINE_MODEL_TYPES:
        return _load_baseline_artifact(resolved_dir, model_type)
    if model_type == TRANSFORMER_MODEL_TYPE:
        return _load_transformer_artifact(resolved_dir, model_type)
    raise ValueError(f"Unsupported model_type in artifact: {model_type}")


def get_cached_artifact(artifact_dir: Path) -> LoadedArtifact:
    resolved_dir = Path(artifact_dir).resolve()
    loaded = _ARTIFACT_CACHE.get(resolved_dir)
    if loaded is None:
        loaded = load_artifact(resolved_dir)
        _ARTIFACT_CACHE[resolved_dir] = loaded
    return loaded


def clear_artifact_cache() -> None:
    _ARTIFACT_CACHE.clear()