from __future__ import annotations

from pathlib import Path
from typing import Any

from signal_backend.inference.predictor import predict_one_from_artifact


def predict_one(model_input: str, artifact_dir: Path) -> dict[str, Any]:
    return predict_one_from_artifact(artifact_dir, model_input=model_input)