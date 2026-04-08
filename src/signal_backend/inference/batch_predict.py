from __future__ import annotations

from pathlib import Path
from typing import Any

from signal_backend.inference.predictor import predict_batch_from_artifact


def predict_batch(model_inputs: list[str], artifact_dir: Path, batch_size: int | None = None) -> list[dict[str, Any]]:
    return predict_batch_from_artifact(
        artifact_dir,
        [{"model_input": model_input} for model_input in model_inputs],
        batch_size=batch_size,
    )