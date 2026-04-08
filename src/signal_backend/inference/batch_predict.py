from __future__ import annotations

from pathlib import Path
from typing import Any

from signal_backend.inference.predict import _load_label_mapping, _predict_probabilities
from signal_backend.models.transformer_classifier import load_saved_transformer


def predict_batch(model_inputs: list[str], artifact_dir: Path, batch_size: int | None = None) -> list[dict[str, Any]]:
    tokenizer, model = load_saved_transformer(Path(artifact_dir))
    id_to_label = _load_label_mapping(Path(artifact_dir))
    probabilities_batch = _predict_probabilities(
        model,
        tokenizer,
        model_inputs,
        batch_size=8 if batch_size is None else batch_size,
    )

    results: list[dict[str, Any]] = []
    for probabilities in probabilities_batch:
        label_id = max(range(len(probabilities)), key=probabilities.__getitem__)
        results.append(
            {
                "label": id_to_label[label_id],
                "label_id": label_id,
                "score": float(probabilities[label_id]),
                "probabilities": probabilities,
            }
        )
    return results
