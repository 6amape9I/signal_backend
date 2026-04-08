from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from signal_backend.inference.artifact_loader import (
    BASELINE_MODEL_TYPES,
    TRANSFORMER_MODEL_TYPE,
    BaselineRuntime,
    LoadedArtifact,
    TransformerRuntime,
    get_cached_artifact,
)
from signal_backend.inference.model_input import build_model_input


def _label_order(loaded_artifact: LoadedArtifact) -> list[str]:
    return [loaded_artifact.id_to_label[index] for index in sorted(loaded_artifact.id_to_label)]


def _scores_to_mapping(label_order: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    return {label: float(score) for label, score in zip(label_order, values, strict=True)}


def _normalize_decision_scores(scores: Any) -> list[list[float]]:
    array = np.asarray(scores)
    if array.ndim == 1:
        return [[float(-value), float(value)] for value in array.tolist()]
    return [[float(item) for item in row] for row in array.tolist()]


def _predict_with_baseline(
    runtime: BaselineRuntime,
    model_type: str,
    model_inputs: Sequence[str],
    label_order: Sequence[str],
    id_to_label: dict[int, str],
) -> list[dict[str, Any]]:
    matrix = runtime.vectorizer.transform(model_inputs)
    predicted_ids = [int(item) for item in runtime.model.predict(matrix).tolist()]

    if model_type == "tfidf_logreg":
        score_rows = [[float(item) for item in row] for row in runtime.model.predict_proba(matrix).tolist()]
        score_type = "probabilities"
    else:
        score_rows = _normalize_decision_scores(runtime.model.decision_function(matrix))
        score_type = "decision_scores"

    return [
        {
            "prediction": id_to_label[predicted_id],
            "label_id": predicted_id,
            "scores": _scores_to_mapping(label_order, score_row),
            "score_type": score_type,
            "label_order": list(label_order),
            "model_type": model_type,
        }
        for predicted_id, score_row in zip(predicted_ids, score_rows, strict=True)
    ]


def _predict_with_transformer(
    runtime: TransformerRuntime,
    model_inputs: Sequence[str],
    label_order: Sequence[str],
    id_to_label: dict[int, str],
    batch_size: int | None,
) -> list[dict[str, Any]]:
    effective_batch_size = 8 if batch_size is None else int(batch_size)
    outputs: list[dict[str, Any]] = []

    with torch.no_grad():
        for start in range(0, len(model_inputs), effective_batch_size):
            batch_inputs = list(model_inputs[start : start + effective_batch_size])
            encoded = runtime.tokenizer(
                batch_inputs,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(runtime.device) for key, value in encoded.items()}
            probabilities = torch.softmax(runtime.model(**encoded).logits, dim=-1)
            for row in probabilities.detach().cpu().tolist():
                label_id = max(range(len(row)), key=row.__getitem__)
                outputs.append(
                    {
                        "prediction": id_to_label[label_id],
                        "label_id": label_id,
                        "scores": _scores_to_mapping(label_order, row),
                        "score_type": "probabilities",
                        "label_order": list(label_order),
                        "model_type": TRANSFORMER_MODEL_TYPE,
                    }
                )

    return outputs


def predict_batch(
    loaded_artifact: LoadedArtifact,
    model_inputs: list[str],
    batch_size: int | None = None,
) -> list[dict[str, Any]]:
    label_order = _label_order(loaded_artifact)
    if loaded_artifact.model_type in BASELINE_MODEL_TYPES:
        return _predict_with_baseline(
            loaded_artifact.runtime,
            loaded_artifact.model_type,
            model_inputs,
            label_order,
            loaded_artifact.id_to_label,
        )
    if loaded_artifact.model_type == TRANSFORMER_MODEL_TYPE:
        return _predict_with_transformer(
            loaded_artifact.runtime,
            model_inputs,
            label_order,
            loaded_artifact.id_to_label,
            batch_size,
        )
    raise ValueError(f"Unsupported model_type: {loaded_artifact.model_type}")


def predict_one(loaded_artifact: LoadedArtifact, model_input: str) -> dict[str, Any]:
    return predict_batch(loaded_artifact, [model_input], batch_size=1)[0]


def predict_one_from_artifact(
    artifact_dir,
    *,
    title: str | None = None,
    text: str | None = None,
    model_input: str | None = None,
) -> dict[str, Any]:
    loaded_artifact = get_cached_artifact(artifact_dir)
    prepared_input = build_model_input(title=title, text=text, model_input=model_input)
    return predict_one(loaded_artifact, prepared_input)


def predict_batch_from_artifact(
    artifact_dir,
    items: list[dict],
    *,
    batch_size: int | None = None,
) -> list[dict[str, Any]]:
    loaded_artifact = get_cached_artifact(artifact_dir)
    prepared_inputs = [
        build_model_input(
            title=item.get("title"),
            text=item.get("text"),
            model_input=item.get("model_input"),
        )
        for item in items
    ]
    return predict_batch(loaded_artifact, prepared_inputs, batch_size=batch_size)