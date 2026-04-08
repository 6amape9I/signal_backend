from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from signal_backend.models.transformer_classifier import load_saved_transformer


def _load_label_mapping(artifact_dir: Path) -> dict[int, str]:
    payload = json.loads((Path(artifact_dir) / "label_mapping.json").read_text(encoding="utf-8"))
    return {int(key): value for key, value in payload["id_to_label"].items()}


def _predict_probabilities(model, tokenizer, model_inputs: list[str], batch_size: int = 8) -> list[list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    outputs: list[list[float]] = []

    with torch.no_grad():
        for start in range(0, len(model_inputs), batch_size):
            batch_inputs = model_inputs[start : start + batch_size]
            encoded = tokenizer(
                batch_inputs,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)
            outputs.extend(probabilities.detach().cpu().tolist())

    return outputs


def predict_one(model_input: str, artifact_dir: Path) -> dict[str, Any]:
    tokenizer, model = load_saved_transformer(Path(artifact_dir))
    id_to_label = _load_label_mapping(Path(artifact_dir))
    probabilities = _predict_probabilities(model, tokenizer, [model_input], batch_size=1)[0]
    label_id = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
        "label": id_to_label[label_id],
        "label_id": label_id,
        "score": float(probabilities[label_id]),
        "probabilities": probabilities,
    }
