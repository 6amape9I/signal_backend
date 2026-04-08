from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def _is_local_model_source(model_name_or_path: str | Path) -> bool:
    return Path(model_name_or_path).exists()


def load_tokenizer(model_name_or_path: str | Path):
    source = str(model_name_or_path)
    if _is_local_model_source(model_name_or_path):
        return AutoTokenizer.from_pretrained(source, local_files_only=True)
    return AutoTokenizer.from_pretrained(source)


def build_transformer_classifier(
    model_name_or_path: str | Path,
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
):
    source = str(model_name_or_path)
    common_kwargs = {
        "num_labels": len(label_to_id),
        "label2id": label_to_id,
        "id2label": {int(key): value for key, value in id_to_label.items()},
    }

    if _is_local_model_source(model_name_or_path):
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                source,
                local_files_only=True,
                **common_kwargs,
            )
            return model, True
        except OSError:
            config = AutoConfig.from_pretrained(source, local_files_only=True)
            config.num_labels = len(label_to_id)
            config.label2id = label_to_id
            config.id2label = {int(key): value for key, value in id_to_label.items()}
            model = AutoModelForSequenceClassification.from_config(config)
            return model, False

    model = AutoModelForSequenceClassification.from_pretrained(source, **common_kwargs)
    return model, True


def load_saved_transformer(artifact_dir: Path):
    artifact_path = Path(artifact_dir)
    tokenizer = AutoTokenizer.from_pretrained(artifact_path / "tokenizer", local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        artifact_path / "best_model",
        local_files_only=True,
    )
    return tokenizer, model
