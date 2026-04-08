from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from signal_backend.data.schemas import DatasetRecord


def _extract_labels(data: pd.DataFrame | Iterable[DatasetRecord] | Iterable[str]) -> list[str]:
    if isinstance(data, pd.DataFrame):
        if "category_teacher_final" not in data.columns:
            raise KeyError("DataFrame must contain 'category_teacher_final'.")
        labels = data["category_teacher_final"].tolist()
    else:
        labels = []
        for item in data:
            if isinstance(item, DatasetRecord):
                labels.append(item.category_teacher_final)
            elif isinstance(item, str):
                labels.append(item)
            else:
                candidate = getattr(item, "category_teacher_final", None)
                if candidate is None:
                    raise TypeError(
                        "Unsupported label source. Expected DataFrame, DatasetRecord, "
                        "objects with 'category_teacher_final', or strings."
                    )
                labels.append(candidate)
    return sorted(set(labels))


def build_label_to_id(
    data: pd.DataFrame | Iterable[DatasetRecord] | Iterable[str],
) -> dict[str, int]:
    return {label: index for index, label in enumerate(_extract_labels(data))}


def build_id_to_label(
    data: pd.DataFrame | Iterable[DatasetRecord] | Iterable[str],
) -> dict[int, str]:
    label_to_id = build_label_to_id(data)
    return {index: label for label, index in label_to_id.items()}
