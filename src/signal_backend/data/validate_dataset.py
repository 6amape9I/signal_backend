from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from signal_backend.data.load_jsonl import load_dataset_dataframe
from signal_backend.data.schemas import REQUIRED_STRING_FIELDS


class DatasetValidationError(ValueError):
    """Raised when dataset-level validation fails."""


@dataclass(slots=True)
class DatasetValidationResult:
    dataset_path: Path
    dataframe: pd.DataFrame
    row_count: int
    class_names: list[str]
    class_distribution: dict[str, int]
    missing_required_field_counts: dict[str, int]
    duplicate_record_ids_count: int
    column_names: list[str]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path.as_posix(),
            "row_count": self.row_count,
            "column_names": self.column_names,
            "class_distribution": self.class_distribution,
            "duplicate_record_ids_count": self.duplicate_record_ids_count,
            "missing_required_field_counts": self.missing_required_field_counts,
            "class_names": self.class_names,
        }


def _count_missing_required_fields(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for field_name in REQUIRED_STRING_FIELDS:
        if field_name not in df.columns:
            counts[field_name] = len(df)
            continue

        values = df[field_name]
        counts[field_name] = int(
            values.isna().sum()
            + values.astype(str).map(lambda value: not value.strip()).sum()
        )
    return counts


def validate_dataset(path: Path) -> DatasetValidationResult:
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise DatasetValidationError(f"Dataset file does not exist: {dataset_path}")
    if not dataset_path.is_file():
        raise DatasetValidationError(f"Dataset path is not a file: {dataset_path}")
    if dataset_path.stat().st_size == 0:
        raise DatasetValidationError(f"Dataset file is empty: {dataset_path}")

    df = load_dataset_dataframe(dataset_path)

    if df.empty:
        raise DatasetValidationError("Dataset contains zero records.")

    missing_required_field_counts = _count_missing_required_fields(df)
    invalid_required_fields = {
        field_name: count
        for field_name, count in missing_required_field_counts.items()
        if count > 0
    }
    if invalid_required_fields:
        raise DatasetValidationError(
            f"Dataset has missing required values: {invalid_required_fields}"
        )

    duplicate_record_ids_count = int(df["record_id"].duplicated().sum())
    if duplicate_record_ids_count > 0:
        raise DatasetValidationError(
            f"Dataset contains duplicate record_id values: {duplicate_record_ids_count}"
        )

    class_distribution_series = (
        df["category_teacher_final"].value_counts(dropna=False).sort_index()
    )
    class_names = class_distribution_series.index.tolist()
    if len(class_names) < 2:
        raise DatasetValidationError(
            "Dataset must contain at least 2 distinct classes in "
            "'category_teacher_final'."
        )

    return DatasetValidationResult(
        dataset_path=dataset_path,
        dataframe=df,
        row_count=len(df),
        class_names=class_names,
        class_distribution={
            str(label): int(count)
            for label, count in class_distribution_series.items()
        },
        missing_required_field_counts=missing_required_field_counts,
        duplicate_record_ids_count=duplicate_record_ids_count,
        column_names=df.columns.tolist(),
    )
