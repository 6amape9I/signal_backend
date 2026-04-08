from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from signal_backend.data.schemas import DatasetRecord


LABEL_COLUMN = "category_teacher_final"
RECORD_ID_COLUMN = "record_id"


class DatasetSplitError(ValueError):
    """Raised when a dataset cannot be split as requested."""


@dataclass(slots=True)
class SplitResult:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    assigned_df: pd.DataFrame
    seed: int
    train_size: float
    val_size: float
    test_size: float


def _to_dataframe(
    data: pd.DataFrame | list[DatasetRecord] | list[dict[str, Any]],
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    rows: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, DatasetRecord):
            rows.append(item.model_dump())
        elif isinstance(item, dict):
            rows.append(item)
        else:
            raise TypeError("Unsupported dataset container for split creation.")
    return pd.DataFrame(rows)


def _validate_split_sizes(
    train_size: float,
    val_size: float,
    test_size: float,
) -> None:
    sizes = {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }

    for name, value in sizes.items():
        if value <= 0 or value >= 1:
            raise DatasetSplitError(f"{name} must be between 0 and 1, got {value}.")

    total = train_size + val_size + test_size
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise DatasetSplitError(f"Split sizes must sum to 1.0, got {total:.12f}.")


def _validate_split_feasibility(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
) -> None:
    if df.empty:
        raise DatasetSplitError("Cannot split an empty dataset.")

    if LABEL_COLUMN not in df.columns:
        raise DatasetSplitError(
            f"DataFrame must contain '{LABEL_COLUMN}' for stratified split."
        )
    if RECORD_ID_COLUMN not in df.columns:
        raise DatasetSplitError(
            f"DataFrame must contain '{RECORD_ID_COLUMN}' for split integrity checks."
        )

    class_counts = df[LABEL_COLUMN].value_counts().sort_index()
    num_classes = int(class_counts.size)
    min_class_count = int(class_counts.min())

    if min_class_count < 3:
        raise DatasetSplitError(
            "Each class must contain at least 3 records to create non-empty "
            f"train/val/test splits. Current minimum class size: {min_class_count}. "
            f"Class distribution: {class_counts.to_dict()}"
        )

    total_rows = len(df)
    for split_name, split_size in (("train", train_size), ("val", val_size), ("test", test_size)):
        if total_rows * split_size < num_classes:
            raise DatasetSplitError(
                f"Split '{split_name}' is too small to represent all {num_classes} "
                f"classes. Requested size {split_size} for {total_rows} rows."
            )


def create_stratified_split(
    data: pd.DataFrame | list[DatasetRecord] | list[dict[str, Any]],
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> SplitResult:
    _validate_split_sizes(train_size, val_size, test_size)

    df = _to_dataframe(data)
    _validate_split_feasibility(df, train_size, val_size, test_size)

    holdout_size = val_size + test_size

    try:
        train_df, holdout_df = train_test_split(
            df,
            test_size=holdout_size,
            random_state=seed,
            stratify=df[LABEL_COLUMN],
        )

        relative_test_size = test_size / holdout_size
        val_df, test_df = train_test_split(
            holdout_df,
            test_size=relative_test_size,
            random_state=seed,
            stratify=holdout_df[LABEL_COLUMN],
        )
    except ValueError as exc:
        raise DatasetSplitError(
            "Unable to create a stratified split with the requested proportions. "
            f"Original error: {exc}"
        ) from exc

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    assigned_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    if len(assigned_df) != len(df):
        raise DatasetSplitError("Split output row count does not match input row count.")

    unique_record_ids = assigned_df[RECORD_ID_COLUMN].nunique()
    if unique_record_ids != len(df):
        raise DatasetSplitError("Split output contains duplicate or missing record_id values.")

    return SplitResult(
        train_df=train_df.drop(columns=["split"]),
        val_df=val_df.drop(columns=["split"]),
        test_df=test_df.drop(columns=["split"]),
        assigned_df=assigned_df,
        seed=seed,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _write_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in df.to_dict(orient="records"):
            cleaned_row = {key: _sanitize_json_value(value) for key, value in row.items()}
            output_file.write(json.dumps(cleaned_row, ensure_ascii=False) + "\n")


def save_split_files(
    split_result: SplitResult,
    processed_dir: Path,
) -> dict[str, Path]:
    output_paths = {
        "train": processed_dir / "train.jsonl",
        "val": processed_dir / "val.jsonl",
        "test": processed_dir / "test.jsonl",
    }

    _write_jsonl(split_result.train_df, output_paths["train"])
    _write_jsonl(split_result.val_df, output_paths["val"])
    _write_jsonl(split_result.test_df, output_paths["test"])
    return output_paths


def _distribution(df: pd.DataFrame) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in df[LABEL_COLUMN].value_counts().sort_index().items()
    }


def build_split_report(split_result: SplitResult) -> dict[str, Any]:
    return {
        "total_rows": len(split_result.assigned_df),
        "train_rows": len(split_result.train_df),
        "val_rows": len(split_result.val_df),
        "test_rows": len(split_result.test_df),
        "seed": split_result.seed,
        "train_size": split_result.train_size,
        "val_size": split_result.val_size,
        "test_size": split_result.test_size,
        "overall_class_distribution": _distribution(split_result.assigned_df),
        "train_class_distribution": _distribution(split_result.train_df),
        "val_class_distribution": _distribution(split_result.val_df),
        "test_class_distribution": _distribution(split_result.test_df),
    }


def save_split_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
