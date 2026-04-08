from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from signal_backend.data.label_mapping import build_id_to_label, build_label_to_id
from signal_backend.data.load_jsonl import load_dataset_dataframe
from signal_backend.paths import TEST_SPLIT_PATH, TRAIN_SPLIT_PATH, VAL_SPLIT_PATH
from signal_backend.training.metrics import EvaluationResult, compute_evaluation_result


TEXT_COLUMN = "model_input"
LABEL_COLUMN = "category_teacher_final"


class SplitFilesError(FileNotFoundError):
    """Raised when train/val/test files are missing."""


@dataclass(slots=True)
class SplitDataBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_path: Path
    val_path: Path
    test_path: Path


@dataclass(slots=True)
class PredictionOutput:
    predicted_labels: list[str]
    predicted_label_ids: list[int]
    probabilities: list[list[float]] | None = None
    decision_scores: list[list[float]] | None = None
    score_type: str | None = None


def load_split_data(
    train_path: Path = TRAIN_SPLIT_PATH,
    val_path: Path = VAL_SPLIT_PATH,
    test_path: Path = TEST_SPLIT_PATH,
) -> SplitDataBundle:
    for required_path in (train_path, val_path, test_path):
        if not Path(required_path).exists():
            raise SplitFilesError(
                f"Missing split file: {required_path}. Run 'python scripts/make_split.py' first."
            )

    return SplitDataBundle(
        train_df=load_dataset_dataframe(train_path),
        val_df=load_dataset_dataframe(val_path),
        test_df=load_dataset_dataframe(test_path),
        train_path=Path(train_path),
        val_path=Path(val_path),
        test_path=Path(test_path),
    )


def build_train_label_mapping(train_df: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    label_to_id = build_label_to_id(train_df)
    id_to_label = build_id_to_label(train_df)
    return label_to_id, id_to_label


def dataframe_texts(df: pd.DataFrame, text_column: str = TEXT_COLUMN) -> list[str]:
    return df[text_column].astype(str).tolist()


def dataframe_labels(df: pd.DataFrame, label_column: str = LABEL_COLUMN) -> list[str]:
    return df[label_column].astype(str).tolist()


def evaluate_dataframe(
    *,
    split_name: str,
    df: pd.DataFrame,
    labels: Sequence[str],
    predictor: Callable[[list[str]], PredictionOutput],
    text_column: str = TEXT_COLUMN,
    label_column: str = LABEL_COLUMN,
) -> tuple[EvaluationResult, PredictionOutput]:
    predictions = predictor(dataframe_texts(df, text_column=text_column))
    result = compute_evaluation_result(
        split_name=split_name,
        y_true=dataframe_labels(df, label_column=label_column),
        y_pred=predictions.predicted_labels,
        labels=labels,
    )
    return result, predictions


def default_test_dataset_path() -> Path:
    return TEST_SPLIT_PATH
