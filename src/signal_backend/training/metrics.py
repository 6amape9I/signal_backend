from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


@dataclass(slots=True)
class EvaluationResult:
    split_name: str
    metrics: dict[str, Any]
    classification_report: dict[str, Any]
    confusion_matrix_rows: list[dict[str, Any]]


def compute_evaluation_result(
    *,
    split_name: str,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> EvaluationResult:
    metrics = {
        "row_count": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)
        ),
    }
    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    rows: list[dict[str, Any]] = []
    for true_index, true_label in enumerate(labels):
        for pred_index, pred_label in enumerate(labels):
            rows.append(
                {
                    "split": split_name,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": int(matrix[true_index, pred_index]),
                }
            )

    return EvaluationResult(
        split_name=split_name,
        metrics=metrics,
        classification_report=report,
        confusion_matrix_rows=rows,
    )


def confusion_rows_to_frame(results: Sequence[EvaluationResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.extend(result.confusion_matrix_rows)
    return pd.DataFrame(rows)
