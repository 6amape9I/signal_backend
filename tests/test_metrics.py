from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_backend.training.metrics import compute_evaluation_result, confusion_rows_to_frame


def test_compute_evaluation_result_returns_expected_keys() -> None:
    result = compute_evaluation_result(
        split_name="test",
        y_true=["a", "a", "b", "b"],
        y_pred=["a", "b", "b", "b"],
        labels=["a", "b"],
    )

    assert result.split_name == "test"
    assert {"row_count", "accuracy", "macro_f1", "weighted_f1"}.issubset(result.metrics)
    assert "a" in result.classification_report
    assert "b" in result.classification_report


def test_confusion_rows_to_frame_contains_long_form_rows() -> None:
    result = compute_evaluation_result(
        split_name="val",
        y_true=["a", "b"],
        y_pred=["a", "b"],
        labels=["a", "b"],
    )

    frame = confusion_rows_to_frame([result])

    assert list(frame.columns) == ["split", "true_label", "pred_label", "count"]
    assert len(frame) == 4
    assert set(frame["split"]) == {"val"}
