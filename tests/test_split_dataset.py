from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_backend.data.split_dataset import create_stratified_split


def _build_dataframe():
    import pandas as pd

    rows = []
    for label in ("economy", "politics", "sports"):
        for index in range(10):
            rows.append(
                {
                    "record_id": f"{label}-{index}",
                    "category_teacher_final": label,
                    "title": f"Title {label} {index}",
                    "text_clean": f"Text {label} {index}",
                    "model_input": f"Input {label} {index}",
                }
            )
    return pd.DataFrame(rows)


def test_split_preserves_total_row_count() -> None:
    dataframe = _build_dataframe()

    split_result = create_stratified_split(dataframe, seed=42)

    assert len(split_result.train_df) + len(split_result.val_df) + len(split_result.test_df) == len(dataframe)


def test_split_is_reproducible_for_same_seed() -> None:
    dataframe = _build_dataframe()

    first_split = create_stratified_split(dataframe, seed=42)
    second_split = create_stratified_split(dataframe, seed=42)

    assert first_split.train_df["record_id"].tolist() == second_split.train_df["record_id"].tolist()
    assert first_split.val_df["record_id"].tolist() == second_split.val_df["record_id"].tolist()
    assert first_split.test_df["record_id"].tolist() == second_split.test_df["record_id"].tolist()


def test_split_subsets_do_not_intersect_by_record_id() -> None:
    dataframe = _build_dataframe()

    split_result = create_stratified_split(dataframe, seed=42)

    train_ids = set(split_result.train_df["record_id"])
    val_ids = set(split_result.val_df["record_id"])
    test_ids = set(split_result.test_df["record_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
