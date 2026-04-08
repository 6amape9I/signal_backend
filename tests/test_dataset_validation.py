from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_backend.data.load_jsonl import DatasetLoadError
from signal_backend.data.validate_dataset import DatasetValidationError, validate_dataset


def _write_dataset(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        (
            '{"record_id":"%s","category_teacher_final":"%s","title":"%s",'
            '"text_clean":"%s","model_input":"%s"}'
        )
        % (
            row["record_id"],
            row["category_teacher_final"],
            row["title"],
            row["text_clean"],
            row["model_input"],
        )
        for row in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_validate_dataset_rejects_empty_category(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_dataset(
        dataset_path,
        [
            {
                "record_id": "1",
                "category_teacher_final": " ",
                "title": "Title 1",
                "text_clean": "Text 1",
                "model_input": "Input 1",
            },
            {
                "record_id": "2",
                "category_teacher_final": "economy",
                "title": "Title 2",
                "text_clean": "Text 2",
                "model_input": "Input 2",
            },
        ],
    )

    with pytest.raises(DatasetLoadError, match="line 1"):
        validate_dataset(dataset_path)


def test_validate_dataset_rejects_empty_model_input(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_dataset(
        dataset_path,
        [
            {
                "record_id": "1",
                "category_teacher_final": "politics",
                "title": "Title 1",
                "text_clean": "Text 1",
                "model_input": " ",
            },
            {
                "record_id": "2",
                "category_teacher_final": "economy",
                "title": "Title 2",
                "text_clean": "Text 2",
                "model_input": "Input 2",
            },
        ],
    )

    with pytest.raises(DatasetLoadError, match="line 1"):
        validate_dataset(dataset_path)


def test_validate_dataset_detects_duplicate_record_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_dataset(
        dataset_path,
        [
            {
                "record_id": "1",
                "category_teacher_final": "politics",
                "title": "Title 1",
                "text_clean": "Text 1",
                "model_input": "Input 1",
            },
            {
                "record_id": "1",
                "category_teacher_final": "economy",
                "title": "Title 2",
                "text_clean": "Text 2",
                "model_input": "Input 2",
            },
        ],
    )

    with pytest.raises(DatasetValidationError, match="duplicate record_id"):
        validate_dataset(dataset_path)
