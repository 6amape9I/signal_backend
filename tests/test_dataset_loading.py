from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_backend.data.load_jsonl import DatasetLoadError, load_dataset_dataframe, load_dataset_records


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_dataset_records_reads_jsonl(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(
        dataset_path,
        [
            '{"record_id":"1","category_teacher_final":"politics","title":"Title 1","text_clean":"Text 1","model_input":"Input 1"}',
            '{"record_id":"2","category_teacher_final":"economy","title":"Title 2","text_clean":"Text 2","model_input":"Input 2"}',
        ],
    )

    records = load_dataset_records(dataset_path)
    dataframe = load_dataset_dataframe(dataset_path)

    assert len(records) == 2
    assert len(dataframe) == 2
    assert {"record_id", "category_teacher_final", "title", "text_clean", "model_input"}.issubset(dataframe.columns)
    assert records[0].record_id == "1"


def test_load_dataset_records_validates_required_fields(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(
        dataset_path,
        [
            '{"record_id":"1","category_teacher_final":"politics","title":"   ","text_clean":"Text 1","model_input":"Input 1"}',
        ],
    )

    with pytest.raises(DatasetLoadError, match="line 1"):
        load_dataset_records(dataset_path)


def test_load_dataset_records_reports_broken_json_line(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(
        dataset_path,
        [
            '{"record_id":"1","category_teacher_final":"politics","title":"Title 1","text_clean":"Text 1","model_input":"Input 1"}',
            '{"record_id":"2","category_teacher_final":"economy"',
        ],
    )

    with pytest.raises(DatasetLoadError, match="Invalid JSON at line 2"):
        load_dataset_records(dataset_path)
