from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from signal_backend.data.schemas import DatasetRecord


class DatasetLoadError(ValueError):
    """Raised when a dataset file cannot be loaded or parsed."""


def _validate_dataset_path(path: Path) -> Path:
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise DatasetLoadError(f"Dataset path is not a file: {resolved_path}")
    return resolved_path


def load_dataset_records(path: Path) -> list[DatasetRecord]:
    dataset_path = _validate_dataset_path(path)
    records: list[DatasetRecord] = []

    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line_number, raw_line in enumerate(dataset_file, start=1):
            line = raw_line.strip()
            if not line:
                raise DatasetLoadError(
                    f"Dataset contains an empty line at line {line_number}."
                )

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetLoadError(
                    f"Invalid JSON at line {line_number}: {exc.msg}"
                ) from exc

            try:
                record = DatasetRecord.model_validate(payload)
            except ValidationError as exc:
                raise DatasetLoadError(
                    f"Record validation failed at line {line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


def load_dataset_dataframe(path: Path) -> pd.DataFrame:
    records = load_dataset_records(path)
    rows = [record.model_dump() for record in records]
    return pd.DataFrame(rows)
