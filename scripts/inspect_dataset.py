from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from signal_backend.data.label_mapping import build_label_to_id
from signal_backend.data.load_jsonl import DatasetLoadError
from signal_backend.data.validate_dataset import (
    DatasetValidationError,
    validate_dataset,
)
from signal_backend.paths import DEFAULT_DATASET_PATH, PROCESSED_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and validate a JSONL dataset.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the source JSONL dataset.",
    )
    return parser.parse_args()


def _print_summary(summary: dict[str, object]) -> None:
    print("Dataset summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    args = parse_args()

    try:
        validation_result = validate_dataset(args.dataset_path)
    except (DatasetLoadError, DatasetValidationError, FileNotFoundError) as exc:
        print(f"Dataset inspection failed: {exc}", file=sys.stderr)
        return 1

    summary = validation_result.to_summary_dict()
    summary["label_to_id"] = build_label_to_id(validation_result.dataframe)

    output_path = PROCESSED_DIR / "dataset_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _print_summary(summary)
    print(f"Saved summary to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
