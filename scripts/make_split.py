from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from signal_backend.config import ConfigError
from signal_backend.data.dataset_settings import (
    apply_dataset_overrides,
    apply_split_overrides,
    load_dataset_settings,
)
from signal_backend.data.load_jsonl import DatasetLoadError
from signal_backend.data.split_dataset import (
    DatasetSplitError,
    build_split_report,
    create_stratified_split,
    save_split_files,
    save_split_report,
)
from signal_backend.data.validate_dataset import (
    DatasetValidationError,
    validate_dataset,
)
from signal_backend.paths import PROCESSED_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reproducible stratified dataset splits.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Path to the source JSONL dataset.",
    )
    parser.add_argument("--train-size", type=float, default=None)
    parser.add_argument("--val-size", type=float, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        settings = load_dataset_settings()
        settings = apply_dataset_overrides(settings, dataset_path=args.dataset_path)
        settings = apply_split_overrides(
            settings,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
        )

        validation_result = validate_dataset(settings.dataset_path)
        split_result = create_stratified_split(
            validation_result.dataframe,
            train_size=settings.split.train_size,
            val_size=settings.split.val_size,
            test_size=settings.split.test_size,
            seed=settings.split.seed,
        )
    except (
        ConfigError,
        DatasetLoadError,
        DatasetValidationError,
        DatasetSplitError,
        FileNotFoundError,
    ) as exc:
        print(f"Split creation failed: {exc}", file=sys.stderr)
        return 1

    save_split_files(split_result, PROCESSED_DIR)
    split_report = build_split_report(split_result)
    report_path = PROCESSED_DIR / "split_report.json"
    save_split_report(split_report, report_path)

    print("Split report:")
    print(json.dumps(split_report, ensure_ascii=False, indent=2))
    print(f"Saved split files to: {PROCESSED_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
