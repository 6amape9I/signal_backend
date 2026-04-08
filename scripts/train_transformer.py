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
from signal_backend.training.train_transformer import (
    load_transformer_training_config,
    train_transformer_from_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transformer classifier.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the transformer YAML config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_transformer_training_config(args.config)
        run_dir = train_transformer_from_config(config)
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        print("Transformer training summary:")
        print(json.dumps({"run_dir": run_dir.as_posix(), **summary}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"Transformer training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
