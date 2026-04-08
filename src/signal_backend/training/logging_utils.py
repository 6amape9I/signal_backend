from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any


LOGGER_NAME = "signal_backend.training"


def setup_training_logger(run_dir: Path) -> tuple[logging.Logger, Path]:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = run_dir / "train.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, run_dir / "train_log.jsonl"


def log_event(event_log_path: Path, *, event: str, payload: dict[str, Any]) -> None:
    with event_log_path.open("a", encoding="utf-8") as event_log:
        event_log.write(json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n")
