from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import yaml

from signal_backend.paths import ARTIFACTS_DIR
from signal_backend.training.metrics import EvaluationResult, confusion_rows_to_frame


def build_run_directory(model_type: str, run_name: str | None, output_dir: Path | None = None) -> Path:
    base_dir = ARTIFACTS_DIR if output_dir is None else Path(output_dir)
    if run_name:
        directory_name = run_name
    else:
        directory_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = base_dir / directory_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_yaml(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def save_label_mapping(label_to_id: dict[str, int], id_to_label: dict[int, str], output_path: Path) -> None:
    payload = {
        "label_to_id": label_to_id,
        "id_to_label": {str(key): value for key, value in id_to_label.items()},
    }
    save_json(payload, output_path)


def save_evaluation_artifacts(
    *,
    model_type: str,
    results: Sequence[EvaluationResult],
    run_dir: Path,
) -> None:
    metrics_payload = {"model_type": model_type}
    report_payload: dict[str, Any] = {}
    for result in results:
        metrics_payload[result.split_name] = result.metrics
        report_payload[result.split_name] = result.classification_report

    save_json(metrics_payload, run_dir / "metrics.json")
    save_json(report_payload, run_dir / "classification_report.json")
    confusion_rows_to_frame(results).to_csv(run_dir / "confusion_matrix.csv", index=False)


def save_run_summary(summary: dict[str, Any], run_dir: Path) -> None:
    save_json(summary, run_dir / "run_summary.json")
