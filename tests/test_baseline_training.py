from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_rows(prefix: str, label: str, count: int) -> list[dict[str, str]]:
    return [
        {
            "record_id": f"{prefix}-{label}-{index}",
            "category_teacher_final": label,
            "title": f"{label} title {index}",
            "text_clean": f"{label} text {index}",
            "model_input": f"{label} keyword {index}",
        }
        for index in range(count)
    ]


def _write_baseline_config(config_path: Path, model_type: str, output_dir: Path, train_path: Path, val_path: Path, test_path: Path) -> None:
    config_path.write_text(
        "\n".join(
            [
                "features:",
                "  max_features: 1000",
                "  ngram_range: [1, 2]",
                "  min_df: 1",
                "  max_df: 1.0",
                "  lowercase: true",
                "model:",
                f"  type: {model_type}",
                "  class_weight: balanced",
                "  C: 1.0",
                "  max_iter: 500",
                "run:",
                f"  run_name: {model_type}_toy",
                f"  output_dir: {output_dir.as_posix()}",
                "  random_state: 42",
                "data:",
                f"  train_path: {train_path.as_posix()}",
                f"  val_path: {val_path.as_posix()}",
                f"  test_path: {test_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )


def _run_training(tmp_path: Path, model_type: str) -> Path:
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    test_path = tmp_path / "test.jsonl"
    output_dir = tmp_path / "artifacts"
    config_path = tmp_path / f"{model_type}.yaml"

    _write_jsonl(train_path, _build_rows("train", "sports", 4) + _build_rows("train", "politics", 4))
    _write_jsonl(val_path, _build_rows("val", "sports", 2) + _build_rows("val", "politics", 2))
    _write_jsonl(test_path, _build_rows("test", "sports", 2) + _build_rows("test", "politics", 2))
    _write_baseline_config(config_path, model_type, output_dir, train_path, val_path, test_path)

    completed = subprocess.run(
        [PYTHON, "scripts/train_baseline.py", "--config", str(config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return output_dir / f"{model_type}_toy"


def test_logreg_baseline_training_creates_artifacts(tmp_path: Path) -> None:
    artifact_dir = _run_training(tmp_path, "tfidf_logreg")

    assert (artifact_dir / "model.joblib").exists()
    assert (artifact_dir / "vectorizer.joblib").exists()
    assert (artifact_dir / "metrics.json").exists()
    assert (artifact_dir / "classification_report.json").exists()
    assert (artifact_dir / "confusion_matrix.csv").exists()
    assert (artifact_dir / "label_mapping.json").exists()
    assert (artifact_dir / "run_summary.json").exists()
    assert (artifact_dir / "train.log").exists()
    assert (artifact_dir / "train_log.jsonl").exists()


def test_linear_svm_baseline_training_creates_artifacts(tmp_path: Path) -> None:
    artifact_dir = _run_training(tmp_path, "tfidf_linear_svm")
    metrics_payload = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    event_lines = (artifact_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()

    assert (artifact_dir / "model.joblib").exists()
    assert (artifact_dir / "vectorizer.joblib").exists()
    assert (artifact_dir / "train.log").exists()
    assert metrics_payload["model_type"] == "tfidf_linear_svm"
    assert "val" in metrics_payload and "test" in metrics_payload
    assert any('"event": "fit_completed"' in line for line in event_lines)
    assert any('"event": "evaluation_completed"' in line for line in event_lines)