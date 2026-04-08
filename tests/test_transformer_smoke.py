from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from transformers import BertConfig, BertTokenizerFast


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from signal_backend.inference.batch_predict import predict_batch
from signal_backend.inference.predict import predict_one


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
            "model_input": f"{label} keyword sample {index}",
        }
        for index in range(count)
    ]


def _create_local_tiny_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "sports", "politics", "keyword", "sample", "title", "text"]
    (model_dir / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(model_dir / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(model_dir)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        pad_token_id=0,
    )
    config.to_json_file(str(model_dir / "config.json"))


def test_transformer_smoke(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny_model"
    _create_local_tiny_model(model_dir)

    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    test_path = tmp_path / "test.jsonl"
    output_dir = tmp_path / "artifacts"
    config_path = tmp_path / "transformer.yaml"

    _write_jsonl(train_path, _build_rows("train", "sports", 4) + _build_rows("train", "politics", 4))
    _write_jsonl(val_path, _build_rows("val", "sports", 2) + _build_rows("val", "politics", 2))
    _write_jsonl(test_path, _build_rows("test", "sports", 2) + _build_rows("test", "politics", 2))
    config_path.write_text(
        "\n".join(
            [
                "data:",
                f"  train_path: {train_path.as_posix()}",
                f"  val_path: {val_path.as_posix()}",
                f"  test_path: {test_path.as_posix()}",
                "model:",
                "  type: transformer_classifier",
                f"  model_name_or_path: {model_dir.as_posix()}",
                "  max_length: 32",
                "training:",
                "  batch_size: 2",
                "  learning_rate: 0.0005",
                "  weight_decay: 0.0",
                "  num_epochs: 1",
                "  gradient_accumulation_steps: 1",
                "  random_seed: 42",
                "  class_weight_mode: balanced",
                "  early_stopping_patience: 1",
                "scheduler:",
                "  warmup_steps: 0",
                "run:",
                "  run_name: transformer_smoke",
                f"  output_dir: {output_dir.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [PYTHON, "scripts/train_transformer.py", "--config", str(config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    artifact_dir = output_dir / "transformer_smoke"
    event_lines = (artifact_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert (artifact_dir / "best_model").exists()
    assert (artifact_dir / "tokenizer").exists()
    assert (artifact_dir / "metrics.json").exists()
    assert (artifact_dir / "classification_report.json").exists()
    assert (artifact_dir / "confusion_matrix.csv").exists()
    assert (artifact_dir / "label_mapping.json").exists()
    assert (artifact_dir / "train.log").exists()
    assert (artifact_dir / "train_log.jsonl").exists()
    assert (artifact_dir / "run_summary.json").exists()
    assert any('"event": "run_started"' in line for line in event_lines)
    assert any('"event": "epoch_completed"' in line for line in event_lines)
    assert any('"event": "artifacts_saved"' in line for line in event_lines)

    one_prediction = predict_one("sports keyword sample", artifact_dir)
    batch_predictions = predict_batch(["sports keyword sample", "politics keyword sample"], artifact_dir, batch_size=2)

    assert "prediction" in one_prediction
    assert len(batch_predictions) == 2