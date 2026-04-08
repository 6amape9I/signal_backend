from __future__ import annotations

from pathlib import Path


def best_model_dir(run_dir: Path) -> Path:
    path = Path(run_dir) / "best_model"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_best_model(model, output_dir: Path) -> None:
    model.save_pretrained(best_model_dir(output_dir))
