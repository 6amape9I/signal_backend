from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signal_backend.data.dataset_settings import (
    apply_dataset_overrides,
    apply_split_overrides,
    load_dataset_settings,
)


def test_load_dataset_settings_reads_yaml_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  path: data/input/custom.jsonl",
                "  label_column: final_label",
                "  text_column: text_field",
                "split:",
                "  train_size: 0.7",
                "  val_size: 0.2",
                "  test_size: 0.1",
                "  seed: 7",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_dataset_settings(config_path)

    assert settings.dataset_path.name == "custom.jsonl"
    assert settings.label_column == "final_label"
    assert settings.text_column == "text_field"
    assert settings.split.train_size == 0.7
    assert settings.split.val_size == 0.2
    assert settings.split.test_size == 0.1
    assert settings.split.seed == 7


def test_apply_overrides_prefer_cli_values(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  path: data/input/from_config.jsonl",
                "split:",
                "  train_size: 0.8",
                "  val_size: 0.1",
                "  test_size: 0.1",
                "  seed: 42",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_dataset_settings(config_path)
    settings = apply_dataset_overrides(settings, dataset_path=tmp_path / "override.jsonl")
    settings = apply_split_overrides(settings, train_size=0.6, val_size=0.2, test_size=0.2, seed=99)

    assert settings.dataset_path == (tmp_path / "override.jsonl").resolve()
    assert settings.split.train_size == 0.6
    assert settings.split.val_size == 0.2
    assert settings.split.test_size == 0.2
    assert settings.split.seed == 99
