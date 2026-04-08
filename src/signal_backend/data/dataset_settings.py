from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from signal_backend.config import ConfigError, load_yaml_config, resolve_path
from signal_backend.paths import DATASET_CONFIG_PATH, DEFAULT_DATASET_PATH


@dataclass(frozen=True, slots=True)
class SplitSettings:
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    seed: int = 42


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    dataset_path: Path = DEFAULT_DATASET_PATH
    label_column: str = "category_teacher_final"
    text_column: str = "model_input"
    split: SplitSettings = field(default_factory=SplitSettings)


def load_dataset_settings(config_path: Path = DATASET_CONFIG_PATH) -> DatasetSettings:
    config = load_yaml_config(config_path)

    dataset_section = config.get("dataset", {})
    split_section = config.get("split", {})

    if not isinstance(dataset_section, dict):
        raise ConfigError("'dataset' section must be a mapping.")
    if not isinstance(split_section, dict):
        raise ConfigError("'split' section must be a mapping.")

    dataset_path_raw = dataset_section.get("path", DEFAULT_DATASET_PATH)
    label_column = str(dataset_section.get("label_column", "category_teacher_final"))
    text_column = str(dataset_section.get("text_column", "model_input"))

    return DatasetSettings(
        dataset_path=resolve_path(dataset_path_raw),
        label_column=label_column,
        text_column=text_column,
        split=SplitSettings(
            train_size=float(split_section.get("train_size", 0.8)),
            val_size=float(split_section.get("val_size", 0.1)),
            test_size=float(split_section.get("test_size", 0.1)),
            seed=int(split_section.get("seed", 42)),
        ),
    )


def apply_dataset_overrides(
    settings: DatasetSettings,
    dataset_path: Path | None = None,
) -> DatasetSettings:
    if dataset_path is None:
        return settings
    return replace(settings, dataset_path=Path(dataset_path).resolve())


def apply_split_overrides(
    settings: DatasetSettings,
    train_size: float | None = None,
    val_size: float | None = None,
    test_size: float | None = None,
    seed: int | None = None,
) -> DatasetSettings:
    split = settings.split
    return replace(
        settings,
        split=SplitSettings(
            train_size=split.train_size if train_size is None else train_size,
            val_size=split.val_size if val_size is None else val_size,
            test_size=split.test_size if test_size is None else test_size,
            seed=split.seed if seed is None else seed,
        ),
    )
