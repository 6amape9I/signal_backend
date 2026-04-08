from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from signal_backend.config import ConfigError, load_yaml_config, resolve_path
from signal_backend.paths import API_CONFIG_PATH


@dataclass(frozen=True, slots=True)
class APISettings:
    default_artifact_dir: Path
    host: str = "127.0.0.1"
    port: int = 8000
    batch_size: int = 8
    max_request_items: int = 128


def load_api_settings(config_path: Path = API_CONFIG_PATH) -> APISettings:
    config = load_yaml_config(config_path)
    default_artifact_dir = config.get("default_artifact_dir")
    if not default_artifact_dir:
        raise ConfigError("'default_artifact_dir' is required in API config.")

    settings = APISettings(
        default_artifact_dir=resolve_path(default_artifact_dir),
        host=str(config.get("host", "127.0.0.1")),
        port=int(config.get("port", 8000)),
        batch_size=int(config.get("batch_size", 8)),
        max_request_items=int(config.get("max_request_items", 128)),
    )
    if settings.batch_size <= 0:
        raise ConfigError("'batch_size' must be greater than 0.")
    if settings.max_request_items <= 0:
        raise ConfigError("'max_request_items' must be greater than 0.")
    if settings.port <= 0:
        raise ConfigError("'port' must be greater than 0.")
    return settings