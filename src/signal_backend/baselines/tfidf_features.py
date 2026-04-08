from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

from signal_backend.config import ConfigError, normalize_optional_string


SUPPORTED_CLASS_WEIGHTS = {None, "balanced"}


def parse_ngram_range(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError("'ngram_range' must be a two-element list like [1, 2].")
    min_n, max_n = int(value[0]), int(value[1])
    if min_n <= 0 or max_n <= 0 or min_n > max_n:
        raise ConfigError(f"Invalid ngram_range: {value}")
    return min_n, max_n


def normalize_class_weight(value: Any) -> str | None:
    normalized = normalize_optional_string(value)
    if normalized not in SUPPORTED_CLASS_WEIGHTS:
        raise ConfigError(
            "'class_weight' must be null, 'none', or 'balanced'."
        )
    return normalized


def build_tfidf_vectorizer(features_config: dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=features_config.get("max_features"),
        ngram_range=parse_ngram_range(features_config.get("ngram_range", [1, 1])),
        min_df=features_config.get("min_df", 1),
        max_df=features_config.get("max_df", 1.0),
        lowercase=bool(features_config.get("lowercase", True)),
    )
