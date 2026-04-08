"""Inference utilities for trained models."""

from signal_backend.inference.artifact_loader import LoadedArtifact, clear_artifact_cache, get_cached_artifact, load_artifact
from signal_backend.inference.model_input import build_model_input
from signal_backend.inference.predictor import (
    predict_batch,
    predict_batch_from_artifact,
    predict_one,
    predict_one_from_artifact,
)

__all__ = [
    "LoadedArtifact",
    "build_model_input",
    "clear_artifact_cache",
    "get_cached_artifact",
    "load_artifact",
    "predict_one",
    "predict_batch",
    "predict_one_from_artifact",
    "predict_batch_from_artifact",
]