from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from signal_backend.config import resolve_path
from signal_backend.inference import predict_batch_from_artifact, predict_one_from_artifact
from signal_backend.serving.schemas import BatchPredictRequest, PredictRequest
from signal_backend.serving.settings import APISettings


class BadRequestError(ValueError):
    """Raised when a request is structurally invalid for the service layer."""


class InputValidationError(ValueError):
    """Raised when model input cannot be constructed from the request payload."""


@dataclass(slots=True)
class InferenceService:
    settings: APISettings

    def health(self) -> dict[str, str]:
        return {
            "status": "ok",
            "default_artifact_dir": self.settings.default_artifact_dir.as_posix(),
        }

    def _resolve_artifact_dir(self, artifact_dir: str | None) -> Path:
        if artifact_dir is not None and not artifact_dir.strip():
            raise BadRequestError("'artifact_dir' must not be empty.")

        resolved = self.settings.default_artifact_dir if artifact_dir is None else resolve_path(artifact_dir)
        if not resolved.exists():
            raise FileNotFoundError(f"Artifact directory does not exist: {resolved}")
        if not resolved.is_dir():
            raise FileNotFoundError(f"Artifact path is not a directory: {resolved}")
        return resolved

    def predict(self, request: PredictRequest) -> dict:
        artifact_dir = self._resolve_artifact_dir(request.artifact_dir)
        try:
            return predict_one_from_artifact(
                artifact_dir,
                title=request.title,
                text=request.text,
                model_input=request.model_input,
            )
        except ValueError as exc:
            raise InputValidationError(str(exc)) from exc

    def batch_predict(self, request: BatchPredictRequest) -> dict[str, list[dict]]:
        if not request.items:
            raise BadRequestError("Request 'items' must not be empty.")
        if len(request.items) > self.settings.max_request_items:
            raise BadRequestError(
                f"Request contains {len(request.items)} items, limit is {self.settings.max_request_items}."
            )

        artifact_dir = self._resolve_artifact_dir(request.artifact_dir)
        try:
            results = predict_batch_from_artifact(
                artifact_dir,
                [item.model_dump() for item in request.items],
                batch_size=self.settings.batch_size,
            )
        except ValueError as exc:
            raise InputValidationError(str(exc)) from exc
        return {"items": results}