from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

try:
    from signal_backend.serving.schemas import (
        BatchPredictRequest,
        BatchPredictResponse,
        HealthResponse,
        PredictRequest,
        PredictResponse,
    )
    from signal_backend.serving.service import BadRequestError, InferenceService, InputValidationError
    from signal_backend.serving.settings import APISettings, load_api_settings
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    SRC_DIR = REPO_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from signal_backend.serving.schemas import (
        BatchPredictRequest,
        BatchPredictResponse,
        HealthResponse,
        PredictRequest,
        PredictResponse,
    )
    from signal_backend.serving.service import BadRequestError, InferenceService, InputValidationError
    from signal_backend.serving.settings import APISettings, load_api_settings


def create_app(settings: APISettings | None = None) -> FastAPI:
    service = InferenceService(load_api_settings() if settings is None else settings)
    app = FastAPI(title="signal_backend API")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(**service.health())

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        try:
            return PredictResponse(**service.predict(request))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except BadRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except InputValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Internal prediction error.") from exc

    @app.post("/batch_predict", response_model=BatchPredictResponse)
    def batch_predict(request: BatchPredictRequest) -> BatchPredictResponse:
        try:
            return BatchPredictResponse(**service.batch_predict(request))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except BadRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except InputValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Internal prediction error.") from exc

    return app


app = create_app()