from __future__ import annotations

from pathlib import Path

import joblib
from fastapi.testclient import TestClient

from apps.api.main import create_app
from signal_backend.baselines.train_logreg import train_tfidf_logreg
from signal_backend.data.label_mapping import build_id_to_label, build_label_to_id
from signal_backend.serving.settings import APISettings
from signal_backend.training.save_artifacts import save_label_mapping, save_run_summary


FEATURES_CONFIG = {
    "max_features": 1000,
    "ngram_range": [1, 2],
    "min_df": 1,
    "max_df": 1.0,
    "lowercase": True,
}


def _create_baseline_artifact(tmp_path: Path) -> Path:
    train_texts = [
        "sports goal win",
        "sports match goal",
        "politics parliament vote",
        "politics election vote",
    ]
    labels = ["sports", "sports", "politics", "politics"]
    label_to_id = build_label_to_id(labels)
    id_to_label = build_id_to_label(labels)
    label_ids = [label_to_id[label] for label in labels]

    artifact_dir = tmp_path / "baseline_api"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    bundle = train_tfidf_logreg(train_texts, label_ids, FEATURES_CONFIG, {"C": 1.0, "max_iter": 200}, 42)

    joblib.dump(bundle.model, artifact_dir / "model.joblib")
    joblib.dump(bundle.vectorizer, artifact_dir / "vectorizer.joblib")
    save_label_mapping(label_to_id, id_to_label, artifact_dir / "label_mapping.json")
    save_run_summary({"model_type": "tfidf_logreg"}, artifact_dir)
    return artifact_dir


def test_api_health_predict_and_batch_predict(tmp_path: Path) -> None:
    artifact_dir = _create_baseline_artifact(tmp_path)
    app = create_app(APISettings(default_artifact_dir=artifact_dir, host="127.0.0.1", port=8000, batch_size=8, max_request_items=16))
    client = TestClient(app)

    health_response = client.get("/health")
    predict_response = client.post("/predict", json={"title": "sports", "text": "goal"})
    batch_response = client.post(
        "/batch_predict",
        json={
            "items": [
                {"title": "sports", "text": "goal"},
                {"model_input": "politics vote"},
            ]
        },
    )

    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"
    assert health_response.json()["default_artifact_dir"] == artifact_dir.as_posix()

    assert predict_response.status_code == 200
    payload = predict_response.json()
    assert payload["prediction"] in {"sports", "politics"}
    assert payload["score_type"] == "probabilities"
    assert set(payload["scores"].keys()) == set(payload["label_order"])

    assert batch_response.status_code == 200
    assert len(batch_response.json()["items"]) == 2


def test_api_returns_validation_error_for_empty_predict_request(tmp_path: Path) -> None:
    artifact_dir = _create_baseline_artifact(tmp_path)
    app = create_app(APISettings(default_artifact_dir=artifact_dir, host="127.0.0.1", port=8000, batch_size=8, max_request_items=16))
    client = TestClient(app)

    response = client.post("/predict", json={})

    assert response.status_code == 422
    assert "model_input" in response.json()["detail"] or "title" in response.json()["detail"]


def test_api_returns_not_found_for_missing_artifact_dir(tmp_path: Path) -> None:
    app = create_app(APISettings(default_artifact_dir=tmp_path / "missing_artifact", host="127.0.0.1", port=8000, batch_size=8, max_request_items=16))
    client = TestClient(app)

    response = client.post("/predict", json={"model_input": "sports goal"})

    assert response.status_code == 404