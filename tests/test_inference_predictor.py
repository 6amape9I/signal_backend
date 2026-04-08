from __future__ import annotations

import sys
from pathlib import Path

import joblib
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from signal_backend.baselines.train_linear_svm import train_tfidf_linear_svm
from signal_backend.baselines.train_logreg import train_tfidf_logreg
from signal_backend.data.label_mapping import build_id_to_label, build_label_to_id
from signal_backend.inference import clear_artifact_cache, get_cached_artifact, load_artifact
from signal_backend.inference.predictor import predict_batch, predict_batch_from_artifact, predict_one, predict_one_from_artifact
from signal_backend.training.save_artifacts import save_label_mapping, save_run_summary


FEATURES_CONFIG = {
    "max_features": 1000,
    "ngram_range": [1, 2],
    "min_df": 1,
    "max_df": 1.0,
    "lowercase": True,
}


def _create_baseline_artifact(tmp_path: Path, model_type: str) -> Path:
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

    artifact_dir = tmp_path / model_type
    artifact_dir.mkdir(parents=True, exist_ok=False)
    if model_type == "tfidf_logreg":
        bundle = train_tfidf_logreg(train_texts, label_ids, FEATURES_CONFIG, {"C": 1.0, "max_iter": 200}, 42)
    else:
        bundle = train_tfidf_linear_svm(train_texts, label_ids, FEATURES_CONFIG, {"C": 1.0}, 42)

    joblib.dump(bundle.model, artifact_dir / "model.joblib")
    joblib.dump(bundle.vectorizer, artifact_dir / "vectorizer.joblib")
    save_label_mapping(label_to_id, id_to_label, artifact_dir / "label_mapping.json")
    save_run_summary({"model_type": model_type}, artifact_dir)
    return artifact_dir


def _create_transformer_artifact(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "transformer_classifier"
    tokenizer_dir = artifact_dir / "tokenizer"
    model_dir = artifact_dir / "best_model"
    artifact_dir.mkdir(parents=True, exist_ok=False)

    label_to_id = {"politics": 0, "sports": 1}
    id_to_label = {0: "politics", 1: "sports"}
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "sports", "politics", "goal", "vote", "text"]
    vocab_path = artifact_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")

    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)
    tokenizer.save_pretrained(tokenizer_dir)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        pad_token_id=0,
        num_labels=2,
        label2id=label_to_id,
        id2label={str(key): value for key, value in id_to_label.items()},
    )
    model = BertForSequenceClassification(config)
    model.save_pretrained(model_dir)

    save_label_mapping(label_to_id, id_to_label, artifact_dir / "label_mapping.json")
    save_run_summary({"model_type": "transformer_classifier"}, artifact_dir)
    return artifact_dir


def test_baseline_predictor_returns_unified_structure(tmp_path: Path) -> None:
    artifact_dir = _create_baseline_artifact(tmp_path, "tfidf_logreg")
    loaded_artifact = load_artifact(artifact_dir)

    prediction = predict_one(loaded_artifact, "sports goal")

    assert prediction["prediction"] in {"sports", "politics"}
    assert prediction["score_type"] == "probabilities"
    assert prediction["model_type"] == "tfidf_logreg"
    assert set(prediction["scores"].keys()) == set(prediction["label_order"])


def test_transformer_predictor_returns_unified_structure(tmp_path: Path) -> None:
    artifact_dir = _create_transformer_artifact(tmp_path)
    loaded_artifact = load_artifact(artifact_dir)

    prediction = predict_one(loaded_artifact, "sports goal text")

    assert prediction["prediction"] in {"sports", "politics"}
    assert prediction["score_type"] == "probabilities"
    assert prediction["model_type"] == "transformer_classifier"
    assert set(prediction["scores"].keys()) == set(prediction["label_order"])


def test_batch_prediction_returns_same_number_of_results(tmp_path: Path) -> None:
    artifact_dir = _create_baseline_artifact(tmp_path, "tfidf_linear_svm")
    loaded_artifact = load_artifact(artifact_dir)

    results = predict_batch(loaded_artifact, ["sports goal", "politics vote"])

    assert len(results) == 2
    assert all(result["score_type"] == "decision_scores" for result in results)


def test_predict_from_artifact_builds_model_input_and_uses_cache(tmp_path: Path) -> None:
    artifact_dir = _create_baseline_artifact(tmp_path, "tfidf_logreg")
    clear_artifact_cache()

    first_loaded = get_cached_artifact(artifact_dir)
    result_one = predict_one_from_artifact(artifact_dir, title="sports", text="goal")
    result_batch = predict_batch_from_artifact(
        artifact_dir,
        [
            {"title": "sports", "text": "goal"},
            {"model_input": "politics vote"},
        ],
    )
    second_loaded = get_cached_artifact(artifact_dir)

    assert first_loaded is second_loaded
    assert result_one["prediction"] in {"sports", "politics"}
    assert len(result_batch) == 2