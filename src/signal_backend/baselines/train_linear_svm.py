from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.svm import LinearSVC

from signal_backend.baselines.tfidf_features import build_tfidf_vectorizer, normalize_class_weight
from signal_backend.training.evaluate import PredictionOutput


@dataclass(slots=True)
class BaselineBundle:
    model: LinearSVC
    vectorizer: Any
    model_type: str


def _normalize_decision_scores(scores: Any) -> list[list[float]]:
    array = np.asarray(scores)
    if array.ndim == 1:
        return [[float(-value), float(value)] for value in array.tolist()]
    return [[float(item) for item in row] for row in array.tolist()]


def train_tfidf_linear_svm(
    train_texts: Sequence[str],
    train_label_ids: Sequence[int],
    features_config: dict[str, Any],
    model_config: dict[str, Any],
    random_state: int,
) -> BaselineBundle:
    vectorizer = build_tfidf_vectorizer(features_config)
    train_matrix = vectorizer.fit_transform(train_texts)
    classifier = LinearSVC(
        C=float(model_config.get("C", 1.0)),
        class_weight=normalize_class_weight(model_config.get("class_weight")),
        random_state=random_state,
    )
    classifier.fit(train_matrix, train_label_ids)
    return BaselineBundle(model=classifier, vectorizer=vectorizer, model_type="tfidf_linear_svm")


def predict_with_linear_svm(
    bundle: BaselineBundle,
    texts: Sequence[str],
    id_to_label: dict[int, str],
) -> PredictionOutput:
    matrix = bundle.vectorizer.transform(texts)
    predicted_ids = bundle.model.predict(matrix).tolist()
    decision_scores = _normalize_decision_scores(bundle.model.decision_function(matrix))
    return PredictionOutput(
        predicted_labels=[id_to_label[int(label_id)] for label_id in predicted_ids],
        predicted_label_ids=[int(label_id) for label_id in predicted_ids],
        decision_scores=decision_scores,
        score_type="decision_scores",
    )
