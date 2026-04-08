from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from signal_backend.baselines.tfidf_features import build_tfidf_vectorizer, normalize_class_weight
from signal_backend.training.evaluate import PredictionOutput


@dataclass(slots=True)
class BaselineBundle:
    model: LogisticRegression
    vectorizer: Any
    model_type: str


def train_tfidf_logreg(
    train_texts: Sequence[str],
    train_label_ids: Sequence[int],
    features_config: dict[str, Any],
    model_config: dict[str, Any],
    random_state: int,
) -> BaselineBundle:
    vectorizer = build_tfidf_vectorizer(features_config)
    train_matrix = vectorizer.fit_transform(train_texts)
    classifier = LogisticRegression(
        C=float(model_config.get("C", 1.0)),
        max_iter=int(model_config.get("max_iter", 1000)),
        class_weight=normalize_class_weight(model_config.get("class_weight")),
        random_state=random_state,
    )
    classifier.fit(train_matrix, train_label_ids)
    return BaselineBundle(model=classifier, vectorizer=vectorizer, model_type="tfidf_logreg")


def predict_with_logreg(
    bundle: BaselineBundle,
    texts: Sequence[str],
    id_to_label: dict[int, str],
) -> PredictionOutput:
    matrix = bundle.vectorizer.transform(texts)
    predicted_ids = bundle.model.predict(matrix).tolist()
    probabilities = bundle.model.predict_proba(matrix).tolist()
    return PredictionOutput(
        predicted_labels=[id_to_label[int(label_id)] for label_id in predicted_ids],
        predicted_label_ids=[int(label_id) for label_id in predicted_ids],
        probabilities=probabilities,
        score_type="probabilities",
    )
