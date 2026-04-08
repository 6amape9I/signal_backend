from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from signal_backend.data.load_jsonl import load_dataset_dataframe
from signal_backend.inference import load_artifact, predict_batch
from signal_backend.paths import TEST_SPLIT_PATH
from signal_backend.training.metrics import compute_evaluation_result
from signal_backend.training.save_artifacts import save_evaluation_artifacts, save_json


BASELINE_MODELS = {"tfidf_logreg", "tfidf_linear_svm"}
TRANSFORMER_MODEL = "transformer_classifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate a saved model on a dataset.")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=TEST_SPLIT_PATH)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--vectorizer-path", type=Path, default=None)
    parser.add_argument("--label-mapping-path", type=Path, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    return parser.parse_args()


def _load_mapping(mapping_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    label_to_id = {str(key): int(value) for key, value in payload["label_to_id"].items()}
    id_to_label = {int(key): value for key, value in payload["id_to_label"].items()}
    return label_to_id, id_to_label


def _load_artifact_metadata(args: argparse.Namespace) -> tuple[str, Path, Path | None, Path | None, Path]:
    if args.artifact_dir is not None:
        artifact_dir = Path(args.artifact_dir)
        run_summary = json.loads((artifact_dir / "run_summary.json").read_text(encoding="utf-8"))
        model_type = run_summary["model_type"]
        model_path = artifact_dir / "model.joblib"
        vectorizer_path = artifact_dir / "vectorizer.joblib"
        label_mapping_path = artifact_dir / "label_mapping.json"
        return model_type, model_path, vectorizer_path, label_mapping_path, artifact_dir

    if not all([args.model_path, args.vectorizer_path, args.label_mapping_path, args.model_type]):
        raise ValueError(
            "Provide either --artifact-dir or the full set of --model-path, --vectorizer-path, --label-mapping-path, and --model-type."
        )
    return args.model_type, Path(args.model_path), Path(args.vectorizer_path), Path(args.label_mapping_path), Path(args.model_path).parent


def _predict_with_baseline(model_type: str, model, vectorizer, texts: list[str], id_to_label: dict[int, str]) -> list[str]:
    matrix = vectorizer.transform(texts)
    predicted_ids = [int(item) for item in model.predict(matrix).tolist()]
    return [id_to_label[label_id] for label_id in predicted_ids]


def main() -> int:
    args = parse_args()

    try:
        model_type, model_path, vectorizer_path, label_mapping_path, root_dir = _load_artifact_metadata(args)
        dataset_df = load_dataset_dataframe(args.dataset_path)
        _, id_to_label = _load_mapping(label_mapping_path)
        labels = [id_to_label[index] for index in sorted(id_to_label)]
        texts = dataset_df["model_input"].astype(str).tolist()
        y_true = dataset_df["category_teacher_final"].astype(str).tolist()

        if args.artifact_dir is not None:
            loaded_artifact = load_artifact(root_dir)
            predictions = predict_batch(loaded_artifact, texts)
            y_pred = [item["prediction"] for item in predictions]
        elif model_type in BASELINE_MODELS:
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            y_pred = _predict_with_baseline(model_type, model, vectorizer, texts, id_to_label)
        elif model_type == TRANSFORMER_MODEL:
            raise ValueError("Transformer evaluation requires --artifact-dir.")
        else:
            raise ValueError(f"Unsupported model_type for evaluation: {model_type}")

        result = compute_evaluation_result(
            split_name="evaluation",
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
        )

        evaluation_dir = root_dir / "evaluations" / datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_dir.mkdir(parents=True, exist_ok=False)
        save_evaluation_artifacts(model_type=model_type, results=[result], run_dir=evaluation_dir)
        save_json(
            {
                "source_artifact_dir": root_dir.as_posix(),
                "dataset_path": Path(args.dataset_path).resolve().as_posix(),
                "model_type": model_type,
                "metrics": result.metrics,
            },
            evaluation_dir / "evaluation_summary.json",
        )

        print("Evaluation summary:")
        print(
            json.dumps(
                {
                    "evaluation_dir": evaluation_dir.as_posix(),
                    "model_type": model_type,
                    "dataset_path": Path(args.dataset_path).resolve().as_posix(),
                    "metrics": result.metrics,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        print(f"Model evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())