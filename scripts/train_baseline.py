from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from signal_backend.baselines.train_linear_svm import predict_with_linear_svm, train_tfidf_linear_svm
from signal_backend.baselines.train_logreg import predict_with_logreg, train_tfidf_logreg
from signal_backend.config import ConfigError, load_yaml_config, resolve_path
from signal_backend.data.dataset_settings import load_dataset_settings
from signal_backend.training.evaluate import (
    build_train_label_mapping,
    dataframe_texts,
    dataframe_labels,
    evaluate_dataframe,
    load_split_data,
)
from signal_backend.training.save_artifacts import (
    build_run_directory,
    save_evaluation_artifacts,
    save_json,
    save_label_mapping,
    save_run_summary,
    save_yaml,
)


SUPPORTED_MODELS = {"tfidf_logreg", "tfidf_linear_svm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline text classifier.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the baseline YAML config.")
    parser.add_argument(
        "--model-type",
        choices=sorted(SUPPORTED_MODELS),
        default=None,
        help="Optional model type override.",
    )
    return parser.parse_args()


def _require_mapping(config: dict, key: str) -> dict:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' section must be a mapping.")
    return value


def _load_effective_config(config_path: Path, model_type_override: str | None) -> dict:
    config = load_yaml_config(config_path)
    features = _require_mapping(config, "features")
    model = _require_mapping(config, "model")
    run = _require_mapping(config, "run")
    data = _require_mapping(config, "data")

    model_type = model_type_override or model.get("type")
    if model_type not in SUPPORTED_MODELS:
        raise ConfigError(f"Unsupported baseline model type: {model_type}")

    random_state = int(run.get("random_state", 42))
    output_dir = resolve_path(run.get("output_dir", "data/artifacts"))
    return {
        "features": features,
        "model": {**model, "type": model_type},
        "run": {**run, "random_state": random_state, "output_dir": output_dir.as_posix()},
        "data": {
            "train_path": resolve_path(data.get("train_path", "data/processed/train.jsonl")).as_posix(),
            "val_path": resolve_path(data.get("val_path", "data/processed/val.jsonl")).as_posix(),
            "test_path": resolve_path(data.get("test_path", "data/processed/test.jsonl")).as_posix(),
        },
    }


def main() -> int:
    args = parse_args()

    try:
        effective_config = _load_effective_config(args.config, args.model_type)
        split_data = load_split_data(
            train_path=Path(effective_config["data"]["train_path"]),
            val_path=Path(effective_config["data"]["val_path"]),
            test_path=Path(effective_config["data"]["test_path"]),
        )
        label_to_id, id_to_label = build_train_label_mapping(split_data.train_df)
        labels = list(label_to_id.keys())
        train_texts = dataframe_texts(split_data.train_df)
        train_label_ids = [label_to_id[label] for label in dataframe_labels(split_data.train_df)]
        random_state = int(effective_config["run"]["random_state"])
        model_type = effective_config["model"]["type"]

        if model_type == "tfidf_logreg":
            bundle = train_tfidf_logreg(
                train_texts,
                train_label_ids,
                effective_config["features"],
                effective_config["model"],
                random_state,
            )
            predictor = lambda texts: predict_with_logreg(bundle, texts, id_to_label)
        else:
            bundle = train_tfidf_linear_svm(
                train_texts,
                train_label_ids,
                effective_config["features"],
                effective_config["model"],
                random_state,
            )
            predictor = lambda texts: predict_with_linear_svm(bundle, texts, id_to_label)

        val_result, _ = evaluate_dataframe(
            split_name="val",
            df=split_data.val_df,
            labels=labels,
            predictor=predictor,
        )
        test_result, _ = evaluate_dataframe(
            split_name="test",
            df=split_data.test_df,
            labels=labels,
            predictor=predictor,
        )

        run_dir = build_run_directory(
            model_type=model_type,
            run_name=effective_config["run"].get("run_name"),
            output_dir=Path(effective_config["run"]["output_dir"]),
        )
        joblib.dump(bundle.model, run_dir / "model.joblib")
        joblib.dump(bundle.vectorizer, run_dir / "vectorizer.joblib")
        save_evaluation_artifacts(model_type=model_type, results=[val_result, test_result], run_dir=run_dir)
        save_label_mapping(label_to_id, id_to_label, run_dir / "label_mapping.json")
        save_yaml(effective_config, run_dir / "config_snapshot.yaml")

        dataset_settings = load_dataset_settings()
        run_summary = {
            "model_type": model_type,
            "dataset_path": dataset_settings.dataset_path.as_posix(),
            "train_path": split_data.train_path.as_posix(),
            "val_path": split_data.val_path.as_posix(),
            "test_path": split_data.test_path.as_posix(),
            "row_counts": {
                "train": len(split_data.train_df),
                "val": len(split_data.val_df),
                "test": len(split_data.test_df),
            },
            "classes": labels,
            "main_metrics": {
                "val": val_result.metrics,
                "test": test_result.metrics,
            },
        }
        save_run_summary(run_summary, run_dir)

        print("Baseline training summary:")
        print(json.dumps({"run_dir": run_dir.as_posix(), **run_summary}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"Baseline training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
