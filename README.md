# signal_backend

`signal_backend` is the backend repository for Signal news classification.

Implemented stages:
- dataset import from JSONL
- dataset validation
- reproducible `train` / `val` / `test` split
- baseline training with TF-IDF models
- transformer training with artifact-ready inference utilities
- unified inference layer for baseline and transformer artifacts
- minimal FastAPI service over the inference layer

Out of scope at this stage:
- deployment
- Docker / reverse proxy
- frontend integration inside this repository

## Dataset Location

The project expects the local dataset at:

`data/input/dataset_clean.jsonl`

The repository root must not be used as a working dataset path.

## Local Base Model

The default offline transformer config points to:

`resources/base_models/mmBERT-base`

This directory stores the minimal local tokenizer/config assets for transformer initialization. If pretrained weights are absent, the training pipeline initializes the classifier from local config instead of silently fetching weights from the network.

## Project Areas

- `src/signal_backend/data/`: dataset loading, validation, split, label mapping, dataset settings
- `src/signal_backend/baselines/`: TF-IDF baseline models
- `src/signal_backend/training/`: shared metrics, evaluation, artifact saving, logging utilities, transformer training loop
- `src/signal_backend/inference/`: artifact loading, model_input building, unified predictor layer, Python inference wrappers
- `src/signal_backend/serving/`: API schemas, settings, service layer
- `src/signal_backend/models/`: transformer model loading helpers
- `configs/data/dataset_config.yaml`: dataset and split defaults
- `configs/train/*.yaml`: baseline and transformer training configs
- `configs/inference/api_config.yaml`: FastAPI defaults
- `scripts/`: runnable CLIs
- `apps/api/main.py`: FastAPI app entrypoint

## Commands

Inspect and validate the dataset:

```bash
python scripts/inspect_dataset.py
```

Build stratified train/val/test splits:

```bash
python scripts/make_split.py
```

Train TF-IDF + Logistic Regression:

```bash
python scripts/train_baseline.py --config configs/train/baseline_tfidf_logreg.yaml
```

Train TF-IDF + Linear SVM:

```bash
python scripts/train_baseline.py --config configs/train/baseline_tfidf_linear_svm.yaml
```

Re-evaluate a saved model on test data:

```bash
python scripts/evaluate_model.py --artifact-dir data/artifacts/baseline_tfidf_logreg
```

Train transformer classifier:

```bash
python scripts/train_transformer.py --config configs/train/transformer_classifier.yaml
```

Run the local API:

```bash
python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

## Unified Inference Layer

Python-level inference works for both baseline and transformer artifacts.

Main entry points:
- `load_artifact(artifact_dir: Path)`
- `build_model_input(title, text, model_input)`
- `predict_one_from_artifact(artifact_dir, *, title=None, text=None, model_input=None)`
- `predict_batch_from_artifact(artifact_dir, items=[...])`

Prediction result format:
- `prediction`
- `label_id`
- `scores`
- `score_type`
- `label_order`
- `model_type`

`scores` are exposed as a mapping `label -> score`. For `tfidf_logreg` and `transformer_classifier` these are probabilities. For `tfidf_linear_svm` these are decision scores.

## FastAPI Endpoints

Available endpoints:
- `GET /health`
- `POST /predict`
- `POST /batch_predict`

Example `/predict` request:

```json
{
  "title": "Команда выиграла матч",
  "text": "Спортсмены забили три гола и уверенно победили.",
  "artifact_dir": "data/artifacts/baseline_tfidf_logreg"
}
```

Example `/predict` response:

```json
{
  "prediction": "Спорт",
  "label_id": 2,
  "scores": {
    "Общество": 0.08,
    "Политика": 0.04,
    "Спорт": 0.81,
    "Экономика_и_бизнес": 0.07
  },
  "score_type": "probabilities",
  "label_order": [
    "Общество",
    "Политика",
    "Спорт",
    "Экономика_и_бизнес"
  ],
  "model_type": "tfidf_logreg"
}
```

Example `/batch_predict` request:

```json
{
  "artifact_dir": "data/artifacts/baseline_tfidf_logreg",
  "items": [
    {
      "title": "Команда выиграла матч",
      "text": "Спортсмены забили три гола и уверенно победили."
    },
    {
      "model_input": "Новый законопроект обсудили в парламенте."
    }
  ]
}
```

## Artifact Format

Each run is stored in:

`data/artifacts/<run_name>/`

Common artifacts:
- `metrics.json`
- `classification_report.json`
- `confusion_matrix.csv`
- `label_mapping.json`
- `config_snapshot.yaml`
- `run_summary.json`

Baseline-specific artifacts:
- `train.log`
- `train_log.jsonl`
- `model.joblib`
- `vectorizer.joblib`

Transformer-specific artifacts:
- `train.log`
- `train_log.jsonl`
- `tokenizer/`
- `best_model/`

Repeated evaluations are written into:

`data/artifacts/<run_name>/evaluations/<timestamp>/`

## Outputs from Data Preparation

`python scripts/inspect_dataset.py` creates:
- `data/processed/dataset_summary.json`

`python scripts/make_split.py` creates:
- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`
- `data/processed/split_report.json`