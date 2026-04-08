# signal_backend

`signal_backend` is the backend repository for Signal news classification.

Implemented stages:
- dataset import from JSONL
- dataset validation
- reproducible `train` / `val` / `test` split
- baseline training with TF-IDF models
- transformer training with artifact-ready inference utilities

Out of scope at this stage:
- HTTP inference API
- deployment
- frontend integration

## Dataset Location

The project expects the local dataset at:

`data/input/dataset_clean.jsonl`

The repository root must not be used as a working dataset path.

## Local Base Model

The default offline transformer config points to:

`resources/base_models/mmBERT-base`

This directory stores the minimal local tokenizer/config assets for transformer initialization. If pretrained weights are absent, the pipeline initializes the classifier from local config instead of silently fetching weights from the network.

## Project Areas

- `src/signal_backend/data/`: dataset loading, validation, split, label mapping, dataset settings
- `src/signal_backend/baselines/`: TF-IDF baseline models
- `src/signal_backend/training/`: shared metrics, evaluation, artifact saving, logging utilities, transformer training loop
- `src/signal_backend/models/`: transformer model loading helpers
- `src/signal_backend/inference/`: Python-level inference helpers for saved transformer artifacts
- `configs/data/dataset_config.yaml`: dataset and split defaults
- `configs/train/*.yaml`: baseline and transformer training configs
- `scripts/`: runnable CLIs

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

## Baseline vs Transformer

Baseline models:
- `tfidf_logreg`: TF-IDF + Logistic Regression with probabilities
- `tfidf_linear_svm`: TF-IDF + Linear SVM with decision scores

Transformer model:
- `transformer_classifier`: Hugging Face sequence classifier trained on `model_input -> category_teacher_final`
- the default config points to the local `resources/base_models/mmBERT-base` directory; if local weights are absent, the pipeline falls back to model initialization from config instead of using silent network behavior

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