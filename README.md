# signal_backend

`signal_backend` is the backend repository for dataset preparation, model training, and future inference delivery for Signal news classification.

Current scope:
- dataset import
- dataset validation
- reproducible `train` / `val` / `test` split

Out of scope at this stage:
- model training
- inference API
- deployment

## Dataset Location

The project expects the local dataset at:

`data/input/dataset_clean.jsonl`

Do not use the repository root as a working dataset path.

## Project Layout

- `src/signal_backend/`: package code
- `configs/data/dataset_config.yaml`: dataset and split defaults
- `scripts/inspect_dataset.py`: dataset inspection and validation CLI
- `scripts/make_split.py`: reproducible split CLI
- `data/processed/`: generated summaries and split files
- `data/artifacts/`: future model artifacts

## Quick Start

1. Place `dataset_clean.jsonl` into `data/input/dataset_clean.jsonl`.
2. Create and activate a virtual environment.
3. Install dependencies from `pyproject.toml`.

## Available Commands

Inspect and validate the dataset:

```bash
python scripts/inspect_dataset.py
```

Build stratified train/val/test splits:

```bash
python scripts/make_split.py
```

## Outputs

`python scripts/inspect_dataset.py` creates:
- `data/processed/dataset_summary.json`

`python scripts/make_split.py` creates:
- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`
- `data/processed/split_report.json`
