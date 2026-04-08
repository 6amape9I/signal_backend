from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
TRAIN_CONFIGS_DIR = CONFIGS_DIR / "train"
RESOURCES_DIR = REPO_ROOT / "resources"
BASE_MODELS_DIR = RESOURCES_DIR / "base_models"
DATA_DIR = REPO_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
DEFAULT_DATASET_PATH = INPUT_DIR / "dataset_clean.jsonl"
DATASET_CONFIG_PATH = CONFIGS_DIR / "data" / "dataset_config.yaml"
TRAIN_SPLIT_PATH = PROCESSED_DIR / "train.jsonl"
VAL_SPLIT_PATH = PROCESSED_DIR / "val.jsonl"
TEST_SPLIT_PATH = PROCESSED_DIR / "test.jsonl"
DATASET_SUMMARY_PATH = PROCESSED_DIR / "dataset_summary.json"
SPLIT_REPORT_PATH = PROCESSED_DIR / "split_report.json"
LOCAL_MMBERT_BASE_DIR = BASE_MODELS_DIR / "mmBERT-base"
