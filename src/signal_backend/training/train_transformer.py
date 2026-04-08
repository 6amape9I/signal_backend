from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from signal_backend.config import ConfigError, load_yaml_config, normalize_optional_string, resolve_path
from signal_backend.models.transformer_classifier import build_transformer_classifier, load_tokenizer
from signal_backend.training.checkpointing import save_best_model
from signal_backend.training.dataset_adapter import TransformerTextDataset
from signal_backend.training.early_stopping import EarlyStopping
from signal_backend.training.evaluate import (
    build_train_label_mapping,
    dataframe_labels,
    dataframe_texts,
    load_split_data,
)
from signal_backend.training.metrics import compute_evaluation_result
from signal_backend.training.save_artifacts import (
    build_run_directory,
    save_evaluation_artifacts,
    save_label_mapping,
    save_run_summary,
    save_yaml,
)


MODEL_TYPE = "transformer_classifier"


def load_transformer_training_config(config_path: Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    data = config.get("data", {})
    model = config.get("model", {})
    training = config.get("training", {})
    scheduler = config.get("scheduler", {})
    run = config.get("run", {})

    for name, section in {
        "data": data,
        "model": model,
        "training": training,
        "scheduler": scheduler,
        "run": run,
    }.items():
        if not isinstance(section, dict):
            raise ConfigError(f"'{name}' section must be a mapping.")

    return {
        "data": {
            "train_path": resolve_path(data.get("train_path", "data/processed/train.jsonl")).as_posix(),
            "val_path": resolve_path(data.get("val_path", "data/processed/val.jsonl")).as_posix(),
            "test_path": resolve_path(data.get("test_path", "data/processed/test.jsonl")).as_posix(),
        },
        "model": {
            "model_name_or_path": str(model.get("model_name_or_path")),
            "max_length": int(model.get("max_length", 128)),
        },
        "training": {
            "batch_size": int(training.get("batch_size", 8)),
            "learning_rate": float(training.get("learning_rate", 2e-5)),
            "weight_decay": float(training.get("weight_decay", 0.01)),
            "num_epochs": int(training.get("num_epochs", 3)),
            "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 1)),
            "random_seed": int(training.get("random_seed", 42)),
            "class_weight_mode": normalize_optional_string(training.get("class_weight_mode")),
            "early_stopping_patience": int(training.get("early_stopping_patience", 2)),
        },
        "scheduler": {
            "warmup_ratio": scheduler.get("warmup_ratio"),
            "warmup_steps": scheduler.get("warmup_steps"),
        },
        "run": {
            "run_name": run.get("run_name"),
            "output_dir": resolve_path(run.get("output_dir", "data/artifacts")).as_posix(),
        },
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(label_ids: list[int], num_labels: int) -> torch.Tensor:
    counts = np.bincount(label_ids, minlength=num_labels)
    total = counts.sum()
    weights = total / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float32)


def get_warmup_steps(total_steps: int, scheduler_config: dict[str, Any]) -> int:
    if scheduler_config.get("warmup_steps") is not None:
        return int(scheduler_config["warmup_steps"])
    warmup_ratio = scheduler_config.get("warmup_ratio")
    if warmup_ratio is None:
        return 0
    return int(total_steps * float(warmup_ratio))


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _train_one_epoch(
    *,
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    loss_fn,
    gradient_accumulation_steps: int,
) -> float:
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0

    for step, batch in enumerate(dataloader, start=1):
        batch = _move_batch_to_device(batch, device)
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = loss_fn(outputs.logits, batch["labels"])
        total_loss += float(loss.item())
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if step % gradient_accumulation_steps == 0 or step == len(dataloader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / max(len(dataloader), 1)


def _evaluate_transformer(
    *,
    model,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str,
    id_to_label: dict[int, str],
):
    model.eval()
    all_true_ids: list[int] = []
    all_pred_ids: list[int] = []
    all_probabilities: list[list[float]] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            probabilities = torch.softmax(outputs.logits, dim=-1)
            pred_ids = torch.argmax(probabilities, dim=-1)
            all_true_ids.extend(batch["labels"].detach().cpu().tolist())
            all_pred_ids.extend(pred_ids.detach().cpu().tolist())
            all_probabilities.extend(probabilities.detach().cpu().tolist())

    labels = [id_to_label[index] for index in sorted(id_to_label)]
    result = compute_evaluation_result(
        split_name=split_name,
        y_true=[id_to_label[label_id] for label_id in all_true_ids],
        y_pred=[id_to_label[label_id] for label_id in all_pred_ids],
        labels=labels,
    )
    return result, all_probabilities


def train_transformer_from_config(config: dict[str, Any]) -> Path:
    split_data = load_split_data(
        train_path=Path(config["data"]["train_path"]),
        val_path=Path(config["data"]["val_path"]),
        test_path=Path(config["data"]["test_path"]),
    )

    set_seed(config["training"]["random_seed"])
    label_to_id, id_to_label = build_train_label_mapping(split_data.train_df)
    train_label_ids = [label_to_id[label] for label in dataframe_labels(split_data.train_df)]
    val_label_ids = [label_to_id[label] for label in dataframe_labels(split_data.val_df)]
    test_label_ids = [label_to_id[label] for label in dataframe_labels(split_data.test_df)]

    tokenizer = load_tokenizer(config["model"]["model_name_or_path"])
    model, loaded_pretrained = build_transformer_classifier(
        config["model"]["model_name_or_path"],
        label_to_id,
        id_to_label,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = TransformerTextDataset(
        dataframe_texts(split_data.train_df),
        train_label_ids,
        tokenizer,
        config["model"]["max_length"],
    )
    val_dataset = TransformerTextDataset(
        dataframe_texts(split_data.val_df),
        val_label_ids,
        tokenizer,
        config["model"]["max_length"],
    )
    test_dataset = TransformerTextDataset(
        dataframe_texts(split_data.test_df),
        test_label_ids,
        tokenizer,
        config["model"]["max_length"],
    )

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    total_steps = math.ceil(len(train_loader) / config["training"]["gradient_accumulation_steps"]) * config["training"]["num_epochs"]
    warmup_steps = get_warmup_steps(total_steps, config["scheduler"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_steps, 1),
    )

    class_weight_mode = config["training"]["class_weight_mode"]
    class_weights = None
    if class_weight_mode == "balanced":
        class_weights = compute_class_weights(train_label_ids, len(label_to_id)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    run_dir = build_run_directory(
        model_type=MODEL_TYPE,
        run_name=config["run"].get("run_name"),
        output_dir=Path(config["run"]["output_dir"]),
    )
    tokenizer.save_pretrained(run_dir / "tokenizer")

    train_log_path = run_dir / "train_log.jsonl"
    early_stopper = EarlyStopping(config["training"]["early_stopping_patience"])
    best_val_result = None

    for epoch in range(1, config["training"]["num_epochs"] + 1):
        train_loss = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            loss_fn=loss_fn,
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        )
        val_result, _ = _evaluate_transformer(
            model=model,
            dataloader=val_loader,
            device=device,
            split_name="val",
            id_to_label=id_to_label,
        )

        with train_log_path.open("a", encoding="utf-8") as train_log_file:
            train_log_file.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_metrics": val_result.metrics,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if best_val_result is None or val_result.metrics["macro_f1"] > best_val_result.metrics["macro_f1"]:
            best_val_result = val_result
            save_best_model(model, run_dir)

        if early_stopper.step(val_result.metrics["macro_f1"]):
            break

    best_model = AutoModelForSequenceClassification.from_pretrained(run_dir / "best_model", local_files_only=True)
    best_model.to(device)
    test_result, _ = _evaluate_transformer(
        model=best_model,
        dataloader=test_loader,
        device=device,
        split_name="test",
        id_to_label=id_to_label,
    )

    save_evaluation_artifacts(model_type=MODEL_TYPE, results=[best_val_result, test_result], run_dir=run_dir)
    save_label_mapping(label_to_id, id_to_label, run_dir / "label_mapping.json")
    save_yaml(config, run_dir / "config_snapshot.yaml")

    run_summary = {
        "model_type": MODEL_TYPE,
        "base_model_name": config["model"]["model_name_or_path"],
        "loaded_pretrained_weights": loaded_pretrained,
        "dataset_paths": {
            "train": split_data.train_path.as_posix(),
            "val": split_data.val_path.as_posix(),
            "test": split_data.test_path.as_posix(),
        },
        "split_sizes": {
            "train": len(split_data.train_df),
            "val": len(split_data.val_df),
            "test": len(split_data.test_df),
        },
        "classes": list(label_to_id.keys()),
        "best_validation_metric": best_val_result.metrics,
        "final_test_metrics": test_result.metrics,
    }
    save_run_summary(run_summary, run_dir)
    return run_dir
