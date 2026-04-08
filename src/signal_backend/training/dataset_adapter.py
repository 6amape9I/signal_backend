from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset


class TransformerTextDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        label_ids: Sequence[int] | None,
        tokenizer,
        max_length: int,
    ) -> None:
        self.texts = list(texts)
        self.label_ids = None if label_ids is None else list(label_ids)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.label_ids is not None:
            item["labels"] = torch.tensor(self.label_ids[index], dtype=torch.long)
        return item
