from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_score: float | None = None
        self.num_bad_epochs = 0

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs > self.patience
