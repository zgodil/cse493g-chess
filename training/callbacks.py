"""Training callbacks: checkpointing and early stopping."""
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


class ModelCheckpoint:
    """Save the best model checkpoint according to a monitored metric."""

    def __init__(self, save_dir: str, monitor: str = 'val_exact_acc',
                 mode: str = 'max'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.best_path: Path | None = None

    def __call__(self, model: nn.Module, optimizer, epoch: int,
                 metrics: Dict[str, float]) -> bool:
        value = metrics[self.monitor]
        improved = (
            (self.mode == 'max' and value > self.best_value) or
            (self.mode == 'min' and value < self.best_value)
        )
        if improved:
            self.best_value = value
            self.best_path = self.save_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, self.best_path)
            print(f"  [ckpt] saved best model  {self.monitor}={value:.4f}")
        return improved


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, monitor: str = 'val_exact_acc', patience: int = 10,
                 mode: str = 'max'):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_value = float('-inf') if mode == 'max' else float('inf')

    def __call__(self, metrics: Dict[str, float]) -> bool:
        value = metrics[self.monitor]
        improved = (
            (self.mode == 'max' and value > self.best_value) or
            (self.mode == 'min' and value < self.best_value)
        )
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"  [early stop] no improvement for {self.patience} epochs")
                return True
        return False
