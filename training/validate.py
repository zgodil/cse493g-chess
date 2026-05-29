"""Validation loop."""
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a validation/test split.

    Returns loss plus exact-match and tolerance-1 recognition accuracy.
    """
    model.eval()
    total_loss = 0.0
    exact_correct = 0
    tol1_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, board_labels, corners, occupancy, orientation in loader:
            images = images.to(device)
            board_labels = board_labels.to(device)
            corners = corners.to(device)
            occupancy = occupancy.to(device)
            orientation = orientation.to(device)

            preds = model(images)
            loss, _ = criterion(preds, (board_labels, corners, occupancy, orientation))
            total_loss += loss.item()

            pred_labels = preds[0].argmax(dim=1)   # (B, 8, 8)
            per_sample_correct = (pred_labels == board_labels).reshape(
                pred_labels.shape[0], -1).sum(dim=1)   # (B,)

            exact_correct += (per_sample_correct == 64).sum().item()
            tol1_correct += (per_sample_correct >= 63).sum().item()
            total_samples += images.size(0)

    n = len(loader)
    return {
        'val_loss': total_loss / n,
        'val_exact_acc': exact_correct / total_samples,
        'val_tol1_acc': tol1_correct / total_samples,
    }
