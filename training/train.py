"""Single-epoch training loop."""
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """Run one training epoch and return aggregated metrics."""
    model.train()
    total_loss = 0.0
    comp_sums = {'primary': 0., 'corners': 0., 'occupancy': 0., 'orientation': 0.}
    exact_correct = 0
    total_samples = 0

    for images, board_labels, corners, occupancy, orientation in loader:
        images = images.to(device)
        board_labels = board_labels.to(device)
        corners = corners.to(device)
        occupancy = occupancy.to(device)
        orientation = orientation.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                preds = model(images)
                loss, comps = criterion(
                    preds, (board_labels, corners, occupancy, orientation))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images)
            loss, comps = criterion(
                preds, (board_labels, corners, occupancy, orientation))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        for k, v in comps.items():
            comp_sums[k] += v

        pred_labels = preds[0].argmax(dim=1)                       # (B, 8, 8)
        exact_correct += (pred_labels == board_labels).all(dim=(1, 2)).sum().item()
        total_samples += images.size(0)

    n = len(loader)
    return {
        'loss': total_loss / n,
        'loss_primary': comp_sums['primary'] / n,
        'loss_corners': comp_sums['corners'] / n,
        'loss_occupancy': comp_sums['occupancy'] / n,
        'loss_orientation': comp_sums['orientation'] / n,
        'train_exact_acc': exact_correct / total_samples,
    }
