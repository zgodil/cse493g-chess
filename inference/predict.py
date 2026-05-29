"""Inference: load checkpoint, predict board state, output FEN."""
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image

from data.transforms import get_val_transforms
from data.utils import board_to_fen
from model.model import ChessBoardRecognizer


def load_model(checkpoint_path: str, device: str = 'cpu') -> ChessBoardRecognizer:
    model = ChessBoardRecognizer(pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def _apply_chess_constraints(board: np.ndarray) -> np.ndarray:
    """Enforce basic chess legality on a predicted 8×8 board.

    Fixes handled:
      - Pawns on rank 1 (row 7) or rank 8 (row 0) are cleared.
      - If more than one king of a color is present, extras are cleared.
    """
    board = board.copy()

    # Pawns cannot be on rank 1 or rank 8
    for col in range(8):
        if board[0, col] in (0, 6):
            board[0, col] = 12
        if board[7, col] in (0, 6):
            board[7, col] = 12

    # At most one king per color
    for king_id in (5, 11):
        positions = list(zip(*np.where(board == king_id)))
        for pos in positions[1:]:     # keep first occurrence, clear extras
            board[pos] = 12

    return board


def predict(
    model: ChessBoardRecognizer,
    image_path: str,
    device: str = 'cpu',
    img_size: int = 224,
) -> Dict:
    """Run inference on a single image.

    Returns a dict with keys:
        fen         — FEN board string
        board       — (8, 8) numpy array of class ids
        corners     — (8,) numpy array of normalized corner coords
        occupancy   — (64,) numpy array of per-square occupancy probs
        orientation — predicted orientation class (0-3)
    """
    transform = get_val_transforms(img_size)
    image = Image.open(image_path).convert('RGB')
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        board_logits, corners, occ_logits, orient_logits = model(x)

    board_arr = board_logits.argmax(dim=1).squeeze(0).cpu().numpy()   # (8, 8)
    board_arr = _apply_chess_constraints(board_arr)

    return {
        'fen': board_to_fen(board_arr),
        'board': board_arr,
        'corners': corners.squeeze(0).cpu().numpy(),
        'occupancy': torch.sigmoid(occ_logits).squeeze(0).cpu().numpy(),
        'orientation': int(orient_logits.argmax(dim=1).item()),
    }
