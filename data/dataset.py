"""ChessReD PyTorch Dataset."""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.utils import annotations_to_board, compute_orientation, normalize_corners


class ChessRedDataset(Dataset):
    """Dataset wrapper for ChessReD.

    Each sample returns:
        image      (C, H, W) float tensor
        board      (8, 8)    long tensor  — piece class ids 0-12
        corners    (8,)      float tensor — normalized [0,1] corner coords
        occupancy  (64,)     float tensor — binary, 1 if square is occupied
        orientation          long scalar  — 0-3 board orientation class
    """

    def __init__(self, dataroot: str, split: str = 'train', transform=None):
        self.dataroot = Path(dataroot)
        self.transform = transform

        with open(self.dataroot / 'annotations.json', 'r') as f:
            data = json.load(f)

        split_ids = set(data['splits'][split]['image_ids'])

        images_by_id = {img['id']: img for img in data['images']}

        pieces_by_image: dict = {}
        for piece in data['annotations']['pieces']:
            pieces_by_image.setdefault(piece['image_id'], []).append(piece)

        corners_by_image: dict = {}
        for ann in data['annotations']['corners']:
            corners_by_image[ann['image_id']] = ann['corners']

        self.samples = []
        for img_id in split_ids:
            img = images_by_id.get(img_id)
            if img is None:
                continue
            self.samples.append({
                'image_id': img_id,
                'path': img['path'],
                'width': img['width'],
                'height': img['height'],
                'pieces': pieces_by_image.get(img_id, []),
                # None when the image lacks corner annotations (~78% of dataset)
                'corners': corners_by_image.get(img_id, None),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        image = Image.open(self.dataroot / s['path']).convert('RGB')

        board = annotations_to_board(s['pieces'])                         # (8, 8) int64
        # NaN signals "no corner annotation" to the loss — masked out during training
        if s['corners'] is not None:
            corners = normalize_corners(s['corners'], s['width'], s['height'])
        else:
            corners = np.full(8, float('nan'), dtype=np.float32)
        occupancy = (board.reshape(-1) != 12).astype(np.float32)          # (64,) float32
        orientation = compute_orientation(s['pieces'])                    # int

        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.from_numpy(board).long(),
            torch.from_numpy(corners).float(),
            torch.from_numpy(occupancy).float(),
            torch.tensor(orientation, dtype=torch.long),
        )
