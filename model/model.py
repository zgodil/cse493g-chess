"""Chess board recognition model — configurable per experiment."""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from model.backbone import ViTBackbone
from model.heads import (BoardHead, CornersHead, FlatBoardHead,
                         OccupancyHead, OrientationHead)


class ChessBoardRecognizer(nn.Module):
    """ViT-B/16 with a configurable board head and optional auxiliary heads.

    Parameters
    ----------
    head : 'conv2d' | 'flat'
        'conv2d' — patch tokens → 2D conv → 8×8×13  (Experiments 2, 2b, 3, ablation)
        'flat'   — CLS token → MLP → 8×8×13         (Experiment 1)
    aux_heads : tuple of strings
        Any subset of ('corners', 'occupancy', 'orientation').
        Heads not listed are set to None and skipped in forward.
    pretrained : bool
        True  → ImageNet-1K ViT-B/16 weights  (Experiments 1, 2, 3, ablation)
        False → random initialisation          (Experiment 2b)

    Forward always returns a 4-tuple:
        (board_logits, corners|None, occ_logits|None, orient_logits|None)
    """

    def __init__(
        self,
        head: str = 'conv2d',
        aux_heads: tuple = ('corners', 'occupancy', 'orientation'),
        pretrained: bool = True,
        num_piece_classes: int = 13,
        num_orient_classes: int = 4,
    ):
        super().__init__()
        self._head_type = head

        self.backbone = ViTBackbone(pretrained=pretrained)
        d = self.backbone.embed_dim   # 768
        g = self.backbone.grid_size   # 14

        if head == 'conv2d':
            self.board_head = BoardHead(d, num_piece_classes, g)
        elif head == 'flat':
            self.board_head = FlatBoardHead(d, num_piece_classes)
        else:
            raise ValueError(f"Unknown head type: {head!r}. Choose 'conv2d' or 'flat'.")

        self.corners_head = (
            CornersHead(d) if 'corners' in aux_heads else None)
        self.occupancy_head = (
            OccupancyHead(d) if 'occupancy' in aux_heads else None)
        self.orientation_head = (
            OrientationHead(d, num_orient_classes) if 'orientation' in aux_heads else None)

    def forward(self, x: torch.Tensor):
        cls, patches = self.backbone(x)
        board_in = patches if self._head_type == 'conv2d' else cls

        return (
            self.board_head(board_in),
            self.corners_head(cls)     if self.corners_head     is not None else None,
            self.occupancy_head(cls)   if self.occupancy_head   is not None else None,
            self.orientation_head(cls) if self.orientation_head is not None else None,
        )


def build_model(cfg: dict) -> ChessBoardRecognizer:
    """Construct a ChessBoardRecognizer from a config dict."""
    m = cfg['model']
    return ChessBoardRecognizer(
        head=m.get('head', 'conv2d'),
        aux_heads=tuple(m.get('aux_heads', ['corners', 'occupancy', 'orientation'])),
        pretrained=m.get('pretrained', True),
        num_piece_classes=m.get('num_piece_classes', 13),
        num_orient_classes=m.get('num_orient_classes', 4),
    )
