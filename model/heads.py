"""Prediction heads attached to ViT CLS token and patch tokens."""
import torch
import torch.nn as nn


class FlatBoardHead(nn.Module):
    """MLP: CLS token → 8×8×num_classes logits with no spatial structure.

    Used in Experiment 1 to test whether ViT improves over a CNN baseline
    even without exploiting the spatial arrangement of patch tokens.
    """

    def __init__(self, embed_dim: int = 768, num_classes: int = 13):
        super().__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 64 * num_classes),
        )

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        # cls: (B, embed_dim)
        return self.mlp(cls).reshape(cls.shape[0], self.num_classes, 8, 8)


class CornersHead(nn.Module):
    """MLP: CLS token → 8 normalized corner coords (4 corners × xy)."""

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 8),
            nn.Sigmoid(),   # outputs in [0, 1]
        )

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.mlp(cls)   # (B, 8)


class OccupancyHead(nn.Module):
    """MLP: CLS token → 64 occupancy logits (one per square)."""

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
        )

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.mlp(cls)   # (B, 64) — raw logits for BCE


class OrientationHead(nn.Module):
    """MLP: CLS token → 4-class orientation logits."""

    def __init__(self, embed_dim: int = 768, num_classes: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.mlp(cls)   # (B, 4)


class BoardHead(nn.Module):
    """2-D conv head: patch tokens → 8×8×num_classes piece logits.

    Flow:
        (B, 196, 768) → reshape (B, 768, 14, 14) → AdaptiveAvgPool 8×8
        → Conv 3×3 → BN → ReLU → Conv 1×1 → (B, num_classes, 8, 8)
    """

    def __init__(self, embed_dim: int = 768, num_classes: int = 13,
                 grid_size: int = 14):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size

        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        B = patch_tokens.shape[0]
        # (B, 196, 768) → (B, 768, 14, 14)
        x = patch_tokens.transpose(1, 2).reshape(
            B, self.embed_dim, self.grid_size, self.grid_size)
        x = self.pool(x)     # (B, 768, 8, 8)
        x = self.conv(x)     # (B, num_classes, 8, 8)
        return x
