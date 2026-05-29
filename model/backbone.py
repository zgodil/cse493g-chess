"""ViT-B/16 backbone that returns CLS token and patch tokens separately."""
import torch
import torch.nn as nn
import torchvision.models as tv_models


class ViTBackbone(nn.Module):
    """Wraps torchvision ViT-B/16 and exposes (cls_token, patch_tokens).

    Input : (B, 3, img_size, img_size)  — expects img_size == 224
    Output: cls_token    (B, 768)
            patch_tokens (B, 196, 768)  — 14×14 spatial grid flattened
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = (tv_models.ViT_B_16_Weights.IMAGENET1K_V1
                   if pretrained else None)
        vit = tv_models.vit_b_16(weights=weights)

        # Keep everything except the classification head
        self.patch_embed = vit.conv_proj       # Conv2d for patch projection
        self.class_token = vit.class_token     # (1, 1, 768)
        self.encoder = vit.encoder            # Transformer + pos embed
        self.embed_dim = 768
        # Number of patches for 224×224 with patch_size=16: 14×14 = 196
        self.grid_size = 14

    def forward(self, x: torch.Tensor):
        # Patch embedding: (B, 768, 14, 14) → (B, 196, 768)
        B = x.shape[0]
        x = self.patch_embed(x)                          # (B, 768, 14, 14)
        x = x.flatten(2).transpose(1, 2)                # (B, 196, 768)

        # Prepend CLS token and run through transformer encoder
        cls = self.class_token.expand(B, -1, -1)         # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)                   # (B, 197, 768)
        x = self.encoder(x)                              # (B, 197, 768)

        cls_token = x[:, 0]       # (B, 768)
        patch_tokens = x[:, 1:]   # (B, 196, 768)
        return cls_token, patch_tokens
