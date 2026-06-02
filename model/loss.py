"""Combined chess recognition loss with support for disabled auxiliary heads."""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessLoss(nn.Module):
    """L = L_primary + w_corners·L_corners + w_occ·L_occ + w_orient·L_orient

    L_primary : CrossEntropy on 8×8 piece classification  (always active)
    L_corners : MSE on corner coords                       (skipped if pred is None)
    L_occ     : BCE on per-square occupancy                (skipped if pred is None)
    L_orient  : CrossEntropy on board orientation          (skipped if pred is None)

    Corners also uses a per-sample NaN mask since only ~22% of images are annotated.
    """

    def __init__(self, w_corners: float = 0.2, w_occ: float = 0.1,
                 w_orient: float = 0.2, label_smoothing: float = 0.0):
        super().__init__()
        self.w_corners = w_corners
        self.w_occ = w_occ
        self.w_orient = w_orient
        self.label_smoothing = label_smoothing

    def forward(
        self,
        preds: Tuple,
        targets: Tuple,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        board_logits, pred_corners, pred_occ, pred_orient = preds
        board_labels, tgt_corners, tgt_occ, tgt_orient = targets

        device = board_logits.device

        l_primary = F.cross_entropy(board_logits, board_labels,
                                    label_smoothing=self.label_smoothing)

        # Corners — head may be absent; when present, mask out NaN-labelled samples
        if pred_corners is not None:
            valid = ~tgt_corners.isnan().any(dim=1)
            l_corners = (F.mse_loss(pred_corners[valid], tgt_corners[valid])
                         if valid.any()
                         else torch.tensor(0.0, device=device))
        else:
            l_corners = torch.tensor(0.0, device=device)

        # Occupancy — head may be absent; all images have occupancy labels
        l_occ = (F.binary_cross_entropy_with_logits(pred_occ, tgt_occ)
                 if pred_occ is not None
                 else torch.tensor(0.0, device=device))

        # Orientation — head may be absent
        l_orient = (F.cross_entropy(pred_orient, tgt_orient)
                    if pred_orient is not None
                    else torch.tensor(0.0, device=device))

        total = (l_primary
                 + self.w_corners * l_corners
                 + self.w_occ * l_occ
                 + self.w_orient * l_orient)

        components = {
            'primary':     l_primary.item(),
            'corners':     l_corners.item(),
            'occupancy':   l_occ.item(),
            'orientation': l_orient.item(),
        }
        return total, components
