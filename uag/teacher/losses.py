"""PyTorch PraNet / SAM2-UNet structure loss on logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def structure_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Weighted BCE-with-logits + weighted IoU loss for NCHW binary segmentation."""
    logits = logits.float()
    mask = mask.float()

    weit = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(logits)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)
    return (wbce + wiou).mean()
