"""Dataset-level binary confusion metrics matching uag/metrics.py."""

from __future__ import annotations

import torch


class ConfusionAccumulator:
    """Accumulate global TP, FP, and FN counts across batches."""

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-7) -> None:
        self.threshold = float(threshold)
        self.smooth = float(smooth)
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    @torch.no_grad()
    def update(self, prob: torch.Tensor, mask: torch.Tensor) -> None:
        y_true = (mask >= 0.5).to(dtype=torch.float32)
        y_pred = (prob >= self.threshold).to(dtype=torch.float32)

        self.tp += float((y_true * y_pred).sum().item())
        self.fp += float(((1.0 - y_true) * y_pred).sum().item())
        self.fn += float((y_true * (1.0 - y_pred)).sum().item())

    def compute(self) -> dict[str, float]:
        s = self.smooth
        precision = (self.tp + s) / (self.tp + self.fp + s)
        recall = (self.tp + s) / (self.tp + self.fn + s)
        f1_score = (2.0 * self.tp + s) / (2.0 * self.tp + self.fp + self.fn + s)
        iou = (self.tp + s) / (self.tp + self.fp + self.fn + s)
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "iou": float(iou),
        }
