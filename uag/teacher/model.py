"""Teacher model builders and trainable-only checkpoint helpers."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch

from uag.models.SAM2UNet import SAM2UNet, resolve_sam2_cfg


def resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' when available else 'cpu'."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_teacher(cfg: dict[str, Any], device: str = "cpu") -> SAM2UNet:
    """Construct SAM2UNet teacher from config and place it on `device`."""
    ckpt = cfg.get("sam2_checkpoint") or None
    model_cfg = resolve_sam2_cfg(str(cfg.get("sam2_cfg", "l")))
    input_size = cfg.get("input_size", 352)
    model = SAM2UNet(
        checkpoint_path=ckpt,
        model_cfg=model_cfg,
        input_size=input_size,
        device=device,
    )
    return model.to(device)


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Extract state_dict excluding frozen parameters (requires_grad=False)."""
    frozen_keys = {
        name for name, param in model.named_parameters() if not param.requires_grad
    }
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if k not in frozen_keys
    }


def load_teacher(
    cfg: dict[str, Any],
    weights: str | Path,
    device: str = "cpu",
) -> SAM2UNet:
    """Build teacher model and load saved trainable (or full) weights."""
    if not cfg.get("sam2_checkpoint"):
        warnings.warn(
            "sam2_checkpoint is empty; frozen SAM2 encoder weights are randomly initialized.",
            stacklevel=2,
        )

    model = build_teacher(cfg, device=device)
    state_dict = torch.load(Path(weights), map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    frozen_keys = {
        name for name, param in model.named_parameters() if not param.requires_grad
    }
    unexpected_missing = [k for k in missing if k not in frozen_keys]
    if unexpected or unexpected_missing:
        raise RuntimeError(
            f"Incompatible teacher checkpoint {weights}: "
            f"unexpected={unexpected}, missing_trainable={unexpected_missing}"
        )
    return model
