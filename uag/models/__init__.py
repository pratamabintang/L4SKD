"""Model architectures and factory helpers for landslide segmentation."""

from __future__ import annotations

from typing import Optional, Sequence

from uag.models.attention_unet import (
    RepeatChannels,
    attn_gating_block,
    build_attention_unet,
    unet_conv2d,
    unet_gating_signal,
)
from uag.models.dual_branch import (
    ChannelSelect,
    build_dual_branch_attention_unet,
)
from uag.models.unet import build_unet

MODEL_BUILDERS = {
    "attention_unet": build_attention_unet,
    "unet": build_unet,
    "dual_branch": build_dual_branch_attention_unet,
}
MODEL_CHOICES = list(MODEL_BUILDERS)
MODEL_DISPLAY_NAMES = {
    "attention_unet": "Attention_Unet",
    "unet": "Unet",
    "dual_branch": "DualBranch_Attention_Unet",
}


def build_model(
    model_name: str,
    input_shape: tuple[int, int, int] = (128, 128, 14),
    filters: int = 4,
    pretrained_weights: Optional[str] = None,
    bands: Optional[Sequence[int]] = None,
):
    """Build an uncompiled segmentation model by architecture name."""
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model {model_name!r}. Choose from {MODEL_CHOICES}."
        )
    kwargs = {
        "input_shape": input_shape,
        "filters": filters,
        "pretrained_weights": pretrained_weights,
    }
    if model_name == "dual_branch":
        kwargs["bands"] = bands
    return MODEL_BUILDERS[model_name](**kwargs)


__all__ = [
    "ChannelSelect",
    "MODEL_BUILDERS",
    "MODEL_CHOICES",
    "MODEL_DISPLAY_NAMES",
    "RepeatChannels",
    "attn_gating_block",
    "build_attention_unet",
    "build_dual_branch_attention_unet",
    "build_model",
    "build_unet",
    "unet_conv2d",
    "unet_gating_signal",
]
