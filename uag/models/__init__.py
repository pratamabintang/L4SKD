"""Model architectures and factory helpers for landslide segmentation."""

from __future__ import annotations

import importlib
from typing import Optional, Sequence

MODEL_CHOICES = ["attention_unet", "unet", "dual_branch"]
MODEL_DISPLAY_NAMES = {
    "attention_unet": "Attention_Unet",
    "unet": "Unet",
    "dual_branch": "DualBranch_Attention_Unet",
}

_LAZY_EXPORTS = {
    "RepeatChannels": ("uag.models.attention_unet", "RepeatChannels"),
    "attn_gating_block": ("uag.models.attention_unet", "attn_gating_block"),
    "build_attention_unet": ("uag.models.attention_unet", "build_attention_unet"),
    "unet_conv2d": ("uag.models.attention_unet", "unet_conv2d"),
    "unet_gating_signal": ("uag.models.attention_unet", "unet_gating_signal"),
    "ChannelSelect": ("uag.models.dual_branch", "ChannelSelect"),
    "build_dual_branch_attention_unet": (
        "uag.models.dual_branch",
        "build_dual_branch_attention_unet",
    ),
    "build_unet": ("uag.models.unet", "build_unet"),
}


def _get_builders() -> dict:
    return {
        "attention_unet": __getattr__("build_attention_unet"),
        "unet": __getattr__("build_unet"),
        "dual_branch": __getattr__("build_dual_branch_attention_unet"),
    }


def __getattr__(name: str):
    if name == "MODEL_BUILDERS":
        return _get_builders()
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_model(
    model_name: str,
    input_shape: tuple[int, int, int] = (128, 128, 14),
    filters: int = 4,
    pretrained_weights: Optional[str] = None,
    bands: Optional[Sequence[int]] = None,
    channel_cfg: Optional[dict[str, int]] = None,
):
    """Build an uncompiled segmentation model by architecture name."""
    builders = _get_builders()
    if model_name not in builders:
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
    elif model_name == "attention_unet" and channel_cfg is not None:
        kwargs["channel_cfg"] = channel_cfg
    return builders[model_name](**kwargs)


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
