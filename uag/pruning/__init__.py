"""HRankPlus-style structured filter pruning utilities for Attention U-Net."""

from uag.pruning.hrank import (
    PRUNABLE_UNITS,
    build_pruned_channel_cfg,
    collect_hrank_scores,
    compute_feature_ranks,
    concat_keep_indices,
    copy_pruned_attention_unet_weights,
    default_attention_unet_channel_cfg,
    estimate_model_stats,
    get_attention_unet_layer_map,
    parse_compress_rates,
    select_kept_indices,
    slice_bn_weights,
    slice_conv2d_transpose_weights,
    slice_conv2d_weights,
)

__all__ = [
    "PRUNABLE_UNITS",
    "build_pruned_channel_cfg",
    "collect_hrank_scores",
    "compute_feature_ranks",
    "concat_keep_indices",
    "copy_pruned_attention_unet_weights",
    "default_attention_unet_channel_cfg",
    "estimate_model_stats",
    "get_attention_unet_layer_map",
    "parse_compress_rates",
    "select_kept_indices",
    "slice_bn_weights",
    "slice_conv2d_transpose_weights",
    "slice_conv2d_weights",
]
