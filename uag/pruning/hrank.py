"""HRankPlus-style structured filter pruning for Attention U-Net."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

PRUNABLE_UNITS: tuple[str, ...] = (
    "conv1_1",
    "conv1_2",
    "conv2_1",
    "conv2_2",
    "conv3_1",
    "conv3_2",
    "conv4_1",
    "conv4_2",
    "center_1",
    "center_2",
    "up1_deconv",
    "up2_deconv",
    "up3_deconv",
    "up4_deconv",
    "inter_1",
    "inter_2",
    "inter_3",
)


def default_attention_unet_channel_cfg(filters: int) -> dict[str, int]:
    """Return the default per-unit channel widths for Attention U-Net at `filters`."""
    f = int(filters)
    if f < 1:
        raise ValueError(f"filters must be >= 1, got {filters}")
    return {
        "conv1_1": f,
        "conv1_2": f,
        "conv2_1": f,
        "conv2_2": f,
        "conv3_1": f * 2,
        "conv3_2": f * 2,
        "conv4_1": f * 2,
        "conv4_2": f * 2,
        "center_1": f * 4,
        "center_2": f * 4,
        "up1_deconv": f,
        "up2_deconv": f * 2,
        "up3_deconv": f,
        "up4_deconv": f,
        "inter_1": f * 4,
        "inter_2": f * 2,
        "inter_3": f,
    }


def _compute_nchw_ranks(nchw: np.ndarray, unit: str | None = None) -> np.ndarray:
    """Compute per-sample, per-channel matrix ranks for an (B, C, H, W) tensor robustly."""
    arr = np.asarray(nchw, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        label = f" for unit {unit!r}" if unit else ""
        print(f"Warning: non-finite feature map values detected{label}; replacing with 0.0 for rank computation.")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return np.linalg.matrix_rank(arr)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as scipy_svd

        b, c = arr.shape[:2]
        ranks = np.zeros((b, c), dtype=np.intp)
        eps = np.finfo(np.float64).eps
        for i in range(b):
            for j in range(c):
                mat = arr[i, j]
                if not np.any(mat):
                    continue
                try:
                    s = scipy_svd(mat, compute_uv=False, lapack_driver="gesvd")
                except np.linalg.LinAlgError:
                    s = scipy_svd(mat, compute_uv=False, lapack_driver="gesvd", check_finite=False)
                tol = float(s.max()) * max(mat.shape) * eps
                ranks[i, j] = int(np.sum(s > tol))
        return ranks


def compute_feature_ranks(feature_map: np.ndarray) -> np.ndarray:
    """Compute mean matrix rank across batch for an NHWC feature map -> shape (C,)."""
    arr = np.asarray(feature_map, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D NHWC tensor, got shape {arr.shape}.")
    # Move channels next to batch: (B, C, H, W) so matrix_rank operates over (H, W).
    nchw = np.transpose(arr, (0, 3, 1, 2))
    ranks = _compute_nchw_ranks(nchw)  # shape (B, C)
    return ranks.mean(axis=0, dtype=np.float64)


def select_kept_indices(
    ranks: np.ndarray,
    compress_rate: float,
    min_channels: int = 2,
) -> np.ndarray:
    """Select sorted channel indices with highest rank for a given compression rate."""
    rate = float(compress_rate)
    if not (0.0 <= rate < 1.0):
        raise ValueError(f"compress_rate must be in [0.0, 1.0), got {compress_rate}")
    ranks_arr = np.asarray(ranks, dtype=np.float64)
    if ranks_arr.ndim != 1 or ranks_arr.size == 0:
        raise ValueError(f"ranks must be a non-empty 1D array, got shape {ranks_arr.shape}")

    c = int(ranks_arr.size)
    floor_keep = min(int(min_channels), c)
    num_keep = max(floor_keep, int(round(c * (1.0 - rate))))
    num_keep = max(1, min(c, num_keep))

    # Stable descending sort by rank (tie-break preserves lower original channel index first).
    order = np.argsort(-ranks_arr, kind="mergesort")
    kept = np.sort(order[:num_keep]).astype(np.int64)
    return kept


def parse_compress_rates(
    spec: str | float | int | dict[str, float],
    unit_names: Sequence[str] = PRUNABLE_UNITS,
) -> dict[str, float]:
    """Parse uniform, comma-separated, or dict/JSON per-unit compression rates."""
    units = list(unit_names)
    if isinstance(spec, (int, float)):
        rate = float(spec)
        if not (0.0 <= rate < 1.0):
            raise ValueError(f"compress_rate must be in [0.0, 1.0), got {rate}")
        return {u: rate for u in units}

    if isinstance(spec, dict):
        default_rate = float(spec.get("default", 0.0))
        unknown = set(spec) - set(units) - {"default"}
        if unknown:
            raise ValueError(f"Unknown unit names in compress_rate dict: {sorted(unknown)}")
        rates = {u: float(spec.get(u, default_rate)) for u in units}
        for u, r in rates.items():
            if not (0.0 <= r < 1.0):
                raise ValueError(f"compress_rate for {u!r} must be in [0.0, 1.0), got {r}")
        return rates

    text = str(spec).strip()
    if not text:
        raise ValueError("compress_rate specification cannot be empty.")

    maybe_path = Path(text)
    if maybe_path.is_file():
        with maybe_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected JSON object in {maybe_path}, got {type(loaded).__name__}.")
        return parse_compress_rates(loaded, unit_names=units)

    if text.startswith("{"):
        loaded = json.loads(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected JSON object, got {type(loaded).__name__}.")
        return parse_compress_rates(loaded, unit_names=units)

    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) != len(units):
            raise ValueError(
                f"Expected {len(units)} comma-separated rates for {units}, got {len(parts)}."
            )
        rates = {u: float(p) for u, p in zip(units, parts)}
        for u, r in rates.items():
            if not (0.0 <= r < 1.0):
                raise ValueError(f"compress_rate for {u!r} must be in [0.0, 1.0), got {r}")
        return rates

    return parse_compress_rates(float(text), unit_names=units)


def build_pruned_channel_cfg(
    orig_cfg: dict[str, int],
    ranks_by_unit: dict[str, np.ndarray],
    rates_by_unit: dict[str, float],
    min_channels: int = 2,
) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    """Compute pruned channel counts and kept channel indices for each unit."""
    pruned_cfg: dict[str, int] = {}
    keep_indices: dict[str, np.ndarray] = {}
    for unit in PRUNABLE_UNITS:
        if unit not in ranks_by_unit:
            raise KeyError(f"Missing HRank scores for unit {unit!r}.")
        ranks = np.asarray(ranks_by_unit[unit], dtype=np.float64)
        expected_c = int(orig_cfg[unit])
        if ranks.shape != (expected_c,):
            raise ValueError(
                f"Rank shape mismatch for {unit!r}: expected ({expected_c},), got {ranks.shape}."
            )
        kept = select_kept_indices(ranks, rates_by_unit[unit], min_channels=min_channels)
        keep_indices[unit] = kept
        pruned_cfg[unit] = int(kept.size)
    return pruned_cfg, keep_indices


def slice_conv2d_weights(
    weights: Sequence[np.ndarray],
    in_idx: np.ndarray,
    out_idx: np.ndarray,
) -> list[np.ndarray]:
    """Slice Conv2D weights (kH, kW, C_in, C_out) and optional bias."""
    kernel = weights[0][:, :, in_idx, :][:, :, :, out_idx]
    if len(weights) == 1:
        return [np.ascontiguousarray(kernel)]
    bias = weights[1][out_idx]
    return [np.ascontiguousarray(kernel), np.ascontiguousarray(bias)]


def slice_conv2d_transpose_weights(
    weights: Sequence[np.ndarray],
    in_idx: np.ndarray,
    out_idx: np.ndarray,
) -> list[np.ndarray]:
    """Slice Conv2DTranspose weights (kH, kW, C_out, C_in) and optional bias."""
    kernel = weights[0][:, :, out_idx, :][:, :, :, in_idx]
    if len(weights) == 1:
        return [np.ascontiguousarray(kernel)]
    bias = weights[1][out_idx]
    return [np.ascontiguousarray(kernel), np.ascontiguousarray(bias)]


def slice_bn_weights(
    weights: Sequence[np.ndarray],
    idx: np.ndarray,
) -> list[np.ndarray]:
    """Slice BatchNormalization weights [gamma, beta, moving_mean, moving_variance]."""
    return [np.ascontiguousarray(w[idx]) for w in weights]


def concat_keep_indices(
    keep_a: np.ndarray,
    keep_b: np.ndarray,
    orig_a_len: int,
) -> np.ndarray:
    """Compute concatenated channel indices for [tensor_a, tensor_b]."""
    return np.concatenate(
        [np.asarray(keep_a, dtype=np.int64), int(orig_a_len) + np.asarray(keep_b, dtype=np.int64)]
    )


def get_attention_unet_layer_map(model) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map weighted layer names and 17 activation tap tensors from an Attention U-Net model."""
    from tensorflow import keras

    layers = list(model.layers)
    name_to_idx = {layer.name: idx for idx, layer in enumerate(layers)}

    weighted: dict[str, Any] = {}
    taps: dict[str, Any] = {}

    named_blocks = ["conv1", "conv2", "conv3", "conv4", "center"]
    for block in named_blocks:
        for sub in ("1", "2"):
            unit = f"{block}_{sub}"
            conv_layer = layers[name_to_idx[unit]]
            bn_layer = layers[name_to_idx[f"{unit}_bn"]]
            act_layer = layers[name_to_idx[f"{unit}_act"]]
            if not isinstance(conv_layer, keras.layers.Conv2D):
                raise TypeError(f"Expected Conv2D for {unit}, got {type(conv_layer).__name__}")
            if not isinstance(bn_layer, keras.layers.BatchNormalization):
                raise TypeError(
                    f"Expected BatchNormalization for {unit}_bn, got {type(bn_layer).__name__}"
                )
            weighted[unit] = conv_layer
            weighted[f"{unit}_bn"] = bn_layer
            taps[unit] = act_layer.output

    for k in (1, 2, 3):
        g_conv = layers[name_to_idx[f"g{k}_conv"]]
        g_bn = layers[name_to_idx[f"g{k}_bn"]]
        xl = layers[name_to_idx[f"xl_{k}"]]
        g_up_idx = name_to_idx[f"g_up_{k}"]
        g_up = layers[g_up_idx]
        psi_idx = name_to_idx[f"psi_{k}"]
        psi = layers[psi_idx]
        q_conv = layers[name_to_idx[f"q_attn_conv_{k}"]]
        q_bn = layers[name_to_idx[f"q_attn_bn_{k}"]]

        phi_g = None
        for idx in range(g_up_idx - 1, -1, -1):
            if isinstance(layers[idx], keras.layers.Conv2D):
                phi_g = layers[idx]
                break
        if phi_g is None:
            raise RuntimeError(f"Could not locate phi_g_{k} before g_up_{k}.")

        act_xg = None
        for idx in range(psi_idx - 1, -1, -1):
            if isinstance(layers[idx], keras.layers.Activation):
                act_xg = layers[idx]
                break
        if act_xg is None:
            raise RuntimeError(f"Could not locate act_xg before psi_{k}.")

        if not isinstance(g_up, keras.layers.Conv2DTranspose):
            raise TypeError(f"Expected Conv2DTranspose for g_up_{k}, got {type(g_up).__name__}")

        weighted[f"g{k}_conv"] = g_conv
        weighted[f"g{k}_bn"] = g_bn
        weighted[f"xl_{k}"] = xl
        weighted[f"phi_g_{k}"] = phi_g
        weighted[f"g_up_{k}"] = g_up
        weighted[f"psi_{k}"] = psi
        weighted[f"q_attn_conv_{k}"] = q_conv
        weighted[f"q_attn_bn_{k}"] = q_bn
        taps[f"inter_{k}"] = act_xg.output

    for k in (1, 2, 3, 4):
        up_idx = name_to_idx[f"up{k}"]
        deconv = None
        for idx in range(up_idx - 1, -1, -1):
            if isinstance(layers[idx], keras.layers.Conv2DTranspose):
                deconv = layers[idx]
                break
        if deconv is None:
            raise RuntimeError(f"Could not locate up{k}_deconv before up{k}.")
        weighted[f"up{k}_deconv"] = deconv
        taps[f"up{k}_deconv"] = deconv.output

    weighted["final"] = layers[name_to_idx["final"]]
    return weighted, taps


def collect_hrank_scores(
    model,
    tf_ds,
    max_batches: int = 10,
) -> dict[str, np.ndarray]:
    """Estimate average feature-map matrix rank per channel across `max_batches`."""
    from tensorflow import keras

    _, taps = get_attention_unet_layer_map(model)
    probe = keras.Model(
        inputs=model.input,
        outputs=[taps[u] for u in PRUNABLE_UNITS],
        name="hrank_probe",
    )

    rank_sums: dict[str, np.ndarray] = {}
    total_samples = 0

    for batch_idx, batch in enumerate(tf_ds):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        batch_n = int(x.shape[0]) if x.shape[0] is not None else int(np.asarray(x).shape[0])
        feats = probe(x, training=False)
        for unit, feat in zip(PRUNABLE_UNITS, feats):
            arr = np.asarray(feat, dtype=np.float64)
            # Compute mean rank per sample in batch, weighted by batch size for exact global mean.
            nchw = np.transpose(arr, (0, 3, 1, 2))
            batch_sum = _compute_nchw_ranks(nchw, unit=unit).sum(axis=0, dtype=np.float64)
            if unit not in rank_sums:
                rank_sums[unit] = batch_sum
            else:
                rank_sums[unit] += batch_sum
        total_samples += batch_n

    if total_samples == 0:
        raise ValueError("Dataset yielded 0 batches; cannot compute HRank scores.")

    return {u: rank_sums[u] / float(total_samples) for u in PRUNABLE_UNITS}


def copy_pruned_attention_unet_weights(
    src_model,
    dst_model,
    orig_cfg: dict[str, int],
    keep_indices: dict[str, np.ndarray],
) -> None:
    """Copy and slice all 49 weighted layers from `src_model` into thinner `dst_model`."""
    src_layers, _ = get_attention_unet_layer_map(src_model)
    dst_layers, _ = get_attention_unet_layer_map(dst_model)

    in_channels = int(src_model.input_shape[-1])
    keep_input = np.arange(in_channels, dtype=np.int64)
    keep_single_out = np.array([0], dtype=np.int64)

    def _copy_conv_bn(unit: str, in_idx: np.ndarray) -> np.ndarray:
        out_idx = keep_indices[unit]
        dst_layers[unit].set_weights(
            slice_conv2d_weights(src_layers[unit].get_weights(), in_idx, out_idx)
        )
        dst_layers[f"{unit}_bn"].set_weights(
            slice_bn_weights(src_layers[f"{unit}_bn"].get_weights(), out_idx)
        )
        return out_idx

    prev = _copy_conv_bn("conv1_1", keep_input)
    prev = _copy_conv_bn("conv1_2", prev)
    prev = _copy_conv_bn("conv2_1", prev)
    prev = _copy_conv_bn("conv2_2", prev)
    prev = _copy_conv_bn("conv3_1", prev)
    prev = _copy_conv_bn("conv3_2", prev)
    prev = _copy_conv_bn("conv4_1", prev)
    prev = _copy_conv_bn("conv4_2", prev)
    prev = _copy_conv_bn("center_1", prev)
    keep_center = _copy_conv_bn("center_2", prev)

    def _copy_gate_and_attn(k: int, x_idx: np.ndarray, g_idx: np.ndarray) -> None:
        inter_idx = keep_indices[f"inter_{k}"]
        dst_layers[f"g{k}_conv"].set_weights(
            slice_conv2d_weights(src_layers[f"g{k}_conv"].get_weights(), g_idx, g_idx)
        )
        dst_layers[f"g{k}_bn"].set_weights(
            slice_bn_weights(src_layers[f"g{k}_bn"].get_weights(), g_idx)
        )
        dst_layers[f"xl_{k}"].set_weights(
            slice_conv2d_weights(src_layers[f"xl_{k}"].get_weights(), x_idx, inter_idx)
        )
        dst_layers[f"phi_g_{k}"].set_weights(
            slice_conv2d_weights(src_layers[f"phi_g_{k}"].get_weights(), g_idx, inter_idx)
        )
        dst_layers[f"g_up_{k}"].set_weights(
            slice_conv2d_transpose_weights(
                src_layers[f"g_up_{k}"].get_weights(), inter_idx, inter_idx
            )
        )
        dst_layers[f"psi_{k}"].set_weights(
            slice_conv2d_weights(src_layers[f"psi_{k}"].get_weights(), inter_idx, keep_single_out)
        )
        dst_layers[f"q_attn_conv_{k}"].set_weights(
            slice_conv2d_weights(src_layers[f"q_attn_conv_{k}"].get_weights(), x_idx, x_idx)
        )
        dst_layers[f"q_attn_bn_{k}"].set_weights(
            slice_bn_weights(src_layers[f"q_attn_bn_{k}"].get_weights(), x_idx)
        )

    # Decoder stage 1
    _copy_gate_and_attn(1, keep_indices["conv4_2"], keep_center)
    dst_layers["up1_deconv"].set_weights(
        slice_conv2d_transpose_weights(
            src_layers["up1_deconv"].get_weights(), keep_center, keep_indices["up1_deconv"]
        )
    )
    keep_up1 = concat_keep_indices(
        keep_indices["up1_deconv"], keep_indices["conv4_2"], orig_cfg["up1_deconv"]
    )

    # Decoder stage 2
    _copy_gate_and_attn(2, keep_indices["conv3_2"], keep_up1)
    dst_layers["up2_deconv"].set_weights(
        slice_conv2d_transpose_weights(
            src_layers["up2_deconv"].get_weights(), keep_up1, keep_indices["up2_deconv"]
        )
    )
    keep_up2 = concat_keep_indices(
        keep_indices["up2_deconv"], keep_indices["conv3_2"], orig_cfg["up2_deconv"]
    )

    # Decoder stage 3 (note: g3 conditions on up1, matching attention_unet.py)
    _copy_gate_and_attn(3, keep_indices["conv2_2"], keep_up1)
    dst_layers["up3_deconv"].set_weights(
        slice_conv2d_transpose_weights(
            src_layers["up3_deconv"].get_weights(), keep_up2, keep_indices["up3_deconv"]
        )
    )
    keep_up3 = concat_keep_indices(
        keep_indices["up3_deconv"], keep_indices["conv2_2"], orig_cfg["up3_deconv"]
    )

    # Decoder stage 4 + final 1x1 head
    dst_layers["up4_deconv"].set_weights(
        slice_conv2d_transpose_weights(
            src_layers["up4_deconv"].get_weights(), keep_up3, keep_indices["up4_deconv"]
        )
    )
    keep_up4 = concat_keep_indices(
        keep_indices["up4_deconv"], keep_indices["conv1_2"], orig_cfg["up4_deconv"]
    )
    dst_layers["final"].set_weights(
        slice_conv2d_weights(src_layers["final"].get_weights(), keep_up4, keep_single_out)
    )


def estimate_model_stats(model) -> dict[str, int]:
    """Return total parameter count and approximate Conv2D/Conv2DTranspose FLOPs (MACs*2)."""
    from tensorflow import keras

    total_flops = 0
    for layer in model.layers:
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.Conv2DTranspose)):
            out_shape = layer.output.shape
            k_shape = layer.kernel.shape
            h_out = int(out_shape[1])
            w_out = int(out_shape[2])
            kh = int(k_shape[0])
            kw = int(k_shape[1])
            if isinstance(layer, keras.layers.Conv2DTranspose):
                c_out = int(k_shape[2])
                c_in = int(k_shape[3])
            else:
                c_in = int(k_shape[2])
                c_out = int(k_shape[3])
            total_flops += 2 * h_out * w_out * kh * kw * c_in * c_out

    return {
        "params": int(model.count_params()),
        "flops": int(total_flops),
    }
