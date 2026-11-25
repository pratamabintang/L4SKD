"""Region-level sufficiency suite (Family 1), overlap metrics (Family 2), and RQ5 comparators."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

DEFAULT_IOU_TAU = 0.5
DEFAULT_DILATION_ALPHA = 0.1
DEFAULT_BOUNDARY_RATIO = 0.02
DEFAULT_SMEASURE_ALPHA = 0.5
SIZE_BIN_SMALL_MAX = 100
SIZE_BIN_MEDIUM_MAX = 500


def dilate_disk(mask_bool: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a 2D boolean mask by a Euclidean disk of integer radius `r`."""
    mask_b = np.asarray(mask_bool, dtype=bool)
    if mask_b.ndim != 2:
        raise ValueError(f"Expected 2D boolean mask, got shape {mask_b.shape}.")
    if radius <= 0:
        return mask_b.copy()
    if not np.any(mask_b):
        return np.zeros_like(mask_b, dtype=bool)
    return distance_transform_edt(~mask_b) <= float(radius)


def mask_to_boundary(
    mask_bool: np.ndarray,
    dilation_ratio: float = DEFAULT_BOUNDARY_RATIO,
) -> np.ndarray:
    """Extract interior boundary band within `round(dilation_ratio * diag)` pixels (Cheng et al.)."""
    mask_b = np.asarray(mask_bool, dtype=bool)
    if mask_b.ndim != 2:
        raise ValueError(f"Expected 2D boolean mask, got shape {mask_b.shape}.")
    if not np.any(mask_b):
        return np.zeros_like(mask_b, dtype=bool)
    h, w = mask_b.shape
    d = max(1, int(round(float(dilation_ratio) * math.hypot(h, w))))
    padded = np.pad(mask_b, pad_width=1, mode="constant", constant_values=False)
    dist = distance_transform_edt(padded)[1:-1, 1:-1]
    return mask_b & (dist <= float(d))


def _squeeze_2d_bool(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask (or HxWx1 / 1xHxW), got shape {mask.shape}.")
    return arr


def _squeeze_2d_float(arr_in: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr_in, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (or HxWx1 / 1xHxW), got shape {arr_in.shape}.")
    return arr


def compute_boundary_iou(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    dilation_ratio: float = DEFAULT_BOUNDARY_RATIO,
) -> float | None:
    """Compute Boundary IoU (Cheng et al., 2021); returns None when |G| == 0."""
    pred = _squeeze_2d_bool(pred_bin)
    gt = _squeeze_2d_bool(gt_bin)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}.")
    if not np.any(gt):
        return None
    if not np.any(pred):
        return 0.0
    g_b = mask_to_boundary(gt, dilation_ratio=dilation_ratio)
    p_b = mask_to_boundary(pred, dilation_ratio=dilation_ratio)
    union = int(np.logical_or(g_b, p_b).sum())
    if union == 0:
        return 0.0
    inter = int(np.logical_and(g_b, p_b).sum())
    return float(inter) / float(union)


def _s_object_component(pred: np.ndarray, gt_mask: np.ndarray) -> float:
    vals = pred[gt_mask]
    x = float(vals.mean())
    sigma_x = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    return (2.0 * x) / (x * x + 1.0 + sigma_x + 1e-12)


def _quadrant_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    n = pred.size
    if n == 0:
        return 1.0
    mu_x = float(pred.mean())
    mu_y = float(gt.mean())
    dx = pred - mu_x
    dy = gt - mu_y
    if n > 1:
        denom_n = float(n - 1)
        sigma_x2 = float(np.sum(dx * dx)) / denom_n
        sigma_y2 = float(np.sum(dy * dy)) / denom_n
        sigma_xy = float(np.sum(dx * dy)) / denom_n
    else:
        sigma_x2 = 0.0
        sigma_y2 = 0.0
        sigma_xy = 0.0

    alpha = 4.0 * mu_x * mu_y * sigma_xy
    beta = (mu_x * mu_x + mu_y * mu_y) * (sigma_x2 + sigma_y2)
    if alpha != 0.0:
        return alpha / (beta + 1e-12)
    if beta == 0.0:
        return 1.0
    return 0.0


def compute_s_measure(
    pred_prob: np.ndarray,
    gt_bin: np.ndarray,
    alpha: float = DEFAULT_SMEASURE_ALPHA,
) -> float:
    """Compute Structure-measure S_alpha (Fan et al., ICCV 2017) on probabilities."""
    pred = np.clip(_squeeze_2d_float(pred_prob), 0.0, 1.0)
    gt_b = _squeeze_2d_bool(gt_bin)
    if pred.shape != gt_b.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt_b.shape}.")
    gt = gt_b.astype(np.float64)

    y = float(gt.mean())
    if y == 0.0:
        return float(np.clip(1.0 - pred.mean(), 0.0, 1.0))
    if y == 1.0:
        return float(np.clip(pred.mean(), 0.0, 1.0))

    o_fg = _s_object_component(pred, gt_b)
    o_bg = _s_object_component(1.0 - pred, ~gt_b)
    s_o = y * o_fg + (1.0 - y) * o_bg

    h, w = gt_b.shape
    ys, xs = np.nonzero(gt_b)
    cy = int(np.clip(np.rint(float(ys.mean())), 1, h - 1)) if h > 1 else 0
    cx = int(np.clip(np.rint(float(xs.mean())), 1, w - 1)) if w > 1 else 0

    total = float(h * w)
    quadrants = (
        (pred[:cy, :cx], gt[:cy, :cx]),
        (pred[:cy, cx:], gt[:cy, cx:]),
        (pred[cy:, :cx], gt[cy:, :cx]),
        (pred[cy:, cx:], gt[cy:, cx:]),
    )
    s_r = sum((p_q.size / total) * _quadrant_ssim(p_q, g_q) for p_q, g_q in quadrants)

    score = float(alpha) * s_o + (1.0 - float(alpha)) * s_r
    return float(np.clip(score, 0.0, 1.0))


def classify_size_bin(gt_area: int) -> str:
    """Classify ground-truth patch area |G| into empty/small/medium/large."""
    if gt_area <= 0:
        return "empty"
    if gt_area < SIZE_BIN_SMALL_MAX:
        return "small"
    if gt_area <= SIZE_BIN_MEDIUM_MAX:
        return "medium"
    return "large"


def compute_sample_sufficiency(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    tau: float = DEFAULT_IOU_TAU,
    alpha: float = DEFAULT_DILATION_ALPHA,
) -> dict[str, Any]:
    """Compute Family 1 sufficiency components on a single (pred, gt) patch pair."""
    tau_f = float(tau)
    alpha_f = float(alpha)
    if not (0.0 <= tau_f <= 1.0):
        raise ValueError(f"tau must be in [0.0, 1.0], got {tau}")
    if alpha_f < 0.0:
        raise ValueError(f"alpha must be >= 0.0, got {alpha}")

    pred = _squeeze_2d_bool(pred_bin)
    gt = _squeeze_2d_bool(gt_bin)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}.")

    gt_area = int(gt.sum())
    pred_area = int(pred.sum())

    if gt_area == 0:
        return {
            "is_positive": False,
            "gt_area": 0,
            "pred_area": pred_area,
            "empty_patch_fp": bool(pred_area > 0),
        }

    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    sample_iou = float(inter) / float(union)
    tolerant_iou = bool(sample_iou >= tau_f)

    h, w = gt.shape
    if pred_area == 0:
        centroid_hit = False
    else:
        ys, xs = np.nonzero(pred)
        cy = int(np.clip(np.rint(float(ys.mean())), 0, h - 1))
        cx = int(np.clip(np.rint(float(xs.mean())), 0, w - 1))
        centroid_hit = bool(gt[cy, cx])

    radius = int(math.ceil(alpha_f * 2.0 * math.sqrt(float(gt_area) / math.pi)))
    if pred_area == 0:
        dilated_recall = 0.0
    else:
        dilated_pred = dilate_disk(pred, radius)
        dilated_recall = float(np.logical_and(gt, dilated_pred).sum()) / float(gt_area)

    return {
        "is_positive": True,
        "gt_area": gt_area,
        "pred_area": pred_area,
        "sample_iou": sample_iou,
        "tolerant_iou": tolerant_iou,
        "centroid_hit": centroid_hit,
        "dilation_radius": radius,
        "dilated_recall": dilated_recall,
    }


def _to_bhw(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 4 and a.shape[-1] == 1:
        return a[..., 0]
    if a.ndim == 4 and a.shape[1] == 1:
        return a[:, 0, :, :]
    if a.ndim == 3:
        return a
    raise ValueError(f"Expected (B, H, W), (B, H, W, 1), or (B, 1, H, W), got {a.shape}.")


class EvaluationSuiteAccumulator:
    """Accumulate Family 1, Family 2, RQ5 boundary comparators, and per-patch records."""

    def __init__(
        self,
        threshold: float = 0.5,
        tau: float = DEFAULT_IOU_TAU,
        alpha: float = DEFAULT_DILATION_ALPHA,
        smooth: float = 1e-7,
        boundary_ratio: float = DEFAULT_BOUNDARY_RATIO,
        smeasure_alpha: float = DEFAULT_SMEASURE_ALPHA,
    ) -> None:
        self.threshold = float(threshold)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.smooth = float(smooth)
        self.boundary_ratio = float(boundary_ratio)
        self.smeasure_alpha = float(smeasure_alpha)
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.num_positive = 0
        self.tolerant_iou_hits = 0
        self.centroid_hits = 0
        self.dilated_recall_sum = 0.0
        self.num_negative = 0
        self.empty_patch_fp_count = 0
        self.boundary_iou_sum = 0.0
        self.s_measure_all_sum = 0.0
        self.s_measure_pos_sum = 0.0
        self.per_patch_records: list[dict[str, Any]] = []

    def update_batch(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> None:
        prob = _to_bhw(y_prob)
        gt = _to_bhw(y_true)
        if prob.shape != gt.shape:
            raise ValueError(f"Batch shape mismatch: prob {prob.shape} vs gt {gt.shape}.")
        batch_size = prob.shape[0]
        if sample_ids is not None and len(sample_ids) != batch_size:
            raise ValueError(
                f"sample_ids length ({len(sample_ids)}) != batch size ({batch_size})."
            )

        pred_b = prob >= self.threshold
        gt_b = gt >= 0.5

        self.tp += float(np.logical_and(pred_b, gt_b).sum())
        self.fp += float(np.logical_and(pred_b, ~gt_b).sum())
        self.fn += float(np.logical_and(~pred_b, gt_b).sum())

        for i in range(batch_size):
            sample_id = (
                str(sample_ids[i])
                if sample_ids is not None
                else f"sample_{len(self.per_patch_records)}"
            )
            res = compute_sample_sufficiency(
                pred_b[i],
                gt_b[i],
                tau=self.tau,
                alpha=self.alpha,
            )
            b_iou = compute_boundary_iou(
                pred_b[i],
                gt_b[i],
                dilation_ratio=self.boundary_ratio,
            )
            s_val = compute_s_measure(
                prob[i],
                gt_b[i],
                alpha=self.smeasure_alpha,
            )
            self.s_measure_all_sum += s_val

            gt_area = int(res["gt_area"])
            size_bin = classify_size_bin(gt_area)

            if res["is_positive"]:
                self.num_positive += 1
                self.tolerant_iou_hits += int(res["tolerant_iou"])
                self.centroid_hits += int(res["centroid_hit"])
                self.dilated_recall_sum += float(res["dilated_recall"])
                self.boundary_iou_sum += float(b_iou if b_iou is not None else 0.0)
                self.s_measure_pos_sum += s_val
                row = {
                    "sample_id": sample_id,
                    "is_positive": True,
                    "size_bin": size_bin,
                    "gt_area": gt_area,
                    "pred_area": int(res["pred_area"]),
                    "sample_iou": float(res["sample_iou"]),
                    "tolerant_iou": bool(res["tolerant_iou"]),
                    "centroid_hit": bool(res["centroid_hit"]),
                    "dilation_radius": int(res["dilation_radius"]),
                    "dilated_recall": float(res["dilated_recall"]),
                    "empty_patch_fp": False,
                    "boundary_iou": b_iou,
                    "s_measure": s_val,
                }
            else:
                self.num_negative += 1
                self.empty_patch_fp_count += int(res["empty_patch_fp"])
                row = {
                    "sample_id": sample_id,
                    "is_positive": False,
                    "size_bin": size_bin,
                    "gt_area": 0,
                    "pred_area": int(res["pred_area"]),
                    "sample_iou": None,
                    "tolerant_iou": None,
                    "centroid_hit": None,
                    "dilation_radius": None,
                    "dilated_recall": None,
                    "empty_patch_fp": bool(res["empty_patch_fp"]),
                    "boundary_iou": None,
                    "s_measure": s_val,
                }
            self.per_patch_records.append(row)

    def _summarize_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        if n == 0:
            return {
                "num_patches": 0,
                "tolerant_iou_rate": None,
                "centroid_hit_rate": None,
                "size_scaled_dilated_recall": None,
                "mean_sample_iou": None,
                "boundary_iou": None,
                "s_measure": None,
            }
        nf = float(n)
        return {
            "num_patches": n,
            "tolerant_iou_rate": sum(int(bool(r["tolerant_iou"])) for r in rows) / nf,
            "centroid_hit_rate": sum(int(bool(r["centroid_hit"])) for r in rows) / nf,
            "size_scaled_dilated_recall": sum(float(r["dilated_recall"]) for r in rows) / nf,
            "mean_sample_iou": sum(float(r["sample_iou"]) for r in rows) / nf,
            "boundary_iou": sum(float(r["boundary_iou"]) for r in rows) / nf,
            "s_measure": sum(float(r["s_measure"]) for r in rows) / nf,
        }

    def save_per_patch_csv(self, path: str | Path) -> Path:
        """Write per-patch evaluation records to CSV and return the Path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "sample_id",
            "is_positive",
            "size_bin",
            "gt_area",
            "pred_area",
            "sample_iou",
            "tolerant_iou",
            "centroid_hit",
            "dilation_radius",
            "dilated_recall",
            "empty_patch_fp",
            "boundary_iou",
            "s_measure",
        ]
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.per_patch_records:
                writer.writerow({k: "" if row[k] is None else row[k] for k in fieldnames})
        return out

    def compute(self) -> dict[str, Any]:
        s = self.smooth
        precision = (self.tp + s) / (self.tp + self.fp + s)
        recall = (self.tp + s) / (self.tp + self.fn + s)
        dice_f1 = (2.0 * self.tp + s) / (2.0 * self.tp + self.fp + self.fn + s)
        iou = (self.tp + s) / (self.tp + self.fp + self.fn + s)

        if self.num_positive > 0:
            pos_n = float(self.num_positive)
            tol_iou_rate: float | None = float(self.tolerant_iou_hits) / pos_n
            centroid_rate: float | None = float(self.centroid_hits) / pos_n
            dilated_rec: float | None = float(self.dilated_recall_sum) / pos_n
            b_iou_mean: float | None = float(self.boundary_iou_sum) / pos_n
            s_pos_mean: float | None = float(self.s_measure_pos_sum) / pos_n
        else:
            tol_iou_rate = None
            centroid_rate = None
            dilated_rec = None
            b_iou_mean = None
            s_pos_mean = None

        total_n = self.num_positive + self.num_negative
        s_all_mean: float | None = (
            float(self.s_measure_all_sum) / float(total_n) if total_n > 0 else None
        )

        empty_fp_rate: float | None = (
            float(self.empty_patch_fp_count) / float(self.num_negative)
            if self.num_negative > 0
            else None
        )

        pos_rows = [r for r in self.per_patch_records if r["is_positive"]]
        size_stratified = {
            "small": self._summarize_rows([r for r in pos_rows if r["size_bin"] == "small"]),
            "medium": self._summarize_rows([r for r in pos_rows if r["size_bin"] == "medium"]),
            "large": self._summarize_rows([r for r in pos_rows if r["size_bin"] == "large"]),
            "medium_large": self._summarize_rows(
                [r for r in pos_rows if r["size_bin"] in {"medium", "large"}]
            ),
        }

        return {
            "region_sufficiency": {
                "tau": self.tau,
                "alpha": self.alpha,
                "num_positive_patches": self.num_positive,
                "tolerant_iou_rate": tol_iou_rate,
                "centroid_hit_rate": centroid_rate,
                "size_scaled_dilated_recall": dilated_rec,
                "num_negative_patches": self.num_negative,
                "empty_patch_fp_count": self.empty_patch_fp_count,
                "empty_patch_fp_rate": empty_fp_rate,
            },
            "standard_overlap": {
                "iou": float(iou),
                "dice_f1": float(dice_f1),
                "precision": float(precision),
                "recall": float(recall),
            },
            "boundary_comparators": {
                "boundary_dilation_ratio": self.boundary_ratio,
                "smeasure_alpha": self.smeasure_alpha,
                "boundary_iou_positive_mean": b_iou_mean,
                "s_measure_all_mean": s_all_mean,
                "s_measure_positive_mean": s_pos_mean,
            },
            "size_stratified": size_stratified,
        }


def _fmt_opt(val: float | None) -> str:
    return "N/A" if val is None else f"{val:.6f}"


def format_evaluation_suite(suite: dict[str, Any]) -> str:
    """Format Family 1, Family 2, RQ5 boundary comparators, and size stratification."""
    rs = suite["region_sufficiency"]
    so = suite["standard_overlap"]
    lines = [
        f"Family 1 - Region Sufficiency Suite (tau={rs['tau']}, alpha={rs['alpha']}, n_pos={rs['num_positive_patches']}):",
        f"  tolerant_iou_rate:          {_fmt_opt(rs['tolerant_iou_rate'])}",
        f"  centroid_hit_rate:          {_fmt_opt(rs['centroid_hit_rate'])}",
        f"  size_scaled_dilated_recall: {_fmt_opt(rs['size_scaled_dilated_recall'])}",
        f"  empty_patch_fp:             {rs['empty_patch_fp_count']}/{rs['num_negative_patches']} ({_fmt_opt(rs['empty_patch_fp_rate'])})",
        "Family 2 - Standard Overlap (comparability only):",
        f"  iou:       {so['iou']:.6f}",
        f"  dice_f1:   {so['dice_f1']:.6f}",
        f"  precision: {so['precision']:.6f}",
        f"  recall:    {so['recall']:.6f}",
    ]
    if "boundary_comparators" in suite:
        bc = suite["boundary_comparators"]
        lines.extend(
            [
                f"RQ5 Comparators (boundary_ratio={bc['boundary_dilation_ratio']}, s_alpha={bc['smeasure_alpha']}):",
                f"  boundary_iou_positive_mean: {_fmt_opt(bc['boundary_iou_positive_mean'])}",
                f"  s_measure_all_mean:         {_fmt_opt(bc['s_measure_all_mean'])}",
                f"  s_measure_positive_mean:    {_fmt_opt(bc['s_measure_positive_mean'])}",
            ]
        )
    if "size_stratified" in suite:
        lines.append("Size-Stratified Region Sufficiency (positive patches):")
        for key in ("small", "medium", "large", "medium_large"):
            s = suite["size_stratified"][key]
            lines.append(
                f"  {key:12s} (n={s['num_patches']:3d}): "
                f"tol_iou={_fmt_opt(s['tolerant_iou_rate'])}, "
                f"cent={_fmt_opt(s['centroid_hit_rate'])}, "
                f"dil_rec={_fmt_opt(s['size_scaled_dilated_recall'])}, "
                f"b_iou={_fmt_opt(s['boundary_iou'])}"
            )
    return "\n".join(lines)
