"""Run inference and generate visual prediction panels / HDF5 masks."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from test import resolve_eval_config
from uag.config import add_common_args
from uag.data.landslide4sense import (
    Landslide4SenseDataset,
    get_normalizer,
    read_image,
)
from uag.models import build_model
from uag.utils import ensure_dir, parse_bands

_STEM_NUM_RE = re.compile(r"(\d+)$")


def _stretch_percentile(arr: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    lo = np.percentile(arr, low)
    hi = np.percentile(arr, high)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _make_display_image(img_path: str, selected_bands: Optional[list[int]]) -> tuple[np.ndarray, str]:
    """Build an RGB composite (B4, B3, B2 = indices 3, 2, 1) if selected, else band 0."""
    raw = read_image(img_path, bands=None)
    has_rgb = selected_bands is None or all(b in selected_bands for b in (1, 2, 3))
    if has_rgb:
        rgb = np.stack(
            [
                _stretch_percentile(raw[..., 3]),
                _stretch_percentile(raw[..., 2]),
                _stretch_percentile(raw[..., 1]),
            ],
            axis=-1,
        )
        return rgb, "RGB (B4, B3, B2)"

    first_band = selected_bands[0] if selected_bands else 0
    gray = _stretch_percentile(raw[..., first_band])
    return gray, f"Band {first_band}"


def _save_prediction_figure(
    display_img: np.ndarray,
    display_title: str,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    if display_img.ndim == 2:
        axes[0].imshow(display_img, cmap="gray")
    else:
        axes[0].imshow(display_img)
    axes[0].set_title(display_title)
    axes[0].axis("off")

    axes[1].imshow(np.squeeze(pred_mask), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    if gt_mask is not None:
        axes[2].imshow(np.squeeze(gt_mask), cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Ground Truth")
    else:
        axes[2].text(0.5, 0.5, "No Ground Truth", ha="center", va="center")
        axes[2].set_title("Ground Truth (N/A)")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _mask_filename(sample_id: str) -> str:
    match = _STEM_NUM_RE.search(sample_id)
    if match:
        return f"mask_{match.group(1)}.h5"
    return f"mask_{sample_id}.h5"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict segmentation masks on Landslide4Sense.")
    add_common_args(parser)
    parser.add_argument("--run-dir", default=None, help="Run directory containing config.json and best.weights.h5.")
    parser.add_argument("--weights", default=None, help="Explicit path to .weights.h5 checkpoint.")
    parser.add_argument("--filters", type=int, default=None, help="Override base conv filter count.")
    parser.add_argument("--split", default="test", help="Dataset split to predict on (default: test).")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to visualize/save (-1 for all).",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory for figures and optional masks.")
    parser.add_argument("--save-masks", action="store_true", help="Save predicted uint8 masks as mask_<k>.h5.")
    parser.set_defaults(
        data_root_override=None,
        norm_override=None,
        loss_override=None,
        threshold_override=None,
        model_override=None,
        blacklist_override=None,
    )
    return parser


def main() -> None:
    import sys

    parser = build_parser()
    args = parser.parse_args()
    argv = set(sys.argv[1:])
    if "--data-root" in argv or any(a.startswith("--data-root=") for a in argv):
        args.data_root_override = args.data_root
    if "--norm" in argv or any(a.startswith("--norm=") for a in argv):
        args.norm_override = args.norm
    if "--loss" in argv or any(a.startswith("--loss=") for a in argv):
        args.loss_override = args.loss
    if "--threshold" in argv or any(a.startswith("--threshold=") for a in argv):
        args.threshold_override = args.threshold
    if "--model" in argv or any(a.startswith("--model=") for a in argv):
        args.model_override = args.model
    if "--blacklist" in argv or any(a.startswith("--blacklist=") for a in argv):
        args.blacklist_override = args.blacklist

    cfg, weights_path, default_dir = resolve_eval_config(args)
    out_dir = ensure_dir(args.out_dir or (default_dir / f"predictions_{args.split}"))
    panels_dir = ensure_dir(out_dir / "panels")
    masks_dir = ensure_dir(out_dir / "masks") if args.save_masks else None

    bands = cfg.get("bands")
    if isinstance(bands, str):
        bands = parse_bands(bands)

    normalizer = get_normalizer(
        mode=cfg.get("norm", "l4s"),
        stats_file=cfg.get("stats_file"),
        bands=bands,
    )
    ds = Landslide4SenseDataset(
        root=cfg["data_root"],
        split=args.split,
        bands=bands,
        normalizer=normalizer,
        exclude_list=cfg.get("blacklist"),
        require_masks=False,
    )

    total = len(ds) if args.num_samples < 0 else min(args.num_samples, len(ds))
    input_shape = tuple(cfg["input_shape"]) if "input_shape" in cfg else ds.image_shape
    filters = int(cfg.get("filters", 4))
    threshold = float(cfg.get("threshold", 0.5))
    batch_size = int(args.batch_size)

    model = build_model(
        cfg.get("model", "attention_unet"),
        input_shape=input_shape,
        filters=filters,
        pretrained_weights=str(weights_path),
        bands=bands,
        channel_cfg=cfg.get("channel_cfg"),
    )

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_imgs = []
        batch_masks = []
        for idx in range(start, end):
            img, mask = ds[idx]
            batch_imgs.append(img)
            batch_masks.append(mask)

        x_batch = np.stack(batch_imgs, axis=0)
        preds = model.predict(x_batch, verbose=0)

        for offset, idx in enumerate(range(start, end)):
            sample_id = ds.ids[idx]
            img_path = ds.samples[idx][0]
            pred_bin = (preds[offset] > threshold).astype(np.uint8)

            display_img, display_title = _make_display_image(img_path, bands)
            _save_prediction_figure(
                display_img=display_img,
                display_title=display_title,
                pred_mask=pred_bin,
                gt_mask=batch_masks[offset],
                out_path=panels_dir / f"{sample_id}.png",
            )

            if masks_dir is not None:
                mask_file = masks_dir / _mask_filename(sample_id)
                with h5py.File(mask_file, "w") as hf:
                    hf.create_dataset("mask", data=np.squeeze(pred_bin).astype(np.uint8))

    print(f"Saved {total} prediction panels to {panels_dir}")
    if masks_dir is not None:
        print(f"Saved predicted HDF5 masks to {masks_dir}")


if __name__ == "__main__":
    main()
