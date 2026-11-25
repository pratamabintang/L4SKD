"""Export SAM2-UNet teacher logits for Landslide4Sense splits to HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from uag.data.landslide4sense import Landslide4SenseDataset, get_normalizer
from uag.models.SAM2UNet import resolve_sam2_cfg
from uag.teacher.data import TorchL4SDataset
from uag.teacher.model import load_teacher, resolve_device
from uag.utils import ensure_dir, load_json, parse_bands


def export_logits(
    cfg: dict[str, Any],
    weights: str | Path,
    splits: list[str],
    out_dir: str | Path,
    batch_size: int = 16,
    device: str = "auto",
) -> dict[str, Path]:
    """Run teacher inference (no augmentation) and save main-head logits per split."""
    resolved_device = resolve_device(device)
    weights_path = Path(weights)
    out_path_dir = ensure_dir(out_dir)

    bands = cfg.get("bands")
    if isinstance(bands, str):
        bands = parse_bands(bands)
    normalizer = get_normalizer(
        mode=cfg.get("norm", "l4s"),
        stats_file=cfg.get("stats_file"),
        bands=bands,
    )

    model = load_teacher(cfg, weights_path, device=resolved_device)
    model.eval()

    sam2_cfg = resolve_sam2_cfg(str(cfg.get("sam2_cfg", "l")))
    input_size = cfg.get("input_size", 352)
    input_size_attr = int(input_size) if input_size is not None else -1

    exported: dict[str, Path] = {}
    for split in splits:
        try:
            ds_raw = Landslide4SenseDataset(
                root=cfg["data_root"],
                split=split,
                bands=bands,
                normalizer=normalizer,
                id_list=None,
                exclude_list=None,
                require_masks=False,
            )
        except FileNotFoundError as exc:
            print(f"Warning: skipping split {split!r}: {exc}")
            continue

        torch_ds = TorchL4SDataset(ds_raw, augment=False)
        loader = DataLoader(
            torch_ds,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=0,
        )

        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for imgs, _, _ in loader:
                imgs = imgs.to(resolved_device)
                out, _, _ = model(imgs)
                logits_f32 = out.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)
                if not np.isfinite(logits_f32).all():
                    n_bad = int((~np.isfinite(logits_f32)).sum())
                    raise ValueError(
                        f"Teacher produced {n_bad} non-finite logit values on split={split!r}."
                    )
                logits_nhwc = np.clip(logits_f32, -30.0, 30.0).astype(np.float16)
                chunks.append(logits_nhwc)

        all_logits = np.concatenate(chunks, axis=0)
        h5_path = out_path_dir / f"teacher_logits_{split}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("logits", data=all_logits, dtype="float16", compression="gzip")
            str_dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset("ids", data=np.asarray(ds_raw.ids, dtype=object), dtype=str_dt)
            f.attrs["bands"] = json.dumps(bands)
            f.attrs["norm"] = str(cfg.get("norm", "l4s"))
            f.attrs["stats_file"] = str(cfg.get("stats_file") or "")
            f.attrs["sam2_cfg"] = sam2_cfg
            f.attrs["input_size"] = input_size_attr
            f.attrs["weights"] = str(weights_path)

        print(f"Exported {split} logits: {all_logits.shape} -> {h5_path}")
        exported[split] = h5_path

    return exported


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export SAM2-UNet teacher logits for Landslide4Sense splits."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Teacher training run directory containing config.json and best.pt.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional explicit path to teacher .pt weights (default: <run-dir>/best.pt).",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated dataset splits to export (default: train,val,test).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run inference on ('auto', 'cuda', or 'cpu').",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for teacher_logits_<split>.h5 (default: <run-dir>).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_json(run_dir / "config.json")
    weights = Path(args.weights) if args.weights else run_dir / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Teacher weights not found: {weights}")
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    export_logits(
        cfg=cfg,
        weights=weights,
        splits=splits,
        out_dir=out_dir,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
