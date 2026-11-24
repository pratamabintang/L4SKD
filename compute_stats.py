"""Compute per-band mean and standard deviation JSON for Landslide4Sense."""

from __future__ import annotations

import argparse

from uag.data.landslide4sense import Landslide4SenseDataset, compute_band_stats
from uag.utils import save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute per-band mean/std statistics on raw Landslide4Sense images."
    )
    parser.add_argument(
        "--data-root",
        default="datasets/landslide4senses",
        help="Path to the Landslide4Sense dataset root directory.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to compute statistics over (default: train).",
    )
    parser.add_argument(
        "--blacklist",
        default=None,
        help="Optional txt file of sample IDs (x in image_x.h5) to exclude.",
    )
    parser.add_argument(
        "--out",
        default="stats.json",
        help="Output JSON path (default: stats.json).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ds = Landslide4SenseDataset(
        root=args.data_root,
        split=args.split,
        bands=None,
        normalizer=None,
        exclude_list=args.blacklist,
        require_masks=False,
    )
    mean, std = compute_band_stats(ds)
    payload = {
        "split": args.split,
        "num_samples": len(ds),
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
    }
    save_json(payload, args.out)
    print(f"Saved band statistics for {len(ds)} samples to {args.out}")


if __name__ == "__main__":
    main()
