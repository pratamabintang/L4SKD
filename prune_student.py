"""HRankPlus-style structured filter pruning for Attention U-Net."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from uag.data.landslide4sense import (
    Landslide4SenseDataset,
    get_normalizer,
    make_tf_dataset,
)
from uag.losses import get_loss
from uag.metrics import get_metrics
from uag.models import build_model
from uag.pruning.hrank import (
    PRUNABLE_UNITS,
    build_pruned_channel_cfg,
    collect_hrank_scores,
    copy_pruned_attention_unet_weights,
    default_attention_unet_channel_cfg,
    estimate_model_stats,
    parse_compress_rates,
)
from uag.utils import ensure_dir, load_json, parse_bands, save_json


def prune_student(args: argparse.Namespace) -> dict[str, Any]:
    """Run HRank scoring, structural pruning, and weight copying for an Attention U-Net run."""
    from tensorflow import keras

    keras.backend.clear_session()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Run config not found: {cfg_path}")
    cfg = load_json(cfg_path)

    model_name = cfg.get("model", "attention_unet")
    if model_name != "attention_unet":
        raise ValueError(
            f"prune_student.py only supports model='attention_unet', got {model_name!r}."
        )

    weights_path = Path(args.weights) if args.weights else run_dir / "best.weights.h5"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    bands = cfg.get("bands")
    if isinstance(bands, str):
        bands = parse_bands(bands)

    normalizer = get_normalizer(
        mode=cfg.get("norm", "l4s"),
        stats_file=cfg.get("stats_file"),
        bands=bands,
    )

    ds_raw = Landslide4SenseDataset(
        root=cfg["data_root"],
        split=args.split,
        bands=bands,
        normalizer=normalizer,
        id_list=cfg.get("train_list") if args.split == "train" else None,
        exclude_list=cfg.get("blacklist"),
        require_masks=True,
    )
    tf_ds = make_tf_dataset(
        ds_raw,
        batch_size=int(args.batch_size),
        shuffle=False,
        augment=False,
        cache=False,
    )

    input_shape = tuple(cfg["input_shape"]) if "input_shape" in cfg else ds_raw.image_shape
    filters = int(cfg.get("filters", 4))
    orig_cfg = {
        **default_attention_unet_channel_cfg(filters),
        **(cfg.get("channel_cfg") or {}),
    }

    src_model = build_model(
        "attention_unet",
        input_shape=input_shape,
        filters=filters,
        pretrained_weights=str(weights_path),
        channel_cfg=orig_cfg,
    )

    print(f"Collecting HRank scores on split={args.split!r} (up to {args.hrank_batches} batches)...")
    ranks_by_unit = collect_hrank_scores(src_model, tf_ds, max_batches=int(args.hrank_batches))

    rates_by_unit = parse_compress_rates(args.compress_rate)
    if args.center_rate is not None:
        c_rate = float(args.center_rate)
        if not (0.0 <= c_rate < 1.0):
            raise ValueError(f"--center-rate must be in [0.0, 1.0), got {c_rate}")
        rates_by_unit["center_1"] = c_rate
        rates_by_unit["center_2"] = c_rate

    pruned_cfg, keep_indices = build_pruned_channel_cfg(
        orig_cfg=orig_cfg,
        ranks_by_unit=ranks_by_unit,
        rates_by_unit=rates_by_unit,
        min_channels=int(args.min_channels),
    )

    dst_model = build_model(
        "attention_unet",
        input_shape=input_shape,
        filters=filters,
        channel_cfg=pruned_cfg,
    )
    copy_pruned_attention_unet_weights(src_model, dst_model, orig_cfg, keep_indices)

    out_dir = ensure_dir(args.output_dir or (run_dir.parent / f"pruned_{run_dir.name}"))
    pruned_weights_path = out_dir / "pruned_init.weights.h5"
    channel_cfg_path = out_dir / "channel_cfg.json"
    hrank_npz_path = out_dir / "hrank_scores.npz"
    report_path = out_dir / "pruning_report.json"

    dst_model.save_weights(str(pruned_weights_path))
    save_json(pruned_cfg, channel_cfg_path)

    npz_payload: dict[str, np.ndarray] = {}
    for unit in PRUNABLE_UNITS:
        npz_payload[f"rank_{unit}"] = ranks_by_unit[unit]
        npz_payload[f"keep_{unit}"] = keep_indices[unit]
    np.savez(hrank_npz_path, **npz_payload)

    src_stats = estimate_model_stats(src_model)
    dst_stats = estimate_model_stats(dst_model)
    param_reduction = 100.0 * (1.0 - dst_stats["params"] / max(src_stats["params"], 1))
    flops_reduction = 100.0 * (1.0 - dst_stats["flops"] / max(src_stats["flops"], 1))

    val_eval: dict[str, dict[str, float]] | None = None
    if args.eval_val:
        val_raw = Landslide4SenseDataset(
            root=cfg["data_root"],
            split="val",
            bands=bands,
            normalizer=normalizer,
            exclude_list=cfg.get("blacklist"),
            require_masks=True,
        )
        val_tf = make_tf_dataset(
            val_raw,
            batch_size=int(args.batch_size),
            shuffle=False,
            augment=False,
            cache=False,
        )
        loss_name = cfg.get("loss", "dice")
        threshold = float(cfg.get("threshold", 0.5))
        lr = float(cfg.get("lr", 1e-3))

        src_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=get_loss(loss_name),
            metrics=get_metrics(threshold=threshold),
        )
        dst_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=get_loss(loss_name),
            metrics=get_metrics(threshold=threshold),
        )
        val_eval = {
            "source": {
                k: float(v) for k, v in src_model.evaluate(val_tf, return_dict=True, verbose=0).items()
            },
            "pruned_pre_finetune": {
                k: float(v) for k, v in dst_model.evaluate(val_tf, return_dict=True, verbose=0).items()
            },
        }

    report: dict[str, Any] = {
        "source_run_dir": str(run_dir),
        "source_weights": str(weights_path),
        "pruned_weights": str(pruned_weights_path),
        "channel_cfg_path": str(channel_cfg_path),
        "hrank_scores_path": str(hrank_npz_path),
        "split": args.split,
        "hrank_batches": int(args.hrank_batches),
        "min_channels": int(args.min_channels),
        "rates_by_unit": rates_by_unit,
        "orig_channel_cfg": orig_cfg,
        "pruned_channel_cfg": pruned_cfg,
        "source_stats": src_stats,
        "pruned_stats": dst_stats,
        "param_reduction_pct": round(param_reduction, 2),
        "flops_reduction_pct": round(flops_reduction, 2),
        "val_eval": val_eval,
    }
    save_json(report, report_path)

    print("\n=== HRank Pruning Summary ===")
    print(f"Source params: {src_stats['params']:,} | Pruned params: {dst_stats['params']:,} (-{param_reduction:.2f}%)")
    print(f"Source FLOPs:  {src_stats['flops']:,} | Pruned FLOPs:  {dst_stats['flops']:,} (-{flops_reduction:.2f}%)")
    print("Layer widths (orig -> pruned):")
    for unit in PRUNABLE_UNITS:
        print(f"  {unit:12s}: {orig_cfg[unit]:3d} -> {pruned_cfg[unit]:3d} (rate={rates_by_unit[unit]:.2f})")
    if val_eval is not None:
        print("Val F1 (source -> pruned pre-finetune):",
              f"{val_eval['source'].get('f1_score', float('nan')):.4f} -> "
              f"{val_eval['pruned_pre_finetune'].get('f1_score', float('nan')):.4f}")

    print("\nNext steps:")
    if cfg.get("teacher_dir"):
        print(
            "  # 1) KD fine-tuning (recommended):\n"
            f"  python train_kd.py --config configs/finetune_pruned_kd.yaml "
            f"--teacher-dir {cfg['teacher_dir']} --pretrained-weights {pruned_weights_path} "
            f"--channel-cfg {channel_cfg_path}"
        )
    print(
        "  # 2) Task-only fine-tuning ablation:\n"
        f"  python train.py --config configs/train_rgb.yaml --filters {filters} "
        f"--lr 2e-4 --epochs 100 --patience 20 --pretrained-weights {pruned_weights_path} "
        f"--channel-cfg {channel_cfg_path} --run-name {run_dir.name}_pruned_task_ft"
    )
    print(
        "  # 3) Small-from-scratch baseline (same pruned architecture):\n"
        f"  python train.py --config configs/train_rgb.yaml --filters {filters} "
        f"--channel-cfg {channel_cfg_path} --run-name {run_dir.name}_scratch_small"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prune a trained Attention U-Net using HRank feature-map matrix rank."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Trained Attention U-Net run directory containing config.json and best.weights.h5.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional explicit .weights.h5 path (default: <run-dir>/best.weights.h5).",
    )
    parser.add_argument(
        "--compress-rate",
        default="0.5",
        help="Uniform float, 17 comma-separated floats, or JSON path/string mapping units to pruning rates.",
    )
    parser.add_argument(
        "--center-rate",
        type=float,
        default=None,
        help="Optional rate override for center_1 and center_2 (8x8 spatial maps cap rank at 8).",
    )
    parser.add_argument(
        "--min-channels",
        type=int,
        default=2,
        help="Minimum surviving channels per prunable unit (default: 2).",
    )
    parser.add_argument(
        "--hrank-batches",
        type=int,
        default=10,
        help="Number of batches used to estimate average feature-map rank (default: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for HRank estimation and optional validation (default: 8).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split used to collect HRank feature maps (default: train).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for pruned_init.weights.h5 and channel_cfg.json (default: runs/pruned_<run_name>).",
    )
    parser.add_argument(
        "--eval-val",
        action="store_true",
        help="Evaluate source and pruned-before-finetuning models on the val split.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prune_student(args)


if __name__ == "__main__":
    main()
