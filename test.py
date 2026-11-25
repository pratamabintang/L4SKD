"""Evaluate a trained Attention U-Net checkpoint on Landslide4Sense."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from uag.config import add_common_args
from uag.data.landslide4sense import (
    Landslide4SenseDataset,
    get_normalizer,
    make_tf_dataset,
)
from uag.losses import get_loss
from uag.metrics import get_metrics
from uag.models import build_model
from uag.pruning.hrank import estimate_model_stats
from uag.region_metrics import EvaluationSuiteAccumulator, format_evaluation_suite
from uag.utils import load_json, parse_bands, save_json


def evaluate(
    cfg: dict[str, Any],
    weights: str | Path,
    split: str = "test",
    threshold: Optional[float] = None,
    exclude_list: Optional[str | Path] = None,
    iou_tau: Optional[float] = None,
    dilation_alpha: Optional[float] = None,
    per_patch_csv: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Build model, load weights, and evaluate on the specified split."""
    from tensorflow import keras

    keras.backend.clear_session()

    bands = cfg.get("bands")
    if isinstance(bands, str):
        bands = parse_bands(bands)

    normalizer = get_normalizer(
        mode=cfg.get("norm", "l4s"),
        stats_file=cfg.get("stats_file"),
        bands=bands,
    )

    resolved_exclude = exclude_list if exclude_list is not None else cfg.get("blacklist")
    ds_raw = Landslide4SenseDataset(
        root=cfg["data_root"],
        split=split,
        bands=bands,
        normalizer=normalizer,
        exclude_list=resolved_exclude,
        require_masks=True,
    )
    batch_size = int(cfg.get("batch_size", 16))
    tf_ds = make_tf_dataset(
        ds_raw,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        cache=False,
    )

    input_shape = tuple(cfg["input_shape"]) if "input_shape" in cfg else ds_raw.image_shape
    model_name = cfg.get("model", "attention_unet")
    filters = int(cfg.get("filters", 4))
    loss_name = cfg.get("loss", "dice")
    lr = float(cfg.get("lr", 1e-3))
    eval_threshold = float(threshold if threshold is not None else cfg.get("threshold", 0.5))

    model = build_model(
        model_name,
        input_shape=input_shape,
        filters=filters,
        pretrained_weights=str(weights),
        bands=bands,
        channel_cfg=cfg.get("channel_cfg"),
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=get_loss(loss_name),
        metrics=get_metrics(threshold=eval_threshold),
    )

    raw_metrics = model.evaluate(tf_ds, return_dict=True)
    metrics_out: dict[str, Any] = {k: float(v) for k, v in raw_metrics.items()}

    resolved_tau = float(iou_tau if iou_tau is not None else cfg.get("iou_tau", 0.5))
    resolved_alpha = float(
        dilation_alpha if dilation_alpha is not None else cfg.get("dilation_alpha", 0.1)
    )
    acc = EvaluationSuiteAccumulator(
        threshold=eval_threshold,
        tau=resolved_tau,
        alpha=resolved_alpha,
    )
    offset = 0
    for batch_x, batch_y in tf_ds:
        preds = model(batch_x, training=False)
        b_size = int(preds.shape[0]) if preds.shape[0] is not None else int(batch_y.shape[0])
        batch_ids = ds_raw.ids[offset : offset + b_size]
        acc.update_batch(preds.numpy(), batch_y.numpy(), sample_ids=batch_ids)
        offset += b_size
    metrics_out.update(acc.compute())

    weights_size_mb = round(Path(weights).stat().st_size / (1024 * 1024), 4)
    metrics_out["model_cost"] = {
        **estimate_model_stats(model),
        "weights_size_mb": weights_size_mb,
    }
    if per_patch_csv is not None:
        acc.save_per_patch_csv(per_patch_csv)
    return metrics_out


def resolve_eval_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    """Merge config.json from --run-dir (if provided) with explicit CLI overrides."""
    cfg: dict[str, Any] = {}
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        cfg_path = run_dir / "config.json"
        if cfg_path.is_file():
            cfg.update(load_json(cfg_path))
        weights_path = Path(args.weights) if args.weights else run_dir / "best.weights.h5"
        out_dir = run_dir
    else:
        if args.weights is None:
            raise ValueError("Provide either --run-dir or --weights.")
        weights_path = Path(args.weights)
        out_dir = weights_path.parent

    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Apply defaults and explicit overrides.
    defaults = {
        "data_root": "datasets/landslide4senses",
        "norm": "l4s",
        "stats_file": None,
        "loss": "dice",
        "threshold": 0.5,
        "iou_tau": 0.5,
        "dilation_alpha": 0.1,
        "model": "attention_unet",
        "blacklist": None,
        "filters": 4,
        "batch_size": 16,
    }
    for key, default_val in defaults.items():
        cfg.setdefault(key, default_val)

    if args.data_root_override is not None:
        cfg["data_root"] = args.data_root_override
    elif "data_root" not in cfg:
        cfg["data_root"] = args.data_root

    if args.bands is not None:
        cfg["bands"] = parse_bands(args.bands)
    if args.norm_override is not None:
        cfg["norm"] = args.norm_override
    if args.stats_file is not None:
        cfg["stats_file"] = args.stats_file
    if args.filters is not None:
        cfg["filters"] = args.filters
    if args.loss_override is not None:
        cfg["loss"] = args.loss_override
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.threshold_override is not None:
        cfg["threshold"] = args.threshold_override
    if args.iou_tau_override is not None:
        cfg["iou_tau"] = args.iou_tau_override
    if args.dilation_alpha_override is not None:
        cfg["dilation_alpha"] = args.dilation_alpha_override
    if args.model_override is not None:
        cfg["model"] = args.model_override
    if args.blacklist_override is not None:
        cfg["blacklist"] = args.blacklist_override

    return cfg, weights_path, out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Attention U-Net on Landslide4Sense.")
    add_common_args(parser)
    parser.add_argument("--run-dir", default=None, help="Run directory containing config.json and best.weights.h5.")
    parser.add_argument("--weights", default=None, help="Explicit path to .weights.h5 checkpoint.")
    parser.add_argument("--filters", type=int, default=None, help="Override base conv filter count.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate (default: test).")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
    # Track explicit overrides separately from common defaults when --run-dir is supplied.
    parser.set_defaults(
        data_root_override=None,
        norm_override=None,
        loss_override=None,
        threshold_override=None,
        iou_tau_override=None,
        dilation_alpha_override=None,
        model_override=None,
        blacklist_override=None,
    )
    return parser


def _parse_with_overrides(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    # Detect whether common flags were explicitly passed via sys.argv.
    import sys

    argv = set(sys.argv[1:])
    if "--data-root" in argv or any(a.startswith("--data-root=") for a in argv):
        args.data_root_override = args.data_root
    if "--norm" in argv or any(a.startswith("--norm=") for a in argv):
        args.norm_override = args.norm
    if "--loss" in argv or any(a.startswith("--loss=") for a in argv):
        args.loss_override = args.loss
    if "--threshold" in argv or any(a.startswith("--threshold=") for a in argv):
        args.threshold_override = args.threshold
    if "--iou-tau" in argv or any(a.startswith("--iou-tau=") for a in argv):
        args.iou_tau_override = args.iou_tau
    if "--dilation-alpha" in argv or any(a.startswith("--dilation-alpha=") for a in argv):
        args.dilation_alpha_override = args.dilation_alpha
    if "--model" in argv or any(a.startswith("--model=") for a in argv):
        args.model_override = args.model
    if "--blacklist" in argv or any(a.startswith("--blacklist=") for a in argv):
        args.blacklist_override = args.blacklist
    return args


def main() -> None:
    parser = build_parser()
    args = _parse_with_overrides(parser)
    cfg, weights_path, out_dir = resolve_eval_config(args)
    per_patch_path = out_dir / f"per_patch_{args.split}.csv"
    metrics = evaluate(
        cfg=cfg,
        weights=weights_path,
        split=args.split,
        threshold=cfg.get("threshold", 0.5),
        exclude_list=cfg.get("blacklist"),
        iou_tau=cfg.get("iou_tau", 0.5),
        dilation_alpha=cfg.get("dilation_alpha", 0.1),
        per_patch_csv=per_patch_path,
    )
    print(f"Evaluation metrics ({args.split}):")
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            print(f"  {key}: {val:.6f}")
    if "region_sufficiency" in metrics and "standard_overlap" in metrics:
        print(format_evaluation_suite(metrics))
    if "model_cost" in metrics:
        mc = metrics["model_cost"]
        print(
            f"Model cost: params={mc['params']:,}, flops={mc['flops']:,}, "
            f"weights_size_mb={mc['weights_size_mb']:.4f}"
        )
    save_json(metrics, out_dir / f"metrics_{args.split}.json")


if __name__ == "__main__":
    main()
