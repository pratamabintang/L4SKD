"""Hyperparameter grid search mirroring the original Attention_Unet_v1 notebook."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from train import train
from uag.config import add_common_args, add_config_arg, parse_args_with_config
from uag.utils import collect_environment, ensure_dir, parse_bands, save_json

CSV_COLUMNS = [
    "model",
    "batch_size",
    "learning_rate",
    "filters",
    "precision_area",
    "recall_area",
    "f1_score_area",
    "iou_score_area",
    "run_dir",
]


def _write_csv(rows: list[dict], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Attention U-Net hyperparameter grid search on Landslide4Sense."
    )
    add_config_arg(parser)
    add_common_args(parser)
    parser.add_argument("--train-list", default=None, help="Optional txt file filtering training sample IDs.")
    parser.add_argument(
        "--filters-list",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
        help="List of base filter counts to search.",
    )
    parser.add_argument(
        "--lr-list",
        type=float,
        nargs="+",
        default=[10e-3, 5e-4, 10e-4, 5e-5, 10e-5],
        help="List of learning rates to search.",
    )
    parser.add_argument(
        "--batch-list",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
        help="List of batch sizes to search.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Maximum epochs per run.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation.")
    parser.add_argument(
        "--color-jitter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply per-band brightness/contrast perturbation to multispectral bands (B1-B12) only, excluding Slope/DEM, when --augment is enabled.",
    )
    parser.add_argument(
        "--color-brightness",
        type=float,
        default=0.15,
        help="Max additive per-band brightness delta for multispectral color perturbation.",
    )
    parser.add_argument(
        "--color-contrast",
        type=float,
        default=0.15,
        help="Max multiplicative per-band contrast delta ([1-c, 1+c]) for multispectral color perturbation.",
    )
    parser.add_argument(
        "--color-prob",
        type=float,
        default=0.5,
        help="Probability of applying multispectral color perturbation per sample.",
    )
    parser.add_argument("--cache", action="store_true", help="Cache decoded samples in memory.")
    parser.add_argument("--cache-file", default=None, help="Optional tf.data cache file path.")
    parser.add_argument("--output-dir", default="runs/grid_search", help="Directory for runs and CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed_float16 policy.")
    parser.add_argument("--deterministic", action="store_true", help="Enable TF op determinism (slower).")
    return parser


def main() -> None:
    args = parse_args_with_config(build_parser())
    bands = parse_bands(args.bands)
    output_dir = ensure_dir(args.output_dir)
    save_json(vars(args), output_dir / "grid_config.json")
    save_json(collect_environment(), output_dir / "environment.json")
    csv_path = output_dir / "results_Att_Unet.csv"

    rows: list[dict] = []
    for filters in args.filters_list:
        for lr in args.lr_list:
            for batch_size in args.batch_list:
                cfg = {
                    "data_root": args.data_root,
                    "bands": bands,
                    "norm": args.norm,
                    "stats_file": args.stats_file,
                    "loss": args.loss,
                    "threshold": args.threshold,
                    "model": args.model,
                    "train_list": args.train_list,
                    "blacklist": args.blacklist,
                    "filters": filters,
                    "lr": lr,
                    "batch_size": batch_size,
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "augment": args.augment,
                    "color_jitter": args.color_jitter,
                    "color_brightness": args.color_brightness,
                    "color_contrast": args.color_contrast,
                    "color_prob": args.color_prob,
                    "cache": args.cache,
                    "cache_file": args.cache_file,
                    "output_dir": str(output_dir),
                    "seed": args.seed,
                    "mixed_precision": args.mixed_precision,
                    "deterministic": args.deterministic,
                    "eval_test": True,
                }
                result = train(cfg)
                metrics = result["test_metrics"] or {}
                display_model = {
                    "attention_unet": "Attention_Unet",
                    "unet": "Unet",
                    "dual_branch": "DualBranch_Attention_Unet",
                }.get(args.model, args.model)
                rows.append(
                    {
                        "model": display_model,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "filters": filters,
                        "precision_area": metrics.get("precision", ""),
                        "recall_area": metrics.get("recall", ""),
                        "f1_score_area": metrics.get("f1_score", ""),
                        "iou_score_area": metrics.get("iou", ""),
                        "run_dir": result["run_dir"],
                    }
                )
                _write_csv(rows, csv_path)

    print(f"Grid search complete. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
