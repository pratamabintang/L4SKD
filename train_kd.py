"""Train student Attention U-Net with offline output-level DIST knowledge distillation."""

from __future__ import annotations

import argparse

import train as base_train
from uag.config import parse_args_with_config
from uag.utils import parse_bands


def build_parser() -> argparse.ArgumentParser:
    parser = base_train.build_parser()
    parser.description = (
        "Train Attention U-Net student with offline DIST knowledge distillation."
    )
    parser.add_argument(
        "--teacher-dir",
        default=None,
        help="Teacher run directory containing teacher_logits_train.h5 and teacher_logits_val.h5.",
    )
    parser.add_argument(
        "--kd-weight",
        type=float,
        default=1.0,
        help="Weight on the DIST knowledge distillation loss term.",
    )
    parser.add_argument(
        "--kd-tau",
        type=float,
        default=1.0,
        help="Temperature tau for DIST softmax probabilities.",
    )
    parser.add_argument(
        "--kd-beta",
        type=float,
        default=1.0,
        help="Weight on DIST inter-class relation loss.",
    )
    parser.add_argument(
        "--kd-gamma",
        type=float,
        default=1.0,
        help="Weight on DIST intra-class relation loss.",
    )
    parser.add_argument(
        "--monitor",
        default="val_task_loss",
        help="Metric to monitor for ModelCheckpoint and EarlyStopping (default: val_task_loss).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parse_args_with_config(parser)
    if not args.teacher_dir:
        parser.error("--teacher-dir (or teacher_dir in config) is required.")
    cfg = vars(args)
    cfg["bands"] = parse_bands(args.bands)
    result = base_train.train(cfg)
    print(f"KD training complete. Run directory: {result['run_dir']}")
    if result["test_metrics"] is not None:
        print("Test metrics:", result["test_metrics"])


if __name__ == "__main__":
    main()
