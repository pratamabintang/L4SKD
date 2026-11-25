"""Shared CLI argument parser helpers and YAML/JSON config support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

IGNORED_KEYS = {"config", "input_shape", "run_dir", "weights_path", "run_name_resolved"}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add shared data, normalization, and loss arguments to an ArgumentParser."""
    parser.add_argument(
        "--data-root",
        default="datasets/landslide4senses",
        help="Path to the Landslide4Sense dataset root directory.",
    )
    parser.add_argument(
        "--bands",
        default=None,
        help="Comma-separated 0-based band indices to select (default: all 14 bands).",
    )
    parser.add_argument(
        "--norm",
        choices=["l4s", "none"],
        default="l4s",
        help="Band normalization mode ('l4s' uses built-in stats, 'none' disables).",
    )
    parser.add_argument(
        "--stats-file",
        default=None,
        help="Optional JSON file with 'mean' and 'std' lists; overrides --norm.",
    )
    parser.add_argument(
        "--loss",
        choices=["dice", "bce_dice", "bce", "structure", "tversky", "focal_tversky"],
        default="dice",
        help="Segmentation loss function.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binary segmentation metrics.",
    )
    parser.add_argument(
        "--iou-tau",
        type=float,
        default=0.5,
        help="Tolerant IoU threshold tau for Family 1 region sufficiency.",
    )
    parser.add_argument(
        "--dilation-alpha",
        type=float,
        default=0.1,
        help="Size-scaled dilation factor alpha for Family 1 region sufficiency.",
    )
    parser.add_argument(
        "--model",
        choices=["attention_unet", "unet", "dual_branch"],
        default="attention_unet",
        help="Model architecture: Attention U-Net, standard U-Net, or Dual-Branch Attention U-Net.",
    )
    parser.add_argument(
        "--blacklist",
        default=None,
        help="Optional txt file of sample IDs (x in image_x.h5) to exclude across splits.",
    )


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Add --config option for YAML/JSON configuration files."""
    parser.add_argument(
        "--config",
        default=None,
        help="YAML/JSON config file; explicit CLI flags override its values.",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON configuration file into a dictionary."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    suffix = cfg_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        import yaml

        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            return {}
    elif suffix == ".json":
        with cfg_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(
            f"Unsupported config extension {suffix!r}; expected .yaml, .yml, or .json."
        )

    if not isinstance(data, dict):
        raise ValueError(f"Config top level must be a mapping/dict, got {type(data).__name__}.")
    return data


def _coerce_value(action: argparse.Action, value: Any, parser: argparse.ArgumentParser, key: str) -> Any:
    if action.type is not None and value is not None:
        try:
            if isinstance(value, (list, tuple)):
                value = [action.type(v) for v in value]
            else:
                value = action.type(value)
        except (TypeError, ValueError) as exc:
            parser.error(f"Invalid value for config key {key!r}: {exc}")

    if action.choices is not None and value is not None:
        if isinstance(value, (list, tuple)):
            for item in value:
                if item not in action.choices:
                    parser.error(
                        f"Invalid choice {item!r} for config key {key!r} (choose from {list(action.choices)})"
                    )
        elif value not in action.choices:
            parser.error(
                f"Invalid choice {value!r} for config key {key!r} (choose from {list(action.choices)})"
            )
    return value


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Parse CLI args with precedence: argparse defaults < config file < explicit CLI flags."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    config_path = pre_args.config
    if config_path is not None:
        raw_cfg = load_config(config_path)
        actions = {a.dest: a for a in parser._actions if a.dest != "help"}
        values: dict[str, Any] = {}
        for raw_key, raw_val in raw_cfg.items():
            key = str(raw_key).replace("-", "_")
            if key in IGNORED_KEYS:
                continue
            if key not in actions:
                parser.error(f"Unknown config key: {key}")
            values[key] = _coerce_value(actions[key], raw_val, parser, key)
        parser.set_defaults(**values)

    args = parser.parse_args(argv)
    args.config = config_path
    return args
