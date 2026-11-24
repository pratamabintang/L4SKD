"""Train Attention U-Net on the Landslide4Sense dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from uag.config import (
    IGNORED_KEYS,
    add_common_args,
    add_config_arg,
    parse_args_with_config,
)
from uag.data.landslide4sense import (
    Landslide4SenseDataset,
    get_normalizer,
    make_tf_dataset,
)
from uag.losses import get_loss
from uag.metrics import get_metrics
from uag.models import build_model
from uag.utils import (
    collect_environment,
    ensure_dir,
    parse_bands,
    plot_history,
    run_name,
    save_json,
    set_seed,
)


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run one training (and optional test evaluation) job from a config dict."""
    import tensorflow as tf
    from tensorflow import keras

    keras.backend.clear_session()
    set_seed(int(cfg.get("seed", 42)))

    if cfg.get("deterministic", False):
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            print(
                "Warning: tf.config.experimental.enable_op_determinism() "
                "is unavailable in this TensorFlow version."
            )

    if cfg.get("mixed_precision", False):
        keras.mixed_precision.set_global_policy("mixed_float16")

    bands = cfg.get("bands")
    if isinstance(bands, str):
        bands = parse_bands(bands)

    normalizer = get_normalizer(
        mode=cfg.get("norm", "l4s"),
        stats_file=cfg.get("stats_file"),
        bands=bands,
    )

    train_ds_raw = Landslide4SenseDataset(
        root=cfg["data_root"],
        split="train",
        bands=bands,
        normalizer=normalizer,
        id_list=cfg.get("train_list"),
        exclude_list=cfg.get("blacklist"),
        require_masks=True,
    )
    val_ds_raw = Landslide4SenseDataset(
        root=cfg["data_root"],
        split="val",
        bands=bands,
        normalizer=normalizer,
        exclude_list=cfg.get("blacklist"),
        require_masks=True,
    )

    input_shape = train_ds_raw.image_shape
    size = input_shape[0]
    model_name = cfg.get("model", "attention_unet")
    filters = int(cfg.get("filters", 4))
    batch_size = int(cfg.get("batch_size", 8))
    lr = float(cfg.get("lr", 1e-3))
    epochs = int(cfg.get("epochs", 200))
    patience = int(cfg.get("patience", 30))
    loss_name = cfg.get("loss", "dice")
    threshold = float(cfg.get("threshold", 0.5))

    color_jitter = bool(cfg.get("color_jitter", True))
    color_brightness = float(cfg.get("color_brightness", 0.15))
    color_contrast = float(cfg.get("color_contrast", 0.15))
    color_prob = float(cfg.get("color_prob", 0.5))

    cache_opt: bool | str = (
        cfg["cache_file"] if cfg.get("cache_file") else bool(cfg.get("cache", False))
    )
    train_tf_ds = make_tf_dataset(
        train_ds_raw,
        batch_size=batch_size,
        shuffle=True,
        augment=bool(cfg.get("augment", False)),
        cache=cache_opt,
        seed=int(cfg.get("seed", 42)),
        color_jitter=color_jitter,
        color_brightness=color_brightness,
        color_contrast=color_contrast,
        color_prob=color_prob,
    )
    val_tf_ds = make_tf_dataset(
        val_ds_raw,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        cache=cache_opt,
    )

    name = cfg.get("run_name") or run_name(
        filters=filters,
        batch_size=batch_size,
        lr=lr,
        size=size,
        model=model_name,
    )
    run_dir = ensure_dir(Path(cfg.get("output_dir", "runs")) / name)
    plots_dir = ensure_dir(run_dir / "plots")
    weights_path = run_dir / "best.weights.h5"

    resolved_cfg = {
        **cfg,
        "model": model_name,
        "bands": bands,
        "input_shape": list(input_shape),
        "filters": filters,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "patience": patience,
        "loss": loss_name,
        "threshold": threshold,
        "color_jitter": color_jitter,
        "color_brightness": color_brightness,
        "color_contrast": color_contrast,
        "color_prob": color_prob,
        "run_name": name,
        "run_dir": str(run_dir),
        "weights_path": str(weights_path),
    }
    save_json(resolved_cfg, run_dir / "config.json")
    save_json(collect_environment(), run_dir / "environment.json")
    try:
        import yaml

        reusable_cfg = {k: v for k, v in resolved_cfg.items() if k not in IGNORED_KEYS}
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(reusable_cfg, f, sort_keys=False)
    except ImportError:
        pass

    model = build_model(
        model_name,
        input_shape=input_shape,
        filters=filters,
        bands=bands,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=get_loss(loss_name),
        metrics=get_metrics(threshold=threshold),
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(weights_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
    ]

    history = model.fit(
        train_tf_ds,
        validation_data=val_tf_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    history_dict = {
        k: [float(v) for v in vals] for k, vals in history.history.items()
    }
    plot_history(history_dict, "f1_score", plots_dir / "f1_score.png")
    plot_history(history_dict, "loss", plots_dir / "loss.png")

    test_metrics: dict[str, float] | None = None
    if cfg.get("eval_test", False):
        test_ds_raw = Landslide4SenseDataset(
            root=cfg["data_root"],
            split="test",
            bands=bands,
            normalizer=normalizer,
            exclude_list=cfg.get("blacklist"),
            require_masks=True,
        )
        test_tf_ds = make_tf_dataset(
            test_ds_raw,
            batch_size=batch_size,
            shuffle=False,
            augment=False,
            cache=False,
        )
        eval_model = build_model(
            model_name,
            input_shape=input_shape,
            filters=filters,
            pretrained_weights=str(weights_path),
            bands=bands,
        )
        eval_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=get_loss(loss_name),
            metrics=get_metrics(threshold=threshold),
        )
        raw_metrics = eval_model.evaluate(test_tf_ds, return_dict=True)
        test_metrics = {k: float(v) for k, v in raw_metrics.items()}
        save_json(test_metrics, run_dir / "test_metrics.json")

    return {
        "run_dir": str(run_dir),
        "weights": str(weights_path),
        "history": history_dict,
        "test_metrics": test_metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Attention U-Net on Landslide4Sense.")
    add_config_arg(parser)
    add_common_args(parser)
    parser.add_argument("--train-list", default=None, help="Optional txt file filtering training sample IDs.")
    parser.add_argument("--filters", type=int, default=4, help="Base number of conv filters.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable random flips/rot90 and multispectral color perturbation.",
    )
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
    parser.add_argument("--output-dir", default="runs", help="Base directory for training runs.")
    parser.add_argument("--run-name", default=None, help="Optional custom run directory name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed_float16 policy.")
    parser.add_argument("--deterministic", action="store_true", help="Enable TF op determinism (slower).")
    parser.add_argument("--eval-test", action="store_true", help="Evaluate best weights on test split.")
    return parser


def main() -> None:
    args = parse_args_with_config(build_parser())
    cfg = vars(args)
    cfg["bands"] = parse_bands(args.bands)
    result = train(cfg)
    print(f"Training complete. Run directory: {result['run_dir']}")
    if result["test_metrics"] is not None:
        print("Test metrics:", result["test_metrics"])


if __name__ == "__main__":
    main()
