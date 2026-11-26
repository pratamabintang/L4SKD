"""Train SAM2-UNet teacher on Landslide4Sense (RGB) with PyTorch."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from export_teacher_logits import export_logits
from uag.config import IGNORED_KEYS, add_config_arg, parse_args_with_config
from uag.data.landslide4sense import Landslide4SenseDataset, get_normalizer
from uag.models.SAM2UNet import resolve_sam2_cfg
from uag.region_metrics import EvaluationSuiteAccumulator
from uag.teacher.data import TorchL4SDataset
from uag.teacher.losses import structure_loss
from uag.teacher.metrics import ConfusionAccumulator
from uag.teacher.model import (
    build_teacher,
    load_teacher,
    resolve_device,
    trainable_state_dict,
)
from uag.utils import (
    collect_environment,
    ensure_dir,
    parse_bands,
    plot_history,
    save_json,
)


def _set_torch_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cfg_alias(sam2_cfg: str) -> str:
    resolved = resolve_sam2_cfg(sam2_cfg)
    stem = Path(resolved).stem
    if stem.startswith("sam2_hiera_"):
        return stem[len("sam2_hiera_") :]
    return stem


def _resolve_amp_dtype(use_amp: bool, device: str) -> tuple[bool, torch.dtype]:
    """Use bfloat16 when supported on CUDA; disable float16 AMP to avoid SAM2 overflow."""
    if not (use_amp and device.startswith("cuda")):
        return False, torch.float32
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    print("Note: CUDA device lacks bfloat16 support; disabling float16 AMP to avoid SAM2-UNet overflow.")
    return False, torch.float32


def _evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float,
    use_amp: bool,
    amp_dtype: torch.dtype = torch.float32,
    include_region_suite: bool = False,
    iou_tau: float = 0.5,
    dilation_alpha: float = 0.1,
    sample_ids: Optional[Sequence[str]] = None,
    per_patch_csv: Optional[str | Path] = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    acc = ConfusionAccumulator(threshold=threshold)
    suite_acc = (
        EvaluationSuiteAccumulator(
            threshold=threshold,
            tau=iou_tau,
            alpha=dilation_alpha,
        )
        if include_region_suite
        else None
    )
    amp_enabled = use_amp and device.startswith("cuda")

    with torch.no_grad():
        for imgs, masks, idxs in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                out, out1, out2 = model(imgs)
                loss = (
                    structure_loss(out, masks)
                    + structure_loss(out1, masks)
                    + structure_loss(out2, masks)
                )
            total_loss += float(loss.item())
            n_batches += 1
            probs = torch.sigmoid(out.float())
            acc.update(probs, masks)
            if suite_acc is not None:
                batch_ids = (
                    [sample_ids[int(idx)] for idx in idxs]
                    if sample_ids is not None
                    else None
                )
                suite_acc.update_batch(
                    probs.cpu().numpy(),
                    masks.cpu().numpy(),
                    sample_ids=batch_ids,
                )

    metrics: dict[str, Any] = {
        "loss": total_loss / max(n_batches, 1),
        **acc.compute(),
    }
    if suite_acc is not None:
        metrics.update(suite_acc.compute())
        if per_patch_csv is not None:
            suite_acc.save_per_patch_csv(per_patch_csv)
    return metrics


def train_teacher(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run SAM2-UNet teacher training, validation, optional test eval, and logit export."""
    seed = int(cfg.get("seed", 42))
    _set_torch_seed(seed)

    device = resolve_device(str(cfg.get("device", "auto")))
    use_amp, amp_dtype = _resolve_amp_dtype(bool(cfg.get("amp", False)), device)

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
    height = input_shape[0]
    sam2_cfg = resolve_sam2_cfg(str(cfg.get("sam2_cfg", "sam2_hiera_l.yaml")))
    sam2_ckpt = cfg.get("sam2_checkpoint") or None
    input_size = int(cfg.get("input_size", 352))
    batch_size = int(cfg.get("batch_size", 12))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    epochs = int(cfg.get("epochs", 50))
    patience = int(cfg.get("patience", 0))
    threshold = float(cfg.get("threshold", 0.5))
    iou_tau = float(cfg.get("iou_tau", 0.5))
    dilation_alpha = float(cfg.get("dilation_alpha", 0.1))
    num_workers = int(cfg.get("num_workers", 2))
    augment = bool(cfg.get("augment", False))

    name = cfg.get("run_name") or (
        f"sam2unet_{_cfg_alias(sam2_cfg)}_rgb_size{height}_in{input_size}_bs{batch_size}_lr{lr}"
    )
    run_dir = ensure_dir(Path(cfg.get("output_dir", "runs")) / name)
    plots_dir = ensure_dir(run_dir / "plots")
    weights_path = run_dir / "best.pt"

    resolved_cfg = {
        **cfg,
        "model": "sam2unet",
        "bands": bands,
        "input_shape": list(input_shape),
        "sam2_cfg": sam2_cfg,
        "sam2_checkpoint": sam2_ckpt,
        "input_size": input_size,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "patience": patience,
        "threshold": threshold,
        "iou_tau": iou_tau,
        "dilation_alpha": dilation_alpha,
        "augment": augment,
        "amp": use_amp,
        "run_name": name,
        "run_dir": str(run_dir),
        "weights_path": str(weights_path),
    }
    save_json(resolved_cfg, run_dir / "config.json")

    env_info = collect_environment()
    env_info["torch"] = torch.__version__
    env_info["cuda"] = bool(torch.cuda.is_available())
    save_json(env_info, run_dir / "environment.json")

    try:
        import yaml

        reusable_cfg = {k: v for k, v in resolved_cfg.items() if k not in IGNORED_KEYS}
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(reusable_cfg, f, sort_keys=False)
    except ImportError:
        pass

    train_torch_ds = TorchL4SDataset(train_ds_raw, augment=augment)
    val_torch_ds = TorchL4SDataset(val_ds_raw, augment=False)
    dl_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_torch_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        generator=dl_generator,
    )
    val_loader = DataLoader(
        val_torch_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    model = build_teacher(resolved_cfg, device=device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    amp_label = str(amp_dtype).replace("torch.", "") if use_amp else "disabled (float32)"
    print(f"Trainable params: {n_trainable:,} | AMP: {amp_label}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=1e-7,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    history_rows: list[dict[str, float]] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_f1s: list[float] = []

    best_val_f1 = -1.0
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_train_batches = 0
        current_lr = float(optimizer.param_groups[0]["lr"])

        for batch_idx, (imgs, masks, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out, out1, out2 = model(imgs)
                loss = (
                    structure_loss(out, masks)
                    + structure_loss(out1, masks)
                    + structure_loss(out2, masks)
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite teacher loss ({loss.item()}) at epoch {epoch}, batch {batch_idx}."
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

            running_loss += float(loss.item())
            n_train_batches += 1

        scheduler.step()
        train_loss = running_loss / max(n_train_batches, 1)

        val_res = _evaluate_loader(
            model,
            val_loader,
            device=device,
            threshold=threshold,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        val_loss = val_res["loss"]
        val_f1 = val_res["f1_score"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "loss": train_loss,
            "val_loss": val_loss,
            "val_precision": val_res["precision"],
            "val_recall": val_res["recall"],
            "val_f1_score": val_f1,
            "val_iou": val_res["iou"],
        }
        history_rows.append(row)
        print(
            f"Epoch {epoch:03d}/{epochs:03d} - lr: {current_lr:.6g} "
            f"- loss: {train_loss:.4f} - val_loss: {val_loss:.4f} "
            f"- val_f1: {val_f1:.4f} - val_iou: {val_res['iou']:.4f}"
        )

        if math.isfinite(val_loss) and val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improve = 0
            torch.save(trainable_state_dict(model), weights_path)
        else:
            epochs_without_improve += 1
            if patience > 0 and epochs_without_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    history_csv = run_dir / "history.csv"
    with history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "lr",
                "loss",
                "val_loss",
                "val_precision",
                "val_recall",
                "val_f1_score",
                "val_iou",
            ],
        )
        writer.writeheader()
        writer.writerows(history_rows)

    history_dict = {
        "loss": train_losses,
        "val_loss": val_losses,
        "f1_score": val_f1s,
    }
    plot_history(history_dict, "loss", plots_dir / "loss.png")
    plot_history(history_dict, "f1_score", plots_dir / "f1_score.png")

    best_model = load_teacher(resolved_cfg, weights_path, device=device)
    teacher_cost = {
        "total_params": int(sum(p.numel() for p in best_model.parameters())),
        "trainable_params": int(
            sum(p.numel() for p in best_model.parameters() if p.requires_grad)
        ),
        "weights_size_mb": round(weights_path.stat().st_size / (1024 * 1024), 4),
    }

    val_metrics = _evaluate_loader(
        best_model,
        val_loader,
        device=device,
        threshold=threshold,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        include_region_suite=True,
        iou_tau=iou_tau,
        dilation_alpha=dilation_alpha,
        sample_ids=val_ds_raw.ids,
        per_patch_csv=run_dir / "per_patch_val.csv",
    )
    val_metrics["model_cost"] = teacher_cost
    save_json(val_metrics, run_dir / "val_metrics.json")

    test_metrics: dict[str, Any] | None = None
    if cfg.get("eval_test", False):
        test_ds_raw = Landslide4SenseDataset(
            root=cfg["data_root"],
            split="test",
            bands=bands,
            normalizer=normalizer,
            exclude_list=cfg.get("blacklist"),
            require_masks=True,
        )
        test_loader = DataLoader(
            TorchL4SDataset(test_ds_raw, augment=False),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.startswith("cuda"),
        )
        test_metrics = _evaluate_loader(
            best_model,
            test_loader,
            device=device,
            threshold=threshold,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            include_region_suite=True,
            iou_tau=iou_tau,
            dilation_alpha=dilation_alpha,
            sample_ids=test_ds_raw.ids,
            per_patch_csv=run_dir / "per_patch_test.csv",
        )
        test_metrics["model_cost"] = teacher_cost
        save_json(test_metrics, run_dir / "test_metrics.json")

    exported: dict[str, Path] | None = None
    if cfg.get("export_logits", True):
        exported = export_logits(
            cfg=resolved_cfg,
            weights=weights_path,
            splits=["train", "val", "test"],
            out_dir=run_dir,
            batch_size=max(batch_size, 16),
            device=device,
        )

    return {
        "run_dir": str(run_dir),
        "weights": str(weights_path),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "exported_logits": {k: str(v) for k, v in exported.items()} if exported else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SAM2-UNet teacher on Landslide4Sense (RGB)."
    )
    add_config_arg(parser)
    parser.add_argument(
        "--data-root",
        default="datasets/landslide4senses",
        help="Path to the Landslide4Sense dataset root directory.",
    )
    parser.add_argument(
        "--bands",
        default="3,2,1",
        help="Comma-separated 0-based RGB band indices (default: 3,2,1).",
    )
    parser.add_argument(
        "--norm",
        choices=["l4s", "none"],
        default="l4s",
        help="Band normalization mode ('l4s' or 'none').",
    )
    parser.add_argument(
        "--stats-file",
        default=None,
        help="Optional JSON stats file overriding --norm.",
    )
    parser.add_argument(
        "--train-list",
        default=None,
        help="Optional txt file filtering training sample IDs.",
    )
    parser.add_argument(
        "--blacklist",
        default=None,
        help="Optional txt file of sample IDs to exclude across splits.",
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
        "--sam2-cfg",
        default="sam2_hiera_l.yaml",
        help="SAM2 config name or alias ('t', 's', 'b+', 'l').",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        default=None,
        help="Path to pretrained SAM2 checkpoint (.pt); empty string disables.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=352,
        help="Teacher square input resolution (divisible by 32; default: 352).",
    )
    parser.add_argument("--batch-size", type=int, default=12, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience on val_f1_score (0 disables).",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable random joint flips and 90-degree rotations.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable PyTorch CUDA automatic mixed precision.",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device ('auto', 'cuda', or 'cpu').",
    )
    parser.add_argument("--output-dir", default="runs", help="Base directory for runs.")
    parser.add_argument("--run-name", default=None, help="Optional custom run directory name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate best checkpoint on test split after training.",
    )
    parser.add_argument(
        "--export-logits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export teacher logits for train/val/test splits after training.",
    )
    return parser


def main() -> None:
    args = parse_args_with_config(build_parser())
    cfg = vars(args)
    cfg["bands"] = parse_bands(args.bands)
    result = train_teacher(cfg)
    print(f"Teacher training complete. Run directory: {result['run_dir']}")
    print("Val metrics:", result["val_metrics"])
    if result["test_metrics"] is not None:
        print("Test metrics:", result["test_metrics"])


if __name__ == "__main__":
    main()
