"""Shared utilities for seeding, filesystem I/O, plotting, and CLI parsing."""

from __future__ import annotations

import json
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def set_seed(seed: int) -> None:
    """Set random seeds across Python, NumPy, and TensorFlow/Keras."""
    import tensorflow as tf
    from tensorflow import keras

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if hasattr(keras.utils, "set_random_seed"):
        keras.utils.set_random_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if needed and return its Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path) -> None:
    """Serialize a Python object to formatted JSON."""
    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> Any:
    """Load JSON from disk."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_history(history_dict: dict[str, list[float]], key: str, out_path: str | Path) -> None:
    """Plot training and validation curves for a history key and save to disk."""
    if key not in history_dict:
        return

    out = Path(out_path)
    ensure_dir(out.parent)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_dict[key], label="train")
    val_key = f"val_{key}"
    if val_key in history_dict:
        ax.plot(history_dict[val_key], label="validation")
    ax.set_title(f"Model {key}")
    ax.set_ylabel(key)
    ax.set_xlabel("epoch")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def run_name(
    filters: int,
    batch_size: int,
    lr: float,
    size: int,
    model: str = "attention_unet",
) -> str:
    """Build a deterministic run directory name."""
    prefixes = {
        "attention_unet": "attunet",
        "unet": "unet",
        "dual_branch": "dualbranch",
    }
    prefix = prefixes.get(model, model)
    return f"{prefix}_size{size}_f{filters}_bs{batch_size}_lr{lr}"


def parse_bands(bands: Optional[str | Sequence[int]]) -> Optional[list[int]]:
    """Parse 0-based band indices from a comma-separated string or integer sequence."""
    if bands is None:
        return None
    if isinstance(bands, (list, tuple)):
        parsed = [int(b) for b in bands]
        return parsed or None
    if not bands.strip():
        return None
    parsed = [int(part.strip()) for part in bands.split(",") if part.strip()]
    return parsed or None


def collect_environment() -> dict[str, Any]:
    """Collect Python, library, GPU, and Git metadata for reproducibility."""
    h5py_version: Optional[str] = None
    try:
        import h5py

        h5py_version = h5py.__version__
    except Exception:
        pass

    tf_version: Optional[str] = None
    keras_version: Optional[str] = None
    gpus: list[str] = []
    try:
        import tensorflow as tf
        from tensorflow import keras

        tf_version = tf.__version__
        keras_version = getattr(keras, "__version__", None)
        gpus = [dev.name for dev in tf.config.list_physical_devices("GPU")]
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            check=False,
        )
        if rev.returncode == 0:
            git_commit = rev.stdout.strip() or None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            check=False,
        )
        if status.returncode == 0:
            git_dirty = bool(status.stdout.strip())
    except Exception:
        pass

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "h5py": h5py_version,
        "tensorflow": tf_version,
        "keras": keras_version,
        "gpus": gpus,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
