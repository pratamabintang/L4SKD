"""Landslide4Sense dataset loader and TensorFlow input pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np

# Channel-wise mean and standard deviation from the Landslide4Sense-2022 repository.
L4S_MEAN: list[float] = [
    -0.4914,
    -0.3074,
    -0.1277,
    -0.0625,
    0.0439,
    0.0803,
    0.0644,
    0.0802,
    0.3000,
    0.4082,
    0.0823,
    0.0516,
    0.3338,
    0.7819,
]
L4S_STD: list[float] = [
    0.9325,
    0.8775,
    0.8860,
    0.8869,
    0.8857,
    0.8418,
    0.8354,
    0.8491,
    0.9061,
    1.6072,
    0.8848,
    0.9232,
    0.9018,
    1.2913,
]
NUM_BANDS = 14
BAND_NAMES: list[str] = [f"B{i}" for i in range(1, 13)] + ["Slope", "DEM"]
MULTISPECTRAL_BAND_INDICES: tuple[int, ...] = tuple(range(12))  # B1..B12
TOPOGRAPHIC_BAND_INDICES: tuple[int, ...] = (12, 13)  # Slope, DEM
SPLIT_DIRS: dict[str, list[str]] = {
    "train": ["train"],
    "val": ["val", "validation"],
    "validation": ["val", "validation"],
    "test": ["test"],
}

_IMAGE_RE = re.compile(r"^image_(\d+)\.h5$")


def _resolve_split_dir(base_dir: Path, split: str) -> Optional[Path]:
    candidates = SPLIT_DIRS.get(split, [split])
    for candidate in candidates:
        path = base_dir / candidate
        if path.is_dir():
            return path
    return None


def _load_id_filter(id_list: str | Path) -> set[str]:
    path = Path(id_list)
    if not path.is_file():
        raise FileNotFoundError(f"ID list file not found: {path}")
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stem = line.strip()
            if stem:
                if stem.endswith(".h5"):
                    stem = stem[:-3]
                ids.add(stem)
    return ids


def load_exclude_ids(exclude_list: str | Path) -> set[int]:
    """Load numeric sample indices `x` (for image_x.h5 / mask_x.h5) to exclude."""
    path = Path(exclude_list)
    if not path.is_file():
        raise FileNotFoundError(f"Blacklist file not found: {path}")

    excluded: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            raw = token
            if token.endswith(".h5"):
                token = token[:-3]
            if token.startswith("image_"):
                token = token[len("image_") :]
            elif token.startswith("mask_"):
                token = token[len("mask_") :]
            if not token.isdigit():
                raise ValueError(
                    f"Invalid sample ID {raw!r} in {path}:{line_no}; "
                    "expected an integer x, image_x(.h5), or mask_x(.h5)."
                )
            excluded.add(int(token))
    return excluded


def list_samples(
    root: str | Path,
    split: str,
    id_list: Optional[str | Path] = None,
    exclude_list: Optional[str | Path] = None,
    require_masks: bool = True,
) -> list[tuple[str, Optional[str]]]:
    """List paired (image_path, mask_path) tuples sorted numerically by sample index."""
    root_path = Path(root)
    images_root = root_path / "images"
    annotations_root = root_path / "annotations"

    img_split_dir = _resolve_split_dir(images_root, split)
    if img_split_dir is None:
        tried = ", ".join(str(images_root / d) for d in SPLIT_DIRS.get(split, [split]))
        raise FileNotFoundError(
            f"Image split directory for '{split}' not found under {images_root} (tried: {tried})"
        )

    ann_split_dir = _resolve_split_dir(annotations_root, split)
    if require_masks and ann_split_dir is None:
        tried = ", ".join(str(annotations_root / d) for d in SPLIT_DIRS.get(split, [split]))
        raise FileNotFoundError(
            f"Annotation split directory for '{split}' not found under {annotations_root} (tried: {tried})"
        )

    allowed_ids = _load_id_filter(id_list) if id_list is not None else None
    excluded_ids = load_exclude_ids(exclude_list) if exclude_list is not None else None

    indexed_samples: list[tuple[int, str, Optional[str]]] = []
    for img_path in img_split_dir.glob("image_*.h5"):
        match = _IMAGE_RE.match(img_path.name)
        if not match:
            continue
        idx = int(match.group(1))
        if excluded_ids is not None and idx in excluded_ids:
            continue
        stem = img_path.stem
        if allowed_ids is not None and stem not in allowed_ids:
            continue

        mask_path_str: Optional[str] = None
        if ann_split_dir is not None:
            candidate_mask = ann_split_dir / f"mask_{idx}.h5"
            if candidate_mask.is_file():
                mask_path_str = str(candidate_mask)
            elif require_masks:
                raise FileNotFoundError(
                    f"Missing mask for {img_path.name}: expected {candidate_mask}"
                )
        elif require_masks:
            raise FileNotFoundError(
                f"Missing annotations directory for split '{split}'"
            )

        indexed_samples.append((idx, str(img_path), mask_path_str))

    indexed_samples.sort(key=lambda item: item[0])
    samples = [(img_p, mask_p) for _, img_p, mask_p in indexed_samples]
    if not samples:
        raise ValueError(
            f"No samples found for split '{split}' in {img_split_dir}"
            + (f" matching {id_list}" if id_list is not None else "")
            + (f" after excluding {exclude_list}" if exclude_list is not None else "")
        )
    return samples


def read_image(path: str | Path, bands: Optional[Sequence[int]] = None) -> np.ndarray:
    """Read an HDF5 image patch ('img') as float32 (H, W, C)."""
    with h5py.File(path, "r") as f:
        img = f["img"][()]
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if bands is not None:
        img = img[..., list(bands)]
    return np.ascontiguousarray(img, dtype=np.float32)


def read_mask(path: str | Path) -> np.ndarray:
    """Read an HDF5 mask ('mask') as float32 (H, W, 1)."""
    with h5py.File(path, "r") as f:
        mask = f["mask"][()]
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 2:
        mask = mask[..., np.newaxis]
    return np.ascontiguousarray(mask, dtype=np.float32)


class Normalizer:
    """Per-band channel standardization."""

    def __init__(
        self,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        bands: Optional[Sequence[int]] = None,
    ) -> None:
        if mean is None or std is None:
            self.mean: Optional[np.ndarray] = None
            self.std: Optional[np.ndarray] = None
            return

        mean_arr = np.asarray(mean, dtype=np.float32)
        std_arr = np.asarray(std, dtype=np.float32)
        if bands is not None:
            idx = list(bands)
            mean_arr = mean_arr[idx]
            std_arr = std_arr[idx]

        std_arr = np.where(std_arr == 0.0, 1.0, std_arr).astype(np.float32)
        self.mean = mean_arr.reshape(1, 1, -1)
        self.std = std_arr.reshape(1, 1, -1)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return img.astype(np.float32, copy=False)
        return ((img - self.mean) / self.std).astype(np.float32, copy=False)


def get_normalizer(
    mode: str = "l4s",
    stats_file: Optional[str | Path] = None,
    bands: Optional[Sequence[int]] = None,
) -> Normalizer:
    """Create a Normalizer from built-in L4S stats, a JSON stats file, or identity."""
    if stats_file is not None:
        with Path(stats_file).open("r", encoding="utf-8") as f:
            stats = json.load(f)
        return Normalizer(mean=stats["mean"], std=stats["std"], bands=bands)

    mode_norm = mode.lower()
    if mode_norm == "l4s":
        return Normalizer(mean=L4S_MEAN, std=L4S_STD, bands=bands)
    if mode_norm == "none":
        return Normalizer(mean=None, std=None, bands=bands)
    raise ValueError(f"Unknown normalization mode: {mode!r}. Expected 'l4s' or 'none'.")


class Landslide4SenseDataset:
    """Framework-agnostic Landslide4Sense HDF5 dataset."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        bands: Optional[Sequence[int]] = None,
        normalizer: Optional[Normalizer] = None,
        id_list: Optional[str | Path] = None,
        exclude_list: Optional[str | Path] = None,
        require_masks: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.bands = list(bands) if bands is not None else None
        if self.bands is not None:
            for b in self.bands:
                if b < 0 or b >= NUM_BANDS:
                    raise ValueError(f"Band index {b} out of range [0, {NUM_BANDS - 1}]")
        self.normalizer = normalizer
        self.exclude_list = Path(exclude_list) if exclude_list is not None else None
        self.require_masks = require_masks
        self.samples = list_samples(
            root=self.root,
            split=split,
            id_list=id_list,
            exclude_list=exclude_list,
            require_masks=require_masks,
        )
        self.ids = [Path(img_path).stem for img_path, _ in self.samples]

        sample_img = read_image(self.samples[0][0], bands=self.bands)
        self._image_shape = tuple(int(dim) for dim in sample_img.shape)

    @property
    def num_bands(self) -> int:
        return self._image_shape[2]

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return self._image_shape

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, Optional[np.ndarray]]:
        img_path, mask_path = self.samples[index]
        img = read_image(img_path, bands=self.bands)
        if self.normalizer is not None:
            img = self.normalizer(img)
        mask = read_mask(mask_path) if mask_path is not None else None
        return img, mask


def get_multispectral_channel_mask(bands: Optional[Sequence[int]] = None) -> np.ndarray:
    """Return a (1, 1, C) float32 mask with 1.0 for multispectral bands (B1-B12) and 0.0 for Slope/DEM."""
    selected = list(range(NUM_BANDS)) if bands is None else list(bands)
    for b in selected:
        if b < 0 or b >= NUM_BANDS:
            raise ValueError(f"Band index {b} out of range [0, {NUM_BANDS - 1}]")
    mask = [1.0 if b in MULTISPECTRAL_BAND_INDICES else 0.0 for b in selected]
    return np.asarray(mask, dtype=np.float32).reshape(1, 1, -1)


def resolve_dual_branch_indices(
    channels: int,
    bands: Optional[Sequence[int]] = None,
    branch1_indices: Optional[Sequence[int]] = None,
    branch2_indices: Optional[Sequence[int]] = None,
) -> tuple[list[int], Optional[list[int]]]:
    """Resolve input tensor channel indices for dual-branch vs single-branch encoding.

    When `bands` is provided, splits tensor positions into multispectral (B1-B12)
    and topographic (Slope, DEM) branches; falls back to single-encoder mode if
    only one modality is present. When `bands` is None and `channels == 14`, splits
    channels 0..11 vs 12..13.
    """
    if channels < 1:
        raise ValueError(f"Input must have at least 1 channel, got {channels}.")

    if branch1_indices is not None or branch2_indices is not None:
        if not branch1_indices or not branch2_indices:
            raise ValueError(
                "Both branch1_indices and branch2_indices must be non-empty when overriding branches."
            )
        b1 = [int(i) for i in branch1_indices]
        b2 = [int(i) for i in branch2_indices]
        for idx in b1 + b2:
            if idx < 0 or idx >= channels:
                raise ValueError(f"Channel index {idx} out of range [0, {channels - 1}].")
        return b1, b2

    if bands is not None:
        selected = [int(b) for b in bands]
        if len(selected) != channels:
            raise ValueError(
                f"Length of bands ({len(selected)}) does not match input channels ({channels})."
            )
        for b in selected:
            if b < 0 or b >= NUM_BANDS:
                raise ValueError(f"Band index {b} out of range [0, {NUM_BANDS - 1}].")
        ms_pos = [pos for pos, b in enumerate(selected) if b in MULTISPECTRAL_BAND_INDICES]
        topo_pos = [pos for pos, b in enumerate(selected) if b in TOPOGRAPHIC_BAND_INDICES]
        if ms_pos and topo_pos:
            return ms_pos, topo_pos
        return list(range(channels)), None

    if channels == NUM_BANDS:
        return list(MULTISPECTRAL_BAND_INDICES), list(TOPOGRAPHIC_BAND_INDICES)
    if channels <= 3:
        return list(range(channels)), None
    return [0, 1, 2], list(range(3, channels))


def make_tf_dataset(
    ds: Landslide4SenseDataset,
    batch_size: int,
    shuffle: bool = False,
    augment: bool = False,
    cache: bool | str = False,
    seed: Optional[int] = None,
    drop_remainder: bool = False,
    color_jitter: bool = True,
    color_brightness: float = 0.15,
    color_contrast: float = 0.15,
    color_prob: float = 0.5,
):
    """Build a tf.data.Dataset from a Landslide4SenseDataset."""
    import tensorflow as tf

    if len(ds) == 0:
        raise ValueError("Cannot build tf.data.Dataset from an empty dataset.")
    if not ds.require_masks or any(mask_p is None for _, mask_p in ds.samples):
        raise ValueError("make_tf_dataset requires masks on all samples.")
    if color_brightness < 0.0:
        raise ValueError(f"color_brightness must be >= 0.0, got {color_brightness}")
    if not (0.0 <= color_contrast <= 1.0):
        raise ValueError(f"color_contrast must be in [0.0, 1.0], got {color_contrast}")
    if not (0.0 <= color_prob <= 1.0):
        raise ValueError(f"color_prob must be in [0.0, 1.0], got {color_prob}")

    height, width, channels = ds.image_shape
    ms_mask_np = get_multispectral_channel_mask(ds.bands)
    apply_color_jitter = bool(
        color_jitter
        and np.any(ms_mask_np)
        and color_prob > 0.0
        and (color_brightness > 0.0 or color_contrast > 0.0)
    )
    ms_mask_tf = tf.constant(ms_mask_np > 0.5, dtype=tf.bool)

    def _py_load(idx_tensor) -> tuple[np.ndarray, np.ndarray]:
        img, mask = ds[int(idx_tensor)]
        if mask is None:
            raise ValueError("Sample returned None mask.")
        return img, mask

    def _load(idx):
        img, mask = tf.numpy_function(_py_load, [idx], [tf.float32, tf.float32])
        img.set_shape((height, width, channels))
        mask.set_shape((height, width, 1))
        return img, mask

    def _augment(img, mask):
        pair = tf.concat([img, mask], axis=-1)
        pair = tf.image.random_flip_left_right(pair, seed=seed)
        pair = tf.image.random_flip_up_down(pair, seed=seed)
        k = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32, seed=seed)
        pair = tf.image.rot90(pair, k=k)
        aug_img = pair[..., :channels]
        aug_mask = pair[..., channels:]

        if apply_color_jitter:
            def _perturb_color():
                perturbed = aug_img
                if color_contrast > 0.0:
                    ch_mean = tf.reduce_mean(perturbed, axis=(0, 1), keepdims=True)
                    contrast_factor = tf.random.uniform(
                        shape=(1, 1, channels),
                        minval=1.0 - color_contrast,
                        maxval=1.0 + color_contrast,
                        dtype=tf.float32,
                        seed=seed,
                    )
                    perturbed = (perturbed - ch_mean) * contrast_factor + ch_mean
                if color_brightness > 0.0:
                    brightness_delta = tf.random.uniform(
                        shape=(1, 1, channels),
                        minval=-color_brightness,
                        maxval=color_brightness,
                        dtype=tf.float32,
                        seed=seed,
                    )
                    perturbed = perturbed + brightness_delta
                return tf.where(ms_mask_tf, perturbed, aug_img)

            if color_prob >= 1.0:
                aug_img = _perturb_color()
            else:
                do_jitter = (
                    tf.random.uniform(shape=(), minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed)
                    < color_prob
                )
                aug_img = tf.cond(do_jitter, _perturb_color, lambda: aug_img)

        aug_img.set_shape((height, width, channels))
        aug_mask.set_shape((height, width, 1))
        return aug_img, aug_mask

    pipeline = tf.data.Dataset.range(len(ds))
    pipeline = pipeline.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if isinstance(cache, str) and cache:
        pipeline = pipeline.cache(cache)
    elif cache is True:
        pipeline = pipeline.cache()

    if shuffle:
        pipeline = pipeline.shuffle(
            buffer_size=len(ds),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    if augment:
        pipeline = pipeline.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    pipeline = pipeline.batch(batch_size, drop_remainder=drop_remainder)
    pipeline = pipeline.prefetch(tf.data.AUTOTUNE)
    return pipeline


def load_split_arrays(
    root: str | Path,
    split: str,
    bands: Optional[Sequence[int]] = None,
    normalizer: Optional[Normalizer] = None,
    id_list: Optional[str | Path] = None,
    exclude_list: Optional[str | Path] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an entire split into in-memory numpy arrays (X, Y)."""
    ds = Landslide4SenseDataset(
        root=root,
        split=split,
        bands=bands,
        normalizer=normalizer,
        id_list=id_list,
        exclude_list=exclude_list,
        require_masks=True,
    )
    height, width, channels = ds.image_shape
    x_arr = np.empty((len(ds), height, width, channels), dtype=np.float32)
    y_arr = np.empty((len(ds), height, width, 1), dtype=np.float32)
    for i in range(len(ds)):
        img, mask = ds[i]
        x_arr[i] = img
        y_arr[i] = mask
    return x_arr, y_arr


def compute_band_stats(ds: Iterable[tuple[np.ndarray, Optional[np.ndarray]]]) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-band mean and standard deviation in float64 across an unnormalized dataset."""
    total_sum: Optional[np.ndarray] = None
    total_sq: Optional[np.ndarray] = None
    total_pixels = 0

    for img, _ in ds:
        arr = np.asarray(img, dtype=np.float64)
        flat = arr.reshape(-1, arr.shape[-1])
        if total_sum is None:
            total_sum = np.zeros(flat.shape[1], dtype=np.float64)
            total_sq = np.zeros(flat.shape[1], dtype=np.float64)
        total_sum += flat.sum(axis=0)
        total_sq += np.square(flat).sum(axis=0)
        total_pixels += flat.shape[0]

    if total_sum is None or total_sq is None or total_pixels == 0:
        raise ValueError("Dataset is empty; cannot compute band statistics.")

    mean = total_sum / total_pixels
    var = np.maximum(total_sq / total_pixels - np.square(mean), 0.0)
    std = np.sqrt(var)
    return mean, std
