# Attention U-Net for Landslide4Sense

Modular TensorFlow/Keras implementation refactored from `Attention_Unet_v1.ipynb`, paired with a Landslide4Sense HDF5 data loader. Works with both `tf.keras` 2 (TF <= 2.15) and Keras 3 (TF >= 2.16) without `segmentation_models`.

## Project Layout

```text
.
├── configs/
│   ├── train_default.yaml         # Default single-run training config
│   ├── train_unet.yaml            # Standard U-Net (no attention gates) training config
│   ├── train_dual_branch.yaml     # Dual-branch Attention U-Net (MS + Topo) config
│   ├── train_class1.yaml          # Landslide-only (class1) training variant
│   └── grid_search.yaml           # Hyperparameter grid search config
├── uag/
│   ├── __init__.py
│   ├── config.py                  # Shared CLI arguments
│   ├── losses.py                  # dice, bce_dice, bce, structure, tversky, focal_tversky losses
│   ├── metrics.py                 # Dataset-level precision, recall, f1_score, iou
│   ├── utils.py                   # Seeding, JSON I/O, plotting, band parsing
│   ├── data/
│   │   ├── __init__.py
│   │   └── landslide4sense.py     # Landslide4Sense loader, Normalizer, tf.data pipeline
│   └── models/
│       ├── __init__.py
│       ├── attention_unet.py      # Attention U-Net + RepeatChannels layer
│       ├── dual_branch.py         # Dual-branch Attention U-Net (MS + Slope/DEM fusion)
│       └── unet.py                # Standard U-Net without attention gates
├── train.py                       # Single training run (train + val, optional test eval)
├── test.py                        # Checkpoint evaluation on any split
├── grid_search.py                 # Hyperparameter grid search + results CSV
├── predict.py                     # Batch inference, RGB/mask panels, optional HDF5 masks
├── compute_stats.py               # Streaming per-band mean/std calculation
└── requirements.txt
```

## Expected Dataset Structure

```text
datasets/landslide4senses/
├── images/
│   ├── train/image_<k>.h5         # key 'img', shape (128, 128, 14)
│   ├── val/image_<k>.h5           # or validation/
│   └── test/image_<k>.h5
├── annotations/
│   ├── train/mask_<k>.h5          # key 'mask', shape (128, 128)
│   ├── val/mask_<k>.h5            # or validation/
│   └── test/mask_<k>.h5
├── class1_train.txt               # optional landslide-positive sample ID filter
├── class1_validation.txt
└── blacklist.txt                  # optional sample indices (x in image_x.h5) to exclude
```

- Both `val/` and `validation/` directory names are supported automatically.
- `--bands` accepts comma-separated **0-based** band indices (e.g., `--bands 1,2,3,12,13`).
- Use `--blacklist datasets/landslide4senses/blacklist.txt` (or `blacklist:` in YAML/JSON configs) to skip listed sample indices `x` (`image_x.h5` / `mask_x.h5`) across `train`, `val`, and `test` splits (`test.py` and `predict.py` also inherit `blacklist` from the run's saved `config.json` unless overridden via `--blacklist`).

## Example Commands

Compute band statistics (optional; built-in L4S stats are used by default):

```bash
python compute_stats.py --data-root datasets/landslide4senses --split train --out stats.json
```

Train a single model:

```bash
python train.py --data-root datasets/landslide4senses --filters 8 --lr 1e-3 --batch-size 8 --epochs 200 --augment --eval-test
```

Evaluate a trained run:

```bash
python test.py --run-dir runs/attunet_size128_f8_bs8_lr0.001 --split test
```

Run the hyperparameter grid search:

```bash
python grid_search.py --data-root datasets/landslide4senses --output-dir runs/grid_search
```

Generate prediction visualizations and optional HDF5 masks:

```bash
python predict.py --run-dir runs/attunet_size128_f8_bs8_lr0.001 --split test --num-samples 20 --save-masks
```

## Configuration & Reproducibility

`train.py` and `grid_search.py` accept `--config` pointing to a `.yaml`/`.yml` or `.json` file. Option precedence is:

1. Argparse defaults
2. Values from `--config`
3. Explicit CLI flags (highest priority)

Examples:

```bash
python train.py --config configs/train_default.yaml
python train.py --config configs/train_unet.yaml
python train.py --config configs/train_default.yaml --loss structure
python train.py --config configs/train_default.yaml --loss focal_tversky
python train.py --config configs/train_default.yaml --lr 5e-4
python grid_search.py --config configs/grid_search.yaml
python grid_search.py --config configs/grid_search.yaml --model unet
```

## Loss Functions

Select the loss via `--loss` (or `loss:` in YAML/JSON configs):

- `dice` (default): Soft Dice loss.
- `bce_dice`: Binary cross-entropy + soft Dice loss.
- `bce`: Binary cross-entropy.
- `structure`: PraNet-style weighted BCE + weighted IoU with edge-aware pixel weights `1 + 5 * |avg_pool2d_31(mask) - mask|`, adapted for NHWC sigmoid probability outputs and per-pixel `reduction='none'` BCE weighting.
- `tversky`: `1 - Tversky index` with `alpha=0.7` on false negatives and `0.3` on false positives.
- `focal_tversky`: `(1 - Tversky)^0.75` (`alpha=0.7`, `gamma=0.75`).

## Model Architectures

Select the model architecture via `--model` (or `model:` in YAML/JSON configs):

- `attention_unet` (default): Attention U-Net with additive attention gates on skip connections (`attunet_*` run prefix).
- `unet`: Standard U-Net using the exact same encoder/decoder depth, filter counts, and BatchNorm blocks without attention gates (`unet_*` run prefix), suitable for ablation comparisons.
- `dual_branch`: Dual-branch Attention U-Net (`dualbranch_*` run prefix) that encodes multispectral bands (`B1`–`B12`) and topographic bands (`Slope`, `DEM`) in separate encoders and fuses features across all 5 resolution levels before the Attention U-Net decoder (falls back to single-encoder mode when only one modality is selected via `--bands`).

`test.py` and `predict.py` automatically read `model` from the run's saved `config.json` (overridable via `--model`).

Every training run writes `config.json`, `config.yaml`, and `environment.json` (Python, NumPy, h5py, TensorFlow/Keras versions, detected GPUs, and Git commit/dirty status) into its run directory, and grid search writes `grid_config.json` and `environment.json`. You can reproduce any past run with:

```bash
python train.py --config runs/<name>/config.yaml
# or using the saved JSON config directly:
python train.py --config runs/<name>/config.json
```

Pass `--deterministic` (or set `deterministic: true` in the config) to enable `tf.config.experimental.enable_op_determinism()` for bitwise-reproducible TensorFlow operations (at some runtime cost).

## Notes

- **Augmentation**: `--augment` applies joint random flips/90-degree rotations to all bands and masks, plus per-band brightness/contrast color perturbation exclusively to multispectral bands `B1`–`B12` (indices `0..11`), leaving `Slope` (`12`) and `DEM` (`13`) untouched. Control this behavior via `--no-color-jitter`, `--color-brightness`, `--color-contrast`, and `--color-prob` (or matching YAML/JSON config keys).
- **Metrics**: `uag/metrics.py` computes dataset-level (micro) pixel Precision, Recall, F1, and IoU by accumulating TP/FP/FN across all batches in an epoch, rather than averaging per-batch scores.
- **Checkpoints**: Saved weights use the `.weights.h5` suffix required by Keras 3 (`best.weights.h5`).
- **Legacy files**: The original `Attention_Unet_v1.ipynb`, `helper.py`, and `losses.py` were removed after refactoring and can be restored from git commit `88b4d3a` (e.g., `git checkout 88b4d3a -- Attention_Unet_v1.ipynb`).
