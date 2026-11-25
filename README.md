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
│   ├── train_teacher.yaml         # SAM2-UNet teacher (RGB) PyTorch training config
│   ├── train_rgb.yaml             # RGB-only Attention U-Net baseline config
│   ├── train_kd.yaml              # Output-level DIST KD (SAM2-UNet -> Attention U-Net) config
│   ├── train_kd_f32.yaml          # Wider Attention U-Net (filters=32) DIST KD config
│   ├── finetune_pruned_kd.yaml    # KD fine-tuning config for HRank-pruned Attention U-Net
│   └── grid_search.yaml           # Hyperparameter grid search config
├── uag/
│   ├── __init__.py
│   ├── config.py                  # Shared CLI arguments
│   ├── losses.py                  # dice, bce_dice, bce, structure, tversky, focal_tversky, dist_loss_binary
│   ├── metrics.py                 # Dataset-level precision, recall, f1_score, iou
│   ├── region_metrics.py          # Family 1 region sufficiency suite + Family 2 standard overlap
│   ├── utils.py                   # Seeding, JSON I/O, plotting, band parsing
│   ├── data/
│   │   ├── __init__.py
│   │   └── landslide4sense.py     # Landslide4Sense loader, Normalizer, teacher logit loader, tf.data pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── attention_unet.py      # Attention U-Net + RepeatChannels layer (supports channel_cfg)
│   │   ├── dual_branch.py         # Dual-branch Attention U-Net (MS + Slope/DEM fusion)
│   │   ├── SAM2UNet.py            # SAM2-UNet PyTorch teacher model
│   │   ├── sam2/                  # Vendored SAM2 backbone & utilities
│   │   ├── sam2_configs/          # SAM2 Hiera Hydra configs
│   │   └── unet.py                # Standard U-Net without attention gates
│   ├── pruning/
│   │   ├── __init__.py
│   │   └── hrank.py               # HRankPlus-style feature-rank scoring & coupled weight slicing
│   └── teacher/
│       ├── __init__.py
│       ├── data.py                # PyTorch Dataset wrapper with geometric augmentation
│       ├── losses.py              # PyTorch PraNet/SAM2-UNet structure loss
│       ├── metrics.py             # PyTorch dataset-level confusion metrics
│       └── model.py               # Teacher builders & trainable-only checkpoint helpers
├── train.py                       # Single training run (train + val, optional test eval)
├── train_teacher.py               # SAM2-UNet PyTorch teacher training + logit export
├── export_teacher_logits.py       # Export teacher logits to HDF5 for any trained teacher run
├── train_kd.py                    # Attention U-Net student training with offline DIST KD
├── prune_student.py               # HRank structured filter pruning for Attention U-Net
├── test.py                        # Checkpoint evaluation on any split
├── grid_search.py                 # Hyperparameter grid search + results CSV
├── predict.py                     # Batch inference, RGB/mask panels, optional HDF5 masks
├── compute_stats.py               # Streaming per-band mean/std calculation
├── requirements.txt
└── requirements-teacher.txt
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

## SAM2-UNet Teacher + DIST Distillation (RGB)

Install teacher dependencies (PyTorch + SAM2 dependencies):

```bash
pip install -r requirements-teacher.txt
```

Download the pretrained SAM2 Hiera-Large checkpoint into `checkpoints/`:

```bash
mkdir -p checkpoints
wget -O checkpoints/sam2_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

Run the three-stage RGB experiment workflow (`bands: [3, 2, 1]`, i.e., B4/B3/B2):

1. Train the SAM2-UNet teacher and export un-augmented main-head logits (`teacher_logits_{train,val,test}.h5`):

```bash
python train_teacher.py --config configs/train_teacher.yaml
```

2. Train the RGB Attention U-Net baseline (no distillation):

```bash
python train.py --config configs/train_rgb.yaml
```

3. Train the Attention U-Net student with offline output-level DIST distillation:

```bash
python train_kd.py --config configs/train_kd.yaml
```

To re-export teacher logits from an existing teacher run directory:

```bash
python export_teacher_logits.py --run-dir runs/sam2unet_l_rgb
```

### Distillation Notes

- **Loss formulation**: `train_kd.py` minimizes `task_loss(mask, y_pred) + kd_weight * dist_loss_binary(y_pred, teacher_logit, tau=kd_tau, beta=kd_beta, gamma=kd_gamma)`. Student sigmoid probabilities are converted back to binary logits `z_s = log(p) - log(1 - p)`, embedded as 2-class logits `[0, z] / tau`, softmaxed, and compared via DIST's inter-class (`beta`) and intra-class (`gamma`) Pearson correlation losses scaled by `tau^2`.
- **Teacher logits**: Exported once from clean, un-augmented `128x128` inputs (teacher internally resizes to `input_size`, e.g., `352`, and resizes logits back to `128x128`). During student training, random flips and 90-degree rotations are applied jointly to images, masks, and teacher logits.
- **Binary DIST caveat**: With a single binary sigmoid head expanded to 2 classes `[0, z]`, the 2-class row-centered Pearson correlation in the inter-class term collapses to approximately sign agreement (`+-1`) with near-zero gradient; the intra-class term (pixel-wise correlation across the batch) provides the main distillation signal. `kd_beta` defaults to `1.0` to match the reference `DIST_KD` defaults, and can be set to `0.0` via `--kd-beta 0` or `kd_beta: 0.0`.

## Structured Pruning Experiment (HRankPlus-style)

As a separate optimization experiment on a wider `filters: 32` RGB student:

1. Train the wider `filters=32` student with DIST KD (reusing the exported teacher logits in `runs/sam2unet_l_rgb`):

```bash
python train_kd.py --config configs/train_kd_f32.yaml
```

2. Score feature-map ranks (HRank) and prune coupled Attention U-Net channels into `runs/pruned_kd_attunet_rgb_f32` (using `--center-rate` for the `8x8` bottleneck where matrix rank is capped at 8):

```bash
python prune_student.py --run-dir runs/kd_attunet_rgb_f32 --compress-rate 0.5 --center-rate 0.25 --eval-val
```

3. Fine-tune the pruned student with KD loss at a lower learning rate (`lr: 2e-4`, recommended):

```bash
python train_kd.py --config configs/finetune_pruned_kd.yaml
```

4. Optional ablations (task-loss-only fine-tuning and small-from-scratch baseline matching the pruned architecture):

```bash
python train.py --config configs/train_rgb.yaml --filters 32 --lr 2e-4 --epochs 100 --patience 20 --pretrained-weights runs/pruned_kd_attunet_rgb_f32/pruned_init.weights.h5 --channel-cfg runs/pruned_kd_attunet_rgb_f32/channel_cfg.json --run-name kd_attunet_rgb_f32_pruned_task_ft
python train.py --config configs/train_rgb.yaml --filters 32 --channel-cfg runs/pruned_kd_attunet_rgb_f32/channel_cfg.json --run-name attunet_rgb_f32_scratch_small
```

## Evaluation Metrics

`test.py`, `train.py` (`--eval-test`), and `train_teacher.py` report two metric families into `metrics_<split>.json`, `test_metrics.json`, and `val_metrics.json`:

1. **Family 1 — Region-Level Sufficiency Suite (`region_sufficiency`)**: Reported as three separate components over positive patches (`|G| > 0`), never collapsed into a single score (defaults `--iou-tau 0.5`, `--dilation-alpha 0.1`):
   - **Tolerant IoU (`tolerant_iou_rate`)**: Fraction of positive patches with `|P ∩ G| / |P ∪ G| >= tau`.
   - **Centroid Hit (`centroid_hit_rate`)**: Fraction of positive patches where predicted mean coordinate `c(P)` falls inside `G`.
   - **Size-Scaled Dilated Recall (`size_scaled_dilated_recall`)**: Mean over positive patches of `|G ∩ (P ⊕ D_r)| / |G|` using Euclidean disk radius `r(G) = ceil(alpha * 2 * sqrt(|G| / pi))`.
   - Negative patches (`|G| == 0`) are excluded from the three sufficiency averages and tracked separately via `num_negative_patches`, `empty_patch_fp_count`, and `empty_patch_fp_rate`.
2. **Family 2 — Standard Overlap (`standard_overlap`)**: Dataset-level pixel `iou`, `dice_f1` (`2|P∩G|/(|P|+|G|)`), `precision`, and `recall` for literature comparability.
3. **RQ5 Boundary Comparators (`boundary_comparators`), Size Stratification (`size_stratified`), and `model_cost`**: Evaluation outputs also include `boundary_iou_positive_mean`, `s_measure_all_mean` / `s_measure_positive_mean`, positive-patch breakdowns across `small` (`< 100` px), `medium` (`100–500` px), `large` (`> 500` px), and `medium_large` (`>= 100` px), per-patch CSV exports (`per_patch_<split>.csv`), and checkpoint/architecture cost (`model_cost`).
