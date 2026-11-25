# Experimental Design: Region-Level Sufficiency in Distilled and Pruned Landslide Detection

This document specifies the experimental matrix, hypothesis tests, execution protocol, and result templates designed to answer the primary research question and sub-questions **RQ1–RQ5** defined in [draft.md](draft.md).

---

## 1. Experimental Matrix (2×2 Factorial Design + Reference Baselines)

To isolate the individual effects of **Knowledge Distillation (KD)** and **Structured Optimization (HRank Pruning + Fine-Tuning)** as well as their **KD × Optimization interaction**, the core study uses a balanced $2 \times 2$ factorial design on the `filters = 32` RGB Attention U-Net student, supplemented by the teacher reference, a capacity-matched scratch baseline (for **RQ4**), and targeted ablations.

| Arm ID | Model / Variant | Distillation | Optimization (Pruning) | Purpose / Role |
| :--- | :--- | :--- | :--- | :--- |
| **T0** | SAM2-UNet (`sam2_hiera_l`) | — (Teacher) | Unpruned (`352 × 352` internal) | Upper-bound teacher reference & offline logit source |
| **S1** | Attention U-Net (`f=32`) | None (Task-only) | Unpruned | No-KD, Unpruned factorial cell |
| **S2** | Attention U-Net (`f=32`) | Output-level DIST KD | Unpruned | KD, Unpruned factorial cell |
| **S3** | Attention U-Net (`f=32`) | None (Task-only) | HRank pruned (`ρ=0.5`, `ρ_center=0.25`) + Task FT | No-KD, Pruned factorial cell |
| **S4** | Attention U-Net (`f=32`) | Output-level DIST KD | HRank pruned (`ρ=0.5`, `ρ_center=0.25`) + KD FT | **Primary candidate** (KD, Pruned factorial cell) |
| **S5** | Attention U-Net (matched `channel_cfg`) | None (Task-only) | Trained from scratch (no pruning) | **RQ4 baseline** (same params/FLOPs as S3/S4) |
| **S4-TaskFT** | Attention U-Net (`f=32`) | KD before pruning only | HRank pruned from S2 + Task-only FT | Fine-tuning ablation (isolates KD role during recovery) |

### Supplementary Sensitivity & Mechanism Ablations

- **A1 — Compression Rate Sweep**: Repeat pruning on **S2** at uniform rates $\rho \in \{0.30, 0.50, 0.70\}$ (with `center_rate = min(ρ, 0.25)` or proportional scaling) followed by KD fine-tuning to trace the sufficiency-vs-FLOPs Pareto frontier.
- **A2 — Binary DIST Inter-Class Ablation**: Compare **S2** and **S4** with `kd_beta = 1.0` (default) versus `kd_beta = 0.0` (intra-class term only) to empirically verify the analytical observation that binary 2-class row-centered correlation is dominated by $\gamma$.

> **[Figure E.1 Placeholder]**
> *Visual schematic of the $2 \times 2$ factorial design (`{No-KD, DIST-KD} × {Unpruned, HRank-Pruned + FT}`) alongside the teacher reference (`T0`), capacity-matched scratch baseline (`S5`), and fine-tuning ablation (`S4-TaskFT`), showing which pairwise contrasts map to RQ1–RQ5.*

---

## 2. Mapping Research Questions to Experimental Contrasts

| Question | Formal Contrast / Test | Decision Criterion & Interpretation |
| :--- | :--- | :--- |
| **RQ1** (Isolated effect of KD before pruning) | **S2 − S1** | Positive $\Delta$ on Family 1 (Tolerant IoU, Centroid Hit, Size-Scaled Dilated Recall) and Family 2 (IoU, Dice/F1) establishes pre-pruning distillation benefit. |
| **RQ2** (Isolated effect of pruning) | **S3 − S1** (without KD) and **S4 − S2** (with KD) | Quantifies accuracy/sufficiency change relative to parameter, FLOP, and checkpoint-size reductions. |
| **RQ3** (KD × Optimization interaction) | Interaction term: $\Delta_{\text{int}} = (\mathbf{S4} - \mathbf{S3}) - (\mathbf{S2} - \mathbf{S1})$ | $\Delta_{\text{int}} > 0$ demonstrates that distillation benefits a pruned student more than an unpruned one (synergy); $\Delta_{\text{int}} < 0$ indicates pruning erases distilled gains. Compare **S4** vs **S4-TaskFT** to test whether post-pruning KD recovery drives survival. |
| **RQ4** (Necessity of teacher vs small scratch) | **S4 − S5** (and **S3 − S5** for reference) | If **S4** significantly outperforms **S5** at identical `channel_cfg` (identical params and FLOPs), distillation transfers knowledge a compact model cannot learn directly from labels. |
| **RQ5** (Metric suite discrimination) | Rank orderings across `{T0, S1, S2, S3, S4, S4-TaskFT, S5}` + size-stratified analysis | Compute Spearman's $\rho$ and Kendall's $\tau$ rank correlations between Family 1 components, Family 2 (IoU / Dice), and literature boundary metrics (Boundary IoU, Structure-measure). Divergent rankings or different sensitivity across landslide size bins (small vs. medium vs. large) answer RQ5. |
| **Primary RQ** | Joint satisfaction of **RQ1–RQ5** | Distillation is justified for edge deployment if **S4 > S3**, **S4 > S5**, and **S4** satisfies pre-specified sufficiency thresholds at substantially reduced FLOPs/params. |

> **Implementation Note for RQ5**: The repository currently implements **Family 1** (`region_sufficiency`) and **Family 2** (`standard_overlap`) in `uag/region_metrics.py`. As noted in `draft.md`, Boundary IoU (Cheng et al., 2021) and Structure-measure (Fan et al., 2017) are cited for the RQ5 ranking comparison; computing those two comparator scores will require a small post-hoc evaluation helper over saved HDF5 prediction masks (`predict.py --save-masks`) or inside `uag/region_metrics.py`.

---

## 3. Controlled Variables, Pre-Experiment Fixed Constants & Statistical Protocol

### Fixed Constants (Committed Prior to Test Evaluation)
- **Input Modality**: Optical RGB only (`bands: [3, 2, 1]` → `B4, B3, B2`), `norm: l4s`, `blacklist: datasets/landslide4senses/blacklist.txt`.
- **Teacher**: `SAM2UNet` (`sam2_hiera_l.yaml`, `sam2_hiera_large.pt`, `input_size: 352`, `epochs: 50`, `batch_size: 12`, `lr: 1e-3`, `weight_decay: 1e-4`, `augment: true`).
- **Student Backbone**: `attention_unet`, `filters: 32`, `loss: dice`, `batch_size: 8`, `epochs: 200` (`patience: 30`), `lr: 1e-3`, `augment: true`, `color_jitter: true`.
- **Distillation**: `kd_weight: 1.0`, `kd_tau: 1.0`, `kd_beta: 1.0`, `kd_gamma: 1.0`, `monitor: val_task_loss`.
- **Pruning & Fine-Tuning**: `compress_rate: 0.5`, `center_rate: 0.25`, `min_channels: 2`, `hrank_batches: 10`, fine-tuning `lr: 2e-4`, `epochs: 100`, `patience: 20`.
- **Evaluation Constants**: Binary threshold `0.5`, Tolerant IoU $\tau = 0.5$, Size-Scaled Dilated Recall $\alpha = 0.1$.

### Multi-Seed & Statistical Protocol
- Run each student arm (**S1, S2, S3, S4, S5, S4-TaskFT**) across **$K = 3$ independent random seeds** (recommended: `seed ∈ {42, 123, 2026}`; optionally `5` seeds if variance is high) using identical split definitions and cached teacher logits from **T0**.
- Report **mean ± standard deviation** across seeds for every metric.
- For patch-level comparisons within a seed (and aggregated across seeds), perform **paired Wilcoxon signed-rank tests** (for continuous `size_scaled_dilated_recall` and per-patch IoU) and **McNemar's test** (for binary per-patch `tolerant_iou` and `centroid_hit` outcomes) on positive test patches, plus **stratified reporting** by ground-truth landslide area $|G|$ (e.g., Small: $< 100\text{ px}$, Medium: $100\text{–}500\text{ px}$, Large: $> 500\text{ px}$, aligned with the scope in §1.2.2).

---

## 4. Reproducible Execution Sequence (Per Seed `$SEED`)

### Step 0 — Teacher Training & Logit Export (Once)
```bash
# Download pretrained SAM2 Hiera-Large checkpoint into checkpoints/
mkdir -p checkpoints
wget -O checkpoints/sam2_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# Train teacher T0 and export clean 128x128 logits (teacher_logits_{train,val,test}.h5)
python train_teacher.py --config configs/train_teacher.yaml --seed 42
```

### Step 1 — Unpruned Student Arms (`S1` and `S2`)
```bash
# S1: No-KD, Unpruned (filters=32)
python train.py --config configs/train_rgb.yaml --filters 32 --seed $SEED --run-name s1_nokd_f32_seed${SEED}

# S2: DIST-KD, Unpruned (filters=32)
python train_kd.py --config configs/train_kd_f32.yaml --seed $SEED --run-name s2_kd_f32_seed${SEED}
```

### Step 2 — HRank Structured Pruning (`S1 -> S3_init`, `S2 -> S4_init`)
```bash
# Prune No-KD student S1
python prune_student.py --run-dir runs/s1_nokd_f32_seed${SEED} --compress-rate 0.5 --center-rate 0.25 --eval-val --output-dir runs/pruned_s1_seed${SEED}

# Prune KD student S2
python prune_student.py --run-dir runs/s2_kd_f32_seed${SEED} --compress-rate 0.5 --center-rate 0.25 --eval-val --output-dir runs/pruned_s2_seed${SEED}
```

### Step 3 — Recovery Fine-Tuning & Scratch-Small Baseline (`S3`, `S4`, `S4-TaskFT`, `S5`)
```bash
# S3: No-KD, Pruned + Task-only FT
python train.py --config configs/train_rgb.yaml --filters 32 --lr 2e-4 --epochs 100 --patience 20 --pretrained-weights runs/pruned_s1_seed${SEED}/pruned_init.weights.h5 --channel-cfg runs/pruned_s1_seed${SEED}/channel_cfg.json --seed $SEED --run-name s3_nokd_pruned_seed${SEED}

# S4: DIST-KD, Pruned + KD FT (recommended)
python train_kd.py --config configs/finetune_pruned_kd.yaml --pretrained-weights runs/pruned_s2_seed${SEED}/pruned_init.weights.h5 --channel-cfg runs/pruned_s2_seed${SEED}/channel_cfg.json --seed $SEED --run-name s4_kd_pruned_seed${SEED}

# S4-TaskFT: Ablation — KD before pruning, task-only FT after pruning
python train.py --config configs/train_rgb.yaml --filters 32 --lr 2e-4 --epochs 100 --patience 20 --pretrained-weights runs/pruned_s2_seed${SEED}/pruned_init.weights.h5 --channel-cfg runs/pruned_s2_seed${SEED}/channel_cfg.json --seed $SEED --run-name s4_taskft_seed${SEED}

# S5: RQ4 Baseline — Trained from scratch at identical pruned channel_cfg
python train.py --config configs/train_rgb.yaml --filters 32 --channel-cfg runs/pruned_s2_seed${SEED}/channel_cfg.json --seed $SEED --run-name s5_scratch_small_seed${SEED}
```

### Step 4 — Evaluation & Prediction Mask Export
```bash
# Re-evaluate any run on test split with fixed pre-experiment (tau=0.5, alpha=0.1)
python test.py --run-dir runs/<run_name> --split test --iou-tau 0.5 --dilation-alpha 0.1

# Export visual panels + uint8 HDF5 prediction masks for qualitative analysis and RQ5 boundary comparators
python predict.py --run-dir runs/<run_name> --split test --num-samples -1 --save-masks
```

---

## 5. Planned Results Tables & Figure Placeholders

### Table 1: Main Quantitative Comparison (Test Split, Mean ± SD across $K$ Seeds)

| Arm | KD | Pruned | Params | FLOPs | Size (MB) | Tol. IoU ($\tau=0.5$) | Centroid Hit | Dilated Recall ($\alpha=0.1$) | Empty-Patch FP | Std. IoU | Dice / F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T0** (SAM2-UNet) | — | No | — | — | — | — | — | — | — | — | — |
| **S1** (No-KD, Full) | No | No | — | — | — | — | — | — | — | — | — |
| **S2** (KD, Full) | Yes | No | — | — | — | — | — | — | — | — | — |
| **S3** (No-KD, Pruned) | No | Yes | — | — | — | — | — | — | — | — | — |
| **S4** (KD, Pruned) | Yes | Yes | — | — | — | — | — | — | — | — | — |
| **S4-TaskFT** (Abl.) | Pre | Yes | — | — | — | — | — | — | — | — | — |
| **S5** (Scratch Small) | No | Match | — | — | — | — | — | — | — | — | — |

### Table 2: Size-Stratified Region Sufficiency (Test Positive Patches by Landslide Area $|G|$)

| Stratum | Patch Count | Arm | Tol. IoU ($\tau=0.5$) | Centroid Hit | Dilated Recall ($\alpha=0.1$) | Std. IoU |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| Small ($|G| < 100$) | — | S1 / S2 / S3 / S4 / S5 | — | — | — | — |
| Medium ($100 \le |G| \le 500$) | — | S1 / S2 / S3 / S4 / S5 | — | — | — | — |
| Large ($|G| > 500$) | — | S1 / S2 / S3 / S4 / S5 | — | — | — | — |

### Table 3: RQ5 Metric-Suite Rank Correlation Matrix (Spearman $\rho$ / Kendall $\tau$ across All Student Arms)

| Metric | Tolerant IoU | Centroid Hit | Dilated Recall | Std. IoU | Dice / F1 | Boundary IoU | S-measure |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tolerant IoU** | 1.00 | — | — | — | — | — | — |
| **Centroid Hit** | — | 1.00 | — | — | — | — | — |
| **Dilated Recall** | — | — | 1.00 | — | — | — | — |
| **Std. IoU / Dice** | — | — | — | 1.00 | 1.00 | — | — |

### Visual & Diagnostic Figure Placeholders

- **[Figure E.2 Placeholder — KD × Pruning Interaction Plot]**: Two-line interaction plot (x-axis: `Unpruned` vs. `HRank-Pruned + FT`; separate lines for `No-KD` and `DIST-KD` with error bars across seeds) for each of the three Family 1 components and Standard IoU, visually testing **RQ3** ($\Delta_{\text{int}}$).
- **[Figure E.3 Placeholder — Efficiency–Sufficiency Pareto Frontier]**: Scatter/line plot of FLOPs (or parameter count) on the x-axis versus Region Sufficiency components (`tolerant_iou_rate`, `centroid_hit_rate`, `size_scaled_dilated_recall`) on the y-axis across **S1–S5**, **S4-TaskFT**, and compression-rate sweep **A1** ($\rho \in \{0.3, 0.5, 0.7\}$).
- **[Figure E.4 Placeholder — Layer-Wise HRank Profiles & Width Reduction]**: Bar chart comparing average feature-map ranks $\bar{r}_c$ and surviving channel counts across the `17` prunable units (`conv1_1` through `inter_3`), illustrating the rank cap at the `8 × 8` bottleneck (`center_1`, `center_2`).
- **[Figure E.5 Placeholder — Qualitative Prediction Panels]**: Side-by-side `RGB | Ground Truth | T0 | S1 | S2 | S3 | S4 | S5` panels from `predict.py` showing: (a) a medium/large scar where **S4** achieves centroid hit and high dilated recall despite coarse boundaries, (b) a spectral-confusion case, and (c) a negative patch illustrating empty-patch false alarms.
- **[Figure E.6 Placeholder — RQ5 Metric Ranking Comparison]**: Bump chart / rank-order plot comparing student rankings under Family 1 (Region Sufficiency), Family 2 (IoU / Dice), and boundary-precision comparators (Boundary IoU, S-measure).

---

## 6. Threats to Validity and Mitigations

1. **Offline Clean-Logit Approximation**: Teacher logits are cached once from un-augmented RGB images; during student training, geometric transforms (`flip`, `rot90`) are applied identically to logits, whereas optical color jitter is applied only to the input image. This treats teacher predictions as invariant under mild photometric jitter, which is realistic for semantic segmentation teachers and identical across **S2** and **S4**.
2. **Binary DIST Inter-Class Collapse**: As analyzed in Chapter II (§2.4), two-class `[0, z]` embedding makes the inter-class term degenerate to approximate sign matching. Running ablation **A2** (`kd_beta = 0.0` vs. `1.0`) verifies that intra-class correlation ($\gamma$) drives distillation performance.
3. **Bottleneck Rank Saturation (`8 × 8`)**: Applying uniform aggressive pruning to `center_1`/`center_2` where max rank is `8` risks noisy filter selection. Fixing `--center-rate 0.25` prior to test evaluation mitigates bottleneck collapse.
4. **Architecture-Dependent Scratch Baseline (`S5`)**: Because **S1** and **S2** may yield slightly different pruned `channel_cfg` layouts under HRank, **S5** must be instantiated using **S4**'s exact `channel_cfg.json` so **RQ4** compares equal parameter/FLOP capacity directly.
5. **Simulation-Only Efficiency**: Consistent with §1.2.2 scope, deployment cost is quantified via parameters, Conv2D/Conv2DTranspose FLOPs, and `.weights.h5` file size rather than on-device latency or power measurements.
