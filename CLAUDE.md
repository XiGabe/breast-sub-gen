# CLAUDE.md - Breast Subtraction Synthesis Project

## Project Overview

3D Contrast-Enhanced Breast MRI Synthesis using MAISI framework. Generate synthetic post-contrast MRI from pre-contrast inputs.

**Method**: `Pre-contrast → Model → 3D Subtraction Map → Post = Pre + Sub`

## Data Structure (`step_4/`)

- **Files**: `{UUID}_{side}_{pre|sub|mask}.nii.gz`
- **Spacing**: `[0.7, 0.7, 1.2]` mm³ (Z,Y,X), RAS orientation
- **Metadata**: `step_4_metadata.csv` with `Is_Flipped`, `Crop_BBox`, `Orig_Shape`
- **Note**: Right-side breasts are flipped to left (`Is_Flipped=1`)

## Cached Dataset

```
embeddings_breast_sub/     # Sub_emb (64³, 4ch) - VAE encoded
processed_pre/             # Pre_aligned (256³, 1ch) - controlnet condition
processed_mask/            # Mask_aligned (256³, 1ch) - loss weighting
dataset_breast_cached.json # Train/Val split (1553/390 samples)
```

## Training Pipeline

### Current Status (2026-02-27)

| Stage | Epochs | Unfrozen | Loss | Status |
|-------|--------|----------|------|--------|
| Stage 1 | 58-70 | up_blocks.2,3 | 0.86 | ✅ Complete |
| Stage 2 | 71-100 | +middle, up1 | 0.86 | ✅ Complete |
| Stage 2.5 | 101-107 | middle, up1, up2, up3 | Top-K | ✅ Complete |
| **Stage 3 (Failed)** | **108-130** | **ALL up + middle** | **Top-K + Std** | **❌ Background white** |
| **Stage 3 (Attempt 1)** | **108-111** | **ALL up + middle** | **Asymmetric Contrast** | **❌ Signal scattered** |
| **Stage 3 (Current)** | **108-130** | **ALL up + middle** | **MSE Contrast** | **🔄 Training** |

**Stage 2.5 Results** (Top-K Loss):
- Val Loss: 0.8526 (best at epoch 101)
- Inference: Tumor signal ~89% of GT intensity
- Background: Pure black ✅
- Issue: Tumor signal slightly weak, boundaries blurry

**Stage 3 (Failed) Analysis**:
- **Problem**: Background turned completely white (MAE 0.83, Pred mean 0.88)
- **Root Cause**: Top-K (30%) ignored 70% of background pixels + up_blocks.0 unfrozen = no constraint on background
- **up_blocks.0** (80M params) controls global image style but had NO loss constraint in background regions
- Std Loss only constrained variance, not mean intensity

**Stage 3 (Attempt 1) - ROI Intensity + Asymmetric Contrast**:
- **Problem**: Asymmetric contrast allowed model to "cheat" by spreading signal thinly
- **Root Cause**: `F.relu(gt-pred).mean()` only penalizes under-estimation, model satisfies it with many weak pixels
- **Result**: Pred mean 0.003 vs GT 0.048 (only 6%!), pixels > 0.1: 1.36% vs 12.4%
- **Status**: ❌ Signal extremely scattered, predictions nearly black

**Stage 3 (Current) - ROI Intensity + MSE Contrast Loss**:
- **Strategy**: Global L1 (10× tumor weight) + ROI Intensity + **MSE Contrast**
- **Global L1** (dominant): Background × 1.0 + Tumor × 10.0 → forces correct spatial distribution
- **ROI Intensity**: `|mean(pred_roi) - mean(gt_roi)|` → ensures overall brightness
- **MSE Contrast** (killer feature): `F.mse_loss(pred_roi, gt_roi)` → **squared penalty on outliers!**
  - Gradient: 2×(pred-gt) → stronger correction for large errors
  - Prevents model from "cheating" by spreading high signal into many weak pixels
  - If GT=1.0, Pred=0.1 → MSE gradient=-1.8 vs L1 gradient=-1.0
- **Unfreeze**: ALL up_blocks + middle_block (150M+ params)
- **Weights**: `weighted_loss=10.0`, `roi_intensity_weight=0.5`, `mse_weight=0.1`
- **Loss Balance**: Global L1 ~90%, ROI ~10%
- **Training Loss**: ~1.07 (stable)

### Key Implementation Details

**Stage 3 Failed Loss** (Top-K + Std Matching):
```python
# Top-K Loss: Focus on hardest 30% pixels ONLY
# Problem: Ignores 70% of pixels (mostly background!)
l1_loss_raw = F.l1_loss(pred, gt, reduction="none") * weights
l1_loss_flat = l1_loss_raw.view(B, -1)
k = int(l1_loss_flat.size(1) * 0.3)
topk_loss, _ = torch.topk(l1_loss_flat, k, dim=1)
loss_topk = topk_loss.mean()

# Std Matching Loss: Force overall contrast
# Problem: Only constrains variance, not mean intensity
std_pred = torch.std(model_output.float(), dim=[1, 2, 3, 4])
std_gt = torch.std(model_gt.float(), dim=[1, 2, 3, 4])
loss_std = F.l1_loss(std_pred, std_gt)

# Combined: up_blocks.0 has NO constraint on background → background turns white
loss = loss_topk + 1.0 * loss_std
```

**Stage 3 Final Loss** (Global L1 + ROI Intensity + MSE Contrast):
```python
# 1. Global Weighted L1 Loss (ALL pixels constrained!)
l1_loss_raw = F.l1_loss(model_output.float(), model_gt.float(), reduction="none")
loss_global = (l1_loss_raw * weights).mean()
# weights: background × 1.0, tumor × 10.0
# Effect: Dominant spatial constraint, prevents signal scattering

# 2. ROI Intensity + MSE Contrast Loss
roi_mask = (interpolate_label > 0.5)
if roi_mask.sum() > 0:
    roi_mask_expanded = roi_mask.repeat(1, images.shape[1], 1, 1, 1)
    pred_roi = model_output.float()[roi_mask_expanded]
    gt_roi = model_gt.float()[roi_mask_expanded]

    # 2a. Mean intensity constraint (macro-level)
    # Keep L1 for stable gradient on global brightness
    loss_intensity = F.l1_loss(pred_roi.mean(), gt_roi.mean())

    # 2b. MSE Contrast Loss - squared penalty on spatial distribution!
    # MSE heavily penalizes large errors with stronger gradients
    # Gradient = 2*(pred-gt), forces model to rebuild high peaks
    # Prevents "cheating" by spreading signal into many weak pixels
    loss_contrast = F.mse_loss(pred_roi, gt_roi)

    # Combined ROI loss (MSE gradients are strong, use low weight)
    loss_roi_total = loss_intensity + 0.1 * loss_contrast
else:
    loss_roi_total = torch.tensor(0.0, device=model_output.device)

# 3. Final Combined Loss
# roi_intensity_weight=0.5 + weighted_loss=10.0 gives: Global L1 ~90%, ROI ~10%
loss = loss_global + 0.5 * loss_roi_total
```

**Why MSE Works:**
- **MSE Gradient = 2×(pred-gt)**: For large errors (GT=1.0, Pred=0.1), gradient is -1.8 vs L1's -1.0
- **Squared Penalty**: (1.0-0.1)² = 0.81 vs L1 = 0.9, but gradient is 80% stronger!
- **Spatial Distribution**: MSE forces pixel-level matching, not just mean matching
- **Global L1 (10× tumor)**: Dominant constraint ensures correct spatial pattern + pure black background
- **Result**: Model must rebuild high-intensity peaks, cannot "spread thinly"

**Dilated Mask Loss** (kernel_size=5): Compensates VAE receptive field edge effects

**Gradient Checkpointing**: Enabled in Stage 3 to save memory at cost of computation time

**Conditioning Embedding**: 3 strided convolutions (256³ → 128³ → 64³), learns features during downsampling

**VAE Output Inversion**: Apply `1.0 - output` after VAE decode (inherent MAISI VAE property)

## Common Commands

### Training
```bash
# Stage 3 (Full Unfreeze + Top-K + Std Loss)
sbatch submit_finetune_stage3.sh

# Monitor Stage 3
tail -f logs/slurm_finetune_s3_*.log
grep "Loss Components" logs/slurm_finetune_s3_*.log | tail -20
grep "Epoch.*Loss:" logs/slurm_finetune_s3_*.log | tail -10
```

### Inference
```bash
# Update checkpoint path in environment config, then:
sbatch submit_infer.sh

# Visualize
conda run -n breast_gen python -m scripts.visualize_inference_results
```

### TensorBoard
```bash
tensorboard --logdir=tensorboard_logs/
```

## Configuration Files

**Training**:
- `configs/config_breast_controlnet_finetune_stage{1,2,2.5,3}.json`
- `configs/environment_breast_sub_finetune_stage{2,2.5,3}.json`

**Inference**:
- `configs/config_breast_infer.json`
- `configs/environment_breast_sub_infer.json`

**Network**:
- `configs/config_network_rflow.json`

## Key Config Parameters

**Stage 3 (Current - ROI Intensity + MSE Contrast)**:
```json
{
    "use_roi_intensity_loss": true,    // Enable ROI Intensity + MSE Contrast Loss
    "roi_intensity_weight": 0.5,       // Reduced: MSE gradients are strong
    "weighted_loss": 10.0,             // FINAL: Tumor ×10 for dominant spatial constraint
    "finetune_unfreeze_layers": ["up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3", "middle_block"],
    "finetune_unet_lr": 1e-5,          // UNet LR (conservative for up0)
    "finetune_controlnet_lr": 1e-4,    // ControlNet LR
    "batch_size": 4,                   // With gradient checkpointing
    "use_gradient_checkpointing": true, // Memory saving
    "val_frequency_schedule": {"107-130": 3},  // Validate every 3 epochs
    "num_inference_steps": 20,
    "cfg_guidance_scale": 1.0
}
```

**Code (train_controlnet.py):**
```python
loss_roi_total = loss_intensity + 0.1 * loss_contrast  // MSE with low weight
loss = loss_global + 0.5 * loss_roi_total  // roi_weight=0.5, weighted_loss=10.0
```

**Weight Tuning History:**
1. **Initial (Asymmetric)**: `roi_weight=15, contrast=2.0` → Combined loss ~18, ROI 94% → UNSTABLE ❌
2. **Fix (Asymmetric)**: `roi_weight=2, contrast=0.5, tumor=5` → Combined loss ~1.5, ROI 38% → **Signal scattered!** ❌
3. **Final (MSE)**: `roi_weight=0.5, mse=0.1, tumor=10` → Combined loss ~1.07, Global L1 90% → STABLE ✅

**Stage 3 (Failed - DO NOT USE)**:
```json
{
    "use_topk_loss": true,         // ❌ Ignored 70% of pixels
    "topk_ratio": 0.3,
    "use_std_loss": true,          // ❌ Only constrained variance
    "std_loss_weight": 1.0
}
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Loss stuck at ~0.86 | Easy background diluting gradients | Use Top-K Loss (Stage 2.5) |
| Tumor signal weak (89%) | Insufficient contrast matching | Use ROI Intensity Loss (Stage 3 new) |
| Tumor boundary blurry | Limited model capacity | Unfreeze more layers (Stage 3) |
| **Background all white** | **Top-K + up0 unfreeze = no background constraint** | **Use Global L1 + ROI Intensity** |
| **Loss explodes to ~18** | **ROI weight too high (15) + Contrast too aggressive (2.0)** | **Reduce: roi_weight=2, contrast=0.5** |
| **"Average trap"** | **Only using mean intensity loss** | **Add Asymmetric Contrast Loss (F.relu)** |
| **Signal scattered (mean 6%)** | **Asymmetric contrast allows weak pixels** | **Use MSE Contrast + Tumor weight 10×** |
| OOM in Stage 3 | Too many parameters unfrozen | Enable gradient checkpointing |
| Loss > 3.0 | Unstable training | Reduce learning rate or loss weights |
| NaN in validation | Problematic sample | Auto-skipped in dataloader |
| Training too slow | Gradient checkpointing overhead | Expected trade-off |

### Stage 3 Post-Mortem (Failed)

**Symptoms:**
- Pred min: 0.50 (should be ~0.0)
- Pred mean: 0.88 (GT mean: 0.05)
- MAE: 0.83 (Stage 2.5: 0.048)
- Peak center: (128, 128, 128) = image center, NOT tumor location

**Root Cause Analysis:**
```
Top-K Loss (30%) + Unfrozen up_blocks.0 = Disaster

1. Top-K ignores 70% easiest pixels → background ignored
2. Std Loss only constrains variance, not mean
3. up_blocks.0 (80M params) controls global style
4. No gradient signal from background → unconstrained
5. Model optimizes Std Loss by filling background
6. Result: Entire image becomes high intensity
```

**Solution (Stage 3 New):**
- Global L1: Every pixel has weight (bg=1.0, tumor=5.0)
- ROI Intensity: Focus on tumor mean, not spatial pattern
- Result: Background constrained + tumor signal boosted

### Stage 3 Epoch 114 Results (MSE Contrast Loss)

**Improvement over Stage 2.5 (Top-K)**:
- Max: 20.8% → 25.1% (**+4.3%**)
- Mean: 22.9% → 29.9% (**+7.0%**)
- 3 samples tested, all show consistent gains (+3.8%~7.1%)

**Issue**: Still no pixels > 0.5 (GT has 69.7% > 0.5)
**Status**: Direction correct, need more epochs or higher MSE weight

## Checkpoint Files

```
# Training checkpoints
weights/breast_sub_controlnet_epoch_58.pt                    # Pre-finetuning
weights/breast_sub_controlnet_finetune_epoch_70.pt           # Stage 1 end
weights/breast_sub_controlnet_finetune_s2_epoch_100.pt       # Stage 2 end
weights/breast_sub_controlnet_finetune_s2_best.pt            # Stage 2 best
weights/breast_sub_controlnet_finetune_s2_5_topk_epoch_107.pt # Stage 2.5 end
weights/breast_sub_controlnet_finetune_s2_5_topk_best.pt     # Stage 2.5 best (Val Loss: 0.8526)

# Stage 3 (Failed - DO NOT USE)
weights/breast_sub_controlnet_finetune_s3_best.pt            # Contains broken UNet state
weights/breast_sub_controlnet_finetune_s3_epoch_*.pt         # Background all white

# Diffusion backbone (NEVER modified by training)
weights/diff_unet_3d_rflow-mr.pt                            # Original MAISI UNet
```

**Note**: Finetuned UNet states are stored in checkpoint as `unet_finetuned_state_dict`. Original `diff_unet_3d_rflow-mr.pt` is never modified.

## Inference Output Format

```
output_breast_sub_infer_s2_5_e107/
  ├── DUKE_XXX_L_pred_sub.nii.gz  # Predicted subtraction
  ├── DUKE_XXX_L_gt_sub.nii.gz    # Ground truth
  └── visualizations/
      └── DUKE_XXX_L_comparison.png
```

## File Organization

```
breast-sub-gen/
├── scripts/              # Training/inference code
├── configs/              # JSON configurations
├── step_4/               # Preprocessed data
├── weights/              # Model checkpoints
├── logs/                 # Training logs
├── output_*/             # Inference outputs
└── dataset_breast_cached.json
```

## Training Strategy Evolution

1. **Stage 1-2**: Progressive unfreezing, standard weighted L1 loss
   - Result: Baseline performance, loss ~0.86

2. **Stage 2.5**: Top-K Loss (30% hard pixels)
   - Result: Val Loss 0.8526, tumor signal 89%
   - Status: ✅ Working, background pure black
   - Issue: Signal slightly weak, boundaries blurry

3. **Stage 3 (Failed)**: Top-K + Std Loss + up_blocks.0 unfrozen
   - Result: Background all white, MAE 0.83
   - Root cause: Top-K ignored background + up0 unconstrained
   - Status: ❌ Abandoned

4. **Stage 3 (Attempt 1)**: Global L1 + ROI Intensity + Asymmetric Contrast
   - Result: Signal scattered (mean 6% of GT), predictions nearly black
   - Root cause: Asymmetric contrast allowed model to spread signal thinly
   - Status: ❌ Failed

5. **Stage 3 (Current)**: Global L1 (10×) + ROI Intensity + MSE Contrast
   - Goal: Background pure black + tumor signal >95% WITHOUT scattering
   - Strategy: MSE squared penalty forces spatial matching, high tumor weight prevents dispersion
   - Start: From Stage 2.5 Epoch 107 (clean UNet)

## References

- MAISI Framework: MONAI Consortium
- Rectified Flow: Fewer sampling steps than DDPM
- Top-K Loss: Hard example mining for sparse signals
- **Ibarra et al. ROI Intensity Loss**: Mean intensity matching in tumor regions

## Code Changes (2026-02-27)

### 1. Modified `train_controlnet.py` (lines 643-670)

**Before (Top-K + Std):**
```python
use_topk = args.controlnet_train.get("use_topk_loss", False)
if use_topk:
    loss_topk = compute_topk_loss(model_output, model_gt, weights, topk_ratio)
    loss = loss_topk
    if use_std_loss:
        std_pred = torch.std(model_output.float(), dim=[1, 2, 3, 4])
        std_gt = torch.std(model_gt.float(), dim=[1, 2, 3, 4])
        loss_std = F.l1_loss(std_pred, std_gt)
        loss = loss_topk + std_loss_weight * loss_std
```

**After (Global L1 + ROI Intensity):**
```python
use_roi_intensity = args.controlnet_train.get("use_roi_intensity_loss", False)
roi_intensity_weight = args.controlnet_train.get("roi_intensity_weight", 1.0)

# Global weighted L1 (constrains ALL pixels)
l1_loss_raw = F.l1_loss(model_output.float(), model_gt.float(), reduction="none")
loss_global = (l1_loss_raw * weights).mean()

if use_roi_intensity:
    roi_mask = (interpolate_label > 0.5)
    if roi_mask.sum() > 0:
        roi_mask_expanded = roi_mask.repeat(1, images.shape[1], 1, 1, 1)
        pred_roi = model_output.float()[roi_mask_expanded]
        gt_roi = model_gt.float()[roi_mask_expanded]
        loss_intensity = F.l1_loss(pred_roi.mean(), gt_roi.mean())
    else:
        loss_intensity = torch.tensor(0.0, device=model_output.device)
    loss = loss_global + roi_intensity_weight * loss_intensity
else:
    loss = loss_global
```

### 2. Modified `config_breast_controlnet_finetune_stage3.json`

**Changes:**
- Removed: `"use_topk_loss"`, `"topk_ratio"`, `"use_std_loss"`, `"std_loss_weight"`
- Added: `"use_roi_intensity_loss": true`, `"roi_intensity_weight": 1.0`
- Changed: `"val_frequency_schedule": {"107-130": 3}` (more frequent validation)

### 3. Fixed `infer_breast_sub.py`

**Issue**: Incorrectly applied `1.0 - x` inversion to both prediction and GT
**Fix**: Removed inversion (training was in latent space without decoding)

**Before:**
```python
pred_sub_np = 1.0 - pred_sub_np  # WRONG
gt_decoded_np = 1.0 - gt_decoded_np  # WRONG
```

**After:**
```python
# No inversion needed - training in latent space
pred_sub_np = (pred_sub + 1.0) / 2.0
gt_decoded_np = (gt_decoded + 1.0) / 2.0
```

### 4. Added Asymmetric Contrast Loss (2026-02-27 16:00)

**Issue**: Only using mean intensity loss leads to "average trap" - model cheats with half bright/half dark pixels

**Solution**: Added Ibarra et al.'s asymmetric contrast loss that only penalizes under-estimated pixels

**Before (Stage 3 Attempt 1):**
```python
# Only mean intensity - vulnerable to cheating!
loss_intensity = F.l1_loss(pred_roi.mean(), gt_roi.mean())
loss_roi_total = loss_intensity
loss = loss_global + roi_intensity_weight * loss_roi_total
```

**After (Stage 3 Current):**
```python
# Mean intensity + asymmetric contrast
loss_intensity = F.l1_loss(pred_roi.mean(), gt_roi.mean())

# F.relu only penalizes dark pixels, ignores bright ones
under_estimated = F.relu(gt_roi - pred_roi)
loss_contrast = under_estimated.mean()

# Combined with balanced weights
loss_roi_total = loss_intensity + 0.5 * loss_contrast  # FINAL: 0.5, not 2.0
loss = loss_global + 2.0 * loss_roi_total  # FINAL: roi_weight=2.0, not 15
```

**Config Changes:**
- `roi_intensity_weight`: 15 → 2 (reduced to prevent dominating training)
- `contrast_weight`: 2.0 → 0.5 (in code, not config)

**Results:**
- Training loss: ~18 (unstable) → ~1.5 (stable)
- ROI占比: 94% → 38% (healthy balance)
- Global L1 占比: 6% → 62% (background control restored)

### 5. Re-enabled Inference Inversion (2026-02-27 15:40)

**Discovery**: Model outputs are inverted (background=1, tumor=0) - this is expected behavior

**Fix**: Added `1.0 - x` inversion back to inference code

```python
# Convert to [0,1] then invert
pred_sub_np = (pred_sub + 1.0) / 2.0
pred_sub_np = np.clip(pred_sub_np, 0, 1)
pred_sub_np = 1.0 - pred_sub_np  # Re-enabled
```

### 6. Switched to MSE Contrast Loss (2026-02-27 17:15)

**Issue**: Asymmetric contrast loss caused signal scattering - model satisfied mean constraint with many weak pixels

**Inference Results (Attempt 1)**:
- Pred mean: 0.003 vs GT mean: 0.048 (only **6%**!)
- Pixels > 0.1: 1.36% vs GT 12.40% (only **11%**!)
- Visual result: Predictions nearly all black

**Root Cause**:
- `F.relu(gt - pred).mean()` only penalizes under-estimation
- Model "cheats" by spreading signal into many weak pixels
- Satisfies mean constraint but fails spatial distribution

**Solution**: MSE Contrast Loss with higher tumor weight

**Before (Asymmetric - Attempt 1):**
```python
# Tumor weight 5×, asymmetric contrast
loss_intensity = F.l1_loss(pred_roi.mean(), gt_roi.mean())
under_estimated = F.relu(gt_roi - pred_roi)
loss_contrast = under_estimated.mean()
loss_roi_total = loss_intensity + 0.5 * loss_contrast
loss = loss_global + 2.0 * loss_roi_total
```

**After (MSE - Current):**
```python
# Tumor weight 10×, MSE contrast (squared penalty!)
loss_intensity = F.l1_loss(pred_roi.mean(), gt_roi.mean())
loss_contrast = F.mse_loss(pred_roi, gt_roi)  # KEY: MSE gradient = 2×(pred-gt)
loss_roi_total = loss_intensity + 0.1 * loss_contrast  # Low weight: MSE gradients strong
loss = loss_global + 0.5 * loss_roi_total  # roi_weight=0.5
```

**Why MSE Works**:
| Metric | L1 Gradient | MSE Gradient |
|--------|-------------|-------------|
| Error 0.1 (GT=1.0, Pred=0.9) | -1.0 | -0.2 |
| Error 0.9 (GT=1.0, Pred=0.1) | -1.0 | **-1.8** |

MSE gives **80% stronger gradient** for large outliers, forcing model to rebuild high peaks!

**Config Changes:**
- `weighted_loss`: 5.0 → **10.0** (dominant spatial constraint)
- `roi_intensity_weight`: 2.0 → **0.5** (MSE gradients strong)
- `mse_weight`: 0.5 → **0.1** (in code, prevents instability)

**Expected Results**:
- Global L1 ~90%: Strong spatial pattern constraint
- ROI ~10%: Sufficient for mean + contrast matching
- Tumor 10×: Prevents signal scattering
