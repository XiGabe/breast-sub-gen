# CLAUDE.md - Breast Subtraction Synthesis Project

## Project Overview

3D Contrast-Enhanced Breast MRI Synthesis using MAISI framework.
**Method**: `Pre-contrast → ControlNet → 3D Subtraction Map → Post = Pre + Sub`

## Data Structure (`step_4/`)

- **Files**: `{UUID}_{side}_{pre|sub|mask}.nii.gz`
- **Spacing**: `[0.7, 0.7, 1.2]` mm³ (Z,Y,X), RAS orientation
- **Note**: Right-side breasts flipped to left (`Is_Flipped=1`)

## Cached Dataset

```
embeddings_breast_sub/     # Sub_emb (64³, 4ch) - VAE encoded
processed_pre/             # Pre_aligned (256³, 1ch) - controlnet condition
processed_mask/            # Mask_aligned (256³, 1ch) - loss weighting
dataset_breast_cached.json # Train/Val split (1553/390 samples)
```

---

## Current Status (2026-03-03)

### Training Progress

| Stage | Epochs | Unfrozen | Loss | Peak% | Tumor% | Status |
|-------|--------|----------|------|-------|--------|--------|
| Stage 2.5 | 101-107 | middle, up1-3 | Top-K | 40% | 28% | ✅ Complete |
| **Stage 3** | 108-130 | ALL up + middle | MSE | **54%** | **46%** | ✅ Complete |
| Stage 4 | 130-150 | +down2,3 | MSE | ~54% | ~46% | ✅ Complete |
| **Stage 4.5** | **143-160** | **same as S4** | **Aggressive** | **TBD** | **TBD** | **🔄 Training** |

### Best Checkpoint: Stage 3 E130

- Peak: 54.1% of GT
- Tumor: 45.8% of GT
- **Issue**: Still severely underestimates intensity

### Key Finding: Why Intensity is Low

**Problem**: UNet barely changed (median diff only 0.55%)
- Stage 4 E142 UNet vs Base UNet: mean change = 0.0008
- UNet learning rate 1e-5 too conservative for domain gap

**Root Cause**: Loss configuration too conservative
```python
weighted_loss = 10.0        # Tumor ×10 - suppresses over-prediction
roi_intensity_weight = 0.5  # MSE Contrast only ~10% of total loss
# Result: Model chooses "safe" under-prediction
```

---

## Stage 4.5: 激进破局 (Aggressive Breakthrough)

### Problem Statement

1. **"拉不动"** - UNet LR 1e-5 too low for domain gap
2. **"不敢亮"** - Tumor weight 10× suppresses high peaks

### Solution: Dual Strategy

```json
{
    "finetune_unet_lr": 1e-4,      // UP from 1e-5 - match ControlNet
    "finetune_controlnet_lr": 1e-4,
    "weighted_loss": 3.0,           // DOWN from 10.0 - release L1 suppression
    "roi_intensity_weight": 1.0,    // UP from 0.5 - MSE Contrast takes over
    "batch_size": 4,
    "n_epochs": 160,
    "start_epoch": 143
}
```

### Expected Effect

| Metric | Stage 4 | Stage 4.5 | Change |
|--------|---------|-----------|--------|
| UNet LR | 1e-5 | 1e-4 | 10× higher |
| Tumor Weight | 10× | 3× | 70% lower |
| ROI % of Total | ~10% | ~22% | 2× higher |

### Training Status (Epoch 144/160)

- **Current Loss**: ~1.1 (stable)
- **ROI %**: 22% (up from 10%) ✅ MSE taking over
- **Next Validation**: Epoch 145

---

## Common Commands

```bash
# Training
sbatch submit_finetune_stage4_5.sh
tail -f logs/slurm_finetune_s4_5_*.log

# Monitor loss components
grep "Loss Components" logs/*.log | tail -20

# Inference
conda run -n breast_gen python -m scripts.infer_breast_sub \
    -e configs/environment_breast_sub_infer.json \
    -c configs/config_breast_infer.json \
    -t configs/config_network_rflow.json

# Visualize
conda run -n breast_gen python -m scripts.visualize_inference_results ./output_breast_sub_infer_*
```

---

## Configuration Files

**Training**:
- `configs/config_breast_controlnet_finetune_stage4_5.json` - Current (aggressive)
- `configs/environment_breast_sub_finetune_stage4_5.json`

**Inference**:
- `configs/config_breast_infer.json`
- `configs/environment_breast_sub_infer.json`

**Network**:
- `configs/config_network_rflow.json`

---

## Key Loss Implementation

```python
# Global Weighted L1 Loss (ALL pixels constrained)
l1_loss_raw = F.l1_loss(model_output.float(), model_gt.float(), reduction="none")
loss_global = (l1_loss_raw * weights).mean()  # bg×1.0, tumor×weighted_loss

# ROI Intensity + MSE Contrast Loss
roi_mask = (interpolate_label > 0.5)
if roi_mask.sum() > 0:
    pred_roi = model_output.float()[roi_mask_expanded]
    gt_roi = model_gt.float()[roi_mask_expanded]
    loss_intensity = F.l1_loss(pred_roi.mean(), gt_roi.mean())
    loss_contrast = F.mse_loss(pred_roi, gt_roi)  # KEY: gradient = 2×(pred-gt)
    loss_roi_total = loss_intensity + 0.1 * loss_contrast
else:
    loss_roi_total = torch.tensor(0.0, device=model_output.device)

# Final Combined
loss = loss_global + roi_intensity_weight * loss_roi_total
```

**Stage 4.5**: `weighted_loss=3.0`, `roi_intensity_weight=1.0` → ROI ~22% of total

**Stage 3/4**: `weighted_loss=10.0`, `roi_intensity_weight=0.5` → ROI ~10% of total

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Background all white | Top-K + up0 unfreeze | Use Global L1 + ROI Intensity |
| Signal scattered (6% of GT) | Asymmetric contrast | Use MSE Contrast + high tumor weight |
| Peak signal weak (54% of GT) | MSE + 10× too conservative | **Reduce tumor weight to 3×** |
| UNet not learning | LR 1e-5 too low | **Increase to 1e-4** |
| Inplace operation error | Gradient checkpointing + unfrozen up0 | Disable gradient checkpointing |
| OOM | Too many parameters | Reduce batch_size or enable checkpointing |

---

## Checkpoint Files

```
# Best checkpoint
weights/breast_sub_controlnet_finetune_s3_epoch_130.pt       # Peak 54% of GT

# Stage 4/4.5 checkpoints
weights/breast_sub_controlnet_finetune_s4_epoch_*.pt         # Stage 4 (E130-150)
weights/breast_sub_controlnet_finetune_s4_epoch_143.pt       # Stage 4.5 start point

# Historical (for reference)
weights/breast_sub_controlnet_finetune_s2_5_topk_epoch_107.pt # Stage 2.5 (40% peak)

# Diffusion backbone (NEVER modified)
weights/diff_unet_3d_rflow-mr.pt
```

**Note**: Finetuned UNet states stored as `unet_finetuned_state_dict` in checkpoint.

---

## Critical Bugs Fixed

### 1. Inference Not Loading Finetuned UNet (2026-03-03)

**Problem**: `infer_breast_sub.py` wasn't loading `unet_finetuned_state_dict` from checkpoint.
All previous inference results were **INVALID**.

**Fix**: Added lines 80-92 to load finetuned UNet state.

### 2. VAE Output Inversion

**Requirement**: Must apply `1.0 - output` after VAE decode (inherent MAISI VAE property).

**Verification**: Removing inversion causes negative correlation (-0.40).

---

## Inference Results (After Bug Fix)

| Checkpoint | Peak% | Tumor% | MAE | Finding |
|-----------|-------|--------|-----|---------|
| Stage 2.5 (E107) | 40% | 28% | 0.0425 | Baseline (Top-K) |
| Stage 3 (E130) | **54%** | **46%** | 0.0475 | **Best so far** |
| Stage 4 (E142) | ~54% | ~46% | ~0.045 | UNet barely changed |

**Intensity Analysis** (Stage 3):
```
Linear fit: Pred ≈ 0.12 * GT + 0.08
99th percentile: GT=0.80, Pred=0.27 (34%) ← Severe suppression
```

---

## Next Steps

**Immediate**: Wait for Stage 4.5 Epoch 145 validation results

**If still under 60%**:
- Consider further reducing tumor weight (3.0 → 1.0)
- Increase contrast weight (0.1 → 0.3)
- Add asymmetric loss that penalizes under-prediction more heavily

**Alternative**: Analyze training data intensity distribution to verify GT is correct

---

## References

- MAISI Framework: MONAI Consortium
- Rectified Flow: Fewer sampling steps than DDPM
- Top-K Loss: Hard example mining for sparse signals
- MSE Gradient: 2× stronger for large errors than L1
