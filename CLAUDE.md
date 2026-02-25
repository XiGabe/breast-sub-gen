# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a 3D Contrast-Enhanced Breast MRI Synthesis project using MAISI (Medical AI for Synthetic Imaging) framework on the MAMA-MIA Dataset. The goal is to generate synthetic 3D post-contrast breast MRI from pre-contrast inputs using diffusion models.

**Core Innovation**: Unlike prior 2D slice-wise DDPM approaches (e.g., Ibarra et al. 2025), this project uses a fully 3D framework to preserve volumetric spatial consistency for accurate tumor localization.

**Method**: `Pre-contrast MRI → Model → 3D Subtraction Map → Post = Pre + Predicted_Sub`

## Data Structure

### Preprocessed Data (`step_4/`)

All preprocessed breast MRI data resides in `step_4/` with standardized RAS orientation and isotropic resampling to `[0.7, 0.7, 1.2] mm³`.

**File Naming Convention**:
- `{UUID}_{side}_pre.nii.gz` - Pre-contrast (condition input)
- `{UUID}_{side}_sub.nii.gz` - Subtraction map (target, sparse signal)
- `{UUID}_{side}_mask.nii.gz` - Tumor segmentation mask

**Metadata**: `step_4/step_4_metadata.csv` contains all information needed to map back to patient physical space:
- `UUID`: Database primary key
- `Original_ID`: Source patient identifier
- `Side`: Original breast side (`L` or `R`)
- `Is_Flipped`: Whether right-side data was flipped (0 or 1) - **must reverse during inference**
- `Crop_BBox`: `[x, y, z, w, h, d]` bounding box for reconstruction
- `Orig_Shape`: Original full-volume dimensions
- `Has_Tumor`: Binary flag for tumor presence (0 or 1)

**Critical**: All right-side breasts have been horizontally flipped to appear as left breasts (`Is_Flipped=1`). This standardization reduces model learning complexity.

## Code Architecture

### MAISI Framework Structure

The codebase extends MAISI-v2 (MONAI-based) for breast-specific synthesis:

```
scripts/
├── diff_model_train.py          # Diffusion model training (DiT backbone)
├── train_controlnet.py          # ControlNet training with region-aware losses
├── diff_model_infer.py          # Inference with CFG support
├── infer_controlnet.py          # ControlNet-specific inference
├── sample.py                    # Core sampling logic (LDMSampler class)
├── utils.py                     # Utilities (transforms, masking, post-processing)
├── augmentation.py              # Data augmentation (tumor-specific transforms)
├── diff_model_setting.py        # Distributed training setup and config loading
├── find_masks.py                # Database query for candidate masks
├── quality_check.py             Statistical quality control (outlier detection)
└── transforms.py                # VAE transform pipelines for CT/MRI

configs/
├── environment_*.json           # Paths (data, models, outputs)
├── config_infer*.json           # Inference parameters
├── config_network_*.json        # Network architecture definitions
├── modality_mapping.json        # Modality label mappings
└── label_dict*.json             # Anatomy label dictionaries
```

### Training Pipeline Architecture

```
Input (256³)                    Latent Space (64³)
─────────────────────────────────────────────────────
Pre (1ch) → Cond. Embedding → Condition Features (64ch)
     │                          (no VAE)
     │
Sub (1ch) → [VAE Encoder] → → → Target Latent (4ch)
                                    ↓
                           [Diffusion Training]
                                    ↓
              [ControlNet] → [DiT/U-Net] → Predicted Velocity
                    ↓              ↓
            (trainable)      (frozen backbone)
```

### Key Design Decisions

1. **Asymmetric Latent Caching**: Only subtraction maps are encoded through VAE to 64³ latent space. Pre-contrast and mask stay at 256³ physical resolution.

2. **No Spatial Augmentation**: Due to offline latent caching, all spatial transforms (crop, flip, rotate) are disabled. Only intensity augmentation is applied to pre-contrast images.

3. **Mask-Weighted Loss**: Tumor regions are downsampled to latent space then dilated to compensate for VAE receptive field edge effects. Weight map: background=1.0, tumor=5.0.

4. **Frozen Backbone Training**: Only ControlNet and Conditioning Embedding are trainable. DiT/U-Net backbone remains frozen to preserve medical imaging priors.

## Common Commands

### VAE Safety Testing

Before training, validate VAE handles sparse subtraction maps correctly:

```bash
# Test encode-decode quality with different input ranges
python -m scripts.00_vae_safety_test \
    --data_dir step_4 \
    --vae_path ./weights/autoencoder_v2.pt \
    --sample_id DUKE_001_L
```

**IMPORTANT Finding**: The MAISI VAE (`autoencoder_v2.pt`) expects **[-1, 1] input range**, but our preprocessed data is in [0, 1]. The conversion is already applied in `diff_model_create_training_data.py` line 172:
```python
pt_nda = pt_nda * 2.0 - 1.0  # Convert [0,1] to [-1,1]
```

### Offline Latent Caching

Encode subtraction maps to latent space once for faster training:

```bash
python -m scripts.diff_model_create_training_data \
    -e configs/environment_breast_sub.json \
    -c configs/config_breast_sub_train.json \
    -t configs/config_network_rflow.json \
    -g 4
```

**Critical Notes**:
- **No data regeneration needed**: Your `step_4/` data is correctly preprocessed in [0, 1] range
- **Intensity transforms skipped**: Since data is already preprocessed, `ScaleIntensityRangePercentilesd` is disabled to avoid redundant processing
- The code automatically converts [0, 1] → [-1, 1] before VAE encoding
- Verify if VAE checkpoint contains `scale_factor`. If present, apply it during encoding and do NOT re-apply during DataLoader.

### ControlNet Training

Train with dilated mask-weighted loss (no CFG dropout):

```bash
python -m scripts.train_controlnet \
    -e configs/environment_breast_sub.json \
    -c configs/config_breast_controlnet.json \
    -t configs/config_network_rflow.json \
    -g 4
```

### Inference (Zero CFG)

Generate synthetic subtraction maps without masks:

```bash
python -m scripts.diff_model_infer \
    -e configs/environment_breast_sub.json \
    -c configs/config_breast_infer.json \
    -t configs/config_network_rflow.json \
    -g 1
```

**Remember**: Set `cfg_guidance_scale=1.0` (zero CFG) since model was trained without unconditional learning.

### Physical Reconstruction

Post-process to reconstruct full contrast-enhanced image:

```python
# Load generated subtraction map
sub_pred = nib.load("generated_sub.nii.gz").get_fdata()

# Load original pre-contrast (patient-specific normalization)
pre_raw = nib.load("patient_pre.nii.gz").get_fdata()

# Reverse normalization (match preprocessing step)
sub_phys = sub_pred * pre_norm_scale  # Match preprocessing scaling
post_synthetic = pre_raw + sub_phys

# Save
nib.save(nib.Nifti1Image(post_synthetic, affine), "synthetic_post.nii.gz")
```

## Configuration Guidelines

### Network Configuration (`config_network_rflow.json`)

```json
{
  "conditioning_embedding_in_channels": 1,  // Single-channel pre-contrast
  "conditioning_embedding_num_channels": [16, 32, 64],
  "use_region_contrasive_loss": false       // Disable for this task
}
```

### ControlNet Training Config

```json
{
  "controlnet_train": {
    "weighted_loss": true,
    "weighted_loss_label": [1],              // Tumor label
    "batch_size": 2,
    "n_epochs": 100,
    "lr": 1e-4
  }
}
```

### Inference Config

```json
{
  "modality": 9,                             // MRI_T1 from modality_mapping.json
  "cfg_guidance_scale": 1.0,                 // Zero CFG (no classifier-free guidance)
  "num_inference_steps": 20,                 // Rectified Flow: 10-50 steps
  "output_size": [256, 256, 256],
  "spacing": [0.7, 0.7, 1.2]
}
```

## Critical Implementation Notes

### VAE Input Range

**IMPORTANT**: MAISI VAE expects **[-1, 1] input range**, but preprocessed data is [0, 1].

```python
# In diff_model_create_training_data.py, line 172:
pt_nda = pt_nda * 2.0 - 1.0  # Convert [0,1] to [-1,1]
```

**Why this matters**: Without this conversion, VAE produces negative artifacts in reconstruction (e.g., [-0.27, 1.45] output for [0, 1] input). The `00_vae_safety_test.py` script detects this issue automatically.

**Note**: No need to regenerate `step_4/` data - the conversion happens at encoding time.

### Mask Downsampling for Latent Loss

Always dilate masks in **physical space first**, then downsample:

```python
# WRONG: Downsample then dilate (loses small tumors)
mask_latent = F.interpolate(mask, size=(64,64,64), mode='nearest')
dilated = F.max_pool3d(mask_latent, kernel_size=3)

# CORRECT: Dilate then downsample (preserves tumors)
from monai.transforms.utils_morphological_ops import dilate
dilated_phys = dilate(mask, iterations=2)
mask_latent = F.interpolate(dilated_phys, size=(64,64,64), mode='nearest')
```

### Modality Configuration

Add breast MRI modalities to `configs/modality_mapping.json`:

```json
{
  "mri_t1_breast": 18,
  "mri_dce_breast": 19,
  "mri_sub_breast": 20
}
```

### Body Region for Breast

In `scripts/find_masks.py`, add breast region:

```python
region_mapping_maisi = {
    # ... existing regions ...
    "breast": 4,
    "chest/breast": 1
}
```

## Troubleshooting

### VAE Encoding Produces All-Black Latents

- **Cause**: VAE expects `[-1, 1]` input but received `[0, 1]`
- **Fix**: Multiply by 2.0 and subtract 1.0 before encoding

### Training Generates Snow in Background

- **Cause**: Subtraction map has non-zero background or CFG was used
- **Fix**: Ensure `Sub[Sub < 0] = 0` in preprocessing; disable CFG in inference

### Tumor Location Shifted

- **Cause**: VAE receptive field edge effect not compensated
- **Fix**: Increase mask dilation iterations or kernel size

### Losses Not Decreasing

- **Cause**: Learning rate too high or backbone not frozen
- **Fix**: Verify `requires_grad=False` for DiT/U-Net parameters; reduce LR to 5e-5

## File Organization

```
breast-sub-gen/
├── scripts/              # All training/inference code
├── configs/              # JSON configuration files
├── step_4/               # Preprocessed data (see metadata CSV)
├── weights/              # Model checkpoints
├── data/                 # Raw data links
└── tutorial/             # Jupyter notebooks for examples
```

## Key References

- MAISI Framework: Based on MONAI Consortium implementation
- Rectified Flow: Requires fewer sampling steps than DDPM
- Sparse Signal Learning: Subtraction maps are ~95% background zeros
