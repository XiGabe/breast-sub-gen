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
Caching Stage (offline):
Input (variable)                Output (fixed)
─────────────────────────────────────────────────────
Pre (1ch)  ──┐
             ├──> Normalize to 256³ ──> Pre_aligned (256³, 1ch)
Mask (1ch) ──┤  (pad < 256, crop > 256)
             │
Sub (1ch)  ──┘   ──> Normalize to 256³ ──> VAE ──> Sub_emb (64³, 4ch)

Training Stage:
Input (256³)                    Latent Space (64³)
─────────────────────────────────────────────────────
Pre_aligned (1ch) → Interpolate → Cond. Embedding → Condition Features
                                              ↓
Sub_emb (4ch) ──────────────────────────────→ Target Latent
                                              ↓
                                     [Diffusion Training]
                                              ↓
                           [ControlNet] → [DiT/U-Net] → Predicted Velocity
                                 ↓              ↓
                         (trainable)      (frozen backbone)
```

### Key Design Decisions

1. **Asymmetric Latent Caching**: Only subtraction maps are encoded through VAE to 64³ latent space. Pre-contrast and mask stay at 256³ physical resolution.

2. **Fixed 256³ Spatial Size (Plan B)**: All samples normalized to exactly 256³ during offline caching:
   - Dimensions < 256: **symmetric padding** with zeros
   - Dimensions > 256: **mask-anchored cropping** (preserves tumor centroid)
   - No dynamic cropping needed during training (simplifies pipeline)
   - Rationale: Only ~10% of samples exceed 256³ in any dimension

3. **Mask-Weighted Loss**: Tumor regions are downsampled to latent space then dilated to compensate for VAE receptive field edge effects. Weight map: background=1.0, tumor=5.0.

4. **Pre as ControlNet Condition**: Pre-contrast images (256³) are used directly as conditioning, not binary labels. During training, they are trilinearly interpolated to latent space.

5. **Intensity Augmentation (Pre Images Only)**: Minimal intensity augmentations applied during training to simulate real-world imaging variations:
   - **RandGaussianNoise**: prob=0.15, std=0.01 (instrument thermal noise)
   - **RandAdjustContrastd**: prob=0.2, gamma∈[0.9, 1.1] (inter-hospital style variations)
   - Applied ONLY to `pre` images, NOT to Sub_emb or Mask
   - Preserves spatial alignment (no spatial augmentations)

6. **Frozen Backbone Training**: Only ControlNet and Conditioning Embedding are trainable. DiT/U-Net backbone remains frozen to preserve medical imaging priors.

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

### Offline Latent Caching (Triplet Mode)

Process sub/pre/mask triplets with spatial alignment to fixed 256³:

```bash
python -m scripts.diff_model_create_training_data_triplet \
    -e configs/environment_breast_sub.json \
    -c configs/config_breast_sub_train.json \
    -t configs/config_network_rflow.json \
    -g 1
```

**Plan B Strategy**:
- All samples normalized to **exactly 256³** during caching
- Dimensions < 256: symmetric padding
- Dimensions > 256: mask-anchored center crop (preserves tumor location)
- Sub: VAE-encoded to 64³ × 4 channels
- Pre and Mask: saved at 256³ (1 channel each)

**Critical Notes**:
- **No data regeneration needed**: Your `step_4/` data is correctly preprocessed in [0, 1] range
- **Intensity transforms skipped**: Since data is already preprocessed, `ScaleIntensityRangePercentilesd` is disabled
- The code automatically converts [0, 1] → [-1, 1] before VAE encoding
- Script is resumable: skips samples that already have output files
- Only ~10% of samples require cropping (most fit within 256³ after preprocessing)

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
  "spacing": [0.7, 0.7, 1.2]                 // Z, Y, X spacing in mm
}
```

### Cached Dataset Format

The `dataset_breast_cached.json` has explicit "training" and "validation" keys:

```json
{
  "training": [
    {
      "image": "embeddings_breast_sub/DUKE_001_L_sub_emb.nii.gz",
      "pre": "processed_pre/DUKE_001_L_pre_aligned.nii.gz",
      "label": "processed_mask/DUKE_001_L_mask_aligned.nii.gz",
      "spacing": [1.2, 0.7, 0.7],
      "modality": "mri"
    },
    ...
  ],
  "validation": [...]
}
```

**Key points**:
- `spacing`: Physical voxel size [Z, Y, X] in mm, used for:
  - Setting output NIfTI affine matrix
  - Passed to model as spatial encoding
  - Output filename includes spacing info
- Paths are relative to project root
- `scripts/utils.py` automatically detects this format (checks for "validation" key)

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

### Inconsistent Tensor Shapes During Training

- **Cause**: Samples not properly normalized to 256³ during caching
- **Fix**: Verify all cached outputs are exactly 256³ (physical) or 64³ (latent). Check `normalize_to_fixed_size()` implementation for mask-anchored cropping logic.

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

---

# Triplet-Based Training Pipeline (Current Implementation)

**Status**: Offline latent caching in progress (started 2026-02-25)

## Overview

This project uses a **triplet-based training approach** where:
- **Sub (subtraction map)**: VAE-encoded to latent space (64³, 4 channels) - diffusion target
- **Pre (pre-contrast)**: Resized and aligned (256³, 1 channel) - controlnet condition
- **Mask**: Resized and aligned (256³, 1 channel) - loss weighting

All three files undergo the SAME spatial transforms to ensure spatial consistency.

## New Files Created

### Scripts
- `scripts/create_breast_dataset.py` - Generates dataset_breast.json with triplet format
- `scripts/diff_model_create_training_data_triplet.py` - Processes sub/pre/mask triplets with spatial alignment
- `scripts/create_stratified_cached_dataset.py` - Generates stratified train/val split with patient-level grouping

### Configs
- `configs/environment_breast_sub.json` - Environment config for triplet caching
- `configs/environment_breast_sub_triplet.json` - Environment config for triplet training
- `configs/config_breast_controlnet.json` - ControlNet training config

### Documentation
- `TRIPLET_IMPLEMENTATION_SUMMARY.md` - Complete implementation guide

## Modified Files
- `scripts/train_controlnet.py` - Uses pre images as controlnet condition (instead of binary labels)
- `scripts/utils.py` - Supports triplet data loading with "pre", "image", "label" keys; **supports cached dataset format** with explicit "training"/"validation" keys; **no dynamic cropping** (all samples fixed at 256³)

## Current Progress (2026-02-25)

### Step 1: ✅ Dataset JSON Created
```bash
python -m scripts.create_breast_dataset --input step_4 --output dataset_breast.json
# Result: dataset_breast.json with 1943 samples
```

### Step 2: ✅ Offline Latent Caching (COMPLETE - Fixed 256³ Strategy)

**Completed**: 2026-02-25 13:02

**Plan B Strategy**:
- All samples normalized to **exactly 256³** (physical space)
- Dimensions < 256: **symmetric padding**
- Dimensions > 256: **mask-anchored center crop** (preserves tumor location)
- Latent space: **64³ × 4 channels** (VAE downsamples by 4x)

**Output**:
- `embeddings_breast_sub/` - Sub_emb files (64³, 4 channels) - **1943 files**
- `processed_pre/` - Pre_aligned files (256³, 1 channel) - **1943 files**
- `processed_mask/` - Mask_aligned files (256³, 1 channel) - **1943 files**

### Step 2.5: ✅ Stratified Train/Val Split (COMPLETE)

**Completed**: 2026-02-25 13:44

**Split Strategy**: Patient-level stratified sampling (三层分层抽样)
- **Layer 1**: Dataset (DUKE/ISPY1/ISPY2/NACT)
- **Layer 2**: Tumor status (has_tumor=0/1)
- **Layer 3**: Patient grouping (Original_ID)

**Result**:
| Split | Samples | Patients |
|-------|---------|----------|
| Train | 1553 (79.9%) | 1203 |
| Val | 390 (20.1%) | 301 |
| **Total** | **1943** | **1504** |

**Validation**:
- ✅ No patient overlap (prevents data leakage)
- ✅ Dataset proportions maintained (DUKE: 29.4%, ISPY1: 9.0%, ISPY2: 58.3%, NACT: 3.3%)
- ✅ Tumor proportions maintained (77.4% with tumor)

**Command**:
```bash
conda run -n breast_gen python -m scripts.create_stratified_cached_dataset \
    --metadata-csv step_4/step_4_metadata.csv \
    --input dataset_breast.json \
    --output dataset_breast_cached.json \
    --embedding-dir ./embeddings_breast_sub \
    --pre-dir ./processed_pre \
    --mask-dir ./processed_mask \
    --val-ratio 0.2 \
    --random-seed 42
```

**Output**: `dataset_breast_cached.json` with explicit "training" and "validation" keys

### Step 3: ⏳ Start ControlNet Training (READY)

The cached dataset is ready. `environment_breast_sub_triplet.json` already points to `dataset_breast_cached.json`.

```bash
conda run -n breast_gen python -m scripts.train_controlnet \
    -e configs/environment_breast_sub_triplet.json \
    -c configs/config_breast_controlnet.json \
    -t configs/config_network_rflow.json \
    -g 4
```

## Resumability

**If caching process is interrupted**: Simply re-run the same command. The script checks if output files already exist and skips completed samples.

**To verify cached data** (works during or after caching):
```python
python3 << 'EOF'
import json
import glob
import os

# Use original dataset to get expected count
with open('dataset_breast.json') as f:
    expected = len(json.load(f)['training'])

sub_count = len(glob.glob('embeddings_breast_sub/*_sub_emb.nii.gz')) if os.path.exists('embeddings_breast_sub') else 0
pre_count = len(glob.glob('processed_pre/*_pre_aligned.nii.gz')) if os.path.exists('processed_pre') else 0
mask_count = len(glob.glob('processed_mask/*_mask_aligned.nii.gz')) if os.path.exists('processed_mask') else 0

print(f'Expected: {expected}')
print(f'Sub: {sub_count}, Pre: {pre_count}, Mask: {mask_count}')
print(f'Progress: {sub_count}/{expected} ({100*sub_count//expected if expected > 0 else 0}%)')
print(f'Status: {"✓ COMPLETE" if sub_count == pre_count == mask_count == expected else "⏳ IN PROGRESS"}')
EOF
```

## Task Persistence

**Important**: Implementation tasks are tracked in Claude's internal task system (not local files). If you restart Claude:
1. Use `/tasks` command to view all tasks
2. Use `/tasks resume <task_id>` to resume any task
3. All code changes remain in the repository
4. Background processes continue running independently

To stop a background process:
```bash
# Find the process
ps aux | grep diff_model_create_training_data_triplet

# Kill it
kill <PID>
```
