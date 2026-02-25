# Triplet-Based Breast Subtraction Training Pipeline - Implementation Summary

## Overview

This implementation adds support for triplet-based training where:
- **Sub (subtraction map)**: VAE-encoded to latent space (64³, 4 channels) - diffusion target
- **Pre (pre-contrast)**: Resized and aligned (256³, 1 channel) - controlnet condition
- **Mask**: Resized and aligned (256³, 1 channel) - loss weighting

All three files undergo the SAME spatial transforms to ensure spatial consistency.

## Files Created

### Scripts
1. **`scripts/create_breast_dataset.py`**
   - Generates `dataset_breast.json` with triplet format from step_4/ files
   - Format: `{"sub": "...", "pre": "...", "mask": "...", "modality": "mri"}`

2. **`scripts/diff_model_create_training_data_triplet.py`**
   - Processes sub/pre/mask triplets with spatial alignment
   - Only Sub goes through VAE encoding
   - Pre and Mask saved as aligned files

3. **`scripts/create_cached_dataset.py`**
   - Generates `dataset_breast_cached.json` after caching is complete
   - Points to cached files (sub_emb, pre_aligned, mask_aligned)

### Configs
1. **`configs/environment_breast_sub.json`** (updated)
   - Added `pre_output_dir` and `mask_output_dir` for triplet caching

2. **`configs/environment_breast_sub_triplet.json`** (new)
   - Environment config for training with cached triplet data

3. **`configs/config_breast_controlnet.json`** (new)
   - ControlNet training config with weighted loss settings

## Files Modified

### 1. `scripts/train_controlnet.py`

**Changes to `compute_model_output()` function:**
- Added `pre_images` parameter
- Uses pre-contrast images as controlnet condition (interpolated to latent space)
- Falls back to `binarize_labels()` if `pre_images` is None

**Changes to training loop:**
- Extracts `pre` from batch if available
- Passes `pre_images` to `compute_model_output()`

### 2. `scripts/utils.py`

**Changes to `add_data_dir2path()` function:**
- Handles triplet format (sub, pre, mask keys)
- Maps sub→image, pre→pre, mask→label for compatibility

**Changes to `prepare_maisi_controlnet_json_dataloader()` function:**
- Detects triplet format from dataset
- Loads pre, image, and label with appropriate transforms
- Maintains backward compatibility with standard format

## Execution Flow

### Step 1: Generate Triplet Dataset JSON

```bash
python -m scripts.create_breast_dataset \
    --input step_4 \
    --output dataset_breast.json
```

Output: `dataset_breast.json` with 1943 samples in triplet format

### Step 2: Execute Offline Latent Caching (8-10 hours)

```bash
python -m scripts.diff_model_create_training_data_triplet \
    -e configs/environment_breast_sub.json \
    -c configs/config_breast_sub_train.json \
    -t configs/config_network_rflow.json \
    -g 1
```

Output directories:
- `embeddings_breast_sub/` - Sub_emb files (64³, 4ch)
- `processed_pre/` - Pre_aligned files (256³, 1ch)
- `processed_mask/` - Mask_aligned files (256³, 1ch)

### Step 3: Generate Cached Dataset JSON

```bash
python -m scripts.create_cached_dataset \
    --input dataset_breast.json \
    --output dataset_breast_cached.json \
    --embedding-dir ./embeddings_breast_sub \
    --pre-dir ./processed_pre \
    --mask-dir ./processed_mask
```

Output: `dataset_breast_cached.json` pointing to cached files

### Step 4: Start ControlNet Training

```bash
python -m scripts.train_controlnet \
    -e configs/environment_breast_sub_triplet.json \
    -c configs/config_breast_controlnet.json \
    -t configs/config_network_rflow.json \
    -g 1
```

## Monitoring Commands

### During caching:
```bash
watch -n 60 'echo "Sub: $(ls embeddings_breast_sub/*_sub_emb.nii.gz 2>/dev/null | wc -l) Pre: $(ls processed_pre/*_pre_aligned.nii.gz 2>/dev/null | wc -l) Mask: $(ls processed_mask/*_mask_aligned.nii.gz 2>/dev/null | wc -l)"'
```

### After caching:
```bash
python3 << 'EOF'
import json
import glob
with open('dataset_breast_cached.json') as f:
    expected = len(json.load(f)['training']))
sub_count = len(glob.glob('embeddings_breast_sub/*_sub_emb.nii.gz'))
pre_count = len(glob.glob('processed_pre/*_pre_aligned.nii.gz'))
mask_count = len(glob.glob('processed_mask/*_mask_aligned.nii.gz'))
print(f'Expected: {expected}')
print(f'Sub: {sub_count}, Pre: {pre_count}, Mask: {mask_count}')
print(f'Status: {"✓" if sub_count == pre_count == mask_count == expected else "✗"}')
EOF
```

## Storage Requirements

```
Cached data (1943 samples):
- Sub_emb:    ~64³ × 4 bytes × 4 ch × 1943 ≈ 0.13 GB
- Pre_aligned: ~256³ × 4 bytes × 1 ch × 1943 ≈ 1.3 GB
- Mask_aligned: ~256³ × 4 bytes × 1 ch × 1943 ≈ 1.3 GB
Total: ~2.7 GB
```

## Key Design Decisions

1. **Asymmetric Latent Caching**: Only subtraction maps are VAE-encoded; pre-contrast and mask stay at physical resolution

2. **Spatial Consistency**: All three files use identical spatial transforms (same random seed if augmentation is added later)

3. **Pre as Conditioning**: Pre-contrast images are used directly as ControlNet condition (interpolated to latent space during training)

4. **Backward Compatibility**: Training code falls back to binary label encoding if pre images are not available

## Configuration Details

### ControlNet Conditioning
- Input: Pre-contrast images at 256³ physical resolution
- Processing: Trilinear interpolation to 64³ latent space during training
- Channels: 1 (single channel input)

### Weighted Loss
- Background weight: 1.0
- Tumor (label=1) weight: 5.0
- Mask downsampled: Nearest-neighbor interpolation to latent space

## Next Steps

1. Run the caching script (Step 2)
2. Verify cached outputs with validation script
3. Start ControlNet training (Step 4)
4. Monitor training via tensorboard: `tensorboard --logdir ./tensorboard_logs`
