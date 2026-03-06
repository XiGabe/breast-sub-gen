# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **3D breast MRI subtraction synthesis** project using a cascade architecture:
1. **3D Tumor Locator** (nnU-Net): Generates predicted tumor masks from pre-contrast MRI
2. **MAISI Diffusion Model + ControlNet**: Generates high-fidelity subtraction images conditioned on pre-contrast MRI + tumor masks

The core philosophy is "Align to Inference, Standardize for Training" - converting multi-center bilateral breast data into standardized unilateral RAS-oriented, isotropic-resolution inputs.

**Data Flow**: Pre-contrast MRI + Tumor Mask → [VAE encode] → Latent Space → [ControlNet + U-Net] → Predicted Subtraction Latent → [VAE decode] → Subtraction Image → Post = Pre + Subtraction

---

## GPU Environment

The project runs on the sablab-gpu partition. To request an interactive GPU session:

```bash
srun --partition=sablab-gpu \
     --gres=gpu:a40:1 \
     --mem=20G \
     --cpus-per-task=4 \
     --pty /bin/bash -i
```

To check GPU availability: `sinfo`

---

## Conda Environment

All packages and dependencies are managed in the `breast_gen` conda environment.
Activate with: `conda activate breast_gen`

---

## Project Structure

```
breast-sub-gen/
├── configs/                    # Configuration JSON files
│   ├── config_network_rflow.json           # Network architecture (dual-channel: 2)
│   ├── config_maisi_controlnet_train_*.json  # Training configs
│   ├── environment_maisi_*.json            # Environment paths
│   └── modality_mapping_breast.json        # Modality mapping for breast MRI
├── scripts/
│   ├── train_controlnet.py     # Main ControlNet training script (dual-channel)
│   ├── infer_controlnet.py     # Inference script (dual-channel)
│   ├── diff_model_setting.py   # Config loading, logging, distributed setup
│   ├── utils.py                # Data loading, transforms, utilities
│   ├── transforms.py           # MAISI VAE transform pipeline
│   ├── augmentation.py         # Tumor augmentation transforms
│   └── sample.py               # Diffusion sampling/inference (dual-channel)
├── data/
│   ├── step_4/                 # Original processed data (256³, preprocessed)
│   ├── embeddings_breast_sub/  # Offline VAE-encoded latents (64³, 4-ch)
│   ├── processed_mask/         # Processed masks (256³, 1-ch)
│   ├── processed_pre/          # Processed pre-contrast images (256³, 1-ch)
│   ├── dataset_breast.json     # Original dataset JSON
│   └── dataset_breast_cached.json  # Cached dataset with patient splits
├── models/                     # Model checkpoints (created during training)
│   └── *_best.pt, *_epoch_*.pt
├── weights/
│   ├── autoencoder_v2.pt       # Pre-trained VAE (encoder+decoder)
│   └── diff_unet_3d_rflow-mr.pt # Pre-trained diffusion U-Net
└── outputs/
    ├── logs/                   # Training logs
    ├── inference_*/            # Inference visualization outputs
    └── tfevent/                # TensorBoard event files
```

---

## Data Pipeline (Status: Completed)

The preprocessing pipeline has created standardized data:
- **Resolution**: 256×256×256 (padded if smaller)
- **Spacing**: [0.7, 0.7, 1.2] mm³
- **Orientation**: RAS (Right-Anterior-Superior)
- **Pre-processing**: Single-side extraction, flip standardization, dual asymmetric normalization

### Offline Caching

The target subtraction images have been pre-encoded through the VAE:
- `data/embeddings_breast_sub/*.nii.gz`: 64×64×64 latents, 4 channels
- `data/processed_pre/*.nii.gz`: Pre-contrast images at 256³ resolution
- `data/processed_mask/*.nii.gz`: Binary tumor masks at 256³ resolution

**Note**: Raw VAE latents (range ~[-8, 9]) are scaled by `scale_factor` at train_controlnet.py:361

---

## Configuration System

Configurations are split across three JSON files loaded in `train_controlnet.py`:

1. **`environment_*.json`**: Paths, data locations, model checkpoint paths
2. **`config_maisi_controlnet_train_*.json`**: Training hyperparameters (batch size, LR, epochs)
3. **`config_network_*.json`**: Network architecture definitions

Key parameters in `config_network_rflow.json`:
- `conditioning_embedding_in_channels`: **2** (dual-channel: Pre + Mask)
- `latent_channels`: 4
- `num_class_embeds`: 128 (for modality embedding)

---

## Training Commands

### Progressive Training (Recommended)

```bash
# Stage 3.1: ControlNet only (50 epochs)
sbatch scripts/submit_stage1.sh

# Stage 3.2: Deep U-Net blocks (50 epochs)
sbatch scripts/submit_stage2.sh

# Stage 3.3: All U-Net blocks (50 epochs)
sbatch scripts/submit_stage3.sh
```

### Single GPU Training

```bash
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr.json \
    --model_config_path configs/config_maisi_controlnet_train_rflow-mr.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1
```

### Multi-GPU Training (DDP)

```bash
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr.json \
    --model_config_path configs/config_maisi_controlnet_train_rflow-mr.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 4
```

---

## Key Scripts Reference

### `scripts/train_controlnet.py`
Main training loop. Key functions:
- `train_controlnet()`: Training loop with pure L1 + background penalty loss
- `compute_model_output()`: Forward pass through ControlNet → U-Net with dual-channel input
- `apply_random_morphological_perturbation()`: Pure PyTorch dilation/erosion augmentation

**Removed**: Region Contrastive Loss (was causing 2x slowdown + OOM)

### `scripts/infer_controlnet.py`
Inference script for generating subtraction images. Uses dual-channel input.

### `scripts/sample.py`
Diffusion sampling/inference with CFG guidance validation.
**VAE后处理关键**: VAE解码输出不在[0, 1]范围，使用实际范围归一化
**WARNING**: `cfg_guidance_scale` should be 0 or 1.0 (model not trained with mask dropout).

### `scripts/visualize_inference.py`
Inference visualization script for model validation:
- Multi-view visualization (axial, sagittal, coronal)
- Statistics output (tumor enhancement intensity)
- Supports all stage checkpoints (1/2/3)
Diffusion sampling/inference with CFG guidance validation.
**WARNING**: `cfg_guidance_scale` should be 0 or 1.0 (model not trained with mask dropout).

**VAE解码后处理** (scripts/sample.py:460-520):
- VAE解码输出不在固定范围[0, 1]，需要基于实际范围归一化
- 使用 `use_crop_body_mask=False` (subtraction generation不应裁剪)
- 归一化: `(synthetic_images - actual_min) / (actual_max - actual_min)`

### `scripts/visualize_inference.py`
推理与可视化脚本，用于验证模型性能：
- 加载Stage 1/2/3 checkpoint
- 生成subtraction图像并创建多视角可视化
- 输出统计信息（肿瘤区域增强强度）

### `scripts/utils.py`
- `prepare_maisi_controlnet_json_dataloader()`: Creates train/val dataloaders from JSON
- `define_instance()`: Instantiates models from config using MONAI ConfigParser
- **Removed**: `binarize_labels()` function (no longer needed with dual-channel)

### `scripts/transforms.py`
Contains `VAE_Transform` class - **NOT used for current cached data pipeline** since preprocessing is complete.

---

## Architecture Notes

### ControlNet Input Processing (Dual-Channel)

**Current implementation (dual-channel)**:
```python
# From train_controlnet.py compute_model_output()
pre_images = batch["pre"].to(device)  # [B, 1, 256, 256, 256]
masks = batch["label"].to(device)      # [B, 1, 256, 256, 256]

# Optional: 30% probability of morphological perturbation
if apply_morphological_perturb and torch.rand(1).item() < 0.3:
    masks = apply_random_morphological_perturbation(masks)

# Dual-channel condition: [Pre-contrast MRI, Tumor Mask]
controlnet_cond = torch.cat([pre_images.float(), masks.float()], dim=1)
# Output: [B, 2, 256, 256, 256]
```

**Pure PyTorch morphology** (avoids MONAI's native ops which can cause errors in forward loops):
- Dilation: `F.max_pool3d` on padded tensor
- Erosion: `1 - F.max_pool3d(1 - mask)` (min pooling via inverted max pool)
- Kernel sizes: 1 or 3 (odd only)

### Loss Computation (The Great Cleanup)

**Implemented loss** (train_controlnet.py:504-530):
```python
# 1. Precise dimension reduction using Max Pooling (preserves ANY tiny lesion!)
roi_mask_latent = F.max_pool3d(labels.float(), kernel_size=4, stride=4) > 0.0

# 2. Base Global L1 with moderate ROI weighting
weights = torch.ones_like(model_output)
weights[roi_mask_latent.repeat(1, model_output.shape[1], 1, 1, 1)] = 3.0
l1_loss_raw = F.l1_loss(model_output.float(), model_gt.float(), reduction="none")
loss = (l1_loss_raw * weights).mean()

# 3. Absolute Background Penalty (suppresses false positives/snow noise)
if roi_mask_latent.sum() > 0:
    bg_mask_expanded = (~roi_mask_latent).repeat(1, model_output.shape[1], 1, 1, 1)
else:
    bg_mask_expanded = torch.ones_like(model_output, dtype=torch.bool)

pred_bg = model_output.float()[bg_mask_expanded]
gt_bg = model_gt.float()[bg_mask_expanded]
false_positive_bg = F.relu(pred_bg - gt_bg)  # Only penalize pred > gt
loss = loss + 5.0 * (false_positive_bg ** 2).mean()
```

**Key improvements**:
- `F.max_pool3d` instead of `F.interpolate` → tiny lesions never lost
- Background penalty suppresses false positives
- No Region Contrastive Loss → 2x faster training, no OOM risk

---

## Training Strategy: Progressive Co-Tuning

Uses **shell script stages** to avoid optimizer momentum loss. Each stage creates a fresh optimizer.

### Stage 3.1: ControlNet Alignment
- U-Net: Frozen | LR: 1e-4 | Epochs: 50 | Val every: 2
- Config: `config_maisi_controlnet_train_stage1.json`
- Output: `models/breast_controlnet_stage1_epoch_{N}.pt`, `*_best.pt`

### Stage 3.2: Deep Semantic Release
- Unfreeze: Deep blocks (`down_blocks.2`, `down_blocks.3`, `middle_block`, `up_blocks`)
- U-Net LR: 5e-5 | Epochs: 50 | Inplace disabled
- Output: `models/breast_controlnet_stage2_epoch_{N}.pt`, `*_best.pt`

### Stage 3.3: Shallow Edge Refinement
- Unfreeze: All blocks
- U-Net LR: 1e-5 | Epochs: 50
- Output: `models/breast_controlnet_stage3_epoch_{N}.pt`, `*_best.pt`

---

## Checkpoint Strategy

| File | Description |
|------|-------------|
| `{exp_name}_epoch_{N}.pt` | Saved every epoch (independent) |
| `{exp_name}_best.pt` | Best validation loss (or training loss if no val) |

---

## Important Constraints

1. **NO spatial augmentation** during training - would break alignment between 256³ physical inputs and 64³ latent targets
2. **Background must be exactly 0** in subtraction images to prevent snow noise in generation
3. **Patient-level splits** are critical - no data leakage between train/val (already handled in dataset JSON)
4. **VAE scale factor** from pretrained checkpoint must be used to scale latents before model input (no additional [-1, 1] normalization needed)
5. **CFG Guidance**: Use `cfg_guidance_scale = 0` or `1.0` only - model was NOT trained with mask dropout, so values > 1.0 will cause artifacts
6. **VAE解码后处理**: 不要使用固定范围裁剪，VAE输出可能在负数范围，需要基于实际范围归一化

---

## Dataset Statistics

**Total**: 1,943 samples | 1,504 patients

| Split | Samples | Patients | With Tumor | No Tumor |
|-------|---------|----------|------------|----------|
| Train | 1,553 (80%) | 1,203 | 77.4% | 22.6% |
| Val    | 390 (20%)  | 301   | 77.2% | 22.8% |

**Key insight**: ~22% of samples have no tumor (mask all zeros). This ensures the model learns to handle:
- `[pre_with_tumor, tumor_mask]` → subtraction with tumor enhancement
- `[pre_healthy, all_zeros_mask]` → near-black subtraction (no enhancement)

**Stage 1 Inference Results** (Epoch 28 best checkpoint):
- 肿瘤区域增强: 750-965 HU (mean ~880 HU)
- 整体平均值: ~450 HU
- 确认模型学会基本subtraction生成模式