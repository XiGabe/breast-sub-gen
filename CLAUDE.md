# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**3D breast MRI subtraction synthesis** using Cascade V3.0 architecture:
- **Stage 1 (Locator)**: 3D tumor segmentation network (nnU-Net or DynUNet)
- **Stage 2 (Renderer)**: Dual-channel ControlNet + U-Net for high-fidelity synthesis

**Data Flow**: `Post = Pre + Subtraction`

---

## GPU Environment

```bash
srun --partition=sablab-gpu --gres=gpu:a40:1 --mem=20G --cpus-per-task=4 --pty /bin/bash -i
```

---

## Conda Environment

```bash
conda activate breast_gen
```

---

## Project Structure

```
breast-sub-gen/
├── configs/                    # Configuration JSON files
│   ├── config_network_rflow.json           # Network architecture (dual-channel: 2)
│   ├── config_maisi_controlnet_train_stage*.json  # Training configs
│   ├── environment_maisi_*_stage*.json     # Environment paths
│   └── modality_mapping_breast.json        # Modality mapping
├── scripts/
│   ├── train_controlnet.py     # Main training script
│   ├── infer_controlnet.py     # Inference script
│   ├── visualize_inference.py  # Visualization script
│   ├── train_locator.py        # Stage 1: 3D tumor locator training
│   └── utils.py                # Utilities
├── data/
│   ├── embeddings_breast_sub/  # VAE-encoded latents (64³, 4-ch, [-1,1])
│   ├── processed_pre/          # Pre-contrast images (256³, 1-ch, [0,1])
│   ├── processed_mask/         # Tumor masks (256³, 1-ch)
│   └── dataset_breast_cached.json  # Dataset JSON
├── models/                     # Model checkpoints
│   ├── locator/                # Stage 1: Tumor locator checkpoints
│   └── renderer/               # Stage 2: ControlNet checkpoints
├── weights/
│   ├── autoencoder_v2.pt       # Pre-trained VAE
│   └── diff_unet_3d_rflow-mr.pt # Pre-trained diffusion U-Net
└── outputs/
    ├── logs/                   # Training logs
    └── tfevent/                # TensorBoard events
```

---

## Training Pipeline (Cascade V3.0)

### Stage 0: Infrastructure & Safety (Completed)
- [x] VAE normalization: Subtraction latents in `[-1, 1]`
- [x] No spatial augmentation (only intensity augmentation on Pre)
- [x] Patient-level dataset splits (no data leakage)

### Stage 1: 3D Tumor Locator (Optional)
- Train nnU-Net or DynUNet on Pre-contrast MRI
- Input: 1ch 256³ Pre-contrast → Output: 1ch 256³ Predicted Mask
- Loss: Dice + Focal Loss for extreme class imbalance

### Stage 2: Architecture Refactoring (Completed)
- [x] Dual-channel ControlNet: `conditioning_embedding_in_channels = 2`
- [x] Mask perturbation augmentation (30% probability)
- [x] Weighted L1 loss with ROI weight = 100

### Stage 3: U-Net Domain Pre-training
- **Status**: Ready to start / In progress
- **ControlNet**: Disabled / Removed
- **U-Net**: Fully unfrozen
- **Input**: Unconditional generation
- **LR**: 5e-5
- **Key**: Must use `weighted_loss=100` to prevent all-black predictions

### Stage 4: Cascade Step-wise Fine-tuning

#### Stage 4.1: ControlNet Alignment (Epochs 0-50)
| Parameter | Value |
|-----------|-------|
| U-Net | Frozen |
| ControlNet LR | 1e-4 |
| U-Net LR | 0 |
| Warmup | 5% |
| Schedule | Cosine Annealing |

#### Stage 4.2: Deep Feature Release (Epochs 50-100)
| Parameter | Value |
|-----------|-------|
| Unfreeze | down_blocks.2, down_blocks.3, middle_block, up_blocks.0, up_blocks.1 |
| ControlNet LR | 5e-5 |
| U-Net LR | 3e-5 |
| Gradient Checkpointing | Required |

#### Stage 4.3: Full Refinement (Epochs 100-150)
| Parameter | Value |
|-----------|-------|
| U-Net | Fully unfrozen |
| ControlNet LR | 3e-5 |
| U-Net LR | 1e-5 |

---

## Training Commands

```bash
# Stage 3: U-Net Domain Pre-training
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage3.json \
    --model_config_path configs/config_maisi_controlnet_train_stage3.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1

# Stage 4.1: ControlNet Alignment
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage4_1.json \
    --model_config_path configs/config_maisi_controlnet_train_stage4_1.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1

# Stage 4.2: Deep Feature Release
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage4_2.json \
    --model_config_path configs/config_maisi_controlnet_train_stage4_2.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1

# Stage 4.3: Full Refinement
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage4_3.json \
    --model_config_path configs/config_maisi_controlnet_train_stage4_3.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1
```

---

## Architecture Notes

### Dual-Channel ControlNet Input

```python
pre_images = batch["pre"].to(device)      # [B, 1, 256, 256, 256]
masks = batch["label"].to(device)          # [B, 1, 256, 256, 256]

# 30% probability: Mask morphological perturbation
if torch.rand(1).item() < 0.3:
    masks = apply_random_morphological_perturbation(masks)

# Concatenate: [B, 2, 256, 256, 256]
controlnet_cond = torch.cat([pre_images.float(), masks.float()], dim=1)
```

### Loss Function (Weighted L1)

```python
# Downscale ROI mask from 256³ to 64³ using Max Pooling
# Ensures any tumor pixel in 4x4x4 region marks the latent voxel
roi_mask_latent = F.max_pool3d(batch["label"].float(), kernel_size=4, stride=4) > 0.0

# L1 loss with reduction="none" for spatial weighting
raw_l1_loss = F.l1_loss(model_output.float(), model_gt.float(), reduction="none")

# Weight mask: Background = 1.0, ROI = 100.0
weight_mask = torch.ones_like(raw_l1_loss)
if roi_mask_latent.sum() > 0:
    roi_mask_expanded = roi_mask_latent.repeat(1, model_output.shape[1], 1, 1, 1)
    weight_mask[roi_mask_expanded] = args.controlnet_train["weighted_loss"]

# Final loss
loss = (raw_l1_loss * weight_mask).mean()
```

---

## Inference Pipeline

```python
# 1. Locator: Predict tumor mask from Pre-contrast
predicted_mask = locator(Pre)  # [1, 1, 256, 256, 256]

# 2. Renderer: Generate subtraction from Pre + Predicted Mask
controlnet_cond = torch.cat([Pre, predicted_mask], dim=1)
predicted_sub_latent = renderer(controlnet_cond, timestep=0)
predicted_sub = vae_decoder(predicted_sub_latent)  # [1, 1, 256, 256, 256]

# 3. Final: Post = Pre + Subtraction
Post = Pre + predicted_sub
```

**CFG Guidance**: Use `cfg_guidance_scale = 1.0` only (no CFG)

---

## Configuration System

| Config Type | Purpose | Key Parameters |
|-------------|---------|-----------------|
| `environment_*.json` | Paths, data locations, `start_epoch` | `train_json`, `val_json`, `vae_path`, `unet_path` |
| `config_maisi_controlnet_train_*.json` | Training hyperparameters | `lr`, `batch_size`, `weighted_loss`, `unfreeze_blocks` |
| `config_network_*.json` | Network architecture | `conditioning_embedding_in_channels: 2`, `latent_channels: 4` |

---

## Important Constraints

1. **NO spatial augmentation** - only intensity augmentation on Pre (15% Gaussian noise, 20% contrast adjustment)
2. **Patient-level splits** - strict three-level stratification (dataset → tumor status → patient ID)
3. **CFG Guidance**: Use `cfg_guidance_scale = 1.0` only
4. **Gradient checkpointing** required for Stage 4.2+
5. **VAE normalization**: Input subtraction latents must be scaled to `[-1, 1]`

---

## Dataset Statistics

**Total**: 1,943 samples | 1,504 patients

| Split | Samples | Patients | With Tumor | No Tumor |
|-------|---------|----------|------------|----------|
| Train | 1,553 (80%) | 1,203 | 77.4% | 22.6% |
| Val   | 390 (20%)  | 301    | 77.2% | 22.8% |

---

## Current Training Status

**Starting from**: Stage 3 (U-Net Domain Pre-training)

- [ ] Stage 3: U-Net Domain Pre-training
- [ ] Stage 4.1: ControlNet Alignment
- [ ] Stage 4.2: Deep Feature Release
- [ ] Stage 4.3: Full Refinement
