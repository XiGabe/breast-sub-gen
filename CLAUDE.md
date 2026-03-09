# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**3D breast MRI subtraction synthesis** using ControlNet + RFlow diffusion model.

**Architecture**: Pre-contrast MRI + Tumor Mask → [VAE encode] → Latent Space → [ControlNet + U-Net] → Predicted Subtraction → [VAE decode] → Subtraction Image

**Data Flow**: Post = Pre + Subtraction

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
│   └── utils.py                # Utilities
├── data/
│   ├── embeddings_breast_sub/  # VAE-encoded latents (64³, 4-ch)
│   ├── processed_pre/          # Pre-contrast images (256³, 1-ch)
│   ├── processed_mask/         # Tumor masks (256³, 1-ch)
│   └── dataset_breast_cached.json  # Dataset JSON
├── models/                     # Model checkpoints
├── weights/
│   ├── autoencoder_v2.pt       # Pre-trained VAE
│   └── diff_unet_3d_rflow-mr.pt # Pre-trained diffusion U-Net
└── outputs/
    ├── logs/                   # Training logs
    └── tfevent/                # TensorBoard events
```

---

## Training Strategy

### Stage 1: ControlNet Alignment (Epochs 1-50)
- **U-Net**: Fully frozen
- **ControlNet LR**: 1e-4
- **LR Schedule**: Warmup (5%) + Cosine Annealing
- **Validation**: Every epoch
- **Loss**: Weighted MSE (ROI: 5.0, Background: 1.0)

### Stage 2: Deep Semantic Release (Epochs 51-100)
- **Unfreeze**: Deep U-Net blocks
- **ControlNet LR**: 5e-5 | **U-Net LR**: 3e-5

### Stage 3: Shallow Edge Refinement (Epochs 101+)
- **Unfreeze**: All U-Net blocks
- **ControlNet LR**: 3e-5 | **U-Net LR**: 1e-5

---

## Training Commands

```bash
# Stage 1
sbatch scripts/submit_stage1.sh

# Manual start
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage1.json \
    --model_config_path configs/config_maisi_controlnet_train_stage1.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1
```

---

## Architecture Notes

### ControlNet Input (Dual-Channel)

```python
# Dual-channel condition: [Pre-contrast MRI, Tumor Mask]
controlnet_cond = torch.cat([pre_images, masks], dim=1)
# Output: [B, 2, 256, 256, 256]
```

### Loss Function (Weighted MSE)

```python
# ROI masking (256³ -> 64³)
roi_mask_latent = F.max_pool3d(labels, kernel_size=4, stride=4) > 0.0

# MSE loss with ROI weighting
raw_mse_loss = F.mse_loss(model_output, model_gt, reduction="none")
weight_mask = torch.ones_like(raw_mse_loss)
weight_mask[roi_mask_latent.repeat(1, 4, 1, 1, 1)] = 5.0

loss = (raw_mse_loss * weight_mask).mean()
```

---

## Configuration System

- **`environment_*.json`**: Paths, data locations, `start_epoch`
- **`config_maisi_controlnet_train_*.json`**: Training hyperparameters
- **`config_network_*.json`**: Network architecture definitions

Key parameters:
- `conditioning_embedding_in_channels`: **2** (dual-channel: Pre + Mask)
- `latent_channels`: 4
- `num_class_embeds`: 128

---

## Important Constraints

1. **NO spatial augmentation** during training
2. **Patient-level splits** critical - no data leakage
3. **CFG Guidance**: Use `cfg_guidance_scale = 0` or `1.0` only
4. **Stage 2+ requires gradient checkpointing**

---

## Dataset Statistics

**Total**: 1,943 samples | 1,504 patients

| Split | Samples | Patients | With Tumor | No Tumor |
|-------|---------|----------|------------|----------|
| Train | 1,553 (80%) | 1,203 | 77.4% | 22.6% |
| Val    | 390 (20%)  | 301   | 77.2% | 22.8% |
