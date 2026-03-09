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

1. **`environment_*.json`**: Paths, data locations, model checkpoint paths, `start_epoch` for multi-stage training
2. **`config_maisi_controlnet_train_*.json`**: Training hyperparameters (batch size, LR, epochs, unfrozen blocks)
3. **`config_network_*.json`**: Network architecture definitions

Key parameters in `config_network_rflow.json`:
- `conditioning_embedding_in_channels`: **2** (dual-channel: Pre + Mask)
- `latent_channels`: 4
- `num_class_embeds`: 128 (for modality embedding)

---

## Training Commands

### Progressive Training (Recommended)

```bash
# Stage 1: ControlNet only (epochs 1-30)
sbatch scripts/submit_stage1.sh

# Stage 2: Deep U-Net blocks (epochs 31-100)
sbatch scripts/submit_stage2.sh

# Stage 3: All U-Net blocks (epochs 101+)
sbatch scripts/submit_stage3.sh
```

### Single GPU Training

```bash
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast.json \
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
Main training loop with gradient checkpointing for Stage 2+ (mixed frozen/unfrozen U-Net parameters).
Key functions:
- `train_controlnet()`: Training loop with L1 + background penalty loss
- `compute_model_output()`: Forward pass through ControlNet → U-Net with dual-channel input, supports `use_checkpoint=True`
- `apply_random_morphological_perturbation()`: Pure PyTorch dilation/erosion augmentation
- `set_unet_frozen_state()`: Selective freeze/unfreeze U-Net blocks for progressive training

### `scripts/infer_controlnet.py`
Inference script for generating subtraction images. Uses dual-channel input.

### `scripts/sample.py`
Diffusion sampling/inference with CFG guidance validation.
**VAE解码后处理**: VAE解码输出不在[0, 1]范围，使用实际范围归一化
**WARNING**: `cfg_guidance_scale` should be 0 or 1.0 (model not trained with mask dropout).

### `scripts/visualize_inference.py`
Inference visualization script for model validation:
- Multi-view visualization (axial, sagittal, coronal)
- Statistics output (tumor enhancement intensity)
- Supports all stage checkpoints (1/2/3)

### `scripts/utils.py`
- `prepare_maisi_controlnet_json_dataloader()`: Creates train/val dataloaders from JSON
- `define_instance()`: Instantiates models from config using MONAI ConfigParser

---

## Architecture Notes

### ControlNet Input Processing (Dual-Channel)

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

**Pure PyTorch morphology**: Dilation via `F.max_pool3d`, Erosion via `1 - F.max_pool3d(1 - mask)`

### Loss Computation

```python
# 1. Precise dimension reduction using Max Pooling
roi_mask_latent = F.max_pool3d(labels.float(), kernel_size=4, stride=4) > 0.0

# 2. Base Global L1 with moderate ROI weighting
weights = torch.ones_like(model_output)
weights[roi_mask_latent.repeat(1, model_output.shape[1], 1, 1, 1)] = 3.0
l1_loss_raw = F.l1_loss(model_output.float(), model_gt.float(), reduction="none")
loss = (l1_loss_raw * weights).mean()

# 3. Absolute Background Penalty (suppresses false positives)
bg_mask_expanded = (~roi_mask_latent).repeat(1, model_output.shape[1], 1, 1, 1) if roi_mask_latent.sum() > 0 else torch.ones_like(model_output, dtype=torch.bool)
pred_bg = model_output.float()[bg_mask_expanded]
gt_bg = model_gt.float()[bg_mask_expanded]
false_positive_bg = F.relu(pred_bg - gt_bg)
loss = loss + 5.0 * (false_positive_bg ** 2).mean()
```

---

## Training Strategy: Progressive Co-Tuning

Uses **shell script stages** with fresh optimizer per stage. Global epoch numbering via `start_epoch` in environment configs.

### Stage 1: ControlNet Alignment (Epochs 1-30)
- **U-Net**: Fully frozen
- **ControlNet LR**: 1e-4
- **LR Schedule**: PolynomialLR (power=2.0)
- **Validation**: Every 2 epochs
- **Output**: `breast_controlnet_stage1_best.pt`

### Stage 2: Deep Semantic Release (Epochs 31-100)
- **Unfreeze**: `down_blocks.2`, `down_blocks.3`, `middle_block`, `up_blocks.0-3`
- **ControlNet LR**: 5e-5 | **U-Net LR**: 3e-5 (60% of ControlNet)
- **LR Schedule**: Warmup (5% epochs) + Cosine Annealing (to 10%)
- **Validation**: Every epoch
- **Gradient Checkpointing**: Enabled (required for mixed frozen/unfrozen params)
- **DDP**: Both ControlNet and U-Net wrapped
- **Output**: `breast_controlnet_stage2_best.pt`

### Stage 3: Shallow Edge Refinement (Epochs 101+)
- **Unfreeze**: All U-Net blocks
- **ControlNet LR**: 3e-5 | **U-Net LR**: 1e-5
- **Validation**: Every epoch
- **Output**: `breast_controlnet_stage3_best.pt`

---

## Checkpoint Strategy

| File | Description |
|------|-------------|
| `{exp_name}_epoch_{N}.pt` | Saved every epoch (local epoch in filename) |
| `{exp_name}_best.pt` | Best validation loss checkpoint |

**Checkpoint contents**:
- `epoch`: Global epoch number
- `train_loss`, `val_loss`: Loss values
- `controlnet_state_dict`: ControlNet weights
- `unet_state_dict`: U-Net weights (Stage 2+: only unfrozen blocks)
- `scale_factor`: VAE scaling factor

---

## Important Constraints

1. **NO spatial augmentation** during training - breaks alignment between 256³ inputs and 64³ latents
2. **Background must be exactly 0** in subtraction images to prevent snow noise
3. **Patient-level splits** critical - no data leakage (handled in dataset JSON)
4. **VAE scale factor** from pretrained checkpoint required (no additional normalization)
5. **CFG Guidance**: Use `cfg_guidance_scale = 0` or `1.0` only (no mask dropout training)
6. **Stage 2+ requires gradient checkpointing** to avoid inplace operation errors with mixed frozen/unfrozen U-Net parameters

---

## Dataset Statistics

**Total**: 1,943 samples | 1,504 patients

| Split | Samples | Patients | With Tumor | No Tumor |
|-------|---------|----------|------------|----------|
| Train | 1,553 (80%) | 1,203 | 77.4% | 22.6% |
| Val    | 390 (20%)  | 301   | 77.2% | 22.8% |

**Stage 1 Results** (Epoch 28 best checkpoint):
- 肿瘤区域增强: 750-965 HU (mean ~880 HU)
- 整体平均值: ~450 HU
- 模型学会基本 subtraction 生成模式

**Stage 2 In Progress** (Started 2026-03-06):
- 70 epochs (31-100)
- Expected runtime: ~16-17 hours on A40
- Status: Training
