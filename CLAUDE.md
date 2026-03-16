# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**3D breast MRI subtraction synthesis** using Cascade V2.0 architecture:
- **Stage 0**: 3D Locator (nnU-Net) for tumor detection
- **Stage 1**: Dual-channel ControlNet + RFlow diffusion model for high-fidelity rendering

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
│   └── locator/               # nnU-Net locator models
├── weights/
│   ├── autoencoder_v2.pt       # Pre-trained VAE
│   └── diff_unet_3d_rflow-mr.pt # Pre-trained diffusion U-Net
└── outputs/
    ├── logs/                   # Training logs
    └── tfevent/                # TensorBoard events
```

---

## Training Pipeline (Cascade V2.0)

### Stage 0: Data Pipeline & Safety Check

- [ ] **0.1 VAE Normalization & Offline Caching**
    - **Target**: Subtraction (Sub) encoded via VAE to $64^3$ latent, values mapped to `[-1, 1]` (apply `Sub * 2.0 - 1.0` before encoding)
    - **Output**: `[-1, 1]`, convert to `[0, 1]` after decoding
    - **Cached files**: `sub_emb.nii.gz` ($64^3$, 4-ch); Pre and Mask at original $256^3$ (1-ch each)

- [ ] **0.2 Disable Spatial Augmentation**
    - **Action**: Remove all spatial deformations (RandomCrop, RandomAffine, RandomFlip)
    - **Rationale**: Ensure absolute spatial alignment between $256^3$ physical images and $64^3$ latents
    - **Alternative**: Apply intensity augmentation only on Pre (RandGaussianNoise 15%, RandAdjustContrast 20%)

- [ ] **0.3 Patient-level Dataset JSON**
    - **Structure**: `{"image": "sub_emb.nii.gz", "condition": "pre_norm.nii.gz", "label": "mask.nii.gz"}`
    - **Split**: Strict 3-layer stratified sampling (Dataset → Tumor Status → Patient ID) to prevent data leakage

---

### Stage 1: 3D Tumor Locator (The Locator)

*Goal: Train a dedicated 3D segmentation network to detect anatomical abnormalities in pre-contrast MRI, eliminating dependency on expert masks.*

- [ ] **1.1 Select and Configure Locator Network**
    - **Tools**: nnU-Net v2 (recommended) or MONAI DynUNet
    - **Input/Output**: Input 1ch $256^3$ Pre-contrast → Output 1ch $256^3$ Predicted Mask

- [ ] **1.2 Set Up Highly Imbalanced Segmentation Loss**
    - **Action**: Use `Dice Loss + Focal Loss` combo to handle 99% background vs 1% tiny tumors

- [ ] **1.3 Evaluate Predictions**
    - **Expectation**: Perfect 100% detection not required; acceptable if the locator roughly框出位置 (框出大概位置即可)

---

### Stage 2: Architecture & Loss Refactoring

- [x] **2.1 Network Config** (`configs/config_network_rflow.json`)
    - `conditioning_embedding_in_channels`: **2** (dual-channel: Pre + Mask)

- [x] **2.2 ControlNet Input with Mask Perturbation** (`scripts/train_controlnet.py`)

    ```python
    pre_images = batch["pre"].to(device)
    masks = batch["label"].to(device)

    # 30% probability for morphological perturbation (simulate Locator errors)
    if torch.rand(1).item() < 0.3:
        masks = apply_random_morphological_perturbation(masks)

    # Concatenate: [B, 2, 256, 256, 256]
    controlnet_cond = torch.cat([pre_images.float(), masks.float()], dim=1)
    ```

- [x] **2.3 High-Fidelity Weighted L1 Loss**

    ```python
    # 1. Precise downsampling with Max Pooling (preserve tiny tumor pixels)
    # 256³ -> 64³ (strict 4x reduction, no overlap)
    # If any tumor pixel exists in 4x4x4 region, set latent point to True
    roi_mask_latent = F.max_pool3d(batch["label"].float(), kernel_size=4, stride=4) > 0.0

    # 2. Base L1 Loss (following MAISI official diffusion training)
    raw_l1_loss = F.l1_loss(model_output.float(), model_gt.float(), reduction="none")

    # 3. Dynamic Gradient Balancing
    weight_mask = torch.ones_like(raw_l1_loss)

    # 4. Solve 3D sparsity (100x strong boost)
    # Tumor occupies ~0.07% in latent space; 100x weight boosts gradient contribution to ~7%
    if roi_mask_latent.sum() > 0:
        roi_mask_expanded = roi_mask_latent.repeat(1, model_output.shape[1], 1, 1, 1)
        weight_mask[roi_mask_expanded] = args.controlnet_train["weighted_loss"]

    # 5. Final Loss (unbiased)
    loss = (raw_l1_loss * weight_mask).mean()
    ```

---

### Stage 3: Progressive Co-Tuning (Cascade Renderer)

*Goal: Abandon unconditional pretraining; use GT Mask as spatial constraint. Five-step progressive strategy.*

#### Stage 3.1: ControlNet 独立探路期 (The Anchor Phase | Epoch 0-50)
- **目标**: 构建安全锚点。U-Net 绝对冻结，强迫 ControlNet 专心学习双通道 cat(Pre, Mask) 的空间坐标映射
- **U-Net**: 100% frozen
- **Unfreeze**: `[]`
- **ControlNet LR**: `1e-4`
- **U-Net LR**: `0.0`
- **ROI Weight**: `weighted_loss = 100` (强制关注 1% 阳性区域)
- **Batch**: 4

#### Stage 3.2: 深层语义适应期 (Bottleneck Adaptation | Epoch 51-70)
- **目标**: 解决"均值稀释"黑洞。在 ControlNet 提供的精准空间信号指引下，放开 U-Net 的深层瓶颈
- **Unfreeze**: `["down_blocks.3", "middle_block", "up_blocks.0"]`
- **ControlNet LR**: `5e-5`
- **U-Net LR**: `1e-5` (必须比 ControlNet 小 10 倍)
- **ROI Weight**: `weighted_loss = 20`
- **Batch**: 4, enable `gradient_checkpointing`

#### Stage 3.3: 中层纹理过渡期 (Mid-level Texture Release | Epoch 71-90)
- **目标**: 释放中层特征路由。让中层网络学习造影剂的异质性分布
- **Unfreeze 追加**: `["down_blocks.2", "down_blocks.3", "middle_block", "up_blocks.0", "up_blocks.1"]`
- **ControlNet LR**: `1e-5`
- **U-Net LR**: `5e-6`
- **ROI Weight**: `weighted_loss = 20`

#### Stage 3.4: 次浅层高频引入期 (High-Frequency Injection | Epoch 91-130)
- **目标**: 攻克高难度病灶 (如毛刺边缘、局灶血管)。释放次浅层的 down.1 和 up.2
- **Unfreeze**: `["down_blocks.1", "down_blocks.2", "down_blocks.3", "middle_block", "up_blocks.0", "up_blocks.1", "up_blocks.2"]`
- **ControlNet LR**: `1e-5` (保持高推力，防止局部最优)
- **U-Net LR**: `5e-6` 到 `8e-6`
- **ROI Weight**: `weighted_loss = 20`
- **注意**: 必须启用 Cosine Annealing 和至少 500 步 Warmup，防止梯度爆炸

#### Stage 3.5: 全解封极限画质期 (Full Refinement | Epoch 131-180)
- **目标**: 打通全局梯度，终极画质释放。让最浅层参与进来，解决最后的稀疏性问题
- **Unfreeze**: Full Unfreeze (全部 blocks)
- **ControlNet LR**: `5e-6`
- **U-Net LR**: `1e-6` 到 `2e-6` (极度保守)
- **ROI Weight**: `weighted_loss = 20`

---

### Stage 4: Fully Automated Inference

- [ ] **4.1 CFG-free Inference**
    - **Action**: Since unconditional generation is abandoned, force `guidance_scale = 1.0`

- [ ] **4.2 Cascade Inference Pipeline**
    - **Step 1**: Input unknown Pre-contrast → **Locator (3D nnU-Net)** → Get `Predicted Mask`
    - **Step 2**: Concatenate Pre + Predicted Mask → **Renderer (MAISI)** → Get `Predicted Subtraction`
    - **Step 3**: Compute `Post_syn = Pre_raw + Sub_pred` (ensure inverse transform back to physical intensity)

---

## Architecture Diagram

```
========================================================================================
                      【Stage 1: 3D Semantic Locator】
========================================================================================
                                     |
   [ Unknown Pre-contrast MRI ] --------+
   (Physical: 1ch, 256³)             |
                                     V
                     +-------------------------------+
                     |   3D nnU-Net / DynUNet        |  <-- (Dice+Focal Loss)
                     |   (Extract anatomical clues) |
                     +-------------------------------+
                                     |
                                     V
                        [ Predicted 3D Tumor Mask ]
                        (Physical: 1ch, 256³)

========================================================================================
                      【Stage 2: 3D High-Fidelity Renderer】
========================================================================================

   [ Pre-contrast ] --------------+       +-------------- [ Predicted Mask ]
   (1ch, 256³)                    |       |               (1ch, 256³)
                                  V       V
                            ( torch.cat dim=1 )
                                      |
                     [ Joint Condition Input (2ch, 256³) ]
                                      |
       +--------------------------------------------------------------+
       | MAISI ControlNet (Condition Encoding)                       |
       | Step Conv: 2ch -> 16ch -> 32ch -> 64ch (down to 64³)        |
       +--------------------------------------------------------------+
                                      |
                                      V
       +--------------------------------------------------------------+
       | MAISI 3D Diffusion Transformer (U-Net)                       |
       | (Latent 64³ with Progressive Unfreeze)                      |
       +--------------------------------------------------------------+
                                      |
                                      V
                           [ Predicted Sub_Latent ]
                                      |
       +--------------------------------------------------------------+
       | VAE Decoder (64³ latent -> Physical space)                 |
       +--------------------------------------------------------------+
                                      |
                                      V
                       [ Predicted 3D Subtraction ]
                               (1ch, 256³)
                                      |
                     ( Post = Pre + Predicted Subtraction )
                                      |
                                      V
                   🌟 【Final Output: High-Fidelity 3D Virtual Enhanced MRI】 🌟
========================================================================================
```

---

## Training Commands

```bash
# Stage 3.1 (ControlNet Anchor)
sbatch scripts/submit_stage1.sh

# Manual start
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage1.json \
    --model_config_path configs/config_maisi_controlnet_train_stage1.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1
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
- `weighted_loss`: **100** (ROI region weight)

---

## Important Constraints

1. **NO spatial augmentation** during training (only intensity augmentation on Pre)
2. **Patient-level splits** critical - no data leakage
3. **CFG Guidance**: Use `guidance_scale = 1.0` only (no unconditional generation)
4. **Stage 3.2+ requires gradient checkpointing**
5. **Mask perturbation**: 30% probability for morphological augmentation

---

## Dataset Statistics

**Total**: 1,943 samples | 1,504 patients

| Split | Samples | Patients | With Tumor | No Tumor |
|-------|---------|----------|------------|----------|
| Train | 1,553 (80%) | 1,203 | 77.4% | 22.6% |
| Val    | 390 (20%)  | 301   | 77.2% | 22.8% |
