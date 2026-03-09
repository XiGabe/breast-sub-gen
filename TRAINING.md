# Training Documentation - 3D Breast MRI Subtraction Generation

## Project Goal

Train dual-channel ControlNet for high-fidelity 3D breast MRI subtraction synthesis

**Architecture**: 3D Locator (nnU-Net) + MAISI ControlNet (dual-channel: Pre+Mask) + Progressive Unfreezing

**Last Updated**: 2026-03-06

---

## Current Status

- [x] Data preprocessing complete (data/step_4)
- [x] Offline caching complete (embeddings_breast_sub, processed_mask, processed_pre)
- [x] Dataset JSON generated (dataset_breast_cached.json)
- [x] **Stage 1 complete** (30 epochs, best checkpoint saved)
- [ ] **Stage 2 in progress** (70 epochs, started 2026-03-06)

---

## Training Stages

### ✅ Stage 1: ControlNet Alignment (Epochs 1-30, Complete)

**Goal**: Train ControlNet with U-Net fully frozen

**Configuration**:
- U-Net: Fully frozen
- ControlNet LR: 1e-4
- LR Schedule: PolynomialLR (power=2.0)
- Validation: Every 2 epochs
- Config: `config_maisi_controlnet_train_stage1.json`

**Results** (Epoch 28 best):
- Train Loss: 1.8885
- Val Loss: 8.4696 (L1: 1.3802, BgPen: 7.0894)
- Output: `models/breast_controlnet_stage1_best.pt`

**Inference Verification**:
- Tumor enhancement: 750-965 HU (mean ~880 HU)
- Overall mean: ~450 HU
- Model learned basic subtraction generation

### 🔄 Stage 2: Deep Semantic Release (Epochs 31-100, In Progress)

**Goal**: Unfreeze deep U-Net blocks to improve brightness consistency

**Configuration**:
- **Unfrozen blocks**: `down_blocks.2`, `down_blocks.3`, `middle_block`, `up_blocks.0-3`
- **ControlNet LR**: 5e-5
- **U-Net LR**: 3e-5 (60% of ControlNet)
- **LR Schedule**: Warmup (5%) + Cosine Annealing (to 10%)
- **Validation**: Every epoch
- **Gradient Checkpointing**: Enabled (required for mixed frozen/unfrozen)
- **DDP**: Both ControlNet and U-Net wrapped
- Config: `config_maisi_controlnet_train_stage2.json`
- Script: `scripts/submit_stage2.sh`

**Started**: 2026-03-06
**Expected Runtime**: ~16-17 hours on A40

### 📋 Stage 3: Shallow Edge Refinement (Epochs 101+, Pending)

**Goal**: Unfreeze all U-Net blocks for edge refinement

**Configuration**:
- **Unfrozen blocks**: All U-Net blocks
- **ControlNet LR**: 3e-5
- **U-Net LR**: 1e-5
- Config: `config_maisi_controlnet_train_stage3.json`
- Script: `scripts/submit_stage3.sh`

---

## Network Architecture

### ControlNet Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `conditioning_embedding_in_channels` | **2** | Dual-channel: [Pre-contrast MRI, Tumor Mask] |
| `latent_channels` | 4 | VAE latent channels |
| `num_class_embeds` | 128 | Modality embedding classes |

### Input/Output Shapes

| Component | Shape | Description |
|-----------|-------|-------------|
| Pre-contrast MRI | [B, 1, 256, 256, 256] | Physical resolution input |
| Tumor Mask | [B, 1, 256, 256, 256] | Binary segmentation mask |
| ControlNet Condition | [B, 2, 256, 256, 256] | Concatenated [Pre, Mask] |
| VAE Latent (target) | [B, 4, 64, 64, 64] | Pre-encoded subtraction |
| Model Output | [B, 4, 64, 64, 64] | Predicted subtraction latent |

---

## Loss Function

**Implemented**: L1 Loss + Background Penalty

```python
# ROI masking via Max Pooling (preserves tiny lesions)
roi_mask_latent = F.max_pool3d(labels.float(), kernel_size=4, stride=4) > 0.0

# L1 with ROI weighting (3.0x)
weights = torch.ones_like(model_output)
weights[roi_mask_latent.repeat(1, model_output.shape[1], 1, 1, 1)] = 3.0
l1_loss = (F.l1_loss(model_output, model_gt, reduction="none") * weights).mean()

# Background penalty (suppresses false positives)
bg_mask = ~roi_mask_latent
false_positive = F.relu(model_output[bg_mask] - model_gt[bg_mask])
bg_penalty = 5.0 * (false_positive ** 2).mean()

loss = l1_loss + bg_penalty
```

---

## Running Training

### Stage-wise (Recommended)

```bash
sbatch scripts/submit_stage1.sh  # Stage 1: epochs 1-30
sbatch scripts/submit_stage2.sh  # Stage 2: epochs 31-100
sbatch scripts/submit_stage3.sh  # Stage 3: epochs 101+
```

### Direct Command

```bash
conda activate breast_gen
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage2.json \
    --model_config_path configs/config_maisi_controlnet_train_stage2.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1
```

### Monitoring

```bash
# Real-time logs
tail -f outputs/logs/breast_controlnet_stage2/train.log

# TensorBoard
tensorboard --logdir outputs/tfevent/

# Check validation results
grep "Validation COMPLETE" outputs/logs/breast_controlnet_stage2/train.log
```

---

## Dataset Statistics

| Split | Samples | Patients | With Tumor | No Tumor |
|-------|---------|----------|------------|----------|
| Train | 1,553 (80%) | 1,203 | 77.4% | 22.6% |
| Val    | 390 (20%)  | 301   | 77.2% | 22.8% |

---

## Modification Log

### 2026-03-06 - Stage 2 Training Started

**Changes**:
- ✅ Global epoch numbering via `start_epoch` in environment configs
- ✅ Warmup + Cosine Annealing LR scheduler (replaces PolynomialLR)
- ✅ Enhanced validation loop with detailed logging and skip counting
- ✅ Gradient checkpointing for U-Net forward pass (required for Stage 2+)
- ✅ DDP wrapping for both ControlNet and U-Net when unfrozen
- ✅ Stage 2 config: 70 epochs, validation every epoch, dual LR (5e-5 / 3e-5)

**Files Modified**:
- `scripts/train_controlnet.py`: Global epoch, checkpoint wrapper, validation enhancements
- `configs/config_maisi_controlnet_train_stage2.json`: n_epochs=70, validation_frequency=1, unet_lr=3e-5
- `configs/environment_*_stage*.json`: Added `start_epoch` field
- `scripts/submit_stage2.sh`: Enhanced output summary

### 2026-03-06 - Stage 1 Complete

**Completed**: 30 epochs training, inference verification
**Best checkpoint**: Epoch 28 (breast_controlnet_stage1_best.pt)
