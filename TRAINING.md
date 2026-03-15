# Training Log

**Project**: 3D Breast MRI Subtraction Synthesis with Cascade V2.0

---

## Stage 0: Data Pipeline & Safety Check

**Status**: Complete

- [x] VAE Normalization: Subtraction mapped to `[-1, 1]`
- [x] Offline Caching: `sub_emb.nii.gz` (64³, 4-ch)
- [x] Spatial Augmentation: Disabled (intensity only on Pre)
- [x] Patient-level Split: 3-layer stratified sampling

---

## Stage 1: 3D Tumor Locator (The Locator)

**Status**: Pending

**Goal**: Train nnU-Net to detect tumors in pre-contrast MRI

- [ ] **1.1** Configure nnU-Net v2 or MONAI DynUNet
- [ ] **1.2** Set Dice + Focal Loss for highly imbalanced segmentation
- [ ] **1.3** Train on pre-contrast images
- [ ] **1.4** Evaluate - acceptable if locator roughly框出位置

**Expected Output**: `models/locator/best_model.pt`

---

## Stage 2: Architecture & Loss Refactoring

**Status**: Complete

- [x] `conditioning_embedding_in_channels = 2` (dual-channel)
- [x] Mask perturbation: 30% probability for morphological augmentation
- [x] Loss: Weighted L1 (ROI: 100.0, Background: 1.0)

---

## Stage 3: Progressive Co-Tuning (Cascade Renderer)

### Stage 3.1: ControlNet Anchor Phase (Epochs 0-50)

**Status**: Ready to start

**Configuration**:
- U-Net: Fully frozen (100%)
- ControlNet LR: `1e-4`
- U-Net LR: `0.0`
- ROI Weight: `100`
- Batch: 4

**Expected Output**: `models/breast_controlnet_stage1_best.pt`

---

### Stage 3.2: Bottleneck Adaptation (Epochs 50-100)

**Status**: Pending

**Configuration**:
- U-Net Unfreeze: `["down_blocks.3", "middle_block", "up_blocks.0"]`
- ControlNet LR: `5e-5`
- U-Net LR: `1e-5`
- Gradient Checkpointing: Enabled

---

### Stage 3.3: Mid-level Texture Release (Epochs 100-150)

**Status**: Pending

**Configuration**:
- U-Net Unfreeze追加: `["down_blocks.2", "down_blocks.3", "middle_block", "up_blocks.0", "up_blocks.1"]`
- ControlNet LR: `1e-5`
- U-Net LR: `5e-6`

---

### Stage 3.4: Full Refinement (Epochs 150-200)

**Status**: Pending

**Configuration**:
- U-Net: Full Unfreeze
- ControlNet LR: `5e-6`
- U-Net LR: `1e-6` to `5e-6`

---

## Stage 4: Fully Automated Inference

**Status**: Pending

**Pipeline**:
1. Input unknown Pre-contrast → **Locator** → `Predicted Mask`
2. Concatenate Pre + Predicted Mask → **Renderer** → `Predicted Subtraction`
3. Compute `Post_syn = Pre_raw + Sub_pred`

---

## Training Commands

```bash
# Stage 3.1 (ControlNet Anchor)
sbatch scripts/submit_stage1.sh

# Manual start
conda activate breast_gen
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage1.json \
    --model_config_path configs/config_maisi_controlnet_train_stage1.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1

# Stage 3.2+ (requires gradient_checkpointing)
# Update environment config start_epoch and model config
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage2.json \
    --model_config_path configs/config_maisi_controlnet_train_stage2.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1
```

---

## Monitoring

```bash
# Real-time logs
tail -f outputs/logs/breast_controlnet_stage1/train.log

# TensorBoard
tensorboard --logdir outputs/tfevent/

# Check validation results
grep "Validation COMPLETE" outputs/logs/breast_controlnet_stage1/train.log

# Check loss values
grep "loss" outputs/logs/breast_controlnet_stage1/train.log
```

---

## Dataset Statistics

| Split | Samples | Patients | With Tumor | No Tumor |
|-------|---------|----------|------------|----------|
| Train | 1,553 (80%) | 1,203 | 77.4% | 22.6% |
| Val    | 390 (20%)  | 301   | 77.2% | 22.8% |

---

## Checkpoints

| Stage | Epoch | Best Loss | Status |
|-------|-------|-----------|--------|
| 0 (Data) | - | - | Complete |
| 1 (Locator) | - | - | Pending |
| 2 (Refactor) | - | - | Complete |
| 3.1 (Anchor) | - | - | Ready |
| 3.2 (Bottleneck) | - | - | Pending |
| 3.3 (Texture) | - | - | Pending |
| 3.4 (Full) | - | - | Pending |
| 4 (Inference) | - | - | Pending |
