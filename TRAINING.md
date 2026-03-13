# Training SOP (Cascade V3.0)

**Project**: 3D Breast MRI Subtraction Synthesis
**Architecture**: 3D Locator + Dual-Channel ControlNet Renderer

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Stage 3: U-Net Domain Pre-training](#stage-3-u-net-domain-pre-training)
3. [Stage 4.1: ControlNet Alignment](#stage-41-controlnet-alignment)
4. [Stage 4.2: Deep Feature Release](#stage-42-deep-feature-release)
5. [Stage 4.3: Full Refinement](#stage-43-full-refinement)
6. [Inference](#inference)
7. [Monitoring](#monitoring)

---

## Quick Start

```bash
# Activate environment
conda activate breast_gen

# Start from a specific stage
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage3.json \
    --model_config_path configs/config_maisi_controlnet_train_stage3.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1
```

---

## Stage 3: U-Net Domain Pre-training

**Objective**: Train U-Net to generate subtraction-like images without ControlNet, solving the "all-black" shortcut problem.

### Configuration

| Parameter | Value |
|-----------|-------|
| ControlNet | Disabled |
| U-Net | Fully Unfrozen |
| LR | 5e-5 |
| weighted_loss | 100 |
| Epochs | 50 |
| Batch Size | 1 |
| Input | Unconditional (no condition) |

### Training Command

```bash
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage3.json \
    --model_config_path configs/config_maisi_controlnet_train_stage3.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1
```

### Expected Behavior

- Model generates random-looking but plausible vascular/tumor patterns
- ROI regions show higher intensity (due to weighted_loss=100)
- Background remains mostly dark (subtraction is sparse)

### Checkpoint Output

```
models/renderer_stage3_best.pt
```

---

## Stage 4.1: ControlNet Alignment

**Objective**: Train ControlNet to align spatial features while U-Net remains frozen.

### Configuration

| Parameter | Value |
|-----------|-------|
| ControlNet | Training |
| U-Net | Frozen |
| ControlNet LR | 1e-4 |
| U-Net LR | 0 |
| weighted_loss | 100 |
| Warmup | 5% |
| Schedule | Cosine Annealing |
| Epochs | 50 |
| Validation | Every epoch |

### Training Command

```bash
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage4_1.json \
    --model_config_path configs/config_maisi_controlnet_train_stage4_1.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1
```

### Expected Behavior

- Tumors appear at mask-specified locations
- Edges may be slightly blurry
- ControlNet learns to map Pre + Mask → subtraction spatial features

### Checkpoint Output

```
models/renderer_stage4_1_best.pt
```

---

## Stage 4.2: Deep Feature Release

**Objective**: Unfreeze deep U-Net blocks to adapt to ControlNet's spatial signals.

### Unfreeze Blocks

```
down_blocks.2, down_blocks.3
middle_block
up_blocks.0, up_blocks.1
```

### Configuration

| Parameter | Value |
|-----------|-------|
| ControlNet | Training |
| U-Net | Partial (deep blocks) |
| ControlNet LR | 5e-5 |
| U-Net LR | 3e-5 |
| weighted_loss | 100 |
| Gradient Checkpointing | Enabled |
| Epochs | 50 (total: 100) |

### Training Command

```bash
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage4_2.json \
    --model_config_path configs/config_maisi_controlnet_train_stage4_2.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1
```

### Expected Behavior

- Deeper semantic features adapt to ControlNet signals
- More realistic enhancement patterns
- Better tumor-background contrast

### Checkpoint Output

```
models/renderer_stage4_2_best.pt
```

---

## Stage 4.3: Full Refinement

**Objective**: Unfreeze all U-Net blocks for high-frequency detail generation.

### Configuration

| Parameter | Value |
|-----------|-------|
| ControlNet | Training |
| U-Net | Fully Unfrozen |
| ControlNet LR | 3e-5 |
| U-Net LR | 1e-5 |
| weighted_loss | 100 |
| Epochs | 50 (total: 150) |

### Training Command

```bash
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage4_3.json \
    --model_config_path configs/config_maisi_controlnet_train_stage4_3.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1
```

### Expected Behavior

- Realistic heterogeneity and micro-vessels
- Sharp tumor edges
- Final high-fidelity synthesis

### Checkpoint Output

```
models/renderer_stage4_3_best.pt
```

---

## Inference

### Single-Sample Inference

```bash
python -m scripts.infer_controlnet \
    --checkpoint models/renderer_stage4_3_best.pt \
    --pre_path data/processed_pre/patient001_pre.nii.gz \
    --mask_path data/processed_mask/patient001_mask.nii.gz \
    --output_path outputs/inference/patient001_post.nii.gz \
    --cfg_scale 1.0
```

### Cascade Inference (With Locator)

```bash
python -m scripts.infer_controlnet \
    --checkpoint models/renderer_stage4_3_best.pt \
    --pre_path data/processed_pre/patient001_pre.nii.gz \
    --use_locator \
    --locator_checkpoint models/locator/best.pt \
    --output_path outputs/inference/patient001_post.nii.gz \
    --cfg_scale 1.0
```

---

## Monitoring

### Real-time Logs

```bash
# Tail training logs
tail -f outputs/logs/stage4_1/train.log

# Filter validation results
grep "Validation" outputs/logs/stage4_1/train.log
```

### TensorBoard

```bash
tensorboard --logdir outputs/tfevent/ --port 6006
```

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| train_loss | Training L1 loss (weighted) | Decreasing |
| val_loss | Validation L1 loss | < 0.05 |
| val_roi_loss | ROI-only L1 loss | < 0.02 |
| val_mae | Mean Absolute Error | < 0.03 |

---

## Checkpoints Summary

| Stage | Epoch Range | Checkpoint | Status |
|-------|-------------|------------|--------|
| Stage 3 | 0-50 | models/renderer_stage3_best.pt | Ready |
| Stage 4.1 | 0-50 | models/renderer_stage4_1_best.pt | Ready |
| Stage 4.2 | 50-100 | models/renderer_stage4_2_best.pt | Ready |
| Stage 4.3 | 100-150 | models/renderer_stage4_3_best.pt | Ready |

---

## Troubleshooting

### Issue: All-black predictions
- **Cause**: weighted_loss not applied correctly
- **Fix**: Ensure ROI mask is downscaled with max_pool3d and weight=100

### Issue: Poor tumor localization
- **Cause**: ControlNet not learning spatial features
- **Fix**: Verify dual-channel input (Pre + Mask concatenation)

### Issue: CUDA OOM
- **Cause**: 3D volumes require large memory
- **Fix**: Enable gradient checkpointing, reduce batch size to 1

### Issue: Validation loss much higher than training
- **Cause**: Overfitting or data distribution mismatch
- **Fix**: Check patient-level split integrity
