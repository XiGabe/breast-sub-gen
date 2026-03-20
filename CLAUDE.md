# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a breast MRI subtraction imaging project using diffusion models (MAISI DiT). The goal is to train a DiT model to generate sparse lesion masks (black background + highlighted lesions) from breast MRI subtraction data.

## Common Commands

### Training (Breast Sparse Training)

```bash
# Submit training job via sbatch
cd /midtier/sablab/scratch/hoc4008/breast-sub-gen
sbatch scripts/train_breast_sparse.sh

# Monitor training logs
tail -f logs/train_<job_id>.err
```

### Training

```bash
conda activate breast_gen
python -m scripts.diff_model_train \
  -e configs/environment_maisi_diff_model_rflow-mr_breast.json \
  -c configs/config_maisi_diff_model_rflow-mr_breast.json \
  -t configs/config_network_rflow.json \
  -g 1
```

### Inference
```bash
source /home/hoc4008/miniconda3/etc/profile.d/conda.sh && conda activate breast_gen && python -m scripts.diff_model_infer -e configs/environment_maisi_diff_model_rflow-mr_breast.json -c configs/config_maisi_diff_model_rflow-mr_breast.json -t configs/config_network_rflow.json -g 1 2>&1
```

## Architecture

### Key Scripts

- `scripts/diff_model_train.py` - Main training script for DiT model (modified for breast data with validation loss)
- `scripts/diff_model_infer.py` - Inference script
- `scripts/sample.py` - Sampling/generation script
- `scripts/diff_model_setting.py` - Configuration utilities (load_config, setup_logging, initialize_distributed)
- `scripts/train_controlnet.py` - ControlNet training
- `scripts/inference.py` - General inference

### Configuration Files

- `configs/config_maisi_diff_model_rflow-mr_breast.json` - Model config (batch_size=8, n_epochs=500)
- `configs/environment_maisi_diff_model_rflow-mr_breast.json` - Environment paths (data, weights, output)
- `configs/config_network_rflow.json` - Network architecture definition
- `configs/modality_mapping.json` - Modality mapping (MRI=9)

### Data

- **Embeddings**: `data/embeddings_breast_sub/` (1943 latent embeddings)
- **Masks**: `data/processed_mask/`
- **Dataset JSON**: `data/dataset_breast_cached.json` (training: 1553, validation: 390)

### Checkpoints

- **Pretrained**: `weights/diff_unet_3d_rflow-mr.pt` (2.1GB)
- **Output**: `models/diff_unet_3d_rflow-mr_breast_epoch_*.pt`

## Training Details

- **Loss**: Plain L1 Loss
- **Batch size**: 16
- **Epochs**: 500
- **Learning rate**: 1e-5
- **Spacing**: [1.2, 0.7, 0.7] (Z, Y, X)
- **Modality**: MRI (8)
- **Validation**: 390 samples (10% of data)

## Key Modifications Made

The training script was modified to:
1. Support loading validation data from `dataset_breast_cached.json`
2. Calculate validation loss after each epoch
3. Save best model based on validation loss
4. Save each epoch checkpoint with epoch number
5. Read spacing/modality directly from dataset JSON (not separate JSON files)
