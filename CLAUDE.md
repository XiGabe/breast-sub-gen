# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a breast MRI subtraction imaging project using diffusion models (MAISI DiT). The goal is to train a DiT model to generate sparse lesion masks (black background + highlighted lesions) from breast MRI subtraction data.

## Architecture

### Model Pipeline

The codebase implements a **Latent Diffusion Model (LDM)** pipeline based on MONAI:

1. **Autoencoder (VAE)** - `scripts/sample.py:ReconModel`
   - Encodes images into latent space via `autoencoder.encode_stage_1_inputs()`
   - Decodes latents back to image space via `autoencoder.decode_stage_2_outputs()`
   - Scale factor normalizes latent magnitudes

2. **Diffusion UNet** - `scripts/diff_model_train.py`, `scripts/sample.py:ldm_conditional_sample_one_mask`
   - Operates in latent space (not pixel space)
   - Uses RFlowScheduler for continuous-time flow matching
   - Takes noisy latent + condition (modality, spacing, body region) as input
   - Outputs noise prediction for denoising

3. **ControlNet** - `scripts/train_controlnet.py`, `scripts/infer_controlnet.py`
   - Provides additional conditioning (body region index, modality embeddings)
   - Two losses: `compute_region_contrasive_loss` (ROI contrastive) + background similarity loss

### Data Flow

```
Raw Image → Autoencoder.encode → Latent z
                                   ↓
                          [Diffusion Process]
                                   ↓
                          Autoencoder.decode → Generated Image
```

For training (`diff_model_train.py`):
- Forward: noise ε added to latent z → UNet predicts ε
- Loss: L1 between predicted and true noise
- Validation: same loss on held-out data

For inference (`sample.py:ldm_conditional_sample_one_mask`):
- Start from random noise latent
- Iteratively denoise using RFlowScheduler
- Decode final latent to image

### Scripts Overview

| Script | Purpose |
|--------|---------|
| `diff_model_setting.py` | Config loading (`load_config`), DDP setup, logging |
| `diff_model_train.py` | DiT training loop, validation, checkpointing |
| `diff_model_infer.py` | DiT inference with sliding window |
| `sample.py` | Core sampling classes: `ReconModel`, `LDMSampler`, `ldm_conditional_sample_one_*` |
| `train_controlnet.py` | ControlNet training with region contrastive loss |
| `infer_controlnet.py` | ControlNet-based conditional generation |
| `inference.py` | MAISI general inference (downloads model, runs inference) |
| `utils.py` | `define_instance` (JSON→class), `dynamic_infer`, label remapping |
| `augmentation.py` | 3D elastic/affine transforms, tumor removal |
| `find_masks.py` | Mask candidate search by body region/anatomy |
| `transforms.py` | CT/MRI intensity normalization transforms |
| `quality_check.py` | Outlier detection for generated samples |
| `utils_plot.py` | Visualization utilities (3D plots, center finding) |

### Configuration System

Three JSON files compose a full config:
- **Environment** (`environment_*.json`): data paths, checkpoint paths, output directories
- **Model** (`config_*.json`): batch_size, epochs, learning rate, scheduler params
- **Network** (`config_network_*.json`): UNet architecture definition (via `define_instance`)

`modality_mapping.json` maps imaging modalities to integer IDs (MRI=9, CT=2/3).

### Key Classes

- `RFlowScheduler` (MONAI) - continuous-time diffusion scheduler
- `ReconModel` - wraps autoencoder for decoding
- `LDMSampler` / `lddm_conditional_sample_one_*` - sampling with/without conditions
- `DiffusionInferer` / `SlidingWindowInferer` - MONAI inference helpers
- `define_instance(args, name)` - instantiates class from JSON config by name
