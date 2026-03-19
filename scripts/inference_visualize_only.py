#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference and Visualization Script for ControlNet Model

This script runs inference on validation samples and creates visualizations.
Does NOT include ground truth analysis - much faster than inference_and_analysis.py.

Usage:
    python -m scripts.inference_visualize_only \
        --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage3_6.json \
        --model_config_path configs/config_maisi_controlnet_train_stage3_6.json \
        --model_def_path configs/config_network_rflow.json \
        --num_samples 10 \
        --output_dir ./outputs/vis_stage3_6 \
        --num_inference_steps 30
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Tuple, Optional

import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from monai.networks.utils import copy_model_state

from .sample import ldm_conditional_sample_one_image
from .utils import define_instance, prepare_maisi_controlnet_json_dataloader
from .diff_model_setting import load_config
from .utils_plot import find_label_center_loc


# ============================================================================
# Configuration
# ============================================================================

TARGET_SIZE = (256, 256, 256)


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(output_dir: str):
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, f"visualize_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("visualize_only")


# ============================================================================
# Visualization Functions
# ============================================================================

def create_visualization(
    pre_img: np.ndarray,
    mask: np.ndarray,
    gen_sub: np.ndarray,
    save_path: str = None,
    view: str = "axial",
    sample_id: str = "sample"
):
    """Create visualization of pre-contrast, mask, and generated subtraction."""
    center_loc = find_label_center_loc(torch.from_numpy(mask))
    if center_loc[0] is None:
        center_loc = [s // 2 for s in mask.shape]

    if view == "axial":
        axis = 2
        slice_idx = int(center_loc[2])
        view_label = "Axial"
    elif view == "sagittal":
        axis = 0
        slice_idx = int(center_loc[0])
        view_label = "Sagittal"
    elif view == "coronal":
        axis = 1
        slice_idx = int(center_loc[1])
        view_label = "Coronal"
    else:
        raise ValueError(f"Unknown view: {view}")

    def extract_slice(data, axis, idx):
        if axis == 0:
            return data[idx, :, :]
        elif axis == 1:
            return data[:, idx, :]
        else:
            return data[:, :, idx]

    pre_slice = extract_slice(pre_img, axis, slice_idx)
    mask_slice = extract_slice(mask, axis, slice_idx)
    gen_sub_slice = extract_slice(gen_sub, axis, slice_idx)

    gen_post = pre_img + gen_sub
    gen_post_slice = extract_slice(gen_post, axis, slice_idx)

    n_cols = 2
    n_rows = 2

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

    def plot_slice(ax, data, title, cmap='gray', vmin=None, vmax=None):
        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()

        if vmax - vmin < 1e-6:
            vmin = vmin - 0.5
            vmax = vmax + 0.5

        im = ax.imshow(data.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return im

    # Row 1: Pre-contrast and Pre + Mask
    ax1 = fig.add_subplot(gs[0, 0])
    plot_slice(ax1, pre_slice, f"Pre-contrast MRI\n{view_label} Slice {slice_idx}", cmap='gray')

    ax2 = fig.add_subplot(gs[0, 1])
    plot_slice(ax2, pre_slice, f"Pre + Tumor Mask\n{view_label} Slice {slice_idx}", cmap='gray')
    # Overlay mask on pre
    mask_slice_plot = extract_slice(mask, axis, slice_idx).T
    ax2.contour(mask_slice_plot, levels=[0.5], colors=['lime'], linewidths=2, origin='lower')

    # Row 2: Generated Subtraction and Generated Post
    ax3 = fig.add_subplot(gs[1, 0])
    vmin_sub = min(gen_sub_slice.min(), -gen_sub_slice.max()) if gen_sub_slice.max() > 0 else gen_sub_slice.min()
    vmax_sub = max(gen_sub_slice.max(), -gen_sub_slice.min()) if gen_sub_slice.min() < 0 else gen_sub_slice.max()
    plot_slice(ax3, gen_sub_slice, f"Generated Subtraction\n[{gen_sub_slice.min():.2f}, {gen_sub_slice.max():.2f}]",
               cmap='RdBu_r', vmin=vmin_sub, vmax=vmax_sub)

    ax4 = fig.add_subplot(gs[1, 1])
    plot_slice(ax4, gen_post_slice, f"Generated Post-contrast\n[{gen_post_slice.min():.2f}, {gen_post_slice.max():.2f}]",
               cmap='gray')

    fig.suptitle(f"Sample: {sample_id} - ControlNet Inference Results\n{view_label} View at Tumor Center",
                 fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_nifti(data: np.ndarray, sample_id: str, output_dir: str, suffix: str):
    """Save data as NIfTI file."""
    # Convert float16 to float32 if needed (nibabel doesn't support float16)
    if data.dtype == np.float16:
        data = data.astype(np.float32)
    # Convert int64 to int32 if needed (nibabel doesn't support int64)
    elif data.dtype == np.int64:
        data = data.astype(np.int32)

    affine = np.eye(4)
    affine[0, 0] = 0.7
    affine[1, 1] = 0.7
    affine[2, 2] = 1.2

    nii = nib.Nifti1Image(data, affine)
    output_path = os.path.join(output_dir, f"{sample_id}_{suffix}.nii.gz")
    nib.save(nii, output_path)
    return output_path


# ============================================================================
# Main Inference Function
# ============================================================================

def run_inference_visualize(
    env_config_path: str,
    model_config_path: str,
    model_def_path: str,
    num_samples: int = 10,
    output_dir: str = "./outputs/visualize_only",
    num_inference_steps: int = 30,
):
    """
    Run inference and create visualizations only (no analysis).

    Args:
        env_config_path: Path to environment config JSON
        model_config_path: Path to model training config JSON
        model_def_path: Path to model definition config JSON
        num_samples: Number of validation samples to process
        output_dir: Directory to save outputs
        num_inference_steps: Number of diffusion sampling steps
    """
    # Setup output directories
    vis_output_dir = os.path.join(output_dir, "visualizations")
    nifti_output_dir = os.path.join(output_dir, "nifti_outputs")

    os.makedirs(vis_output_dir, exist_ok=True)
    os.makedirs(nifti_output_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("=" * 80)
    logger.info("Starting Inference Visualization (No Analysis)")
    logger.info("=" * 80)

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load configurations
    logger.info("Loading configurations...")
    args = load_config(env_config_path, model_config_path, model_def_path)

    # Override output directory if specified
    args.output_dir = output_dir

    # Step 1: Load models
    logger.info("-" * 80)
    logger.info("Loading models...")

    # Load VAE
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=False)
    if "unet_state_dict" in checkpoint_autoencoder.keys():
        checkpoint_autoencoder = checkpoint_autoencoder["unet_state_dict"]
    autoencoder.load_state_dict(checkpoint_autoencoder)
    autoencoder.eval()
    logger.info(f"Loaded VAE from {args.trained_autoencoder_path}")

    # Load diffusion U-Net
    unet = define_instance(args, "diffusion_unet_def").to(device)
    checkpoint_diffusion_unet = torch.load(args.trained_diffusion_path, weights_only=False)
    unet.load_state_dict(checkpoint_diffusion_unet["unet_state_dict"], strict=False)
    scale_factor = checkpoint_diffusion_unet["scale_factor"].to(device)
    unet.eval()
    logger.info(f"Loaded diffusion U-Net from {args.trained_diffusion_path}")
    logger.info(f"  Scale factor: {scale_factor.item():.6f}")

    # Load ControlNet
    controlnet = define_instance(args, "controlnet_def").to(device)
    checkpoint_controlnet = torch.load(args.trained_controlnet_path, weights_only=False)
    copy_model_state(controlnet, unet.state_dict())
    controlnet.load_state_dict(checkpoint_controlnet["controlnet_state_dict"], strict=False)
    controlnet.eval()
    logger.info(f"Loaded ControlNet from {args.trained_controlnet_path}")

    # Load fine-tuned U-Net if available
    if "unet_state_dict" in checkpoint_controlnet and checkpoint_controlnet["unet_state_dict"] is not None:
        unet.load_state_dict(checkpoint_controlnet["unet_state_dict"], strict=False)
        logger.info("Loaded fine-tuned U-Net from ControlNet checkpoint")
    else:
        logger.info("Using base pretrained U-Net (no fine-tuning)")

    # Load noise scheduler
    noise_scheduler = define_instance(args, "noise_scheduler")

    # Step 2: Load validation dataloader
    logger.info("-" * 80)
    logger.info("Loading validation dataloader...")

    if unet.num_class_embeds is not None:
        if args.modality_mapping_path is None or not os.path.exists(args.modality_mapping_path):
            raise ValueError(f"Please check if {args.modality_mapping_path} exists.")
        with open(args.modality_mapping_path, "r") as f:
            args.modality_mapping = json.load(f)
    else:
        args.modality_mapping = None

    _, val_loader = prepare_maisi_controlnet_json_dataloader(
        json_data_list=args.json_data_list,
        data_base_dir=args.data_base_dir,
        batch_size=1,
        fold=0,
        cache_rate=0.0,
        rank=0,
        world_size=1,
        modality_mapping=args.modality_mapping
    )
    logger.info(f"Validation dataset size: {len(val_loader.dataset)} samples")

    # Step 3: Process samples
    logger.info("-" * 80)
    logger.info(f"Running inference on {num_samples} validation samples...")

    processed_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if processed_count >= num_samples:
                break

            try:
                # Extract data from batch
                pre_images = batch["pre"].to(device)
                masks = batch["label"].to(device)

                # Get sample ID
                sample_id = f"sample_{batch_idx}"
                if "pre" in batch and hasattr(batch["pre"], "meta"):
                    meta = batch["pre"].meta
                    if "filename_or_obj" in meta:
                        pre_path = meta["filename_or_obj"]
                        if isinstance(pre_path, (list, tuple)):
                            pre_path = pre_path[0] if len(pre_path) > 0 else ""
                        basename = os.path.basename(pre_path)
                        sample_id = basename.replace("_pre_aligned.nii.gz", "").replace("_pre.nii.gz", "")

                logger.info(f"\nProcessing sample {processed_count + 1}/{num_samples}: {sample_id}")

                # Prepare additional tensors
                batch_size = pre_images.shape[0]
                spacing_tensor = torch.tensor([[1.2, 0.7, 0.7]] * batch_size).to(device)
                modality_tensor = torch.tensor([9] * batch_size).to(device)

                latent_shape = [4, 64, 64, 64]
                output_size = [256, 256, 256]

                # Generate subtraction image
                logger.info(f"  Generating subtraction with {num_inference_steps} diffusion steps...")
                generated_sub, _ = ldm_conditional_sample_one_image(
                    autoencoder=autoencoder,
                    diffusion_unet=unet,
                    controlnet=controlnet,
                    noise_scheduler=noise_scheduler,
                    scale_factor=scale_factor,
                    device=device,
                    combine_label_or=masks,
                    pre_images_or=pre_images,
                    spacing_tensor=spacing_tensor,
                    latent_shape=latent_shape,
                    output_size=output_size,
                    noise_factor=1.0,
                    modality_tensor=modality_tensor,
                    num_inference_steps=num_inference_steps,
                    autoencoder_sliding_window_infer_size=[80, 80, 32],
                    autoencoder_sliding_window_infer_overlap=0.4,
                    cfg_guidance_scale=0
                )

                # Convert to numpy
                pre_np = pre_images[0, 0].cpu().numpy()
                mask_np = masks[0, 0].cpu().numpy()
                gen_sub_np = generated_sub[0, 0].cpu().numpy()

                logger.info(f"  Generated subtraction statistics:")
                logger.info(f"    Range: [{gen_sub_np.min():.4f}, {gen_sub_np.max():.4f}]")
                logger.info(f"    Mean: {gen_sub_np.mean():.4f}, Std: {gen_sub_np.std():.4f}")

                if mask_np.sum() > 0:
                    tumor_values = gen_sub_np[mask_np > 0]
                    logger.info(f"    Tumor region: [{tumor_values.min():.4f}, {tumor_values.max():.4f}], mean: {tumor_values.mean():.4f}")

                if np.abs(gen_sub_np).max() < 1e-6:
                    logger.error(f"  CRITICAL: Generated subtraction is ALL ZEROS!")

                # Save NIfTI files
                save_nifti(gen_sub_np, sample_id, nifti_output_dir, "generated_sub")
                save_nifti(pre_np, sample_id, nifti_output_dir, "pre")
                save_nifti(mask_np, sample_id, nifti_output_dir, "mask")
                logger.info(f"  Saved NIfTI files to {nifti_output_dir}")

                # Create visualizations (3 views)
                for view in ["axial", "sagittal", "coronal"]:
                    vis_path = os.path.join(vis_output_dir, f"{sample_id}_{view}.png")
                    create_visualization(
                        pre_img=pre_np,
                        mask=mask_np,
                        gen_sub=gen_sub_np,
                        save_path=vis_path,
                        view=view,
                        sample_id=sample_id
                    )
                    logger.info(f"  Saved {view} view: {vis_path}")

                processed_count += 1

            except Exception as e:
                logger.error(f"Error processing sample {batch_idx}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

    # Summary
    logger.info("=" * 80)
    logger.info(f"Inference visualization completed!")
    logger.info(f"  Processed: {processed_count}/{num_samples} samples")
    logger.info(f"  Visualizations: {vis_output_dir}")
    logger.info(f"  NIfTI outputs: {nifti_output_dir}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="ControlNet Inference Visualization (No Analysis)")
    parser.add_argument("--env_config_path", type=str, required=True,
                        help="Path to environment config JSON")
    parser.add_argument("--model_config_path", type=str, required=True,
                        help="Path to model config JSON")
    parser.add_argument("--model_def_path", type=str, required=True,
                        help="Path to model definition config JSON")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of validation samples to process")
    parser.add_argument("--output_dir", type=str, default="./outputs/visualize_only",
                        help="Output directory for visualizations")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="Number of diffusion sampling steps")

    args = parser.parse_args()

    run_inference_visualize(
        env_config_path=args.env_config_path,
        model_config_path=args.model_config_path,
        model_def_path=args.model_def_path,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
    )


if __name__ == "__main__":
    main()
