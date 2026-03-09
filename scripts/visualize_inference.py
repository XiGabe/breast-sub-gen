#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference and Visualization Script for Stage 1 ControlNet Model

This script runs inference on validation samples and creates multi-panel visualizations
comparing generated subtraction images with ground truth.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from monai.data import MetaTensor, decollate_batch
from monai.networks.utils import copy_model_state
from torch.utils.data import DataLoader

from .sample import ldm_conditional_sample_one_image, ReconModel
from .utils import define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp
from .diff_model_setting import load_config
from .utils_plot import find_label_center_loc


def setup_logging(output_dir):
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("visualize_inference")


def load_ground_truth_latents(sample_id, data_base_dir):
    """
    Load pre-encoded ground truth subtraction latents from disk.

    Args:
        sample_id (str): Sample identifier
        data_base_dir (str): Base data directory

    Returns:
        torch.Tensor: Ground truth latents with shape [4, 64, 64, 64]
    """
    # The ground truth latents are stored in embeddings_breast_sub/
    # Each sample has corresponding latents with matching ID
    gt_path = os.path.join(data_base_dir, "embeddings_breast_sub", f"{sample_id}_sub_emb.nii.gz")

    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth latent not found: {gt_path}")

    # Load the NIfTI file
    gt_nii = nib.load(gt_path)
    gt_data = gt_nii.get_fdata()

    # Convert to tensor and reorder from [H, W, D, C] to [C, H, W, D]
    # IMPORTANT: Ensure float32 dtype to avoid "Input type (Half) and bias type (float) mismatch" error
    gt_latent = torch.from_numpy(gt_data).float()  # Convert to float32

    # Check if data is 4D (has channel dimension) or 3D
    if gt_latent.ndim == 4:
        # Shape is [H, W, D, C], need to permute to [C, H, W, D]
        gt_latent = gt_latent.permute(3, 0, 1, 2)
    elif gt_latent.ndim == 3:
        # Shape is [H, W, D], need to add channel dimension
        gt_latent = gt_latent.unsqueeze(0)

    return gt_latent


def decode_with_vae(vae, latents, scale_factor, device, infer_size=[80, 80, 32], overlap=0.4):
    """
    Decode latent representations using VAE with sliding window inference.

    Args:
        vae: Autoencoder model
        latents (torch.Tensor): Latent tensor [B, C, H, W, D] or [C, H, W, D]
        scale_factor (torch.Tensor): Scale factor from diffusion checkpoint
        device: Torch device
        infer_size (list): Sliding window size
        overlap (float): Sliding window overlap ratio

    Returns:
        torch.Tensor: Decoded image with shape [B, 1, H_out, W_out, D_out]
    """
    from monai.inferers import SlidingWindowInferer
    from .sample import dynamic_infer

    # Ensure batch dimension
    if latents.ndim == 4:
        latents = latents.unsqueeze(0)  # [C, H, W, D] -> [B, C, H, W, D]

    # CRITICAL: Convert to float32 to avoid dtype mismatch with VAE bias weights
    latents = latents.to(device).float()

    # Create reconstruction model (handles scale factor normalization)
    # IMPORTANT: Ensure VAE is in float32 mode to avoid dtype mismatch
    # Convert VAE to float32 (no-op if already float32)
    vae_float = vae.float()
    recon_model = ReconModel(autoencoder=vae_float, scale_factor=scale_factor).to(device).float()

    # Setup sliding window inferer
    inferer = SlidingWindowInferer(
        roi_size=infer_size,
        sw_batch_size=1,
        progress=False,
        mode="gaussian",
        overlap=overlap,
        sw_device=device,
        device=torch.device("cpu"),
    )

    # Decode with sliding window
    with torch.no_grad():
        decoded = dynamic_infer(inferer, recon_model, latents)

    # For MRI: clip to [0, 1] range (autoencoder output range)
    decoded = torch.clip(decoded, 0.0, 1.0)

    # Keep in [0, 1] range for NIfTI output (removed * 1000.0 scaling)

    return decoded


def save_nifti_with_metadata(data, sample_id, output_dir, data_type="generated_sub",
                             reference_nii_path=None, logger=None):
    """Save numpy array as NIfTI file with proper affine and spacing metadata."""
    import nibabel as nib
    import os

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{sample_id}_{data_type}.nii.gz"
    output_path = os.path.join(output_dir, filename)

    # Get affine from reference if available
    if reference_nii_path and os.path.exists(reference_nii_path):
        try:
            ref_nii = nib.load(reference_nii_path)
            affine = ref_nii.affine
        except Exception as e:
            if logger:
                logger.warning(f"Failed to load reference NIfTI: {e}, using default affine")
            affine = _get_standard_affine()
    else:
        affine = _get_standard_affine()

    # Ensure data is float32
    data = data.astype(np.float32)

    # Create and save NIfTI image
    nii_img = nib.Nifti1Image(data, affine=affine)
    nib.save(nii_img, output_path)

    if logger:
        logger.info(f"  ✓ Saved NIfTI: {output_path} [range: {data.min():.4f}, {data.max():.4f}]")

    return output_path


def _get_standard_affine():
    """Get standard affine matrix for breast MRI data.
    Standard spacing: [1.2, 0.7, 0.7] mm³
    """
    affine = np.eye(4)
    affine[0, 0] = 0.7  # Row spacing
    affine[1, 1] = 0.7  # Column spacing
    affine[2, 2] = 1.2  # Slice spacing
    return affine


def create_comparison_visualization(
    pre_img, mask, gen_sub, gt_sub=None, save_path=None, view="axial", sample_id="sample"
):
    """
    Create multi-panel visualization comparing generated and (optionally) ground truth.

    Args:
        pre_img (np.ndarray): Pre-contrast MRI [H, W, D]
        mask (np.ndarray): Tumor mask [H, W, D]
        gen_sub (np.ndarray): Generated subtraction [H, W, D]
        gt_sub (np.ndarray): Ground truth subtraction [H, W, D] (optional)
        save_path (str): Path to save the visualization
        view (str): One of 'axial', 'sagittal', 'coronal'
        sample_id (str): Sample identifier for title
    """
    # Find tumor center for slice selection
    center_loc = find_label_center_loc(torch.from_numpy(mask))
    if center_loc[0] is None:
        # No tumor found, use middle slices
        center_loc = [s // 2 for s in mask.shape]

    # Determine axis and slice index based on view
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

    # Extract slices for each view
    def extract_slice(data, axis, idx):
        if axis == 0:
            return data[idx, :, :]
        elif axis == 1:
            return data[:, idx, :]
        else:  # axis == 2
            return data[:, :, idx]

    # Extract slices
    pre_slice = extract_slice(pre_img, axis, slice_idx)
    mask_slice = extract_slice(mask, axis, slice_idx)
    gen_sub_slice = extract_slice(gen_sub, axis, slice_idx)

    # Compute post-contrast image
    gen_post = pre_img + gen_sub
    gen_post_slice = extract_slice(gen_post, axis, slice_idx)

    # Determine layout based on whether ground truth is available
    has_gt = gt_sub is not None
    if has_gt:
        gt_sub_slice = extract_slice(gt_sub, axis, slice_idx)
        gt_post = pre_img + gt_sub
        gt_post_slice = extract_slice(gt_post, axis, slice_idx)
        n_cols = 3
        n_rows = 2
    else:
        n_cols = 2
        n_rows = 2

    # Create figure
    fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

    # Helper for plotting slices
    def plot_slice(ax, data, title, cmap='gray', vmin=None, vmax=None, overlay=None, auto_scale=True):
        if auto_scale and vmin is None:
            vmin = data.min()
        if auto_scale and vmax is None:
            vmax = data.max()

        # For very small ranges, expand them to make the image visible
        if auto_scale and vmax - vmin < 1e-6:
            vmin = vmin - 0.5
            vmax = vmax + 0.5

        im = ax.imshow(data.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

        # Add tumor mask overlay if provided
        if overlay is not None:
            mask_overlay = extract_slice(overlay, axis, slice_idx)
            # Create green contour for tumor
            ax.contour(mask_overlay.T, levels=[0.5], colors=['lime'], linewidths=2, origin='lower')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return im

    # Row 1, Col 1: Pre-contrast MRI
    ax1 = fig.add_subplot(gs[0, 0])
    plot_slice(ax1, pre_slice, f"Pre-contrast MRI\n{view_label} Slice {slice_idx} [range: {pre_slice.min():.2f}, {pre_slice.max():.2f}]",
               cmap='gray')

    # Row 1, Col 2: Pre-contrast with tumor mask overlay
    ax2 = fig.add_subplot(gs[0, 1])
    plot_slice(ax2, pre_slice, f"Pre-contrast + Tumor Mask\n{view_label} Slice {slice_idx}",
               cmap='gray', overlay=mask)

    if has_gt:
        # Row 1, Col 3: Generated subtraction (heatmap)
        ax3 = fig.add_subplot(gs[0, 2])
        plot_slice(ax3, gen_sub_slice, f"Generated Subtraction\n[{gen_sub_slice.min():.1f}, {gen_sub_slice.max():.1f}]",
                   cmap='RdBu_r')

        # Row 2, Col 1: Ground truth subtraction (heatmap)
        ax4 = fig.add_subplot(gs[1, 0])
        plot_slice(ax4, gt_sub_slice, f"Ground Truth Subtraction\n[{gt_sub_slice.min():.1f}, {gt_sub_slice.max():.1f}]",
                   cmap='RdBu_r')

        # Row 2, Col 2: Generated post-contrast
        ax5 = fig.add_subplot(gs[1, 1])
        plot_slice(ax5, gen_post_slice, f"Generated Post-contrast\n[{gen_post_slice.min():.1f}, {gen_post_slice.max():.1f}]",
                   cmap='gray')

        # Row 2, Col 3: Ground truth post-contrast
        ax6 = fig.add_subplot(gs[1, 2])
        plot_slice(ax6, gt_post_slice, f"Ground Truth Post-contrast\n[{gt_post_slice.min():.1f}, {gt_post_slice.max():.1f}]",
                   cmap='gray')

        title_addition = f"\n{view_label} View at Tumor Center"
    else:
        # Row 2, Col 1: Generated subtraction (heatmap)
        ax3 = fig.add_subplot(gs[1, 0])
        plot_slice(ax3, gen_sub_slice, f"Generated Subtraction\n[{gen_sub_slice.min():.1f}, {gen_sub_slice.max():.1f}]",
                   cmap='RdBu_r')

        # Row 2, Col 2: Generated post-contrast
        ax4 = fig.add_subplot(gs[1, 1])
        plot_slice(ax4, gen_post_slice, f"Generated Post-contrast\n[{gen_post_slice.min():.1f}, {gen_post_slice.max():.1f}]",
                   cmap='gray')

        title_addition = f"\n{view_label} View at Tumor Center\n(Ground truth decoding skipped due to dtype mismatch)"

    # Add main title
    fig.suptitle(f"Sample: {sample_id} - Stage 1 ControlNet Inference Results{title_addition}",
                 fontsize=14, fontweight='bold', y=0.98)

    # Save figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(
    env_config_path: str,
    model_config_path: str,
    model_def_path: str,
    num_samples: int = 5,
    output_dir: str = None,
    num_gpus: int = 1,
    num_inference_steps: int = 30,
):
    """
    Main function to run inference and create visualizations.

    Args:
        env_config_path: Path to environment config JSON
        model_config_path: Path to model training config JSON
        model_def_path: Path to model definition config JSON
        num_samples: Number of validation samples to process
        output_dir: Directory to save visualizations (overrides config)
        num_gpus: Number of GPUs to use
        num_inference_steps: Number of diffusion sampling steps
    """
    # Setup logging
    temp_output = output_dir or "./outputs/inference_stage1_vis"
    os.makedirs(temp_output, exist_ok=True)
    logger = setup_logging(temp_output)
    logger.info("=" * 80)
    logger.info("Starting Stage 1 ControlNet Inference and Visualization")
    logger.info("=" * 80)

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load configurations
    logger.info("Loading configurations...")
    args = load_config(env_config_path, model_config_path, model_def_path)

    # Override output directory if specified
    if output_dir:
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
    logger.info(f"✓ Loaded VAE from {args.trained_autoencoder_path}")

    # Load diffusion U-Net
    unet = define_instance(args, "diffusion_unet_def").to(device)
    checkpoint_diffusion_unet = torch.load(args.trained_diffusion_path, weights_only=False)
    unet.load_state_dict(checkpoint_diffusion_unet["unet_state_dict"], strict=False)
    scale_factor = checkpoint_diffusion_unet["scale_factor"].to(device)
    unet.eval()
    logger.info(f"✓ Loaded diffusion U-Net from {args.trained_diffusion_path}")
    logger.info(f"  Scale factor: {scale_factor.item():.6f}")

    # Load ControlNet
    controlnet = define_instance(args, "controlnet_def").to(device)
    checkpoint_controlnet = torch.load(args.trained_controlnet_path, weights_only=False)
    copy_model_state(controlnet, unet.state_dict())
    controlnet.load_state_dict(checkpoint_controlnet["controlnet_state_dict"], strict=False)
    controlnet.eval()
    logger.info(f"✓ Loaded ControlNet from {args.trained_controlnet_path}")

    # Load fine-tuned U-Net if available
    if "unet_state_dict" in checkpoint_controlnet and checkpoint_controlnet["unet_state_dict"] is not None:
        unet.load_state_dict(checkpoint_controlnet["unet_state_dict"], strict=False)
        logger.info("✓ Loaded fine-tuned U-Net from ControlNet checkpoint")
    else:
        logger.info("  Using base pretrained U-Net (no fine-tuning)")

    # Load noise scheduler
    noise_scheduler = define_instance(args, "noise_scheduler")

    # Step 2: Load validation dataloader
    logger.info("-" * 80)
    logger.info("Loading validation dataloader...")

    # Setup modality mapping if needed
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
        batch_size=1,  # Process one sample at a time for visualization
        fold=0,
        cache_rate=0.0,
        rank=0,
        world_size=1,
        modality_mapping=args.modality_mapping
    )
    logger.info(f"✓ Validation dataset size: {len(val_loader.dataset)} samples")

    # Step 3: Process samples
    logger.info("-" * 80)
    logger.info(f"Running inference on {num_samples} validation samples...")

    summary_notes = []
    processed_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if processed_count >= num_samples:
                break

            try:
                # Extract data from batch
                pre_images = batch["pre"].to(device)  # [B, 1, H, W, D]
                masks = batch["label"].to(device)      # [B, 1, H, W, D]

                # Get sample ID from metadata (if available)
                sample_id = f"sample_{batch_idx}"

                # Try to extract sample ID from various metadata sources
                if "label" in batch and hasattr(batch["label"], "meta"):
                    meta = batch["label"].meta
                    if "filename_or_obj" in meta:
                        pre_path = meta["filename_or_obj"]
                        if isinstance(pre_path, (list, tuple)):
                            pre_path = pre_path[0] if len(pre_path) > 0 else ""
                        # Extract ID like "DUKE_001_L" from "DUKE_001_L_mask_aligned.nii.gz"
                        basename = os.path.basename(pre_path)
                        sample_id = basename.replace("_mask_aligned.nii.gz", "").replace("_mask.nii.gz", "")
                        logger.info(f"  Extracted sample ID from label metadata: {sample_id}")
                    elif "saved_label" in meta:
                        # Alternative metadata key
                        pre_path = meta["saved_label"]
                        basename = os.path.basename(pre_path)
                        sample_id = basename.replace("_mask_aligned.nii.gz", "").replace("_mask.nii.gz", "")
                        logger.info(f"  Extracted sample ID from saved_label metadata: {sample_id}")

                # Fallback: try extracting from pre image metadata
                if sample_id == f"sample_{batch_idx}" and "pre" in batch and hasattr(batch["pre"], "meta"):
                    meta = batch["pre"].meta
                    if "filename_or_obj" in meta:
                        pre_path = meta["filename_or_obj"]
                        if isinstance(pre_path, (list, tuple)):
                            pre_path = pre_path[0] if len(pre_path) > 0 else ""
                        # Extract ID like "DUKE_001_L" from "DUKE_001_L_pre_aligned.nii.gz"
                        basename = os.path.basename(pre_path)
                        sample_id = basename.replace("_pre_aligned.nii.gz", "").replace("_pre.nii.gz", "")
                        logger.info(f"  Extracted sample ID from pre metadata: {sample_id}")

                logger.info(f"\nProcessing sample {processed_count + 1}/{num_samples}: {sample_id}")

                logger.info(f"\nProcessing sample {processed_count + 1}/{num_samples}: {sample_id}")

                # Prepare additional tensors
                batch_size = pre_images.shape[0]
                spacing_tensor = torch.tensor([[1.2, 0.7, 0.7]] * batch_size).to(device)  # Standard breast MRI spacing
                modality_tensor = torch.tensor([9] * batch_size).to(device)  # MRI modality

                # Set latent and output shapes
                latent_shape = [4, 64, 64, 64]  # [C, H//4, W//4, D//4]
                output_size = [256, 256, 256]

                # Generate subtraction image using ControlNet
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
                    cfg_guidance_scale=0  # CRITICAL: Use 0 for this model
                )

                # Convert to numpy
                pre_np = pre_images[0, 0].cpu().numpy()
                mask_np = masks[0, 0].cpu().numpy()
                gen_sub_np = generated_sub[0, 0].cpu().numpy()

                # DEBUG: Print detailed statistics
                logger.info(f"  Generated subtraction statistics:")
                logger.info(f"    Range: [{gen_sub_np.min():.4f}, {gen_sub_np.max():.4f}]")
                logger.info(f"    Mean: {gen_sub_np.mean():.4f}, Std: {gen_sub_np.std():.4f}")
                logger.info(f"    Non-zero voxels: {(gen_sub_np != 0).sum()} / {gen_sub_np.size} ({100 * (gen_sub_np != 0).sum() / gen_sub_np.size:.2f}%)")

                # Check tumor region values
                if mask_np.sum() > 0:
                    tumor_values = gen_sub_np[mask_np > 0]
                    logger.info(f"    Tumor region: [{tumor_values.min():.4f}, {tumor_values.max():.4f}], mean: {tumor_values.mean():.4f}")

                # Check if all zeros
                if np.abs(gen_sub_np).max() < 1e-6:
                    logger.error(f"  ✗ CRITICAL: Generated subtraction is ALL ZEROS! Model has not learned anything.")

                # Try to load and decode ground truth latents (may fail due to dtype mismatch)
                gt_sub_np = None
                try:
                    logger.info(f"  Loading ground truth latents...")
                    gt_latent = load_ground_truth_latents(sample_id, args.data_base_dir[0])
                    logger.info(f"  Decoding ground truth with VAE...")
                    gt_subtraction = decode_with_vae(
                        vae=autoencoder,
                        latents=gt_latent,
                        scale_factor=scale_factor,
                        device=device,
                        infer_size=[80, 80, 32],
                        overlap=0.4
                    )
                    gt_sub_np = gt_subtraction[0, 0].cpu().numpy()
                    logger.info(f"  ✓ Ground truth decoded successfully")
                except Exception as e:
                    logger.warning(f"  ⚠ Ground truth decoding failed (continuing without GT): {str(e)[:100]}")

                # ========== Save NIfTI files ==========
                nifti_output_dir = os.path.join(args.output_dir, "nifti_outputs")
                os.makedirs(nifti_output_dir, exist_ok=True)

                # Try to get reference pre-contrast path for metadata
                reference_pre_path = None
                if "pre" in batch and hasattr(batch["pre"], "meta"):
                    meta = batch["pre"].meta
                    if "filename_or_obj" in meta:
                        reference_pre_path = meta["filename_or_obj"]
                        if isinstance(reference_pre_path, (list, tuple)):
                            reference_pre_path = reference_pre_path[0] if len(reference_pre_path) > 0 else None

                # Save generated subtraction
                save_nifti_with_metadata(gen_sub_np, sample_id, nifti_output_dir, "generated_sub", reference_pre_path, logger)

                # Save ground truth (if available)
                if gt_sub_np is not None:
                    save_nifti_with_metadata(gt_sub_np, sample_id, nifti_output_dir, "gt_sub", reference_pre_path, logger)

                # Save pre-contrast and mask for reference
                save_nifti_with_metadata(pre_np, sample_id, nifti_output_dir, "pre", reference_pre_path, logger)
                save_nifti_with_metadata(mask_np, sample_id, nifti_output_dir, "mask", reference_pre_path, logger)
                # ========== End NIfTI saving ==========

                # Create visualizations for each view
                for view in ["axial", "sagittal", "coronal"]:
                    vis_path = os.path.join(args.output_dir, f"{sample_id}_{view}.png")
                    create_comparison_visualization(
                        pre_img=pre_np,
                        mask=mask_np,
                        gen_sub=gen_sub_np,
                        gt_sub=gt_sub_np,
                        save_path=vis_path,
                        view=view,
                        sample_id=sample_id
                    )
                    logger.info(f"  ✓ Saved {view} view: {vis_path}")

                # Compute quality metrics (only if ground truth available)
                tumor_mask = mask_np > 0
                if gt_sub_np is not None and tumor_mask.sum() > 0:
                    # Metrics within tumor region
                    gen_tumor = gen_sub_np[tumor_mask]
                    gt_tumor = gt_sub_np[tumor_mask]

                    mae_tumor = np.abs(gen_tumor - gt_tumor).mean()

                    # Background metrics (false positives)
                    bg_mask = ~tumor_mask
                    gen_bg = gen_sub_np[bg_mask]
                    gt_bg = gt_sub_np[bg_mask]
                    mae_bg = np.abs(gen_bg - gt_bg).mean()
                    fp_mean = np.maximum(gen_bg - gt_bg, 0).mean()  # False positive enhancement

                    note = (
                        f"{sample_id}: Tumor MAE={mae_tumor:.2f}, "
                        f"BG MAE={mae_bg:.2f}, FP={fp_mean:.2f}, "
                        f"Tumor voxels={tumor_mask.sum()}"
                    )
                elif gt_sub_np is None:
                    note = f"{sample_id}: No ground truth available (dtype mismatch)"
                elif tumor_mask.sum() == 0:
                    note = f"{sample_id}: No tumor detected"
                else:
                    note = f"{sample_id}: Processed"

                summary_notes.append(note)
                logger.info(f"  Quality: {note}")

                processed_count += 1

                # Clean up memory
                del generated_sub
                if gt_sub_np is not None:
                    del gt_subtraction, gt_latent
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"  ✗ Error processing sample: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Step 4: Generate summary report
    logger.info("-" * 80)
    logger.info("Generating summary report...")

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Stage 1 ControlNet Inference - Summary Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {args.trained_controlnet_path}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Samples processed: {processed_count}/{num_samples}\n")
        f.write(f"Inference steps: {num_inference_steps}\n")
        f.write(f"Output directory: {args.output_dir}\n\n")
        f.write("Sample Quality Metrics:\n")
        f.write("-" * 80 + "\n")
        for note in summary_notes:
            f.write(f"  {note}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("\nVisualization Notes:\n")
        f.write("-" * 80 + "\n")
        f.write("  - Red-Blue colormap: Red = positive enhancement, Blue = negative\n")
        f.write("  - Green contour: Tumor mask boundary\n")
        f.write("  - Subtraction range shown: [-100, 300] HU\n")
        f.write("  - Post-contrast = Pre-contrast + Subtraction\n\n")
        f.write("Expected Quality (Stage 1 - ControlNet only):\n")
        f.write("  - Tumor regions: Should show enhancement pattern\n")
        f.write("  - Background: May have some false positives (snow noise)\n")
        f.write("  - Overall: Interpretable but not perfect\n")

    logger.info(f"✓ Summary report saved: {summary_path}")
    logger.info("=" * 80)
    logger.info("Inference and visualization complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference and create visualizations for Stage 1 ControlNet"
    )
    parser.add_argument(
        "--env_config",
        type=str,
        required=True,
        help="Path to environment config JSON"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to model training config JSON"
    )
    parser.add_argument(
        "--model_def",
        type=str,
        required=True,
        help="Path to model definition config JSON"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of validation samples to process (default: 5)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for visualizations (overrides config)"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of diffusion sampling steps (default: 30)"
    )

    args = parser.parse_args()

    main(
        env_config_path=args.env_config,
        model_config_path=args.model_config,
        model_def_path=args.model_def,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        num_inference_steps=args.num_inference_steps,
    )
