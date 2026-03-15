#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference and Analysis Script for Stage 1 ControlNet Model

This script runs inference on validation samples, creates visualizations,
compares with ground truth, and generates a comprehensive report.

Combines:
- visualize_inference.py: Model inference and visualization
- analyze_inference_outputs.py: Ground truth comparison and analysis
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from monai.data import MetaTensor, decollate_batch
from monai.networks.utils import copy_model_state
from torch.utils.data import DataLoader
import pandas as pd

from .sample import ldm_conditional_sample_one_image, ReconModel
from .utils import define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp
from .diff_model_setting import load_config
from .utils_plot import find_label_center_loc


# ============================================================================
# Configuration
# ============================================================================

# Paths for ground truth
DEFAULT_GT_DIR = "./data/step_4"  # Ground truth subtraction images
DEFAULT_PROCESSED_PRE_DIR = "./data/processed_pre"
TARGET_SIZE = (256, 256, 256)


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(output_dir):
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, f"inference_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("inference_analysis")


# ============================================================================
# Ground Truth Loading Functions
# ============================================================================

def load_ground_truth_latents(sample_id, data_base_dir):
    """
    Load pre-encoded ground truth subtraction latents from disk.

    Args:
        sample_id (str): Sample identifier
        data_base_dir (str): Base data directory

    Returns:
        torch.Tensor: Ground truth latents with shape [4, 64, 64, 64]
    """
    gt_path = os.path.join(data_base_dir, "embeddings_breast_sub", f"{sample_id}_sub_emb.nii.gz")

    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth latent not found: {gt_path}")

    gt_nii = nib.load(gt_path)
    gt_data = gt_nii.get_fdata()

    gt_latent = torch.from_numpy(gt_data).float()

    if gt_latent.ndim == 4:
        gt_latent = gt_latent.permute(3, 0, 1, 2)
    elif gt_latent.ndim == 3:
        gt_latent = gt_latent.unsqueeze(0)

    return gt_latent


def load_gt_subtraction(sample_id: str, gt_dir: str) -> Optional[np.ndarray]:
    """Load ground truth subtraction from step_4 directory."""
    gt_path = os.path.join(gt_dir, f"{sample_id}_sub.nii.gz")
    if not os.path.exists(gt_path):
        return None

    gt_nii = nib.load(gt_path)
    gt_data = gt_nii.get_fdata().astype(np.float32)

    if gt_data.shape != TARGET_SIZE:
        gt_data = resize_to_target(gt_data, TARGET_SIZE)

    return gt_data


def resize_to_target(data: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
    """Resize data to target size using center crop/pad."""
    result = np.zeros(target_size, dtype=data.dtype)

    slices_in = []
    slices_out = []

    for dim, (src_sz, tgt_sz) in enumerate(zip(data.shape, target_size)):
        if src_sz >= tgt_sz:
            start = (src_sz - tgt_sz) // 2
            end = start + tgt_sz
            slices_in.append(slice(start, end))
            slices_out.append(slice(0, tgt_sz))
        else:
            start = (tgt_sz - src_sz) // 2
            end = start + src_sz
            slices_in.append(slice(0, src_sz))
            slices_out.append(slice(start, end))

    result[tuple(slices_out)] = data[tuple(slices_in)]
    return result


# ============================================================================
# VAE Decoding
# ============================================================================

def decode_with_vae(vae, latents, scale_factor, device, infer_size=[160, 160, 64], overlap=0.6):
    """Decode latent representations using VAE with sliding window inference."""
    from monai.inferers import SlidingWindowInferer
    from .sample import dynamic_infer

    if latents.ndim == 4:
        latents = latents.unsqueeze(0)

    latents = latents.to(device).float()

    vae_float = vae.float()
    recon_model = ReconModel(autoencoder=vae_float, scale_factor=scale_factor).to(device).float()

    inferer = SlidingWindowInferer(
        roi_size=infer_size,
        sw_batch_size=1,
        progress=False,
        mode="gaussian",
        overlap=overlap,
        sw_device=device,
        device=torch.device("cpu"),
    )

    with torch.no_grad():
        decoded = dynamic_infer(inferer, recon_model, latents)

    decoded = torch.clip(decoded, 0.0, 1.0)
    return decoded


# ============================================================================
# NIfTI Saving
# ============================================================================

def save_nifti_with_metadata(data, sample_id, output_dir, data_type="generated_sub",
                             reference_nii_path=None, logger=None):
    """Save numpy array as NIfTI file with proper affine and spacing metadata."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{sample_id}_{data_type}.nii.gz"
    output_path = os.path.join(output_dir, filename)

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

    data = data.astype(np.float32)
    nii_img = nib.Nifti1Image(data, affine=affine)
    nib.save(nii_img, output_path)

    if logger:
        logger.info(f"  Saved NIfTI: {output_path} [range: {data.min():.4f}, {data.max():.4f}]")

    return output_path


def _get_standard_affine():
    """Get standard affine matrix for breast MRI data."""
    affine = np.eye(4)
    affine[0, 0] = 0.7
    affine[1, 1] = 0.7
    affine[2, 2] = 1.2
    return affine


# ============================================================================
# Visualization Functions
# ============================================================================

def create_comparison_visualization(
    pre_img, mask, gen_sub, gt_sub=None, save_path=None, view="axial", sample_id="sample"
):
    """Create multi-panel visualization comparing generated and ground truth."""
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

    fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

    def plot_slice(ax, data, title, cmap='gray', vmin=None, vmax=None, overlay=None, auto_scale=True):
        if auto_scale and vmin is None:
            vmin = data.min()
        if auto_scale and vmax is None:
            vmax = data.max()

        if auto_scale and vmax - vmin < 1e-6:
            vmin = vmin - 0.5
            vmax = vmax + 0.5

        im = ax.imshow(data.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

        if overlay is not None:
            mask_overlay = extract_slice(overlay, axis, slice_idx)
            ax.contour(mask_overlay.T, levels=[0.5], colors=['lime'], linewidths=2, origin='lower')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return im

    ax1 = fig.add_subplot(gs[0, 0])
    plot_slice(ax1, pre_slice, f"Pre-contrast MRI\n{view_label} Slice {slice_idx}",
               cmap='gray')

    ax2 = fig.add_subplot(gs[0, 1])
    plot_slice(ax2, pre_slice, f"Pre-contrast + Tumor Mask\n{view_label} Slice {slice_idx}",
               cmap='gray', overlay=mask)

    if has_gt:
        ax3 = fig.add_subplot(gs[0, 2])
        plot_slice(ax3, gen_sub_slice, f"Generated Subtraction\n[{gen_sub_slice.min():.1f}, {gen_sub_slice.max():.1f}]",
                   cmap='RdBu_r')

        ax4 = fig.add_subplot(gs[1, 0])
        plot_slice(ax4, gt_sub_slice, f"Ground Truth Subtraction\n[{gt_sub_slice.min():.1f}, {gt_sub_slice.max():.1f}]",
                   cmap='RdBu_r')

        ax5 = fig.add_subplot(gs[1, 1])
        plot_slice(ax5, gen_post_slice, f"Generated Post-contrast\n[{gen_post_slice.min():.1f}, {gen_post_slice.max():.1f}]",
                   cmap='gray')

        ax6 = fig.add_subplot(gs[1, 2])
        plot_slice(ax6, gt_post_slice, f"Ground Truth Post-contrast\n[{gt_post_slice.min():.1f}, {gt_post_slice.max():.1f}]",
                   cmap='gray')

        title_addition = f"\n{view_label} View at Tumor Center"
    else:
        ax3 = fig.add_subplot(gs[1, 0])
        plot_slice(ax3, gen_sub_slice, f"Generated Subtraction\n[{gen_sub_slice.min():.1f}, {gen_sub_slice.max():.1f}]",
                   cmap='RdBu_r')

        ax4 = fig.add_subplot(gs[1, 1])
        plot_slice(ax4, gen_post_slice, f"Generated Post-contrast\n[{gen_post_slice.min():.1f}, {gen_post_slice.max():.1f}]",
                   cmap='gray')

        title_addition = f"\n{view_label} View"

    fig.suptitle(f"Sample: {sample_id} - Stage 1 ControlNet Inference Results{title_addition}",
                 fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Analysis Functions (from analyze_inference_outputs.py)
# ============================================================================

def create_masks(pre: np.ndarray, mask: np.ndarray,
                breast_threshold: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create region masks."""
    tumor_mask = mask > 0
    breast_mask = (pre > breast_threshold) & (~tumor_mask)
    background_mask = pre <= breast_threshold
    return tumor_mask, breast_mask, background_mask


def compute_statistics(data: np.ndarray, mask: np.ndarray, name: str) -> Dict:
    """Compute statistics for a specific region."""
    if mask.sum() == 0:
        return {
            f"{name}_count": 0,
            f"{name}_min": np.nan,
            f"{name}_max": np.nan,
            f"{name}_mean": np.nan,
            f"{name}_std": np.nan,
            f"{name}_median": np.nan,
        }

    values = data[mask]

    return {
        f"{name}_count": int(mask.sum()),
        f"{name}_min": float(values.min()),
        f"{name}_max": float(values.max()),
        f"{name}_mean": float(values.mean()),
        f"{name}_std": float(values.std()),
        f"{name}_median": float(np.median(values)),
        f"{name}_p05": float(np.percentile(values, 5)),
        f"{name}_p25": float(np.percentile(values, 25)),
        f"{name}_p75": float(np.percentile(values, 75)),
        f"{name}_p95": float(np.percentile(values, 95)),
    }


def compute_background_suppression_metrics(data: np.ndarray, bg_mask: np.ndarray) -> Dict:
    """Compute background suppression quality metrics."""
    if bg_mask.sum() == 0:
        return {}

    bg_values = data[bg_mask]

    return {
        "bg_pct_below_001": float((bg_values < 0.01).sum() / len(bg_values) * 100),
        "bg_pct_below_005": float((bg_values < 0.05).sum() / len(bg_values) * 100),
        "bg_pct_below_010": float((bg_values < 0.10).sum() / len(bg_values) * 100),
        "bg_pct_below_020": float((bg_values < 0.20).sum() / len(bg_values) * 100),
        "bg_pct_above_050": float((bg_values > 0.50).sum() / len(bg_values) * 100),
        "bg_pct_above_100": float((bg_values > 1.00).sum() / len(bg_values) * 100),
    }


def compute_mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                region_name: str) -> Dict:
    """Compute Mean Absolute Error for a specific region."""
    if mask.sum() == 0:
        return {
            f"{region_name}_mae": np.nan,
            f"{region_name}_mae_pct": np.nan,
        }

    pred_vals = pred[mask]
    gt_vals = gt[mask]

    mae = float(np.abs(pred_vals - gt_vals).mean())
    gt_range = gt_vals.max() - gt_vals.min()
    mae_pct = float(mae / gt_range * 100) if gt_range > 0 else np.nan

    return {
        f"{region_name}_mae": mae,
        f"{region_name}_mae_pct": mae_pct,
    }


def generate_histogram_comparison(sample_id: str, gt_sub: np.ndarray,
                                   stage1_sub: np.ndarray,
                                   pre: np.ndarray, mask: np.ndarray,
                                   output_dir: str):
    """Generate histogram comparison plot with GT."""
    tumor_mask, breast_mask, bg_mask = create_masks(pre, mask)

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Global histograms
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(gt_sub.flatten(), bins=100, alpha=0.5, label='Ground Truth', color='green', density=True)
    ax1.hist(stage1_sub.flatten(), bins=100, alpha=0.5, label='Stage 1', color='blue', density=True)
    ax1.set_xlabel('Intensity')
    ax1.set_ylabel('Density')
    ax1.set_title('Global Intensity Distribution')
    ax1.legend()
    ax1.set_xlim(0, 1)

    # Tumor region histograms
    ax2 = fig.add_subplot(gs[1, 0])
    if tumor_mask.sum() > 0:
        ax2.hist(gt_sub[tumor_mask], bins=50, alpha=0.5, label='Ground Truth', color='green', density=True)
        ax2.hist(stage1_sub[tumor_mask], bins=50, alpha=0.5, label='Stage 1', color='blue', density=True)
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Density')
    ax2.set_title('Tumor Region Intensity Distribution')
    ax2.legend()
    ax2.set_xlim(0, 1)

    # Background region histograms
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.hist(gt_sub[bg_mask], bins=100, alpha=0.5, label='Ground Truth', color='green', density=True)
    ax3.hist(stage1_sub[bg_mask], bins=100, alpha=0.5, label='Stage 1', color='blue', density=True)
    ax3.set_xlabel('Intensity')
    ax3.set_ylabel('Density')
    ax3.set_title('Background Region Intensity Distribution')
    ax3.legend()
    ax3.set_xlim(0, 0.5)

    # CDF plots
    ax4 = fig.add_subplot(gs[0, 1])
    for data, label, color in [(gt_sub, 'Ground Truth', 'green'),
                                (stage1_sub, 'Stage 1', 'blue')]:
        sorted_vals = np.sort(data.flatten())
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax4.plot(sorted_vals, cdf, label=label, color=color, linewidth=2)
    ax4.set_xlabel('Intensity')
    ax4.set_ylabel('CDF')
    ax4.set_title('Global CDF')
    ax4.legend()
    ax4.set_xlim(0, 1)
    ax4.grid(True, alpha=0.3)

    # Tumor CDF
    ax5 = fig.add_subplot(gs[1, 1])
    if tumor_mask.sum() > 0:
        for data, label, color in [(gt_sub, 'Ground Truth', 'green'),
                                    (stage1_sub, 'Stage 1', 'blue')]:
            sorted_vals = np.sort(data[tumor_mask])
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax5.plot(sorted_vals, cdf, label=label, color=color, linewidth=2)
    ax5.set_xlabel('Intensity')
    ax5.set_ylabel('CDF')
    ax5.set_title('Tumor Region CDF')
    ax5.legend()
    ax5.set_xlim(0, 1)
    ax5.grid(True, alpha=0.3)

    # Background CDF
    ax6 = fig.add_subplot(gs[2, 1])
    for data, label, color in [(gt_sub, 'Ground Truth', 'green'),
                                (stage1_sub, 'Stage 1', 'blue')]:
        sorted_vals = np.sort(data[bg_mask])
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax6.plot(sorted_vals, cdf, label=label, color=color, linewidth=2)
    ax6.set_xlabel('Intensity')
    ax6.set_ylabel('CDF')
    ax6.set_title('Background Region CDF')
    ax6.legend()
    ax6.set_xlim(0, 0.5)
    ax6.grid(True, alpha=0.3)

    # Background suppression quality bar chart
    ax7 = fig.add_subplot(gs[0, 2])
    metrics = ['< 0.01', '< 0.05', '< 0.10', '< 0.20']
    gt_metrics = [
        (gt_sub[bg_mask] < 0.01).sum() / bg_mask.sum() * 100,
        (gt_sub[bg_mask] < 0.05).sum() / bg_mask.sum() * 100,
        (gt_sub[bg_mask] < 0.10).sum() / bg_mask.sum() * 100,
        (gt_sub[bg_mask] < 0.20).sum() / bg_mask.sum() * 100,
    ]
    s1_metrics = [
        (stage1_sub[bg_mask] < 0.01).sum() / bg_mask.sum() * 100,
        (stage1_sub[bg_mask] < 0.05).sum() / bg_mask.sum() * 100,
        (stage1_sub[bg_mask] < 0.10).sum() / bg_mask.sum() * 100,
        (stage1_sub[bg_mask] < 0.20).sum() / bg_mask.sum() * 100,
    ]

    x = np.arange(len(metrics))
    width = 0.35
    ax7.bar(x - width/2, gt_metrics, width, label='Ground Truth', color='green')
    ax7.bar(x + width/2, s1_metrics, width, label='Stage 1', color='blue')
    ax7.set_ylabel('Percentage (%)')
    ax7.set_title('Background Suppression Quality\n(Higher is better)')
    ax7.set_xticks(x)
    ax7.set_xticklabels(metrics)
    ax7.legend()
    ax7.set_ylim(0, 100)
    ax7.grid(axis='y', alpha=0.3)

    # MAE comparison
    ax8 = fig.add_subplot(gs[1, 2])
    regions = ['Global', 'Background', 'Breast', 'Tumor', 'Foreground']
    s1_mae = [
        np.abs(stage1_sub - gt_sub).mean(),
        np.abs(stage1_sub[bg_mask] - gt_sub[bg_mask]).mean(),
        np.abs(stage1_sub[breast_mask] - gt_sub[breast_mask]).mean() if breast_mask.sum() > 0 else 0,
        np.abs(stage1_sub[tumor_mask] - gt_sub[tumor_mask]).mean() if tumor_mask.sum() > 0 else 0,
        np.abs(stage1_sub[tumor_mask | breast_mask] - gt_sub[tumor_mask | breast_mask]).mean(),
    ]

    x = np.arange(len(regions))
    ax8.bar(x, s1_mae, label='Stage 1 MAE', color='blue')
    ax8.set_ylabel('MAE')
    ax8.set_title('Mean Absolute Error by Region\n(Lower is better)')
    ax8.set_xticks(x)
    ax8.set_xticklabels(regions, rotation=45, ha='right')
    ax8.grid(axis='y', alpha=0.3)
    for i, v in enumerate(s1_mae):
        ax8.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    # Percentile comparison
    ax9 = fig.add_subplot(gs[2, 2])
    percentiles = [5, 25, 50, 75, 95]
    gt_p = np.percentile(gt_sub[tumor_mask], percentiles) if tumor_mask.sum() > 0 else [0, 0, 0, 0, 0]
    s1_p = np.percentile(stage1_sub[tumor_mask], percentiles) if tumor_mask.sum() > 0 else [0, 0, 0, 0, 0]

    x = np.arange(len(percentiles))
    width = 0.35
    ax9.bar(x - width/2, gt_p, width, label='Ground Truth', color='green')
    ax9.bar(x + width/2, s1_p, width, label='Stage 1', color='blue')
    ax9.set_ylabel('Intensity')
    ax9.set_title('Tumor Region Percentiles')
    ax9.set_xticks(x)
    ax9.set_xticklabels([f'P{p}' for p in percentiles])
    ax9.legend()
    ax9.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Sample: {sample_id} - Stage 1 vs Ground Truth Analysis', fontsize=16, y=1.02)

    output_path = os.path.join(output_dir, f"{sample_id}_distributions.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_slice_comparison(sample_id: str, gt_sub: np.ndarray,
                               stage1_sub: np.ndarray,
                               pre: np.ndarray, mask: np.ndarray,
                               output_dir: str):
    """Generate slice comparison with GT (3 columns: Pre, GT, Stage 1)."""
    tumor_mask, breast_mask, bg_mask = create_masks(pre, mask)

    h, w, d = pre.shape
    axial_slice = d // 2
    coronal_slice = w // 2
    sagittal_slice = h // 2

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Row 1: Axial view
    im = axes[0, 0].imshow(pre[:, :, axial_slice].T, cmap='gray', origin='lower')
    axes[0, 0].set_title('Pre-contrast (Axial)')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

    im = axes[0, 1].imshow(gt_sub[:, :, axial_slice].T, cmap='hot', vmin=0, vmax=1, origin='lower')
    axes[0, 1].contour(pre[:, :, axial_slice].T, levels=[0.05], colors='cyan', linewidths=0.5, alpha=0.5, origin='lower')
    axes[0, 1].contour(mask[:, :, axial_slice].T, levels=[0.5], colors='lime', linewidths=1, origin='lower')
    axes[0, 1].set_title(f'Ground Truth (Axial #{axial_slice})')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    im = axes[0, 2].imshow(stage1_sub[:, :, axial_slice].T, cmap='hot', vmin=0, vmax=1, origin='lower')
    axes[0, 2].contour(pre[:, :, axial_slice].T, levels=[0.05], colors='cyan', linewidths=0.5, alpha=0.5, origin='lower')
    axes[0, 2].contour(mask[:, :, axial_slice].T, levels=[0.5], colors='lime', linewidths=1, origin='lower')
    axes[0, 2].set_title('Stage 1 (Axial)')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    # Row 2: Coronal view
    im = axes[1, 0].imshow(pre[:, coronal_slice, :].T, cmap='gray', origin='lower')
    axes[1, 0].set_title('Pre-contrast (Coronal)')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(gt_sub[:, coronal_slice, :].T, cmap='hot', vmin=0, vmax=1, origin='lower')
    axes[1, 1].contour(pre[:, coronal_slice, :].T, levels=[0.05], colors='cyan', linewidths=0.5, alpha=0.5, origin='lower')
    axes[1, 1].contour(mask[:, coronal_slice, :].T, levels=[0.5], colors='lime', linewidths=1, origin='lower')
    axes[1, 1].set_title(f'Ground Truth (Coronal #{coronal_slice})')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    im = axes[1, 2].imshow(stage1_sub[:, coronal_slice, :].T, cmap='hot', vmin=0, vmax=1, origin='lower')
    axes[1, 2].contour(pre[:, coronal_slice, :].T, levels=[0.05], colors='cyan', linewidths=0.5, alpha=0.5, origin='lower')
    axes[1, 2].contour(mask[:, coronal_slice, :].T, levels=[0.5], colors='lime', linewidths=1, origin='lower')
    axes[1, 2].set_title('Stage 1 (Coronal)')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    # Row 3: Sagittal view
    im = axes[2, 0].imshow(pre[sagittal_slice, :, :].T, cmap='gray', origin='lower')
    axes[2, 0].set_title('Pre-contrast (Sagittal)')
    axes[2, 0].axis('off')
    plt.colorbar(im, ax=axes[2, 0], fraction=0.046)

    im = axes[2, 1].imshow(gt_sub[sagittal_slice, :, :].T, cmap='hot', vmin=0, vmax=1, origin='lower')
    axes[2, 1].contour(pre[sagittal_slice, :, :].T, levels=[0.05], colors='cyan', linewidths=0.5, alpha=0.5, origin='lower')
    axes[2, 1].contour(mask[sagittal_slice, :, :].T, levels=[0.5], colors='lime', linewidths=1, origin='lower')
    axes[2, 1].set_title(f'Ground Truth (Sagittal #{sagittal_slice})')
    axes[2, 1].axis('off')
    plt.colorbar(im, ax=axes[2, 1], fraction=0.046)

    im = axes[2, 2].imshow(stage1_sub[sagittal_slice, :, :].T, cmap='hot', vmin=0, vmax=1, origin='lower')
    axes[2, 2].contour(pre[sagittal_slice, :, :].T, levels=[0.05], colors='cyan', linewidths=0.5, alpha=0.5, origin='lower')
    axes[2, 2].contour(mask[sagittal_slice, :, :].T, levels=[0.5], colors='lime', linewidths=1, origin='lower')
    axes[2, 2].set_title('Stage 1 (Sagittal)')
    axes[2, 2].axis('off')
    plt.colorbar(im, ax=axes[2, 2], fraction=0.046)

    plt.suptitle(f'Sample: {sample_id} - Stage 1 vs Ground Truth Slice Comparison', fontsize=16, y=1.02)

    output_path = os.path.join(output_dir, f"{sample_id}_slices.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def analyze_sample(sample_id: str, stage1_sub: np.ndarray, pre: np.ndarray, mask: np.ndarray,
                   gt_dir: str) -> Dict:
    """Analyze a single sample with ground truth."""
    results = {"sample_id": sample_id}

    # Load Ground Truth
    gt_sub = load_gt_subtraction(sample_id, gt_dir)

    if gt_sub is None:
        print(f"Warning: Could not load GT for {sample_id}")
        results["has_gt"] = False
        return results

    results["has_gt"] = True

    # Create region masks
    tumor_mask, breast_mask, bg_mask = create_masks(pre, mask)

    # Analyze Ground Truth
    gt_stats = {}
    gt_stats.update(compute_statistics(gt_sub, tumor_mask, "tumor"))
    gt_stats.update(compute_statistics(gt_sub, breast_mask, "breast"))
    gt_stats.update(compute_statistics(gt_sub, bg_mask, "background"))
    gt_stats.update(compute_statistics(gt_sub, tumor_mask | breast_mask, "foreground"))
    gt_stats.update(compute_background_suppression_metrics(gt_sub, bg_mask))

    for key, value in gt_stats.items():
        results[f"gt_{key}"] = value

    # Analyze Stage 1
    stage1_stats = {}
    stage1_stats.update(compute_statistics(stage1_sub, tumor_mask, "tumor"))
    stage1_stats.update(compute_statistics(stage1_sub, breast_mask, "breast"))
    stage1_stats.update(compute_statistics(stage1_sub, bg_mask, "background"))
    stage1_stats.update(compute_statistics(stage1_sub, tumor_mask | breast_mask, "foreground"))
    stage1_stats.update(compute_background_suppression_metrics(stage1_sub, bg_mask))
    # MAE against GT
    stage1_stats.update(compute_mae(stage1_sub, gt_sub, tumor_mask, "tumor"))
    stage1_stats.update(compute_mae(stage1_sub, gt_sub, breast_mask, "breast"))
    stage1_stats.update(compute_mae(stage1_sub, gt_sub, bg_mask, "background"))
    stage1_stats.update(compute_mae(stage1_sub, gt_sub, tumor_mask | breast_mask, "foreground"))
    stage1_stats.update(compute_mae(stage1_sub, gt_sub, np.ones_like(gt_sub, dtype=bool), "global"))

    for key, value in stage1_stats.items():
        results[f"stage1_{key}"] = value

    return results, gt_sub


def generate_summary_report(all_results: list, output_dir: str):
    """Generate a comprehensive summary report."""
    report = []
    report.append("=" * 80)
    report.append("STAGE 1 INFERENCE VS GROUND TRUTH ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    # Filter results with GT
    valid_results = [r for r in all_results if r.get("has_gt", False)]
    report.append(f"## SUMMARY (N={len(valid_results)} samples with GT / {len(all_results)} total)")
    report.append("")

    if not valid_results:
        report.append("No samples with ground truth available for analysis.")
        report_text = "\n".join(report)
        report_path = os.path.join(output_dir, "summary_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)
        print(report_text)
        return report_text

    # Helper functions
    def avg_metric(key):
        values = [r[key] for r in valid_results if key in r and not np.isnan(r[key])]
        return np.mean(values) if values else np.nan

    def format_val(val):
        return f"{val:.4f}" if not np.isnan(val) else "N/A"

    # Global MAE
    report.append("### GLOBAL MAE (Lower is better)")
    report.append("-" * 40)
    s1_global_mae = avg_metric("stage1_global_mae")
    report.append(f"  Stage 1: {format_val(s1_global_mae)}")
    report.append("")

    # Background suppression MAE
    report.append("### BACKGROUND MAE (Lower is better)")
    report.append("-" * 40)
    s1_bg_mae = avg_metric("stage1_background_mae")
    report.append(f"  Stage 1: {format_val(s1_bg_mae)}")
    report.append("")

    # Tumor enhancement MAE
    report.append("### TUMOR MAE (Lower is better)")
    report.append("-" * 40)
    s1_tumor_mae = avg_metric("stage1_tumor_mae")
    report.append(f"  Stage 1: {format_val(s1_tumor_mae)}")
    report.append("")

    # Background suppression quality
    report.append("### BACKGROUND SUPPRESSION QUALITY (Higher % below threshold = better)")
    report.append("-" * 40)
    for thresh in [0.01, 0.05, 0.10, 0.20]:
        key = f"bg_pct_below_{int(thresh*100):03d}"
        gt_pct = avg_metric(f"gt_{key}")
        s1_pct = avg_metric(f"stage1_{key}")
        report.append(f"  Below {thresh}:")
        report.append(f"    GT:       {format_val(gt_pct)}%")
        report.append(f"    Stage 1:  {format_val(s1_pct)}%")
    report.append("")

    # Tumor intensity statistics
    report.append("### TUMOR REGION INTENSITY STATISTICS")
    report.append("-" * 40)
    for stat in ["mean", "std", "median", "p95"]:
        gt_val = avg_metric(f"gt_tumor_{stat}")
        s1_val = avg_metric(f"stage1_tumor_{stat}")
        report.append(f"  {stat.upper()}:")
        report.append(f"    GT:       {format_val(gt_val)}")
        report.append(f"    Stage 1:  {format_val(s1_val)}")
    report.append("")

    # Per-sample details
    report.append("=" * 80)
    report.append("PER-SAMPLE DETAILS")
    report.append("=" * 80)
    report.append("")

    for result in valid_results:
        sample_id = result.get("sample_id", "Unknown")
        report.append(f"## {sample_id}")
        report.append("")

        report.append("### MAE by Region")
        report.append(f"  {'Region':<15} {'MAE':<12}")
        report.append("  " + "-" * 30)

        for region, region_key in [("Global", "global"), ("Background", "background"),
                                    ("Breast", "breast"), ("Tumor", "tumor"), ("Foreground", "foreground")]:
            s1_mae = result.get(f"stage1_{region_key}_mae", np.nan)
            report.append(f"  {region:<15} {format_val(s1_mae):<12}")

        report.append("")
        report.append("### Background Suppression (% voxels below threshold)")
        report.append(f"  {'Threshold':<12} {'GT':<10} {'Stage 1':<10}")
        report.append("  " + "-" * 35)

        for thresh in [0.01, 0.05, 0.10, 0.20]:
            key = f"bg_pct_below_{int(thresh*100):03d}"
            gt_pct = result.get(f"gt_{key}", np.nan)
            s1_pct = result.get(f"stage1_{key}", np.nan)
            report.append(f"  {thresh:<12.2f} {format_val(gt_pct):<10} {format_val(s1_pct):<10}")

        report.append("")
        report.append("### Tumor Region Statistics")
        report.append(f"  {'Metric':<10} {'GT':<10} {'Stage 1':<10}")
        report.append("  " + "-" * 35)

        for stat in ["mean", "std", "median", "p95"]:
            gt_val = result.get(f"gt_tumor_{stat}", np.nan)
            s1_val = result.get(f"stage1_tumor_{stat}", np.nan)
            report.append(f"  {stat:<10} {format_val(gt_val):<10} {format_val(s1_val):<10}")

        report.append("")
        report.append("-" * 80)
        report.append("")

    # Key findings
    report.append("=" * 80)
    report.append("KEY FINDINGS")
    report.append("=" * 80)
    report.append("")

    findings = []

    # Background suppression check
    gt_bg_suppress = avg_metric("gt_bg_pct_below_010")
    s1_bg_suppress = avg_metric("stage1_bg_pct_below_010")

    if gt_bg_suppress > 50:
        if s1_bg_suppress < 20:
            findings.append(f"WARNING: POOR BACKGROUND SUPPRESSION: Only {s1_bg_suppress:.1f}% of background < 0.10 (GT: {gt_bg_suppress:.1f}%)")
        elif s1_bg_suppress < gt_bg_suppress * 0.5:
            findings.append(f"WARNING: WEAK BACKGROUND SUPPRESSION: {s1_bg_suppress:.1f}% of background < 0.10 (GT: {gt_bg_suppress:.1f}%)")

    # Tumor enhancement check
    gt_tumor_mean = avg_metric("gt_tumor_mean")
    s1_tumor_mean = avg_metric("stage1_tumor_mean")

    if not np.isnan(gt_tumor_mean):
        if abs(s1_tumor_mean - gt_tumor_mean) / gt_tumor_mean > 0.3:
            findings.append(f"WARNING: TUMOR INTENSITY BIAS: {s1_tumor_mean:.3f} vs GT {gt_tumor_mean:.3f} ({abs(s1_tumor_mean/gt_tumor_mean - 1)*100:.1f}% error)")
        elif abs(s1_tumor_mean - gt_tumor_mean) / gt_tumor_mean > 0.1:
            findings.append(f"NOTICE: TUMOR INTENSITY DIFFERENCE: {s1_tumor_mean:.3f} vs GT {gt_tumor_mean:.3f} ({abs(s1_tumor_mean/gt_tumor_mean - 1)*100:.1f}% error)")

    for finding in findings:
        report.append(finding)
    report.append("")

    if not findings:
        report.append("OK: No major issues detected. Stage 1 performs reasonably well.")

    report.append("")
    report.append("=" * 80)

    # Write report
    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    # Print to console
    print(report_text)

    # Save statistics to CSV
    if valid_results:
        df = pd.DataFrame(valid_results)
        csv_path = os.path.join(output_dir, "statistics_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nStatistics saved to: {csv_path}")

    return report_text


# ============================================================================
# Main Function
# ============================================================================

def main(
    env_config_path: str,
    model_config_path: str,
    model_def_path: str,
    num_samples: int = 5,
    output_dir: str = None,
    num_gpus: int = 1,
    num_inference_steps: int = 30,
    gt_dir: str = None,
):
    """
    Main function to run inference, visualization, and analysis.

    Args:
        env_config_path: Path to environment config JSON
        model_config_path: Path to model training config JSON
        model_def_path: Path to model definition config JSON
        num_samples: Number of validation samples to process
        output_dir: Directory to save outputs
        num_gpus: Number of GPUs to use
        num_inference_steps: Number of diffusion sampling steps
        gt_dir: Ground truth directory (default: ./data/step_4)
    """
    # Setup output directories
    temp_output = output_dir or "./outputs/inference_analysis"
    vis_output_dir = os.path.join(temp_output, "visualizations")
    analysis_output_dir = os.path.join(temp_output, "analysis")
    nifti_output_dir = os.path.join(temp_output, "nifti_outputs")

    os.makedirs(vis_output_dir, exist_ok=True)
    os.makedirs(analysis_output_dir, exist_ok=True)
    os.makedirs(nifti_output_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(temp_output)
    logger.info("=" * 80)
    logger.info("Starting Stage 1 ControlNet Inference and Analysis")
    logger.info("=" * 80)

    # Use default GT directory if not specified
    if gt_dir is None:
        gt_dir = DEFAULT_GT_DIR

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

    all_results = []
    processed_count = 0
    sample_ids = []

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
                if "label" in batch and hasattr(batch["label"], "meta"):
                    meta = batch["label"].meta
                    if "filename_or_obj" in meta:
                        pre_path = meta["filename_or_obj"]
                        if isinstance(pre_path, (list, tuple)):
                            pre_path = pre_path[0] if len(pre_path) > 0 else ""
                        basename = os.path.basename(pre_path)
                        sample_id = basename.replace("_mask_aligned.nii.gz", "").replace("_mask.nii.gz", "")

                if sample_id == f"sample_{batch_idx}" and "pre" in batch and hasattr(batch["pre"], "meta"):
                    meta = batch["pre"].meta
                    if "filename_or_obj" in meta:
                        pre_path = meta["filename_or_obj"]
                        if isinstance(pre_path, (list, tuple)):
                            pre_path = pre_path[0] if len(pre_path) > 0 else ""
                        basename = os.path.basename(pre_path)
                        sample_id = basename.replace("_pre_aligned.nii.gz", "").replace("_pre.nii.gz", "")

                logger.info(f"\nProcessing sample {processed_count + 1}/{num_samples}: {sample_id}")
                sample_ids.append(sample_id)

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

                # ========== Save NIfTI files ==========
                reference_pre_path = None
                if "pre" in batch and hasattr(batch["pre"], "meta"):
                    meta = batch["pre"].meta
                    if "filename_or_obj" in meta:
                        reference_pre_path = meta["filename_or_obj"]
                        if isinstance(reference_pre_path, (list, tuple)):
                            reference_pre_path = reference_pre_path[0] if len(reference_pre_path) > 0 else None

                save_nifti_with_metadata(gen_sub_np, sample_id, nifti_output_dir, "generated_sub", reference_pre_path, logger)
                save_nifti_with_metadata(pre_np, sample_id, nifti_output_dir, "pre", reference_pre_path, logger)
                save_nifti_with_metadata(mask_np, sample_id, nifti_output_dir, "mask", reference_pre_path, logger)
                # ========== End NIfTI saving ==========

                # Create visualizations
                for view in ["axial", "sagittal", "coronal"]:
                    vis_path = os.path.join(vis_output_dir, f"{sample_id}_{view}.png")
                    create_comparison_visualization(
                        pre_img=pre_np,
                        mask=mask_np,
                        gen_sub=gen_sub_np,
                        gt_sub=None,  # Will be loaded during analysis
                        save_path=vis_path,
                        view=view,
                        sample_id=sample_id
                    )
                    logger.info(f"  Saved {view} view: {vis_path}")

                # Analyze sample with ground truth
                logger.info(f"  Analyzing against ground truth...")
                result, _ = analyze_sample(sample_id, gen_sub_np, pre_np, mask_np, gt_dir)
                all_results.append(result)

                if result.get("has_gt", False):
                    # Get GT for visualization
                    gt_sub = load_gt_subtraction(sample_id, gt_dir)

                    # Update visualizations with GT
                    for view in ["axial", "sagittal", "coronal"]:
                        vis_path = os.path.join(vis_output_dir, f"{sample_id}_{view}.png")
                        create_comparison_visualization(
                            pre_img=pre_np,
                            mask=mask_np,
                            gen_sub=gen_sub_np,
                            gt_sub=gt_sub,
                            save_path=vis_path,
                            view=view,
                            sample_id=sample_id
                        )

                    # Generate analysis visualizations
                    generate_histogram_comparison(sample_id, gt_sub, gen_sub_np, pre_np, mask_np, analysis_output_dir)
                    generate_slice_comparison(sample_id, gt_sub, gen_sub_np, pre_np, mask_np, analysis_output_dir)

                    # Log key metrics
                    logger.info(f"  Analysis Results:")
                    logger.info(f"    Global MAE: {result.get('stage1_global_mae', np.nan):.4f}")
                    logger.info(f"    Tumor MAE: {result.get('stage1_tumor_mae', np.nan):.4f}")
                    logger.info(f"    Background MAE: {result.get('stage1_background_mae', np.nan):.4f}")
                else:
                    logger.warning(f"  Ground truth not available for comparison")

                processed_count += 1

                # Clean up memory
                del generated_sub
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"  Error processing sample: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Step 4: Generate summary report
    logger.info("-" * 80)
    logger.info("Generating summary report...")

    generate_summary_report(all_results, analysis_output_dir)

    logger.info("=" * 80)
    logger.info("Inference and Analysis complete!")
    logger.info(f"Results saved to: {temp_output}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference, visualization, and analysis for Stage 1 ControlNet"
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
        help="Output directory for results (default: ./outputs/inference_analysis)"
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
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Ground truth directory (default: ./data/step_4)"
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
        gt_dir=args.gt_dir,
    )
