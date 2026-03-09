#!/usr/bin/env python
"""
Analyze Stage 1 vs Stage 2 inference outputs with Ground Truth comparison.

This script compares the outputs from Stage 1 and Stage 2 models against GT,
focusing on:
1. Background suppression effectiveness
2. Tumor enhancement intensity
3. Spatial distribution patterns
4. Statistical differences between stages
5. MAE metrics against ground truth
"""

import os
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

# Define paths
STAGE1_DIR = "./outputs/inference_stage1_final/nifti_outputs"
STAGE2_DIR = "./outputs/inference_stage2_final/nifti_outputs"
GT_DIR = "./data/step_4"
PROCESSED_PRE_DIR = "./data/processed_pre"
OUTPUT_DIR = "./outputs/analysis_report"
SAMPLES = ["DUKE_021_L", "DUKE_022_L", "DUKE_041_L"]
TARGET_SIZE = (256, 256, 256)


def load_nifti(file_path: str) -> Optional[np.ndarray]:
    """Load NIfTI file and return data array."""
    if not os.path.exists(file_path):
        return None
    nii = nib.load(file_path)
    return nii.get_fdata()


def load_gt_subtraction(sample_id: str) -> Optional[np.ndarray]:
    """Load ground truth subtraction from step_4 and resize to match inference outputs.

    Since step_4 contains the original subtraction images (not padded to 256³),
    we need to apply the same padding/crop transform that was used for preprocessing.

    The preprocessing pipeline uses ResizeWithPadOrCropd to get to 256³.
    """
    gt_path = os.path.join(GT_DIR, f"{sample_id}_sub.nii.gz")
    if not os.path.exists(gt_path):
        print(f"Warning: GT subtraction not found: {gt_path}")
        return None

    # Load GT subtraction
    gt_nii = nib.load(gt_path)
    gt_data = gt_nii.get_fdata().astype(np.float32)

    # Get reference shape from processed_pre to understand the transform
    ref_path = os.path.join(PROCESSED_PRE_DIR, f"{sample_id}_pre_aligned.nii.gz")
    if not os.path.exists(ref_path):
        print(f"Warning: Reference processed_pre not found: {ref_path}")
        # Apply simple center crop/pad to 256³
        return resize_to_target(gt_data, TARGET_SIZE)

    ref_nii = nib.load(ref_path)
    ref_shape = ref_nii.shape

    if ref_shape != TARGET_SIZE:
        print(f"Warning: Reference shape {ref_shape} != target {TARGET_SIZE}")

    # Apply resize to match target size
    # Since affines are identical, we just need to pad/crop spatial dimensions
    return resize_to_target(gt_data, TARGET_SIZE)


def resize_to_target(data: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
    """Resize data to target size using center crop/pad.

    Args:
        data: Input data array [H, W, D]
        target_size: Target size (H, W, D)

    Returns:
        Resized data array
    """
    result = np.zeros(target_size, dtype=data.dtype)

    # For each dimension, compute slice indices
    slices_in = []
    slices_out = []

    for dim, (src_sz, tgt_sz) in enumerate(zip(data.shape, target_size)):
        if src_sz >= tgt_sz:
            # Crop: center crop
            start = (src_sz - tgt_sz) // 2
            end = start + tgt_sz
            slices_in.append(slice(start, end))
            slices_out.append(slice(0, tgt_sz))
        else:
            # Pad: center pad
            start = (tgt_sz - src_sz) // 2
            end = start + src_sz
            slices_in.append(slice(0, src_sz))
            slices_out.append(slice(start, end))

    # Apply slicing
    result[tuple(slices_out)] = data[tuple(slices_in)]

    return result


def create_masks(pre: np.ndarray, mask: np.ndarray,
                breast_threshold: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create region masks.

    Args:
        pre: Pre-contrast image [H, W, D]
        mask: Tumor mask [H, W, D]
        breast_threshold: Threshold for breast tissue detection

    Returns:
        tumor_mask, breast_mask, background_mask
    """
    tumor_mask = mask > 0
    breast_mask = (pre > breast_threshold) & (~tumor_mask)
    background_mask = pre <= breast_threshold

    return tumor_mask, breast_mask, background_mask


def compute_statistics(data: np.ndarray, mask: np.ndarray, name: str) -> Dict:
    """Compute statistics for a specific region.

    Args:
        data: Full volume data
        mask: Region mask
        name: Region name

    Returns:
        Dictionary of statistics
    """
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
    """Compute background suppression quality metrics.

    Args:
        data: Full volume data
        bg_mask: Background mask

    Returns:
        Dictionary of metrics
    """
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
    """Compute Mean Absolute Error for a specific region.

    Args:
        pred: Predicted volume
        gt: Ground truth volume
        mask: Region mask
        region_name: Region name for prefix

    Returns:
        Dictionary with MAE metrics
    """
    if mask.sum() == 0:
        return {
            f"{region_name}_mae": np.nan,
            f"{region_name}_mae_pct": np.nan,
        }

    pred_vals = pred[mask]
    gt_vals = gt[mask]

    mae = float(np.abs(pred_vals - gt_vals).mean())
    # Normalize by GT range
    gt_range = gt_vals.max() - gt_vals.min()
    mae_pct = float(mae / gt_range * 100) if gt_range > 0 else np.nan

    return {
        f"{region_name}_mae": mae,
        f"{region_name}_mae_pct": mae_pct,
    }


def analyze_sample(sample_id: str) -> Tuple:
    """Analyze a single sample with ground truth.

    Args:
        sample_id: Sample identifier (e.g., "DUKE_021_L")

    Returns:
        (results_dict, gt_sub, stage1_sub, stage2_sub, pre, mask)
    """
    results = {"sample_id": sample_id}

    # Load Stage 1
    stage1_sub = load_nifti(os.path.join(STAGE1_DIR, f"{sample_id}_generated_sub.nii.gz"))
    stage1_pre = load_nifti(os.path.join(STAGE1_DIR, f"{sample_id}_pre.nii.gz"))
    stage1_mask = load_nifti(os.path.join(STAGE1_DIR, f"{sample_id}_mask.nii.gz"))

    # Load Stage 2
    stage2_sub = load_nifti(os.path.join(STAGE2_DIR, f"{sample_id}_generated_sub.nii.gz"))

    # Load Ground Truth from step_4
    gt_sub = load_gt_subtraction(sample_id)

    if stage1_sub is None or stage2_sub is None or gt_sub is None:
        print(f"Warning: Could not load data for {sample_id}")
        return results, None, None, None, None, None

    # Use Stage 1 pre and mask for consistency
    pre = stage1_pre
    mask = stage1_mask

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

    # Analyze Stage 2
    stage2_stats = {}
    stage2_stats.update(compute_statistics(stage2_sub, tumor_mask, "tumor"))
    stage2_stats.update(compute_statistics(stage2_sub, breast_mask, "breast"))
    stage2_stats.update(compute_statistics(stage2_sub, bg_mask, "background"))
    stage2_stats.update(compute_statistics(stage2_sub, tumor_mask | breast_mask, "foreground"))
    stage2_stats.update(compute_background_suppression_metrics(stage2_sub, bg_mask))
    # MAE against GT
    stage2_stats.update(compute_mae(stage2_sub, gt_sub, tumor_mask, "tumor"))
    stage2_stats.update(compute_mae(stage2_sub, gt_sub, breast_mask, "breast"))
    stage2_stats.update(compute_mae(stage2_sub, gt_sub, bg_mask, "background"))
    stage2_stats.update(compute_mae(stage2_sub, gt_sub, tumor_mask | breast_mask, "foreground"))
    stage2_stats.update(compute_mae(stage2_sub, gt_sub, np.ones_like(gt_sub, dtype=bool), "global"))

    for key, value in stage2_stats.items():
        results[f"stage2_{key}"] = value

    return results, gt_sub, stage1_sub, stage2_sub, pre, mask


def generate_histogram_comparison(sample_id: str, gt_sub: np.ndarray,
                                   stage1_sub: np.ndarray, stage2_sub: np.ndarray,
                                   pre: np.ndarray, mask: np.ndarray):
    """Generate histogram comparison plot with GT."""
    tumor_mask, breast_mask, bg_mask = create_masks(pre, mask)

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Global histograms (top row)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(gt_sub.flatten(), bins=100, alpha=0.5, label='GT', color='green', density=True)
    ax1.hist(stage1_sub.flatten(), bins=100, alpha=0.5, label='Stage 1', color='blue', density=True)
    ax1.hist(stage2_sub.flatten(), bins=100, alpha=0.5, label='Stage 2', color='red', density=True)
    ax1.set_xlabel('Intensity')
    ax1.set_ylabel('Density')
    ax1.set_title('Global Intensity Distribution')
    ax1.legend()
    ax1.set_xlim(0, 1)

    # Tumor region histograms (middle row)
    ax2 = fig.add_subplot(gs[1, 0])
    if tumor_mask.sum() > 0:
        ax2.hist(gt_sub[tumor_mask], bins=50, alpha=0.5, label='GT', color='green', density=True)
        ax2.hist(stage1_sub[tumor_mask], bins=50, alpha=0.5, label='Stage 1', color='blue', density=True)
        ax2.hist(stage2_sub[tumor_mask], bins=50, alpha=0.5, label='Stage 2', color='red', density=True)
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Density')
    ax2.set_title('Tumor Region Intensity Distribution')
    ax2.legend()
    ax2.set_xlim(0, 1)

    # Background region histograms (bottom row)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.hist(gt_sub[bg_mask], bins=100, alpha=0.5, label='GT', color='green', density=True)
    ax3.hist(stage1_sub[bg_mask], bins=100, alpha=0.5, label='Stage 1', color='blue', density=True)
    ax3.hist(stage2_sub[bg_mask], bins=100, alpha=0.5, label='Stage 2', color='red', density=True)
    ax3.set_xlabel('Intensity')
    ax3.set_ylabel('Density')
    ax3.set_title('Background Region Intensity Distribution')
    ax3.legend()
    ax3.set_xlim(0, 0.5)  # Zoom in to see low values

    # CDF plots (right column)
    ax4 = fig.add_subplot(gs[0, 1])
    for data, label, color in [(gt_sub, 'GT', 'green'),
                                (stage1_sub, 'Stage 1', 'blue'),
                                (stage2_sub, 'Stage 2', 'red')]:
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
        for data, label, color in [(gt_sub, 'GT', 'green'),
                                    (stage1_sub, 'Stage 1', 'blue'),
                                    (stage2_sub, 'Stage 2', 'red')]:
            sorted_vals = np.sort(data[tumor_mask])
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax5.plot(sorted_vals, cdf, label=label, color=color, linewidth=2)
    ax5.set_xlabel('Intensity')
    ax5.set_ylabel('CDF')
    ax5.set_title('Tumor Region CDF')
    ax5.legend()
    ax5.set_xlim(0, 1)
    ax5.grid(True, alpha=0.3)

    # Background CDF (zoomed)
    ax6 = fig.add_subplot(gs[2, 1])
    for data, label, color in [(gt_sub, 'GT', 'green'),
                                (stage1_sub, 'Stage 1', 'blue'),
                                (stage2_sub, 'Stage 2', 'red')]:
        sorted_vals = np.sort(data[bg_mask])
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax6.plot(sorted_vals, cdf, label=label, color=color, linewidth=2)
    ax6.set_xlabel('Intensity')
    ax6.set_ylabel('CDF')
    ax6.set_title('Background Region CDF')
    ax6.legend()
    ax6.set_xlim(0, 0.5)
    ax6.grid(True, alpha=0.3)

    # Background suppression quality bar chart (right column)
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
    s2_metrics = [
        (stage2_sub[bg_mask] < 0.01).sum() / bg_mask.sum() * 100,
        (stage2_sub[bg_mask] < 0.05).sum() / bg_mask.sum() * 100,
        (stage2_sub[bg_mask] < 0.10).sum() / bg_mask.sum() * 100,
        (stage2_sub[bg_mask] < 0.20).sum() / bg_mask.sum() * 100,
    ]

    x = np.arange(len(metrics))
    width = 0.25
    ax7.bar(x - width, gt_metrics, width, label='GT', color='green')
    ax7.bar(x, s1_metrics, width, label='Stage 1', color='blue')
    ax7.bar(x + width, s2_metrics, width, label='Stage 2', color='red')
    ax7.set_ylabel('Percentage (%)')
    ax7.set_title('Background Suppression Quality\n(Higher is better)')
    ax7.set_xticks(x)
    ax7.set_xticklabels(metrics)
    ax7.legend()
    ax7.set_ylim(0, 100)
    ax7.grid(axis='y', alpha=0.3)

    # MAE comparison (middle right)
    ax8 = fig.add_subplot(gs[1, 2])
    regions = ['Global', 'Background', 'Breast', 'Tumor', 'Foreground']
    s1_mae = [
        np.abs(stage1_sub - gt_sub).mean(),
        np.abs(stage1_sub[bg_mask] - gt_sub[bg_mask]).mean(),
        np.abs(stage1_sub[breast_mask] - gt_sub[breast_mask]).mean() if breast_mask.sum() > 0 else 0,
        np.abs(stage1_sub[tumor_mask] - gt_sub[tumor_mask]).mean() if tumor_mask.sum() > 0 else 0,
        np.abs(stage1_sub[tumor_mask | breast_mask] - gt_sub[tumor_mask | breast_mask]).mean(),
    ]
    s2_mae = [
        np.abs(stage2_sub - gt_sub).mean(),
        np.abs(stage2_sub[bg_mask] - gt_sub[bg_mask]).mean(),
        np.abs(stage2_sub[breast_mask] - gt_sub[breast_mask]).mean() if breast_mask.sum() > 0 else 0,
        np.abs(stage2_sub[tumor_mask] - gt_sub[tumor_mask]).mean() if tumor_mask.sum() > 0 else 0,
        np.abs(stage2_sub[tumor_mask | breast_mask] - gt_sub[tumor_mask | breast_mask]).mean(),
    ]

    x = np.arange(len(regions))
    width = 0.35
    ax8.bar(x - width/2, s1_mae, width, label='Stage 1', color='blue')
    ax8.bar(x + width/2, s2_mae, width, label='Stage 2', color='red')
    ax8.set_ylabel('MAE')
    ax8.set_title('Mean Absolute Error by Region\n(Lower is better)')
    ax8.set_xticks(x)
    ax8.set_xticklabels(regions, rotation=45, ha='right')
    ax8.legend()
    ax8.grid(axis='y', alpha=0.3)

    # Percentile comparison (bottom right)
    ax9 = fig.add_subplot(gs[2, 2])
    percentiles = [5, 25, 50, 75, 95]
    gt_p = np.percentile(gt_sub[tumor_mask], percentiles) if tumor_mask.sum() > 0 else [0, 0, 0, 0, 0]
    s1_p = np.percentile(stage1_sub[tumor_mask], percentiles) if tumor_mask.sum() > 0 else [0, 0, 0, 0, 0]
    s2_p = np.percentile(stage2_sub[tumor_mask], percentiles) if tumor_mask.sum() > 0 else [0, 0, 0, 0, 0]

    x = np.arange(len(percentiles))
    width = 0.25
    ax9.bar(x - width, gt_p, width, label='GT', color='green')
    ax9.bar(x, s1_p, width, label='Stage 1', color='blue')
    ax9.bar(x + width, s2_p, width, label='Stage 2', color='red')
    ax9.set_ylabel('Intensity')
    ax9.set_title('Tumor Region Percentiles')
    ax9.set_xticks(x)
    ax9.set_xticklabels([f'P{p}' for p in percentiles])
    ax9.legend()
    ax9.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Sample: {sample_id} - Intensity Distribution Analysis', fontsize=16, y=1.02)

    output_path = os.path.join(OUTPUT_DIR, f"{sample_id}_distributions.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_slice_comparison(sample_id: str, gt_sub: np.ndarray,
                               stage1_sub: np.ndarray, stage2_sub: np.ndarray,
                               pre: np.ndarray, mask: np.ndarray):
    """Generate slice comparison with GT (4 columns)."""
    tumor_mask, breast_mask, bg_mask = create_masks(pre, mask)

    # Find center slices
    h, w, d = pre.shape
    axial_slice = d // 2
    coronal_slice = w // 2
    sagittal_slice = h // 2

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # Row 1: Axial view
    # Pre-contrast
    im = axes[0, 0].imshow(pre[:, :, axial_slice].T, cmap='gray', origin='lower')
    axes[0, 0].set_title('Pre-contrast (Axial)')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

    # GT
    im = axes[0, 1].imshow(gt_sub[:, :, axial_slice].T, cmap='hot', vmin=0, vmax=1, origin='lower')
    axes[0, 1].contour(pre[:, :, axial_slice].T, levels=[0.05], colors='cyan', linewidths=0.5, alpha=0.5, origin='lower')
    axes[0, 1].contour(mask[:, :, axial_slice].T, levels=[0.5], colors='lime', linewidths=1, origin='lower')
    axes[0, 1].set_title(f'Ground Truth (Axial #{axial_slice})')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    # Stage 1
    im = axes[0, 2].imshow(stage1_sub[:, :, axial_slice].T, cmap='hot', vmin=0, vmax=1, origin='lower')
    axes[0, 2].contour(pre[:, :, axial_slice].T, levels=[0.05], colors='cyan', linewidths=0.5, alpha=0.5, origin='lower')
    axes[0, 2].contour(mask[:, :, axial_slice].T, levels=[0.5], colors='lime', linewidths=1, origin='lower')
    axes[0, 2].set_title('Stage 1 (Axial)')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    # Stage 2
    im = axes[0, 3].imshow(stage2_sub[:, :, axial_slice].T, cmap='hot', vmin=0, vmax=1, origin='lower')
    axes[0, 3].contour(pre[:, :, axial_slice].T, levels=[0.05], colors='cyan', linewidths=0.5, alpha=0.5, origin='lower')
    axes[0, 3].contour(mask[:, :, axial_slice].T, levels=[0.5], colors='lime', linewidths=1, origin='lower')
    axes[0, 3].set_title('Stage 2 (Axial)')
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

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

    im = axes[1, 3].imshow(stage2_sub[:, coronal_slice, :].T, cmap='hot', vmin=0, vmax=1, origin='lower')
    axes[1, 3].contour(pre[:, coronal_slice, :].T, levels=[0.05], colors='cyan', linewidths=0.5, alpha=0.5, origin='lower')
    axes[1, 3].contour(mask[:, coronal_slice, :].T, levels=[0.5], colors='lime', linewidths=1, origin='lower')
    axes[1, 3].set_title('Stage 2 (Coronal)')
    axes[1, 3].axis('off')
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046)

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

    im = axes[2, 3].imshow(stage2_sub[sagittal_slice, :, :].T, cmap='hot', vmin=0, vmax=1, origin='lower')
    axes[2, 3].contour(pre[sagittal_slice, :, :].T, levels=[0.05], colors='cyan', linewidths=0.5, alpha=0.5, origin='lower')
    axes[2, 3].contour(mask[sagittal_slice, :, :].T, levels=[0.5], colors='lime', linewidths=1, origin='lower')
    axes[2, 3].set_title('Stage 2 (Sagittal)')
    axes[2, 3].axis('off')
    plt.colorbar(im, ax=axes[2, 3], fraction=0.046)

    plt.suptitle(f'Sample: {sample_id} - Slice Comparison', fontsize=16, y=1.02)

    output_path = os.path.join(OUTPUT_DIR, f"{sample_id}_slices.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_report(all_results: list):
    """Generate a comprehensive summary report."""
    report = []
    report.append("=" * 80)
    report.append("STAGE 1 VS STAGE 2 VS GROUND TRUTH ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    # Aggregate statistics
    report.append("## AGGREGATE STATISTICS (N={} samples)".format(len(all_results)))
    report.append("")

    # Helper to compute average across samples
    def avg_metric(key):
        values = [r[key] for r in all_results if key in r and not np.isnan(r[key])]
        return np.mean(values) if values else np.nan

    def format_val(val):
        return f"{val:.4f}" if not np.isnan(val) else "N/A"

    # Global MAE comparison
    report.append("### GLOBAL MAE (Lower is better)")
    report.append("-" * 40)
    s1_global_mae = avg_metric("stage1_global_mae")
    s2_global_mae = avg_metric("stage2_global_mae")
    report.append(f"  Stage 1: {format_val(s1_global_mae)}")
    report.append(f"  Stage 2: {format_val(s2_global_mae)}")
    if not np.isnan(s1_global_mae) and not np.isnan(s2_global_mae):
        if s2_global_mae < s1_global_mae:
            report.append(f"  Winner: Stage 2 ({abs(s2_global_mae/s1_global_mae - 1)*100:.1f}% better)")
        else:
            report.append(f"  Winner: Stage 1 ({abs(s1_global_mae/s2_global_mae - 1)*100:.1f}% better)")
    report.append("")

    # Background suppression MAE
    report.append("### BACKGROUND MAE (Lower is better)")
    report.append("-" * 40)
    s1_bg_mae = avg_metric("stage1_background_mae")
    s2_bg_mae = avg_metric("stage2_background_mae")
    report.append(f"  Stage 1: {format_val(s1_bg_mae)}")
    report.append(f"  Stage 2: {format_val(s2_bg_mae)}")
    if not np.isnan(s1_bg_mae) and not np.isnan(s2_bg_mae):
        if s2_bg_mae < s1_bg_mae:
            report.append(f"  Winner: Stage 2 ({abs(s2_bg_mae/s1_bg_mae - 1)*100:.1f}% better)")
        else:
            report.append(f"  Winner: Stage 1 ({abs(s1_bg_mae/s2_bg_mae - 1)*100:.1f}% better)")
    report.append("")

    # Tumor enhancement MAE
    report.append("### TUMOR MAE (Lower is better)")
    report.append("-" * 40)
    s1_tumor_mae = avg_metric("stage1_tumor_mae")
    s2_tumor_mae = avg_metric("stage2_tumor_mae")
    report.append(f"  Stage 1: {format_val(s1_tumor_mae)}")
    report.append(f"  Stage 2: {format_val(s2_tumor_mae)}")
    if not np.isnan(s1_tumor_mae) and not np.isnan(s2_tumor_mae):
        if s2_tumor_mae < s1_tumor_mae:
            report.append(f"  Winner: Stage 2 ({abs(s2_tumor_mae/s1_tumor_mae - 1)*100:.1f}% better)")
        else:
            report.append(f"  Winner: Stage 1 ({abs(s1_tumor_mae/s2_tumor_mae - 1)*100:.1f}% better)")
    report.append("")

    # Background suppression quality
    report.append("### BACKGROUND SUPPRESSION QUALITY (Higher % below threshold = better)")
    report.append("-" * 40)
    for thresh in [0.01, 0.05, 0.10, 0.20]:
        key = f"bg_pct_below_{int(thresh*100):03d}"
        gt_pct = avg_metric(f"gt_{key}")
        s1_pct = avg_metric(f"stage1_{key}")
        s2_pct = avg_metric(f"stage2_{key}")
        report.append(f"  Below {thresh}:")
        report.append(f"    GT:       {format_val(gt_pct)}%")
        report.append(f"    Stage 1:  {format_val(s1_pct)}%")
        report.append(f"    Stage 2:  {format_val(s2_pct)}%")
    report.append("")

    # Tumor intensity statistics
    report.append("### TUMOR REGION INTENSITY STATISTICS")
    report.append("-" * 40)
    for stat in ["mean", "std", "median", "p95"]:
        gt_val = avg_metric(f"gt_tumor_{stat}")
        s1_val = avg_metric(f"stage1_tumor_{stat}")
        s2_val = avg_metric(f"stage2_tumor_{stat}")
        report.append(f"  {stat.upper()}:")
        report.append(f"    GT:       {format_val(gt_val)}")
        report.append(f"    Stage 1:  {format_val(s1_val)}")
        report.append(f"    Stage 2:  {format_val(s2_val)}")
    report.append("")

    # Per-sample details
    report.append("=" * 80)
    report.append("PER-SAMPLE DETAILS")
    report.append("=" * 80)
    report.append("")

    for result in all_results:
        sample_id = result.get("sample_id", "Unknown")
        report.append(f"## {sample_id}")
        report.append("")

        report.append("### MAE Comparison")
        report.append(f"  {'Region':<15} {'Stage 1':<12} {'Stage 2':<12} {'Better':<10}")
        report.append("  " + "-" * 50)

        for region, region_key in [("Global", "global"), ("Background", "background"),
                                    ("Breast", "breast"), ("Tumor", "tumor"), ("Foreground", "foreground")]:
            s1_mae = result.get(f"stage1_{region_key}_mae", np.nan)
            s2_mae = result.get(f"stage2_{region_key}_mae", np.nan)

            if not np.isnan(s1_mae) and not np.isnan(s2_mae):
                better = "S1" if s1_mae < s2_mae else "S2"
                diff = abs(s1_mae - s2_mae) / max(s1_mae, s2_mae) * 100
                report.append(f"  {region:<15} {s1_mae:<12.4f} {s2_mae:<12.4f} {better} ({diff:.1f}%)")
            else:
                report.append(f"  {region:<15} {format_val(s1_mae):<12} {format_val(s2_mae):<12} N/A")

        report.append("")
        report.append("### Background Suppression (% voxels below threshold)")
        report.append(f"  {'Threshold':<12} {'GT':<10} {'Stage 1':<10} {'Stage 2':<10}")
        report.append("  " + "-" * 45)

        for thresh in [0.01, 0.05, 0.10, 0.20]:
            key = f"bg_pct_below_{int(thresh*100):03d}"
            gt_pct = result.get(f"gt_{key}", np.nan)
            s1_pct = result.get(f"stage1_{key}", np.nan)
            s2_pct = result.get(f"stage2_{key}", np.nan)
            report.append(f"  {thresh:<12.2f} {format_val(gt_pct):<10} {format_val(s1_pct):<10} {format_val(s2_pct):<10}")

        report.append("")
        report.append("### Tumor Region Statistics")
        report.append(f"  {'Metric':<10} {'GT':<10} {'Stage 1':<10} {'Stage 2':<10}")
        report.append("  " + "-" * 45)

        for stat in ["mean", "std", "median", "p95"]:
            gt_val = result.get(f"gt_tumor_{stat}", np.nan)
            s1_val = result.get(f"stage1_tumor_{stat}", np.nan)
            s2_val = result.get(f"stage2_tumor_{stat}", np.nan)
            report.append(f"  {stat:<10} {format_val(gt_val):<10} {format_val(s1_val):<10} {format_val(s2_val):<10}")

        report.append("")
        report.append("-" * 80)
        report.append("")

    # Key findings
    report.append("=" * 80)
    report.append("KEY FINDINGS")
    report.append("=" * 80)
    report.append("")

    findings = []

    # Check regression
    if not np.isnan(s2_global_mae) and not np.isnan(s1_global_mae):
        if s2_global_mae > s1_global_mae * 1.1:
            findings.append(f"⚠️  STAGE 2 REGRESSION: Stage 2 has {abs(s2_global_mae/s1_global_mae - 1)*100:.1f}% HIGHER global MAE than Stage 1")
        elif s2_global_mae < s1_global_mae * 0.9:
            findings.append(f"✓ STAGE 2 IMPROVEMENT: Stage 2 has {abs(s1_global_mae/s2_global_mae - 1)*100:.1f}% LOWER global MAE than Stage 1")

    # Background suppression check
    gt_bg_suppress = avg_metric("gt_bg_pct_below_010")
    s1_bg_suppress = avg_metric("stage1_bg_pct_below_010")
    s2_bg_suppress = avg_metric("stage2_bg_pct_below_010")

    if gt_bg_suppress > 50:
        if s1_bg_suppress < 20:
            findings.append(f"⚠️  POOR BACKGROUND SUPPRESSION (Stage 1): Only {s1_bg_suppress:.1f}% of background < 0.10 (GT: {gt_bg_suppress:.1f}%)")
        if s2_bg_suppress < 20:
            findings.append(f"⚠️  POOR BACKGROUND SUPPRESSION (Stage 2): Only {s2_bg_suppress:.1f}% of background < 0.10 (GT: {gt_bg_suppress:.1f}%)")

    # Tumor enhancement check
    gt_tumor_mean = avg_metric("gt_tumor_mean")
    s1_tumor_mean = avg_metric("stage1_tumor_mean")
    s2_tumor_mean = avg_metric("stage2_tumor_mean")

    if not np.isnan(gt_tumor_mean):
        if abs(s1_tumor_mean - gt_tumor_mean) / gt_tumor_mean > 0.3:
            findings.append(f"⚠️  STAGE 1 TUMOR INTENSITY BIAS: {s1_tumor_mean:.3f} vs GT {gt_tumor_mean:.3f} ({abs(s1_tumor_mean/gt_tumor_mean - 1)*100:.1f}% error)")
        if abs(s2_tumor_mean - gt_tumor_mean) / gt_tumor_mean > 0.3:
            findings.append(f"⚠️  STAGE 2 TUMOR INTENSITY BIAS: {s2_tumor_mean:.3f} vs GT {gt_tumor_mean:.3f} ({abs(s2_tumor_mean/gt_tumor_mean - 1)*100:.1f}% error)")

    for finding in findings:
        report.append(finding)
    report.append("")

    if not findings:
        report.append("No significant issues detected. Both stages perform similarly.")

    report.append("")
    report.append("=" * 80)

    # Write report
    report_text = "\n".join(report)
    report_path = os.path.join(OUTPUT_DIR, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    # Print to console
    print(report_text)

    return report_text


def main():
    """Main analysis function."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("STAGE 1 VS STAGE 2 VS GROUND TRUTH ANALYSIS")
    print("=" * 80)
    print("")

    all_results = []

    for sample_id in SAMPLES:
        print(f"\n{'=' * 80}")
        print(f"Analyzing: {sample_id}")
        print('=' * 80)

        results, gt_sub, stage1_sub, stage2_sub, pre, mask = analyze_sample(sample_id)

        if gt_sub is None:
            print(f"Skipping {sample_id} due to missing data")
            continue

        all_results.append(results)

        # Generate visualizations
        generate_histogram_comparison(sample_id, gt_sub, stage1_sub, stage2_sub, pre, mask)
        generate_slice_comparison(sample_id, gt_sub, stage1_sub, stage2_sub, pre, mask)

        # Print key stats
        print(f"\nKey Statistics for {sample_id}:")
        print(f"  GT Tumor Mean: {results.get('gt_tumor_mean', np.nan):.4f}")
        print(f"  Stage 1 Tumor Mean: {results.get('stage1_tumor_mean', np.nan):.4f}")
        print(f"  Stage 2 Tumor Mean: {results.get('stage2_tumor_mean', np.nan):.4f}")
        print(f"  GT Background < 0.10: {results.get('gt_bg_pct_below_010', np.nan):.2f}%")
        print(f"  Stage 1 Background < 0.10: {results.get('stage1_bg_pct_below_010', np.nan):.2f}%")
        print(f"  Stage 2 Background < 0.10: {results.get('stage2_bg_pct_below_010', np.nan):.2f}%")
        print(f"  Stage 1 Global MAE: {results.get('stage1_global_mae', np.nan):.4f}")
        print(f"  Stage 2 Global MAE: {results.get('stage2_global_mae', np.nan):.4f}")

    # Save statistics to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(OUTPUT_DIR, "statistics_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nStatistics saved to: {csv_path}")

        # Generate summary report
        print("\n")
        generate_summary_report(all_results)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
