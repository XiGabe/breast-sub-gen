#!/usr/bin/env python3
"""
Visualize inference results: compare predicted subtraction maps with ground truth.
"""

import os
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_nii(path):
    """Load NIfTI file and return data array."""
    try:
        return nib.load(path).get_fdata()
    except Exception as e:
        print(f"  ERROR loading {path}: {e}")
        return None

def find_sample_files(output_dir, sample_id):
    """Find prediction and GT files for a sample."""
    pred_path = os.path.join(output_dir, f"{sample_id}_pred_sub.nii.gz")
    gt_path = os.path.join(output_dir, f"{sample_id}_gt_sub.nii.gz")
    return pred_path, gt_path

def find_pre_image(sample_id):
    """Find the pre-contrast image for a sample."""
    # Try different possible paths
    possible_paths = [
        f"processed_pre/{sample_id}_pre_aligned.nii.gz",
        f"./processed_pre/{sample_id}_pre_aligned.nii.gz",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def get_tumor_center(mask_data):
    """Find the center of mass of the tumor region."""
    # Assuming tumor has values > 0
    tumor_indices = np.where(mask_data > 0.1 * mask_data.max())
    if len(tumor_indices[0]) == 0:
        # No tumor found, return center of volume
        return [d // 2 for d in mask_data.shape]
    return [int(np.mean(idx)) for idx in tumor_indices]

def visualize_sample(sample_id, pred_path, gt_path, pre_path=None, output_dir="output_breast_sub_infer"):
    """Create visualization for a single sample."""
    # Load data
    pred = load_nii(pred_path)
    gt = load_nii(gt_path)

    # Skip if files are corrupted
    if pred is None or gt is None:
        print(f"  Skipping {sample_id} due to corrupted files")
        return None

    # Get tumor center from GT for better visualization
    center = get_tumor_center(gt)

    # Extract slices at tumor center
    z_slice = center[0]
    y_slice = center[1]
    x_slice = center[2]

    # Create figure with 3 rows (axial, sagittal, coronal) x 3 cols (pre, gt, pred)
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle(f'Sample: {sample_id}\nSlice locations: Z={z_slice}, Y={y_slice}, X={x_slice}',
                 fontsize=16, fontweight='bold')

    # Common normalization for subtraction maps
    vmin_sub, vmax_sub = 0, 1

    # Row 1: Axial view (Z slice)
    # Axial - Pre (if available)
    pre = None
    if pre_path and os.path.exists(pre_path):
        pre = load_nii(pre_path)
        if pre is not None:
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(pre[z_slice, :, :], cmap='gray', origin='lower')
            ax1.set_title('Pre-contrast (Axial)', fontweight='bold')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Axial - GT
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(gt[z_slice, :, :], cmap='hot', vmin=vmin_sub, vmax=vmax_sub, origin='lower')
    ax2.set_title('Ground Truth Sub (Axial)', fontweight='bold', color='green')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Axial - Prediction
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(pred[z_slice, :, :], cmap='hot', vmin=vmin_sub, vmax=vmax_sub, origin='lower')
    ax3.set_title('Predicted Sub (Axial)', fontweight='bold', color='blue')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Axial - Difference
    ax4 = fig.add_subplot(gs[0, 3])
    diff = np.abs(pred[z_slice, :, :] - gt[z_slice, :, :])
    im4 = ax4.imshow(diff, cmap='Reds', vmin=0, vmax=0.5, origin='lower')
    ax4.set_title('|Pred - GT| (Axial)', fontweight='bold', color='red')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # Row 2: Sagittal view (Y slice)
    if pre is not None:
        ax5 = fig.add_subplot(gs[1, 0])
        im5 = ax5.imshow(pre[:, y_slice, :], cmap='gray', origin='lower')
        ax5.set_title('Pre-contrast (Sagittal)', fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(gt[:, y_slice, :], cmap='hot', vmin=vmin_sub, vmax=vmax_sub, origin='lower')
    ax6.set_title('Ground Truth Sub (Sagittal)', fontweight='bold', color='green')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    ax7 = fig.add_subplot(gs[1, 2])
    im7 = ax7.imshow(pred[:, y_slice, :], cmap='hot', vmin=vmin_sub, vmax=vmax_sub, origin='lower')
    ax7.set_title('Predicted Sub (Sagittal)', fontweight='bold', color='blue')
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    ax8 = fig.add_subplot(gs[1, 3])
    diff = np.abs(pred[:, y_slice, :] - gt[:, y_slice, :])
    im8 = ax8.imshow(diff, cmap='Reds', vmin=0, vmax=0.5, origin='lower')
    ax8.set_title('|Pred - GT| (Sagittal)', fontweight='bold', color='red')
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

    # Row 3: Coronal view (X slice)
    if pre is not None:
        ax9 = fig.add_subplot(gs[2, 0])
        im9 = ax9.imshow(pre[:, :, x_slice], cmap='gray', origin='lower')
        ax9.set_title('Pre-contrast (Coronal)', fontweight='bold')
        ax9.axis('off')
        plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)

    ax10 = fig.add_subplot(gs[2, 1])
    im10 = ax10.imshow(gt[:, :, x_slice], cmap='hot', vmin=vmin_sub, vmax=vmax_sub, origin='lower')
    ax10.set_title('Ground Truth Sub (Coronal)', fontweight='bold', color='green')
    ax10.axis('off')
    plt.colorbar(im10, ax=ax10, fraction=0.046, pad=0.04)

    ax11 = fig.add_subplot(gs[2, 2])
    im11 = ax11.imshow(pred[:, :, x_slice], cmap='hot', vmin=vmin_sub, vmax=vmax_sub, origin='lower')
    ax11.set_title('Predicted Sub (Coronal)', fontweight='bold', color='blue')
    ax11.axis('off')
    plt.colorbar(im11, ax=ax11, fraction=0.046, pad=0.04)

    ax12 = fig.add_subplot(gs[2, 3])
    diff = np.abs(pred[:, :, x_slice] - gt[:, :, x_slice])
    im12 = ax12.imshow(diff, cmap='Reds', vmin=0, vmax=0.5, origin='lower')
    ax12.set_title('|Pred - GT| (Coronal)', fontweight='bold', color='red')
    ax12.axis('off')
    plt.colorbar(im12, ax=ax12, fraction=0.046, pad=0.04)

    # Calculate metrics
    mae = np.mean(np.abs(pred - gt))
    mse = np.mean((pred - gt) ** 2)
    gt_max = gt.max()
    pred_max = pred.max()

    # Add metrics as text
    metrics_text = f'Metrics:\nMAE: {mae:.4f}\nMSE: {mse:.4f}\nGT Max: {gt_max:.4f}\nPred Max: {pred_max:.4f}'
    fig.text(0.02, 0.02, metrics_text, fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return fig

def main():
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output_breast_sub_infer_s3_e122"

    # Get all prediction files
    pred_files = sorted([f for f in os.listdir(output_dir) if f.endswith('_pred_sub.nii.gz')])
    sample_ids = [f.replace('_pred_sub.nii.gz', '') for f in pred_files]

    print(f"Found {len(sample_ids)} samples to visualize")

    # Create visualization output directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Visualize each sample
    skipped = []
    for sample_id in sample_ids:
        print(f"Visualizing {sample_id}...")
        pred_path, gt_path = find_sample_files(output_dir, sample_id)
        pre_path = find_pre_image(sample_id)

        fig = visualize_sample(sample_id, pred_path, gt_path, pre_path, output_dir)

        if fig is None:
            skipped.append(sample_id)
            continue

        # Save figure
        output_path = os.path.join(viz_dir, f"{sample_id}_comparison.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_path}")

    if skipped:
        print(f"\nSkipped {len(skipped)} samples due to corrupted files:")
        for s in skipped:
            print(f"  - {s}")

    print(f"\nAll visualizations saved to: {viz_dir}")

if __name__ == "__main__":
    main()
