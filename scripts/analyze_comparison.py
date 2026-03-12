#!/usr/bin/env python
"""
Analyze Stage 1 and Stage 2 inference outputs with Ground Truth comparison.

This script compares Stage 1 and Stage 2 model outputs against GT, focusing on:
1. Background suppression effectiveness
2. Tumor enhancement intensity
3. Spatial distribution patterns
4. Statistical analysis
5. MAE metrics against ground truth
"""

import os
import json
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

TARGET_SIZE = (256, 256, 256)


def load_nifti(file_path: str) -> Optional[np.ndarray]:
    """Load NIfTI file and return data array."""
    if not os.path.exists(file_path):
        return None
    nii = nib.load(file_path)
    return nii.get_fdata()


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


def load_gt_subtraction(sample_id: str, gt_dir: str) -> Optional[np.ndarray]:
    """Load ground truth subtraction from step_4 directory."""
    gt_path = os.path.join(gt_dir, f"{sample_id}_sub.nii.gz")
    if not os.path.exists(gt_path):
        print(f"Warning: GT subtraction not found: {gt_path}")
        return None

    gt_nii = nib.load(gt_path)
    gt_data = gt_nii.get_fdata().astype(np.float32)

    if gt_data.shape != TARGET_SIZE:
        print(f"Warning: GT shape {gt_data.shape} != target {TARGET_SIZE}, resizing...")
        gt_data = resize_to_target(gt_data, TARGET_SIZE)

    return gt_data


def create_masks(pre: np.ndarray, mask: np.ndarray, breast_threshold: float = 0.05):
    """Create region masks."""
    tumor_mask = mask > 0
    breast_mask = (pre > breast_threshold) & (~tumor_mask)
    background_mask = pre <= breast_threshold
    return tumor_mask, breast_mask, background_mask


def compute_statistics(data: np.ndarray, mask: np.ndarray, name: str) -> Dict:
    """Compute statistics for a specific region."""
    if mask.sum() == 0:
        return {f"{name}_count": 0, f"{name}_min": np.nan, f"{name}_max": np.nan,
                f"{name}_mean": np.nan, f"{name}_std": np.nan, f"{name}_median": np.nan}

    values = data[mask]
    return {
        f"{name}_count": int(mask.sum()),
        f"{name}_min": float(values.min()),
        f"{name}_max": float(values.max()),
        f"{name}_mean": float(values.mean()),
        f"{name}_std": float(values.std()),
        f"{name}_median": float(np.median(values)),
    }


def compute_background_suppression_metrics(data: np.ndarray, bg_mask: np.ndarray) -> Dict:
    """Compute background suppression quality metrics."""
    if bg_mask.sum() == 0:
        return {}

    bg_values = data[bg_mask]
    return {
        "bg_pct_below_010": float((bg_values < 0.10).sum() / len(bg_values) * 100),
        "bg_pct_below_020": float((bg_values < 0.20).sum() / len(bg_values) * 100),
    }


def compute_mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, region_name: str) -> Dict:
    """Compute Mean Absolute Error for a specific region."""
    if mask.sum() == 0:
        return {f"{region_name}_mae": np.nan}

    pred_vals = pred[mask]
    gt_vals = gt[mask]
    mae = float(np.abs(pred_vals - gt_vals).mean())

    return {f"{region_name}_mae": mae}


def analyze_sample(sample_id: str, stage1_dir: str, stage2_dir: str, gt_dir: str) -> Tuple:
    """Analyze a single sample with ground truth."""
    results = {"sample_id": sample_id}

    # Load Stage 1
    stage1_sub = load_nifti(os.path.join(stage1_dir, f"{sample_id}_generated_sub.nii.gz"))
    stage1_pre = load_nifti(os.path.join(stage1_dir, f"{sample_id}_pre.nii.gz"))
    stage1_mask = load_nifti(os.path.join(stage1_dir, f"{sample_id}_mask.nii.gz"))

    # Load Stage 2
    stage2_sub = load_nifti(os.path.join(stage2_dir, f"{sample_id}_generated_sub.nii.gz"))

    # Load Ground Truth
    gt_sub = load_gt_subtraction(sample_id, gt_dir)

    if stage1_sub is None or gt_sub is None:
        print(f"Warning: Could not load data for {sample_id}")
        return results, None, None, None, None, None

    # Use Stage 1 pre and mask for consistency
    pre = stage1_pre
    mask = stage1_mask

    # Create region masks
    tumor_mask, breast_mask, bg_mask = create_masks(pre, mask)

    # Analyze Ground Truth
    gt_stats = compute_statistics(gt_sub, tumor_mask, "gt_tumor")
    gt_stats.update(compute_statistics(gt_sub, bg_mask, "gt_bg"))
    gt_stats.update(compute_background_suppression_metrics(gt_sub, bg_mask))
    results.update(gt_stats)

    # Analyze Stage 1
    s1_stats = compute_statistics(stage1_sub, tumor_mask, "s1_tumor")
    s1_stats.update(compute_statistics(stage1_sub, bg_mask, "s1_bg"))
    s1_stats.update(compute_background_suppression_metrics(stage1_sub, bg_mask))
    s1_stats.update(compute_mae(stage1_sub, gt_sub, tumor_mask, "s1_tumor"))
    s1_stats.update(compute_mae(stage1_sub, gt_sub, bg_mask, "s1_bg"))
    s1_stats.update(compute_mae(stage1_sub, gt_sub, np.ones_like(gt_sub, dtype=bool), "s1_global"))
    results.update(s1_stats)

    # Analyze Stage 2
    if stage2_sub is not None:
        s2_stats = compute_statistics(stage2_sub, tumor_mask, "s2_tumor")
        s2_stats.update(compute_statistics(stage2_sub, bg_mask, "s2_bg"))
        s2_stats.update(compute_background_suppression_metrics(stage2_sub, bg_mask))
        s2_stats.update(compute_mae(stage2_sub, gt_sub, tumor_mask, "s2_tumor"))
        s2_stats.update(compute_mae(stage2_sub, gt_sub, bg_mask, "s2_bg"))
        s2_stats.update(compute_mae(stage2_sub, gt_sub, np.ones_like(gt_sub, dtype=bool), "s2_global"))
        results.update(s2_stats)

    return results, gt_sub, stage1_sub, stage2_sub, pre, mask


def generate_comparison_visualization(sample_id: str, gt_sub: np.ndarray, stage1_sub: np.ndarray,
                                      stage2_sub: np.ndarray, pre: np.ndarray, mask: np.ndarray,
                                      output_dir: str):
    """Generate comparison visualization for GT, Stage1 and Stage2."""
    tumor_mask, breast_mask, bg_mask = create_masks(pre, mask)

    # Find center slices
    axial_slice = pre.shape[2] // 2
    coronal_slice = pre.shape[1] // 2
    sagittal_slice = pre.shape[0] // 2

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # Column labels
    col_titles = ['Pre+Mask', 'Ground Truth', 'Stage 1', 'Stage 2']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight='bold')

    # Row 1: Axial view
    axes[0, 0].imshow(pre[:, :, axial_slice].T, cmap='gray', origin='lower')
    axes[0, 0].contour(mask[:, :, axial_slice].T, levels=[0.5], colors='lime', linewidths=1, origin='lower')
    axes[0, 0].axis('off')

    for idx, data in enumerate([gt_sub, stage1_sub, stage2_sub]):
        if data is not None:
            im = axes[0, idx + 1].imshow(data[:, :, axial_slice].T, cmap='hot', vmin=0, vmax=1, origin='lower')
            axes[0, idx + 1].contour(mask[:, :, axial_slice].T, levels=[0.5], colors='lime', linewidths=1, origin='lower')
            axes[0, idx + 1].axis('off')
            plt.colorbar(im, ax=axes[0, idx + 1], fraction=0.046)

    # Row 2: Coronal view
    axes[1, 0].imshow(pre[:, coronal_slice, :].T, cmap='gray', origin='lower')
    axes[1, 0].contour(mask[:, coronal_slice, :].T, levels=[0.5], colors='lime', linewidths=1, origin='lower')
    axes[1, 0].axis('off')

    for idx, data in enumerate([gt_sub, stage1_sub, stage2_sub]):
        if data is not None:
            im = axes[1, idx + 1].imshow(data[:, coronal_slice, :].T, cmap='hot', vmin=0, vmax=1, origin='lower')
            axes[1, idx + 1].contour(mask[:, coronal_slice, :].T, levels=[0.5], colors='lime', linewidths=1, origin='lower')
            axes[1, idx + 1].axis('off')
            plt.colorbar(im, ax=axes[1, idx + 1], fraction=0.046)

    # Row 3: Sagittal view
    axes[2, 0].imshow(pre[sagittal_slice, :, :].T, cmap='gray', origin='lower')
    axes[2, 0].contour(mask[sagittal_slice, :, :].T, levels=[0.5], colors='lime', linewidths=1, origin='lower')
    axes[2, 0].axis('off')

    for idx, data in enumerate([gt_sub, stage1_sub, stage2_sub]):
        if data is not None:
            im = axes[2, idx + 1].imshow(data[sagittal_slice, :, :].T, cmap='hot', vmin=0, vmax=1, origin='lower')
            axes[2, idx + 1].contour(mask[sagittal_slice, :, :].T, levels=[0.5], colors='lime', linewidths=1, origin='lower')
            axes[2, idx + 1].axis('off')
            plt.colorbar(im, ax=axes[2, idx + 1], fraction=0.046)

    plt.suptitle(f'Sample: {sample_id} - GT vs Stage 1 vs Stage 2 Comparison', fontsize=16, y=1.02)
    plt.savefig(os.path.join(output_dir, f"{sample_id}_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/{sample_id}_comparison.png")


def generate_histogram_comparison(sample_id: str, gt_sub: np.ndarray, stage1_sub: np.ndarray,
                                  stage2_sub: np.ndarray, pre: np.ndarray, mask: np.ndarray,
                                  output_dir: str):
    """Generate histogram comparison plot."""
    tumor_mask, breast_mask, bg_mask = create_masks(pre, mask)

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    colors = {'GT': 'green', 'Stage1': 'blue', 'Stage2': 'red'}
    data_dict = {'GT': gt_sub, 'Stage1': stage1_sub}

    if stage2_sub is not None:
        data_dict['Stage2'] = stage2_sub

    # Global histogram
    ax1 = fig.add_subplot(gs[0, 0])
    for label, data in data_dict.items():
        ax1.hist(data.flatten(), bins=100, alpha=0.5, label=label, color=colors[label], density=True)
    ax1.set_xlabel('Intensity')
    ax1.set_ylabel('Density')
    ax1.set_title('Global Intensity Distribution')
    ax1.legend()
    ax1.set_xlim(0, 1)

    # Tumor region histogram
    ax2 = fig.add_subplot(gs[0, 1])
    if tumor_mask.sum() > 0:
        for label, data in data_dict.items():
            ax2.hist(data[tumor_mask], bins=50, alpha=0.5, label=label, color=colors[label], density=True)
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Density')
    ax2.set_title('Tumor Region Intensity Distribution')
    ax2.legend()
    ax2.set_xlim(0, 1)

    # Background region histogram
    ax3 = fig.add_subplot(gs[0, 2])
    for label, data in data_dict.items():
        ax3.hist(data[bg_mask], bins=100, alpha=0.5, label=label, color=colors[label], density=True)
    ax3.set_xlabel('Intensity')
    ax3.set_ylabel('Density')
    ax3.set_title('Background Region Intensity Distribution')
    ax3.legend()
    ax3.set_xlim(0, 0.3)

    # MAE comparison
    ax4 = fig.add_subplot(gs[1, 0])
    regions = ['Global', 'Background', 'Tumor']

    s1_mae = [
        np.abs(stage1_sub - gt_sub).mean(),
        np.abs(stage1_sub[bg_mask] - gt_sub[bg_mask]).mean(),
        np.abs(stage1_sub[tumor_mask] - gt_sub[tumor_mask]).mean() if tumor_mask.sum() > 0 else 0,
    ]

    mae_data = [s1_mae]
    mae_labels = ['Stage1']

    if stage2_sub is not None:
        s2_mae = [
            np.abs(stage2_sub - gt_sub).mean(),
            np.abs(stage2_sub[bg_mask] - gt_sub[bg_mask]).mean(),
            np.abs(stage2_sub[tumor_mask] - gt_sub[tumor_mask]).mean() if tumor_mask.sum() > 0 else 0,
        ]
        mae_data.append(s2_mae)
        mae_labels.append('Stage2')

    x = np.arange(len(regions))
    width = 0.35 / (len(mae_data) - 1) if len(mae_data) > 1 else 0.35

    for i, (mae_vals, label) in enumerate(zip(mae_data, mae_labels)):
        offset = (i - len(mae_data) // 2) * width
        bars = ax4.bar(x + offset, mae_vals, width, label=label, color=colors.get(label, 'gray'))

    ax4.set_ylabel('MAE')
    ax4.set_title('Mean Absolute Error by Region\n(Lower is better)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(regions)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    # Background suppression comparison
    ax5 = fig.add_subplot(gs[1, 1])
    metrics = ['< 0.10', '< 0.20']

    s1_bg = [
        (stage1_sub[bg_mask] < 0.10).sum() / bg_mask.sum() * 100,
        (stage1_sub[bg_mask] < 0.20).sum() / bg_mask.sum() * 100,
    ]
    gt_bg = [
        (gt_sub[bg_mask] < 0.10).sum() / bg_mask.sum() * 100,
        (gt_sub[bg_mask] < 0.20).sum() / bg_mask.sum() * 100,
    ]

    # Collect all data with labels
    all_bg_data = [('GT', gt_bg, 'green'), ('Stage1', s1_bg, 'blue')]

    if stage2_sub is not None:
        s2_bg = [
            (stage2_sub[bg_mask] < 0.10).sum() / bg_mask.sum() * 100,
            (stage2_sub[bg_mask] < 0.20).sum() / bg_mask.sum() * 100,
        ]
        all_bg_data.append(('Stage2', s2_bg, 'red'))

    x = np.arange(len(metrics))
    width = 0.8 / len(all_bg_data)

    for i, (label, bg_vals, color) in enumerate(all_bg_data):
        offset = (i - len(all_bg_data) / 2 + 0.5) * width
        ax5.bar(x + offset, bg_vals, width, label=label, color=color)

    ax5.set_ylabel('Percentage (%)')
    ax5.set_title('Background Suppression\n(Higher is better)')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)

    # Tumor statistics
    ax6 = fig.add_subplot(gs[1, 2])
    stats = ['mean', 'median']

    s1_tumor = [stage1_sub[tumor_mask].mean(), np.median(stage1_sub[tumor_mask])] if tumor_mask.sum() > 0 else [0, 0]
    gt_tumor = [gt_sub[tumor_mask].mean(), np.median(gt_sub[tumor_mask])] if tumor_mask.sum() > 0 else [0, 0]

    all_stat_data = [('GT', gt_tumor, 'green'), ('Stage1', s1_tumor, 'blue')]

    if stage2_sub is not None:
        s2_tumor = [stage2_sub[tumor_mask].mean(), np.median(stage2_sub[tumor_mask])] if tumor_mask.sum() > 0 else [0, 0]
        all_stat_data.append(('Stage2', s2_tumor, 'red'))

    x = np.arange(len(stats))
    width = 0.8 / len(all_stat_data)

    for i, (label, stat_vals, color) in enumerate(all_stat_data):
        offset = (i - len(all_stat_data) / 2 + 0.5) * width
        ax6.bar(x + offset, stat_vals, width, label=label, color=color)

    ax6.set_ylabel('Intensity')
    ax6.set_title('Tumor Region Statistics')
    ax6.set_xticks(x)
    ax6.set_xticklabels(stats)
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Sample: {sample_id} - Analysis', fontsize=14, y=1.02)
    plt.savefig(os.path.join(output_dir, f"{sample_id}_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/{sample_id}_analysis.png")


def generate_summary_report(all_results: list, output_dir: str):
    """Generate a comprehensive summary report."""
    report = []
    report.append("=" * 80)
    report.append("STAGE 1 vs STAGE 2 vs GROUND TRUTH ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    def avg_metric(key):
        values = [r[key] for r in all_results if key in r and not np.isnan(r.get(key, np.nan))]
        return np.mean(values) if values else np.nan

    def format_val(val):
        return f"{val:.4f}" if not np.isnan(val) else "N/A"

    # Aggregate statistics
    report.append("## AGGREGATE MAE COMPARISON (Lower is better)")
    report.append("-" * 50)
    report.append(f"{'Region':<15} {'Stage1':<12} {'Stage2':<12}")
    report.append("-" * 50)

    for region, key in [("Global", "global"), ("Background", "bg"), ("Tumor", "tumor")]:
        s1_mae = avg_metric(f"s1_{key}_mae")
        s2_mae = avg_metric(f"s2_{key}_mae") if "s2_global_mae" in all_results[0] else np.nan
        report.append(f"{region:<15} {format_val(s1_mae):<12} {format_val(s2_mae):<12}")

    report.append("")
    report.append("## BACKGROUND SUPPRESSION (% voxels below 0.10)")
    report.append("-" * 50)
    gt_bg = avg_metric("gt_bg_pct_below_010")
    s1_bg = avg_metric("s1_bg_pct_below_010")
    s2_bg = avg_metric("s2_bg_pct_below_010") if "s2_bg_pct_below_010" in all_results[0] else np.nan
    report.append(f"GT:       {format_val(gt_bg)}%")
    report.append(f"Stage 1:  {format_val(s1_bg)}%")
    report.append(f"Stage 2:  {format_val(s2_bg)}%")

    report.append("")
    report.append("## TUMOR REGION MEAN INTENSITY")
    report.append("-" * 50)
    gt_tumor = avg_metric("gt_tumor_mean")
    s1_tumor = avg_metric("s1_tumor_mean")
    s2_tumor = avg_metric("s2_tumor_mean") if "s2_tumor_mean" in all_results[0] else np.nan
    report.append(f"GT:       {format_val(gt_tumor)}")
    report.append(f"Stage 1:  {format_val(s1_tumor)}")
    report.append(f"Stage 2:  {format_val(s2_tumor)}")

    report.append("")
    report.append("=" * 80)
    report.append("PER-SAMPLE DETAILS")
    report.append("=" * 80)
    report.append("")

    for result in all_results:
        sample_id = result.get("sample_id", "Unknown")
        report.append(f"## {sample_id}")
        report.append("")
        report.append(f"  GT Tumor Mean:     {format_val(result.get('gt_tumor_mean', np.nan))}")
        report.append(f"  Stage1 Tumor Mean: {format_val(result.get('s1_tumor_mean', np.nan))}")
        report.append(f"  Stage2 Tumor Mean: {format_val(result.get('s2_tumor_mean', np.nan))}")
        report.append("")
        report.append(f"  Stage1 Global MAE: {format_val(result.get('s1_global_mae', np.nan))}")
        report.append(f"  Stage2 Global MAE: {format_val(result.get('s2_global_mae', np.nan))}")
        report.append("")
        report.append(f"  Stage1 Tumor MAE: {format_val(result.get('s1_tumor_mae', np.nan))}")
        report.append(f"  Stage2 Tumor MAE: {format_val(result.get('s2_tumor_mae', np.nan))}")
        report.append("")
        report.append("-" * 50)

    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Analyze Stage 1 vs Stage 2 vs GT")
    parser.add_argument("--stage1_dir", type=str,
                        default="./outputs/logs/breast_controlnet_stage1/inference_stage1_vis/nifti_outputs",
                        help="Stage 1 inference output directory")
    parser.add_argument("--stage2_dir", type=str,
                        default="./outputs/logs/breast_controlnet_stage2/inference_stage2_vis/nifti_outputs",
                        help="Stage 2 inference output directory")
    parser.add_argument("--gt_dir", type=str,
                        default="./data/step_4",
                        help="Ground truth directory")
    parser.add_argument("--samples", type=str, nargs="+",
                        default=["DUKE_021_L", "DUKE_021_R", "DUKE_022_L"],
                        help="Sample IDs to analyze")
    parser.add_argument("--output_dir", type=str,
                        default="./outputs/analysis_comparison",
                        help="Output directory")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("STAGE 1 vs STAGE 2 vs GROUND TRUTH ANALYSIS")
    print("=" * 80)
    print(f"Stage1 dir: {args.stage1_dir}")
    print(f"Stage2 dir: {args.stage2_dir}")
    print(f"GT dir: {args.gt_dir}")
    print("")

    all_results = []

    for sample_id in args.samples:
        print(f"\nAnalyzing: {sample_id}")
        results, gt_sub, s1_sub, s2_sub, pre, mask = analyze_sample(
            sample_id, args.stage1_dir, args.stage2_dir, args.gt_dir)

        if gt_sub is None:
            print(f"Skipping {sample_id} due to missing data")
            continue

        all_results.append(results)

        # Generate visualizations
        generate_comparison_visualization(sample_id, gt_sub, s1_sub, s2_sub, pre, mask, args.output_dir)
        generate_histogram_comparison(sample_id, gt_sub, s1_sub, s2_sub, pre, mask, args.output_dir)

    # Save statistics to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(args.output_dir, "statistics_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nStatistics saved to: {csv_path}")

        # Generate summary report
        generate_summary_report(all_results, args.output_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
