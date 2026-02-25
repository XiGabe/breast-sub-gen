#!/usr/bin/env python3
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
VAE Safety Test for Breast Subtraction Maps

This script tests the VAE's encode-decode quality with different input ranges
to ensure proper handling of sparse subtraction maps.

Background:
-----------
- step_4/ data is pre-normalized to [0, 1]
- Current MAISI training pipeline applies additional ScaleIntensityRangePercentilesd
- This may cause double normalization or incorrect input ranges

Usage:
------
python -m scripts.00_vae_safety_test \\
    --data_dir /path/to/step_4 \\
    --vae_path /path/to/autoencoder.pt \\
    [--sample_id DUKE_001_L]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

# Import MAISI utilities
try:
    from scripts.utils import define_instance
    from scripts.diff_model_setting import load_config
    from monai.inferers import SlidingWindowInferer
except ImportError as e:
    print(f"Warning: Could not import MAISI utilities: {e}")
    print("This script requires being run from the project root directory.")
    sys.exit(1)


def load_subtraction_map(data_dir: str, sample_id: str | None = None) -> tuple[np.ndarray, str, str]:
    """
    Load a subtraction map from the step_4 directory.

    Args:
        data_dir: Path to step_4 directory
        sample_id: Optional specific sample ID (e.g., "DUKE_001_L")

    Returns:
        Tuple of (numpy array, sample_id, filepath)
    """
    if sample_id:
        pattern = os.path.join(data_dir, f"{sample_id}_sub.nii.gz")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No subtraction map found for sample_id={sample_id}")
        filepath = matches[0]
    else:
        pattern = os.path.join(data_dir, "*_sub.nii.gz")
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No subtraction maps found in {data_dir}")
        filepath = matches[0]
        # Extract sample_id from filename
        sample_id = os.path.basename(filepath).replace("_sub.nii.gz", "")

    img = nib.load(filepath)
    data = img.get_fdata()
    return data, sample_id, filepath


def load_vae_from_configs(
    env_config: str,
    model_config: str,
    network_config: str,
    vae_path: str,
    device: torch.device
) -> tuple[torch.nn.Module, dict]:
    """
    Load VAE using MAISI config system.

    Args:
        env_config: Path to environment config JSON
        model_config: Path to model config JSON
        network_config: Path to network config JSON
        vae_path: Path to VAE checkpoint
        device: Torch device

    Returns:
        Tuple of (VAE model, checkpoint dict with scale_factor if present)
    """
    # Load merged config
    args = load_config(env_config, model_config, network_config)

    # Instantiate autoencoder
    autoencoder = define_instance(args, "autoencoder_def").to(device)

    # Load checkpoint
    checkpoint = torch.load(vae_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "unet_state_dict" in checkpoint:
            state_dict = checkpoint["unet_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        autoencoder.load_state_dict(state_dict)
    else:
        # Checkpoint is the state_dict itself
        autoencoder.load_state_dict(checkpoint)
        checkpoint = {"state_dict": checkpoint}

    autoencoder.eval()

    return autoencoder, checkpoint


def prepare_input(
    data: np.ndarray,
    input_range: str,
    device: torch.device
) -> torch.Tensor:
    """
    Prepare input tensor for VAE encoding.

    Args:
        data: Input numpy array
        input_range: Either "0to1" or "-1to1"
        device: Torch device

    Returns:
        Prepared torch tensor [1, 1, X, Y, Z]
    """
    # Convert to tensor and add batch/channel dims
    x = torch.from_numpy(data).float().to(device)
    x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, X, Y, Z]

    # Apply range transformation
    if input_range == "-1to1":
        # Convert [0, 1] -> [-1, 1]
        x = x * 2.0 - 1.0
    elif input_range != "0to1":
        raise ValueError(f"Invalid input_range: {input_range}")

    return x


def reverse_range_transform(x: torch.Tensor, input_range: str) -> torch.Tensor:
    """
    Reverse the range transformation after decoding.

    Args:
        x: Decoded tensor
        input_range: Either "0to1" or "-1to1"

    Returns:
        Tensor with original range
    """
    if input_range == "-1to1":
        # Convert [-1, 1] -> [0, 1]
        x = (x + 1.0) / 2.0
        # Clip to valid range
        x = torch.clamp(x, 0.0, 1.0)
    return x


def dynamic_infer(inferer, model, images):
    """
    Perform dynamic inference using a model and an inferer.

    Args:
        inferer: An inference object, typically a monai SlidingWindowInferer
        model: The model used for inference
        images: The input images for inference, shape [N,C,H,W,D] or [N,C,H,W]

    Returns:
        The output from the model or the inferer, depending on the input size
    """
    import math
    if torch.numel(images[0:1, 0:1, ...]) <= math.prod(inferer.roi_size):
        return model(images)
    else:
        # Extract the spatial dimensions from the images tensor (H, W, D)
        spatial_dims = images.shape[2:]
        orig_roi = inferer.roi_size

        # Check that roi has the same number of dimensions as spatial_dims
        if len(orig_roi) != len(spatial_dims):
            raise ValueError(f"ROI length ({len(orig_roi)}) does not match spatial dimensions ({len(spatial_dims)}).")

        # Iterate and adjust each ROI dimension
        adjusted_roi = [min(roi_dim, img_dim) for roi_dim, img_dim in zip(orig_roi, spatial_dims)]
        inferer.roi_size = adjusted_roi
        output = inferer(network=model, inputs=images)
        inferer.roi_size = orig_roi
        return output


def encode_decode(
    vae: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    target_shape: tuple | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode and decode using VAE with sliding window inference.

    Args:
        vae: VAE model
        x: Input tensor [1, 1, X, Y, Z]
        device: Torch device
        target_shape: Optional target shape for output (D, H, W)

    Returns:
        Tuple of (latent z, decoded output)
    """
    # Create sliding window inferer for large volumes
    inferer = SlidingWindowInferer(
        roi_size=[320, 320, 160],
        sw_batch_size=1,
        progress=False,
        mode="gaussian",
        overlap=0.4,
        sw_device=device,
        device=device,
    )

    # Use the same autocast approach as the original MAISI code
    # Mixed precision for encode/decode pass (CUDA AMP)
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            # Encode
            z = dynamic_infer(inferer, vae.encode_stage_2_inputs, x)

            # Decode
            decoded = dynamic_infer(inferer, vae.decode_stage_2_outputs, z)

            # Handle shape mismatch by center cropping or padding
            if target_shape is not None:
                decoded = decoded.squeeze()  # [1, 1, D, H, W] -> [D, H, W]
                current_shape = decoded.shape

                # Center crop if output is larger than target
                if current_shape[0] > target_shape[0]:
                    start = (current_shape[0] - target_shape[0]) // 2
                    decoded = decoded[start:start + target_shape[0], :, :]
                elif current_shape[0] < target_shape[0]:
                    # Pad if output is smaller
                    pad_total = target_shape[0] - current_shape[0]
                    pad_before = pad_total // 2
                    pad_after = pad_total - pad_before
                    decoded = F.pad(decoded, (0, 0, 0, 0, pad_before, pad_after))

                if current_shape[1] > target_shape[1]:
                    start = (current_shape[1] - target_shape[1]) // 2
                    decoded = decoded[:, start:start + target_shape[1], :]
                elif current_shape[1] < target_shape[1]:
                    pad_total = target_shape[1] - current_shape[1]
                    pad_before = pad_total // 2
                    pad_after = pad_total - pad_before
                    decoded = F.pad(decoded, (0, 0, pad_before, pad_after, 0, 0))

                if current_shape[2] > target_shape[2]:
                    start = (current_shape[2] - target_shape[2]) // 2
                    decoded = decoded[:, :, start:start + target_shape[2]]
                elif current_shape[2] < target_shape[2]:
                    pad_total = target_shape[2] - current_shape[2]
                    pad_before = pad_total // 2
                    pad_after = pad_total - pad_before
                    decoded = F.pad(decoded, (pad_before, pad_after, 0, 0, 0, 0))

    return z, decoded


def calculate_mae(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return float(np.mean(np.abs(original - reconstructed)))


def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_val = max(original.max(), reconstructed.max())
    return float(20 * np.log10(max_val) - 10 * np.log10(mse))


def print_test_results(
    test_name: str,
    input_range: str,
    input_data: torch.Tensor,
    latent: torch.Tensor,
    reconstructed: np.ndarray,
    original: np.ndarray
) -> dict:
    """
    Print formatted test results and return metrics.

    Returns:
        Dict with mae, psnr, and pass status
    """
    mae = calculate_mae(original, reconstructed)
    psnr = calculate_psnr(original, reconstructed)

    print(f"\n{'='*60}")
    print(f"{test_name}")
    print(f"{'='*60}")
    print(f"输入范围 (Input Range): {input_range}")
    print(f"  输入数据范围: [{input_data.min():.6f}, {input_data.max():.6f}]")
    print(f"  潜空间范围: [{latent.min():.6f}, {latent.max():.6f}]")
    print(f"  潜空间形状: {latent.shape}")
    print(f"  重建数据范围: [{reconstructed.min():.6f}, {reconstructed.max():.6f}]")
    print(f"  MAE (平均绝对误差): {mae:.6e}")
    print(f"  PSNR (峰值信噪比): {psnr:.2f} dB")

    # Determine PASS/FAIL based on MAE and output range validity
    # For sparse subtraction maps, use a more lenient threshold
    mae_threshold = 0.10  # More lenient for sparse data (~95% background)

    # Check if output range is valid (no negative values for [0,1] input)
    valid_range = True
    if input_range == "[0, 1]" and reconstructed.min() < -0.05:
        valid_range = False
        print(f"  警告 (Warning): [0,1] 输入产生了负值输出，VAE 可能期望 [-1,1] 输入")

    if mae < mae_threshold and valid_range:
        status = "PASS ✓"
        passed = True
    else:
        status = "FAIL ✗"
        passed = False

    print(f"  状态 (Status): {status}")

    return {"mae": mae, "psnr": psnr, "passed": passed, "valid_range": valid_range}


def save_vae_visualization(
    original: np.ndarray,
    input_encoded: torch.Tensor,
    latent: torch.Tensor,
    reconstructed: np.ndarray,
    sample_id: str,
    output_dir: str = "./vae_vis_output"
) -> None:
    """
    Save VAE encoding/decoding visualization images.

    Args:
        original: Original [0, 1] subtraction map (D, H, W)
        input_encoded: Input for VAE encoding [-1, 1] as tensor [1, 1, D, H, W]
        latent: Latent space representation [1, C, D, H, W]
        reconstructed: Reconstructed [0, 1] output (D, H, W)
        sample_id: Sample identifier for filename
        output_dir: Output directory for visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get middle slices for each axis
    def get_middle_slices(volume):
        """Get middle slices from axial, sagittal, coronal views."""
        d, h, w = volume.shape
        axial = volume[d // 2, :, :]
        sagittal = volume[:, h // 2, :]
        coronal = volume[:, :, w // 2]
        return axial, sagittal, coronal

    # Extract slices
    orig_axial, orig_sag, orig_cor = get_middle_slices(original)
    recon_axial, recon_sag, recon_cor = get_middle_slices(reconstructed)

    # Get encoded input (convert back to [0,1] for visualization)
    input_np = input_encoded.squeeze().cpu().numpy()
    input_np = (input_np + 1.0) / 2.0  # [-1,1] -> [0,1]
    input_axial, input_sag, input_cor = get_middle_slices(input_np)

    # Get latent (use mean across channels for visualization)
    latent_np = latent.squeeze().float().cpu().numpy()  # [C, D, H, W]
    latent_mean = latent_np.mean(axis=0)  # [D, H, W]
    latent_axial, latent_sag, latent_cor = get_middle_slices(latent_mean)

    # Normalize latent for better visualization
    def normalize_for_vis(slice_data):
        """Normalize slice data to [0, 1] for visualization."""
        slice_min, slice_max = slice_data.min(), slice_data.max()
        if slice_max - slice_min > 0:
            return (slice_data - slice_min) / (slice_max - slice_min)
        return slice_data

    latent_axial = normalize_for_vis(latent_axial)
    latent_sag = normalize_for_vis(latent_sag)
    latent_cor = normalize_for_vis(latent_cor)

    # Create figure with 4 rows (original, encoded, latent, reconstructed) and 3 columns (axial, sagittal, coronal)
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))

    # Row titles
    row_titles = ['Original [0,1]', 'VAE Input [-1,1]→[0,1] for vis',
                  'Latent Space (mean of 4 channels)', 'Reconstructed [0,1]']
    col_titles = ['Axial (middle slice)', 'Sagittal (middle slice)', 'Coronal (middle slice)']

    for i, row_title in enumerate(row_titles):
        axes[i, 0].set_ylabel(row_title, fontsize=10, fontweight='bold')

    for j, col_title in enumerate(col_titles):
        axes[0, j].set_title(col_title, fontsize=10)

    # Plot original
    axes[0, 0].imshow(orig_axial, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].imshow(orig_sag, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].imshow(orig_cor, cmap='gray', vmin=0, vmax=1)

    # Plot encoded input
    axes[1, 0].imshow(input_axial, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].imshow(input_sag, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].imshow(input_cor, cmap='gray', vmin=0, vmax=1)

    # Plot latent
    axes[2, 0].imshow(latent_axial, cmap='viridis')
    axes[2, 1].imshow(latent_sag, cmap='viridis')
    axes[2, 2].imshow(latent_cor, cmap='viridis')

    # Plot reconstructed
    axes[3, 0].imshow(recon_axial, cmap='gray', vmin=0, vmax=1)
    axes[3, 1].imshow(recon_sag, cmap='gray', vmin=0, vmax=1)
    axes[3, 2].imshow(recon_cor, cmap='gray', vmin=0, vmax=1)

    # Add error map (difference)
    diff = np.abs(original - reconstructed)
    diff_axial, diff_sag, diff_cor = get_middle_slices(diff)

    # Add colorbars for latent and error map
    for i in range(4):
        for j in range(3):
            axes[i, j].axis('off')

    plt.suptitle(f'VAE Encoding/Decoding Visualization: {sample_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, f'{sample_id}_vae_visualization.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  保存可视化图片到: {fig_path}")
    plt.close()

    # Also save as NIfTI files for inspection in ITK-SNAP or similar
    affine = np.eye(4)

    # Ensure all arrays are float32 for NIfTI compatibility
    original_f32 = original.astype(np.float32)
    input_np_f32 = input_np.astype(np.float32)
    latent_mean_f32 = latent_mean.astype(np.float32)
    reconstructed_f32 = reconstructed.astype(np.float32)

    nib.save(nib.Nifti1Image(original_f32, affine), os.path.join(output_dir, f'{sample_id}_01_original.nii.gz'))
    nib.save(nib.Nifti1Image(input_np_f32, affine), os.path.join(output_dir, f'{sample_id}_02_vae_input.nii.gz'))
    nib.save(nib.Nifti1Image(latent_mean_f32, affine), os.path.join(output_dir, f'{sample_id}_03_latent_mean.nii.gz'))
    nib.save(nib.Nifti1Image(reconstructed_f32, affine), os.path.join(output_dir, f'{sample_id}_04_reconstructed.nii.gz'))
    nib.save(nib.Nifti1Image(diff, affine), os.path.join(output_dir, f'{sample_id}_05_error_map.nii.gz'))

    print(f"  保存 NIfTI 文件到: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="VAE Safety Test for Breast Subtraction Maps")
    parser.add_argument(
        "-d", "--data_dir",
        type=str,
        default="./step_4",
        help="Path to step_4 directory with preprocessed data"
    )
    parser.add_argument(
        "-v", "--vae_path",
        type=str,
        default="./models/autoencoder_v2.pt",
        help="Path to VAE checkpoint"
    )
    parser.add_argument(
        "-s", "--sample_id",
        type=str,
        default=None,
        help="Specific sample ID to test (e.g., DUKE_001_L). If not provided, uses first found."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for testing"
    )
    parser.add_argument(
        "-e", "--env_config",
        type=str,
        default="configs/environment_maisi_diff_model_rflow-mr.json",
        help="Path to environment config JSON"
    )
    parser.add_argument(
        "-c", "--model_config",
        type=str,
        default="configs/config_maisi_diff_model_rflow-mr.json",
        help="Path to model config JSON"
    )
    parser.add_argument(
        "-t", "--network_config",
        type=str,
        default="configs/config_network_rflow.json",
        help="Path to network config JSON"
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print("VAE 安全测试 (VAE Safety Test)")
    print(f"{'='*60}")
    print(f"设备 (Device): {device}")
    print(f"数据目录 (Data Directory): {args.data_dir}")
    print(f"VAE 路径 (VAE Path): {args.vae_path}")

    # Check if VAE file exists
    if not os.path.exists(args.vae_path):
        print(f"\n[ERROR] VAE checkpoint not found: {args.vae_path}")
        print(f"Please provide a valid VAE checkpoint path.")
        sys.exit(1)

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"\n[ERROR] Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Load subtraction map
    print(f"\n正在加载减影图 (Loading subtraction map)...")
    try:
        data, sample_id, filepath = load_subtraction_map(args.data_dir, args.sample_id)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"样本 ID (Sample ID): {sample_id}")
    print(f"文件路径 (Filepath): {filepath}")
    print(f"原始形状 (Original Shape): {data.shape}")
    print(f"原始范围 (Original Range): [{data.min():.6f}, {data.max():.6f}]")
    print(f"稀疏度 (Sparsity - zero ratio): {np.mean(data == 0) * 100:.1f}%")

    # Load VAE
    print(f"\n正在加载 VAE (Loading VAE)...")
    print(f"环境配置 (Env Config): {args.env_config}")
    print(f"模型配置 (Model Config): {args.model_config}")
    print(f"网络配置 (Network Config): {args.network_config}")

    try:
        vae, checkpoint = load_vae_from_configs(
            args.env_config,
            args.model_config,
            args.network_config,
            args.vae_path,
            device
        )
        print("VAE 加载成功 (VAE loaded successfully)")
    except Exception as e:
        print(f"[ERROR] Failed to load VAE: {e}")
        print("\nFalling back to basic test mode (showing expected output)...")
        # Show expected output format without actual testing
        print(f"\n=== VAE 安全测试结果 ===")
        print(f"测试 VAE: {args.vae_path}")
        print(f"样本: {sample_id}_sub.nii.gz")

        print(f"\n测试 1: 输入范围 [0, 1]")
        print("  输入范围: [{:.6f}, {:.6f}]".format(data.min(), data.max()))
        print("  [VAE not loaded - cannot perform actual test]")

        print(f"\n测试 2: 输入范围 [-1, 1]")
        print("  输入范围: [{:.6f}, {:.6f}]".format(data.min() * 2 - 1, data.max() * 2 - 1))
        print("  [VAE not loaded - cannot perform actual test]")

        print(f"\n推荐: 请检查 VAE 路径和配置文件是否正确")
        return

    # Check for scale_factor
    scale_factor = checkpoint.get("scale_factor", None)
    if scale_factor is not None:
        print(f"[INFO] VAE checkpoint contains scale_factor: {scale_factor}")
        print("       This scale factor should be considered when comparing results.")
    else:
        print("[INFO] VAE checkpoint does NOT contain scale_factor")

    # Run tests
    print(f"\n{'='*60}")
    print("开始测试 (Starting Tests)")
    print(f"{'='*60}")

    results = {}

    # Test 1: [0, 1] range
    print(f"\n正在运行测试 1: 输入范围 [0, 1]...")
    x_0to1 = prepare_input(data, "0to1", device)
    target_shape = data.shape  # (D, H, W)
    try:
        z_0to1, decoded_0to1 = encode_decode(vae, x_0to1, device, target_shape=target_shape)
        reconstructed_0to1 = decoded_0to1.cpu().numpy()
        results["0to1"] = print_test_results(
            "测试 1: 输入范围 [0, 1]",
            "[0, 1]",
            x_0to1,
            z_0to1,
            reconstructed_0to1,
            data
        )
    except Exception as e:
        print(f"[ERROR] Test 1 failed: {e}")
        results["0to1"] = {"mae": float('inf'), "psnr": 0, "passed": False}

    # Test 2: [-1, 1] range
    print(f"\n正在运行测试 2: 输入范围 [-1, 1]...")
    x_neg1to1 = prepare_input(data, "-1to1", device)
    try:
        z_neg1to1, decoded_neg1to1 = encode_decode(vae, x_neg1to1, device, target_shape=target_shape)
        decoded_neg1to1 = reverse_range_transform(decoded_neg1to1, "-1to1")
        reconstructed_neg1to1 = decoded_neg1to1.cpu().numpy()
        results["-1to1"] = print_test_results(
            "测试 2: 输入范围 [-1, 1]",
            "[-1, 1]",
            x_neg1to1,
            z_neg1to1,
            reconstructed_neg1to1,
            data
        )

        # Save visualization for the correct input range ([-1, 1])
        print(f"\n保存可视化结果...")
        save_vae_visualization(
            original=data,
            input_encoded=x_neg1to1,
            latent=z_neg1to1,
            reconstructed=reconstructed_neg1to1,
            sample_id=sample_id,
            output_dir="./vae_vis_output"
        )
    except Exception as e:
        print(f"[ERROR] Test 2 failed: {e}")
        results["-1to1"] = {"mae": float('inf'), "psnr": 0, "passed": False}

    # Summary
    print(f"\n{'='*60}")
    print("测试总结 (Test Summary)")
    print(f"{'='*60}")

    print(f"\n样本: {sample_id}_sub.nii.gz")
    print(f"VAE: {args.vae_path}")

    if results.get("0to1", {}).get("passed"):
        range_0to1_status = "PASS ✓"
    else:
        range_0to1_status = "FAIL ✗"

    if results.get("-1to1", {}).get("passed"):
        range_neg1to1_status = "PASS ✓"
    else:
        range_neg1to1_status = "FAIL ✗"

    print(f"\n  [0, 1] 输入范围:  {range_0to1_status} (MAE: {results.get('0to1', {}).get('mae', float('inf')):.6e})")
    print(f"  [-1, 1] 输入范围: {range_neg1to1_status} (MAE: {results.get('-1to1', {}).get('mae', float('inf')):.6e})")

    # Make recommendation
    print(f"\n推荐 (Recommendation):")

    mae_0to1 = results.get("0to1", {}).get("mae", float('inf'))
    mae_neg1to1 = results.get("-1to1", {}).get("mae", float('inf'))
    valid_range_0to1 = results.get("0to1", {}).get("valid_range", True)

    # Check if [0,1] input produced invalid (negative) output
    if not valid_range_0to1:
        print("  → 检测到 [0,1] 输入产生了负值输出 → VAE 期望 [-1, 1] 输入范围")
        print("  → DETECTED: [0,1] input produced negative output → VAE expects [-1, 1] range")
        print("  → 需要在 VAE 编码前转换: input * 2.0 - 1.0")
        print("  → Need to apply transform before VAE encoding: input * 2.0 - 1.0")
        print("  → 在 scripts/diff_model_create_training_data.py 中，第 169 行附近添加:")
        print("     pt_nda = pt_nda * 2.0 - 1.0  # Convert [0,1] to [-1,1]")
    elif results.get("0to1", {}).get("passed") and not results.get("-1to1", {}).get("passed"):
        print("  → 使用 [0, 1] 输入范围 (Use [0, 1] input range)")
        print("  → VAE 期望 [0, 1] 范围的输入")
    elif results.get("-1to1", {}).get("passed") and not results.get("0to1", {}).get("passed"):
        print("  → 使用 [-1, 1] 输入范围 (Use [-1, 1] input range)")
        print("  → VAE 期望 [-1, 1] 范围的输入")
        print("  → 需要在 transforms.py 中添加: input * 2.0 - 1.0")
    elif results.get("0to1", {}).get("passed") and results.get("-1to1", {}).get("passed"):
        if mae_0to1 < mae_neg1to1:
            print("  → 使用 [0, 1] 输入范围 (Use [0, 1] input range) - 略优")
        else:
            print("  → 使用 [-1, 1] 输入范围 (Use [-1, 1] input range) - 略优")
    else:
        print("  → 两个测试都失败！VAE 可能有其他问题。")
        if not valid_range_0to1:
            print("  → 但 [0,1] 输入产生负值输出表明 VAE 期望 [-1,1] 输入")
        print("  → 检查 VAE checkpoint 和训练数据范围")

    print(f"\n{'='*60}")
    print("测试完成 (Test Complete)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
