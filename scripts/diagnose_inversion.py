#!/usr/bin/env python3
"""
Diagnostic script to check if subtraction maps are inverted.
Checks the physical direction of subtraction maps and VAE behavior.
"""

import argparse
import json
import nibabel as nib
import numpy as np
import torch
from pathlib import Path

def diagnose_subtraction_map(
    sub_path: str,
    pre_path: str,
    mask_path: str,
    autoencoder_path: str = None
):
    """
    Check if subtraction maps have correct physical direction.
    """
    print("="*60)
    print("DIAGNOSING SUBTRACTION MAP INVERSION")
    print("="*60)

    # Load data
    sub_data = nib.load(sub_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata()

    print(f"\n1. SUBTRACTION MAP STATISTICS:")
    print(f"   Shape: {sub_data.shape}")
    print(f"   Range: [{sub_data.min():.4f}, {sub_data.max():.4f}]")
    print(f"   Mean: {sub_data.mean():.4f}")
    print(f"   Std: {sub_data.std():.4f}")

    # Extract tumor region statistics
    tumor_mask = mask_data > 0.5
    if tumor_mask.sum() > 0:
        tumor_values = sub_data[tumor_mask]
        bg_values = sub_data[~tumor_mask]

        print(f"\n2. TUMOR REGION (mask > 0.5):")
        print(f"   Voxel count: {tumor_mask.sum()}")
        print(f"   Range: [{tumor_values.min():.4f}, {tumor_values.max():.4f}]")
        print(f"   Mean: {tumor_values.mean():.4f}")
        print(f"   Median: {np.median(tumor_values):.4f}")

        print(f"\n3. BACKGROUND REGION (mask <= 0.5):")
        print(f"   Voxel count: {(~tumor_mask).sum()}")
        print(f"   Range: [{bg_values.min():.4f}, {bg_values.max():.4f}]")
        print(f"   Mean: {bg_values.mean():.4f}")

        # Key diagnostic: tumor vs background relationship
        print(f"\n4. PHYSICAL DIRECTION CHECK:")
        if tumor_values.mean() > bg_values.mean():
            print(f"   ✓ Tumor ({tumor_values.mean():.4f}) > Background ({bg_values.mean():.4f})")
            print(f"   => Subtraction = Post - Pre (CORRECT)")
            print(f"   => No inversion needed!")
        elif tumor_values.mean() < bg_values.mean():
            print(f"   ✗ Tumor ({tumor_values.mean():.4f}) < Background ({bg_values.mean():.4f})")
            print(f"   => Subtraction = Pre - Post (INVERTED)")
            print(f"   => INVERSION REQUIRED: 1.0 - pred")
        else:
            print(f"   ⚠ Tumor ≈ Background (possible issue)")

        # Check for negative tumor values
        negative_tumor_ratio = (tumor_values < 0).sum() / tumor_values.size
        print(f"\n5. NEGATIVE VALUE CHECK:")
        print(f"   Tumor region negative ratio: {negative_tumor_ratio*100:.2f}%")
        if negative_tumor_ratio > 0.5:
            print(f"   ⚠ WARNING: >0.5% of tumor voxels are negative!")
            print(f"   => Possible inversion in data")

    # Check VAE behavior if provided
    if autoencoder_path and Path(autoencoder_path).exists():
        print(f"\n6. VAE ENCODE-DECODE TEST:")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load VAE
        from scripts.utils import define_instance
        from scripts.diff_model_setting import load_config

        # Simple autoencoder loading
        ckpt = torch.load(autoencoder_path, weights_only=False)
        if "unet_state_dict" in ckpt:
            ckpt = ckpt["unet_state_dict"]

        # Print key shapes to understand architecture
        print(f"   VAE checkpoint keys: {list(ckpt.keys())[:5]}...")

        # Test encode-decode roundtrip
        # Note: This is a simplified test - adjust based on actual VAE architecture
        print(f"   (Detailed VAE test requires full model architecture)")

    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)

    # Final recommendation
    print("\nRECOMMENDATION:")
    if tumor_mask.sum() > 0:
        if tumor_values.mean() < bg_values.mean() or negative_tumor_ratio > 0.5:
            print("✓ The `1.0 - pred_sub` inversion in infer_breast_sub.py is CORRECT")
            print("  Your data has inverted subtraction direction (Pre - Post)")
        else:
            print("✗ The `1.0 - pred_sub` inversion may be INCORRECT")
            print("  Your data has standard subtraction direction (Post - Pre)")
            print("  Consider removing line 204 from infer_breast_sub.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose subtraction map inversion")
    parser.add_argument("--sub", type=str, required=True, help="Path to subtraction map")
    parser.add_argument("--pre", type=str, help="Path to pre-contrast image (optional)")
    parser.add_argument("--mask", type=str, required=True, help="Path to tumor mask")
    parser.add_argument("--vae", type=str, help="Path to VAE checkpoint (optional)")

    args = parser.parse_args()

    diagnose_subtraction_map(
        sub_path=args.sub,
        pre_path=args.pre,
        mask_path=args.mask,
        autoencoder_path=args.vae
    )
