#!/usr/bin/env python3
"""
Create dataset_breast.json with triplet format for breast subtraction training.

The output JSON format includes sub, pre, and mask paths for each sample:
{
  "training": [
    {
      "sub": "step_4/DUKE_001_L_sub.nii.gz",
      "pre": "step_4/DUKE_001_L_pre.nii.gz",
      "mask": "step_4/DUKE_001_L_mask.nii.gz",
      "modality": "mri"
    },
    ...
  ]
}

This ensures all three files (sub, pre, mask) are processed together
with spatial consistency during offline latent caching.
"""

import argparse
import json
import os
from pathlib import Path


def create_breast_dataset(
    input_dir: str,
    output_json: str = "dataset_breast.json",
    validate_files: bool = True
) -> dict:
    """
    Create dataset_breast.json by scanning step_4/ directory for triplets.

    Args:
        input_dir: Path to step_4/ directory containing pre/sub/mask files
        output_json: Output JSON filename
        validate_files: If True, check that all three files exist for each sample

    Returns:
        Dictionary with "training" key containing list of sample dicts
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Find all unique sample IDs
    # Files are named: {UUID}_{side}_{type}.nii.gz
    # Types: sub, pre, mask
    sample_ids = set()

    for file_path in input_path.glob("*_sub.nii.gz"):
        # Extract sample ID: DUKE_001_L_sub.nii.gz -> DUKE_001_L
        # file_path.stem removes .gz only, need to also remove .nii
        name_without_gz = file_path.stem  # DUKE_001_L_sub.nii
        name_without_nii = name_without_gz.replace(".nii", "")  # DUKE_001_L_sub
        sample_id = name_without_nii.replace("_sub", "")  # DUKE_001_L
        sample_ids.add(sample_id)

    # Convert to sorted list for reproducibility
    sample_ids = sorted(list(sample_ids))

    # Build dataset with validation
    dataset = []
    missing_files = []

    for sample_id in sample_ids:
        sub_path = f"step_4/{sample_id}_sub.nii.gz"
        pre_path = f"step_4/{sample_id}_pre.nii.gz"
        mask_path = f"step_4/{sample_id}_mask.nii.gz"

        # Validate files exist (check relative paths)
        if validate_files:
            sub_exists = (input_path / f"{sample_id}_sub.nii.gz").exists()
            pre_exists = (input_path / f"{sample_id}_pre.nii.gz").exists()
            mask_exists = (input_path / f"{sample_id}_mask.nii.gz").exists()

            if not (sub_exists and pre_exists and mask_exists):
                missing = []
                if not sub_exists:
                    missing.append("sub")
                if not pre_exists:
                    missing.append("pre")
                if not mask_exists:
                    missing.append("mask")
                missing_files.append(f"{sample_id}: missing {', '.join(missing)}")
                continue

        dataset.append({
            "sub": sub_path,
            "pre": pre_path,
            "mask": mask_path,
            "modality": "mri"
        })

    # Report missing files
    if missing_files:
        print(f"Warning: {len(missing_files)} samples with missing files:")
        for mf in missing_files[:10]:  # Show first 10
            print(f"  - {mf}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")

    # Prepare output dictionary
    output_data = {
        "training": dataset,
        "description": "Breast subtraction dataset with sub/pre/mask triplets"
    }

    # Write to file
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDataset created: {output_json}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Input directory: {input_dir}")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Create dataset_breast.json for triplet-based training"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="step_4",
        help="Input directory containing pre/sub/mask files (default: step_4/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset_breast.json",
        help="Output JSON filename (default: dataset_breast.json)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip file existence validation"
    )

    args = parser.parse_args()

    create_breast_dataset(
        input_dir=args.input,
        output_json=args.output,
        validate_files=not args.no_validate
    )


if __name__ == "__main__":
    main()
