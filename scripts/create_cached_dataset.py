#!/usr/bin/env python3
"""
Generate cached dataset JSON after triplet latent caching is complete.

Reads dataset_breast.json (pointing to step_4/ files) and generates
dataset_breast_cached.json (pointing to cached files).

Output format:
{
  "training": [
    {
      "image": "embeddings_breast_sub/DUKE_001_L_sub_emb.nii.gz",
      "pre": "processed_pre/DUKE_001_L_pre_aligned.nii.gz",
      "label": "processed_mask/DUKE_001_L_mask_aligned.nii.gz",
      "spacing": [1.2, 0.7, 0.7],
      "modality": "mri"
    },
    ...
  ]
}
"""

import argparse
import json
import os
from pathlib import Path


def create_cached_dataset(
    input_json: str = "dataset_breast.json",
    output_json: str = "dataset_breast_cached.json",
    embedding_dir: str = "./embeddings_breast_sub",
    pre_dir: str = "./processed_pre",
    mask_dir: str = "./processed_mask",
    spacing: list = [1.2, 0.7, 0.7],
    validate_files: bool = True
) -> dict:
    """
    Create cached dataset JSON pointing to cached triplet files.

    Args:
        input_json: Input JSON file with step_4/ paths
        output_json: Output JSON file with cached paths
        embedding_dir: Directory containing sub_emb files
        pre_dir: Directory containing pre_aligned files
        mask_dir: Directory containing mask_aligned files
        spacing: Spacing values [x, y, z] for the data
        validate_files: If True, check that cached files exist

    Returns:
        Dictionary with "training" key containing list of sample dicts
    """
    # Load original dataset
    with open(input_json, "r") as f:
        input_data = json.load(f)

    training_data = input_data["training"]
    cached_data = []
    missing_count = 0

    for sample in training_data:
        # Extract base ID from filename
        # step_4/DUKE_001_L_sub.nii.gz -> DUKE_001_L
        sub_path = sample["sub"]
        base_id = os.path.basename(sub_path).replace("_sub.nii.gz", "")

        # Build paths to cached files
        sub_emb_path = os.path.join(embedding_dir, f"{base_id}_sub_emb.nii.gz")
        pre_aligned_path = os.path.join(pre_dir, f"{base_id}_pre_aligned.nii.gz")
        mask_aligned_path = os.path.join(mask_dir, f"{base_id}_mask_aligned.nii.gz")

        # Validate files exist
        if validate_files:
            sub_exists = os.path.isfile(sub_emb_path)
            pre_exists = os.path.isfile(pre_aligned_path)
            mask_exists = os.path.isfile(mask_aligned_path)

            if not (sub_exists and pre_exists and mask_exists):
                missing_count += 1
                if missing_count <= 5:  # Show first 5
                    missing = []
                    if not sub_exists:
                        missing.append("sub_emb")
                    if not pre_exists:
                        missing.append("pre_aligned")
                    if not mask_exists:
                        missing.append("mask_aligned")
                    print(f"Warning: {base_id} missing: {', '.join(missing)}")
                continue

        # Create cached sample entry
        cached_sample = {
            "image": sub_emb_path,
            "pre": pre_aligned_path,
            "label": mask_aligned_path,
            "spacing": spacing,
            "modality": sample.get("modality", "mri")
        }
        cached_data.append(cached_sample)

    if missing_count > 5:
        print(f"... and {missing_count - 5} more samples with missing files")

    # Prepare output dictionary
    output_data = {
        "training": cached_data,
        "description": "Cached breast subtraction dataset with sub_emb/pre_aligned/mask_aligned"
    }

    # Write to file
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nCached dataset created: {output_json}")
    print(f"  Total samples: {len(cached_data)}")
    print(f"  Missing samples skipped: {missing_count}")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Create cached dataset JSON after triplet caching"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset_breast.json",
        help="Input JSON file with step_4 paths (default: dataset_breast.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset_breast_cached.json",
        help="Output JSON file with cached paths (default: dataset_breast_cached.json)"
    )
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default="./embeddings_breast_sub",
        help="Directory containing sub_emb files"
    )
    parser.add_argument(
        "--pre-dir",
        type=str,
        default="./processed_pre",
        help="Directory containing pre_aligned files"
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="./processed_mask",
        help="Directory containing mask_aligned files"
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=[1.2, 0.7, 0.7],
        help="Spacing values [x, y, z] (default: 1.2 0.7 0.7)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip file existence validation"
    )

    args = parser.parse_args()

    create_cached_dataset(
        input_json=args.input,
        output_json=args.output,
        embedding_dir=args.embedding_dir,
        pre_dir=args.pre_dir,
        mask_dir=args.mask_dir,
        spacing=args.spacing,
        validate_files=not args.no_validate
    )


if __name__ == "__main__":
    main()
