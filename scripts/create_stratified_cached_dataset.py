#!/usr/bin/env python3
"""
Create stratified train/validation split for cached breast dataset.

This script performs patient-level stratified sampling to ensure:
1. No patient appears in both train and val (prevents data leakage)
2. Dataset proportions are maintained (DUKE/ISPY1/ISPY2/NACT)
3. Tumor proportions are maintained (has_tumor=0/1)

Usage:
    python -m scripts.create_stratified_cached_dataset \
        --metadata-csv step_4/step_4_metadata.csv \
        --input dataset_breast.json \
        --output dataset_breast_cached.json \
        --embedding-dir ./embeddings_breast_sub \
        --pre-dir ./processed_pre \
        --mask-dir ./processed_mask \
        --val-ratio 0.2
"""

import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def extract_dataset_from_uuid(uuid: str) -> str:
    """Extract dataset name from UUID prefix."""
    # UUID format: DATASET_XXX_L or DATASET_XXX_R
    # e.g., "DUKE_001_L" -> "DUKE", "ISPY1_1234_R" -> "ISPY1"
    parts = uuid.split('_')
    if len(parts) >= 2:
        dataset = parts[0]
        # Handle ISPY1 vs ISPY2
        if dataset.startswith("ISPY"):
            if len(parts) > 1 and parts[1].isdigit():
                # Check if it's ISPY1 or ISPY2
                if parts[1] == "1":
                    return "ISPY1"
                elif parts[1] == "2":
                    return "ISPY2"
        return dataset
    return "UNKNOWN"


def create_stratified_cached_dataset(
    metadata_csv: str,
    input_json: str,
    output_json: str,
    embedding_dir: str,
    pre_dir: str,
    mask_dir: str,
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[dict, pd.DataFrame]:
    """
    Create stratified cached dataset with train/val split.

    Args:
        metadata_csv: Path to step_4_metadata.csv
        input_json: Path to original dataset_breast.json
        output_json: Path for output dataset_breast_cached.json
        embedding_dir: Directory containing cached embeddings (sub_emb)
        pre_dir: Directory containing cached pre images
        mask_dir: Directory containing cached masks
        val_ratio: Fraction of data for validation (default 0.2)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (cached_dataset_dict, patient_df)
    """

    # 1. Read metadata CSV
    print(f"Reading metadata from {metadata_csv}...")
    metadata_df = pd.read_csv(metadata_csv)
    print(f"  Loaded {len(metadata_df)} samples")

    # 2. Extract dataset from UUID
    metadata_df['dataset'] = metadata_df['UUID'].apply(extract_dataset_from_uuid)

    # 3. Read original dataset JSON to get sample list
    print(f"Reading original dataset from {input_json}...")
    with open(input_json, 'r') as f:
        original_dataset = json.load(f)

    # Get list of samples (should all be in "training" key)
    samples = original_dataset.get('training', [])
    print(f"  Found {len(samples)} samples in original dataset")

    # 4. Create sample to UUID mapping
    # Original format: "step_4/DUKE_001_L_sub.nii.gz" -> "DUKE_001_L"
    sample_to_uuid = {}
    for sample in samples:
        sub_path = sample.get('sub', '')
        # Extract UUID from path
        # e.g., "step_4/DUKE_001_L_sub.nii.gz" -> "DUKE_001_L"
        filename = Path(sub_path).name
        uuid = filename.replace('_sub.nii.gz', '')
        sample_to_uuid[uuid] = sample

    # 5. Build patient-level dataframe
    print("\nBuilding patient-level dataframe...")
    patient_data = []

    for uuid in metadata_df['UUID'].unique():
        # Get metadata for this sample
        sample_meta = metadata_df[metadata_df['UUID'] == uuid].iloc[0]

        original_id = sample_meta['Original_ID']
        dataset = sample_meta['dataset']
        has_tumor = sample_meta['Has_Tumor']

        # Check if this sample has cached files
        sub_emb_path = Path(embedding_dir) / f"{uuid}_sub_emb.nii.gz"
        pre_aligned_path = Path(pre_dir) / f"{uuid}_pre_aligned.nii.gz"
        mask_aligned_path = Path(mask_dir) / f"{uuid}_mask_aligned.nii.gz"

        if not (sub_emb_path.exists() and pre_aligned_path.exists() and mask_aligned_path.exists()):
            print(f"  Warning: Missing cached files for {uuid}, skipping")
            continue

        patient_data.append({
            'uuid': uuid,
            'original_id': original_id,
            'dataset': dataset,
            'has_tumor': has_tumor,
            'stratum_key': f"{dataset}_{has_tumor}"
        })

    patient_df = pd.DataFrame(patient_data)
    print(f"  Found {len(patient_df)} samples with complete cached data")

    # 6. Group by patient (Original_ID)
    print("\nGrouping samples by patient...")
    patient_groups = patient_df.groupby('original_id').agg({
        'uuid': list,
        'dataset': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],  # Primary dataset
        'has_tumor': 'sum',  # Total tumors across samples
        'stratum_key': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]  # Primary stratum
    }).reset_index()

    patient_groups.columns = ['original_id', 'uuids', 'dataset', 'total_tumors', 'stratum_key']
    patient_groups['num_samples'] = patient_groups['uuids'].apply(len)

    print(f"  Found {len(patient_groups)} unique patients")
    print(f"  Samples per patient: min={patient_groups['num_samples'].min()}, "
          f"max={patient_groups['num_samples'].max()}, "
          f"mean={patient_groups['num_samples'].mean():.2f}")

    # 7. Stratified split at patient level
    print(f"\nPerforming stratified split (val_ratio={val_ratio})...")
    train_patients_df, val_patients_df = train_test_split(
        patient_groups,
        test_size=val_ratio,
        stratify=patient_groups['stratum_key'],
        random_state=random_seed
    )

    print(f"  Train patients: {len(train_patients_df)}")
    print(f"  Val patients: {len(val_patients_df)}")

    # 8. Assign samples to train/val based on patient split
    train_uuids = set()
    for uuids in train_patients_df['uuids']:
        train_uuids.update(uuids)

    val_uuids = set()
    for uuids in val_patients_df['uuids']:
        val_uuids.update(uuids)

    # 9. Build cached dataset JSON
    print("\nBuilding cached dataset JSON...")
    cached_dataset = {
        "training": [],
        "validation": []
    }

    for _, row in patient_df.iterrows():
        uuid = row['uuid']

        # Build cached paths
        sample_entry = {
            "image": str(Path(embedding_dir) / f"{uuid}_sub_emb.nii.gz"),
            "pre": str(Path(pre_dir) / f"{uuid}_pre_aligned.nii.gz"),
            "label": str(Path(mask_dir) / f"{uuid}_mask_aligned.nii.gz"),
            "spacing": [1.2, 0.7, 0.7],  # Z, Y, X spacing from preprocessing
            "modality": "mri"
        }

        if uuid in train_uuids:
            cached_dataset["training"].append(sample_entry)
        elif uuid in val_uuids:
            cached_dataset["validation"].append(sample_entry)
        else:
            print(f"  Warning: {uuid} not in train or val split")

    print(f"  Training samples: {len(cached_dataset['training'])}")
    print(f"  Validation samples: {len(cached_dataset['validation'])}")

    # 10. Save to file
    print(f"\nSaving to {output_json}...")
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(cached_dataset, f, indent=2)

    return cached_dataset, patient_df


def print_split_statistics(metadata_csv: str, cached_dataset: dict, patient_df: pd.DataFrame):
    """Print detailed statistics about the train/val split."""

    # Get metadata for additional info
    metadata_df = pd.read_csv(metadata_csv)
    metadata_df['dataset'] = metadata_df['UUID'].apply(extract_dataset_from_uuid)

    # Build UUID -> split mapping
    uuid_to_split = {}
    for entry in cached_dataset['training']:
        # Extract UUID from path
        path = Path(entry['image'])
        uuid = path.name.replace('_sub_emb.nii.gz', '')
        uuid_to_split[uuid] = 'train'

    for entry in cached_dataset['validation']:
        path = Path(entry['image'])
        uuid = path.name.replace('_sub_emb.nii.gz', '')
        uuid_to_split[uuid] = 'val'

    # Merge split info with metadata
    metadata_df['split'] = metadata_df['UUID'].map(uuid_to_split)
    metadata_df = metadata_df[metadata_df['split'].notna()]

    print("\n" + "=" * 80)
    print("Stratified Split Results:")
    print("=" * 80)
    print(f"{'':<20} {'Train':>12} {'Val':>12} {'Total':>12} {'Diff':>12}")
    print("-" * 80)

    # Overall statistics
    total_train = len(metadata_df[metadata_df['split'] == 'train'])
    total_val = len(metadata_df[metadata_df['split'] == 'val'])
    total = total_train + total_val
    diff = abs(total_train - total_val * (total_train / total_val if total_val > 0 else 1))

    print(f"{'Total Samples:':<20} {total_train:>12} {total_val:>12} {total:>12} {0:>11.1f}%")

    # Patient statistics
    train_patients = metadata_df[metadata_df['split'] == 'train']['Original_ID'].nunique()
    val_patients = metadata_df[metadata_df['split'] == 'val']['Original_ID'].nunique()
    total_patients = train_patients + val_patients

    print(f"{'Total Patients:':<20} {train_patients:>12} {val_patients:>12} {total_patients:>12} {0:>11.1f}%")
    print()

    # Dataset distribution
    print("Dataset Distribution:")
    for dataset in ['DUKE', 'ISPY1', 'ISPY2', 'NACT']:
        ds_data = metadata_df[metadata_df['dataset'] == dataset]
        ds_train = len(ds_data[ds_data['split'] == 'train'])
        ds_val = len(ds_data[ds_data['split'] == 'val'])
        ds_total = ds_train + ds_val
        ds_train_pct = 100 * ds_train / total if total > 0 else 0
        ds_val_pct = 100 * ds_val / total if total > 0 else 0
        ds_diff = abs(ds_train_pct - ds_val_pct)

        print(f"  {dataset + ':':<18} {ds_train:>5} ({ds_train_pct:>4.1f}%) {ds_val:>5} ({ds_val_pct:>4.1f}%) "
              f"{ds_total:>5} ({100*ds_total/total:>4.1f}%) {ds_diff:>9.1f}%")
    print()

    # Tumor distribution
    print("Tumor Distribution:")
    for has_tumor in [1, 0]:
        tumor_label = "Has Tumor" if has_tumor == 1 else "No Tumor"
        tumor_data = metadata_df[metadata_df['Has_Tumor'] == has_tumor]
        tumor_train = len(tumor_data[tumor_data['split'] == 'train'])
        tumor_val = len(tumor_data[tumor_data['split'] == 'val'])
        tumor_total = tumor_train + tumor_val
        tumor_train_pct = 100 * tumor_train / total if total > 0 else 0
        tumor_val_pct = 100 * tumor_val / total if total > 0 else 0
        tumor_diff = abs(tumor_train_pct - tumor_val_pct)

        print(f"  {tumor_label + ':':<18} {tumor_train:>5} ({tumor_train_pct:>4.1f}%) {tumor_val:>5} ({tumor_val_pct:>4.1f}%) "
              f"{tumor_total:>5} ({100*tumor_total/total:>4.1f}%) {tumor_diff:>9.1f}%")
    print()

    # Check for patient overlap
    train_patient_ids = set(metadata_df[metadata_df['split'] == 'train']['Original_ID'])
    val_patient_ids = set(metadata_df[metadata_df['split'] == 'val']['Original_ID'])
    overlap = train_patient_ids & val_patient_ids

    print("Patient Overlap Check: ", end="")
    if len(overlap) == 0:
        print("✅ No overlap")
    else:
        print(f"❌ OVERLAP DETECTED: {len(overlap)} patients in both splits")
        print(f"  Overlapping patients: {sorted(list(overlap))[:10]}...")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified train/validation split for cached breast dataset"
    )
    parser.add_argument(
        "--metadata-csv",
        default="step_4/step_4_metadata.csv",
        help="Path to step_4_metadata.csv"
    )
    parser.add_argument(
        "--input",
        default="dataset_breast.json",
        help="Path to original dataset_breast.json"
    )
    parser.add_argument(
        "--output",
        default="dataset_breast_cached.json",
        help="Path for output dataset_breast_cached.json"
    )
    parser.add_argument(
        "--embedding-dir",
        default="./embeddings_breast_sub",
        help="Directory containing cached embeddings (sub_emb)"
    )
    parser.add_argument(
        "--pre-dir",
        default="./processed_pre",
        help="Directory containing cached pre images"
    )
    parser.add_argument(
        "--mask-dir",
        default="./processed_mask",
        help="Directory containing cached masks"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default 0.2)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Create stratified cached dataset
    cached_dataset, patient_df = create_stratified_cached_dataset(
        metadata_csv=args.metadata_csv,
        input_json=args.input,
        output_json=args.output,
        embedding_dir=args.embedding_dir,
        pre_dir=args.pre_dir,
        mask_dir=args.mask_dir,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed
    )

    # Print statistics
    print_split_statistics(
        metadata_csv=args.metadata_csv,
        cached_dataset=cached_dataset,
        patient_df=patient_df
    )

    print(f"\n✅ Stratified cached dataset created successfully!")
    print(f"   Output: {args.output}")


if __name__ == "__main__":
    main()
