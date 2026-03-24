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
Offline latent caching script for triplet-based breast subtraction training.

Processes sub/pre/mask triplets with spatial alignment:
- All samples normalized to FIXED 256³
- Pad dimensions < 256
- Crop dimensions > 256 using MASK-ANCHORED center cropping (preserves tumor location)
- Sub: VAE-encoded to latent space (64³, 4 channels) - diffusion target
- Pre: Saved at 256³ (1 channel) - controlnet condition
- Mask: Saved at 256³ (1 channel) - loss weighting

All three files undergo the SAME spatial transforms for alignment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import monai
import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai.transforms import Compose
from monai.utils import set_determinism
from monai.inferers.inferer import SlidingWindowInferer

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .transforms import define_fixed_intensity_transform, SUPPORT_MODALITIES
from .utils import define_instance, dynamic_infer


def create_triplet_transforms(modality: str = 'unknown', skip_intensity: bool = False):
    """Create base MONAI transforms for reading triplet data."""
    if 'mri' in modality:
        modality = 'mri'
    if 'ct' in modality:
        modality = 'ct'

    if skip_intensity:
        intensity_transforms = []
    elif modality in SUPPORT_MODALITIES:
        intensity_transforms = define_fixed_intensity_transform(modality=modality)
    else:
        intensity_transforms = []

    base_transform = Compose(
        [monai.transforms.LoadImaged(keys="image"),
         monai.transforms.EnsureChannelFirstd(keys="image"),
         monai.transforms.Orientationd(keys="image", axcodes="RAS"),
         monai.transforms.EnsureTyped(keys="image", dtype=torch.float32)]
        + intensity_transforms
    )

    return base_transform


def normalize_to_fixed_size(
    data: np.ndarray,
    mask: np.ndarray,
    target_size: tuple = (256, 256, 256),
    mode: str = 'constant'
) -> tuple[np.ndarray, dict]:
    """
    Normalize spatial dimensions to EXACTLY target_size.

    Strategy:
    1. Pad dimensions < target_size to reach target
    2. Crop dimensions > target_size using MASK-ANCHORED center (tumor preservation)

    Args:
        data: Input data array [C, D, H, W]
        mask: Mask array [C, D, H, W] with tumor labels
        target_size: Target spatial size (D, H, W)
        mode: Padding mode ('constant', 'edge', etc.)

    Returns:
        tuple: (Normalized data array with shape [C, target_size], transform_params dict)
              transform_params contains: original_dim, pad (if any), crop (if any), target_size
    """
    transform_params = {
        "original_dim": list(data.shape[1:]),  # [D, H, W]
        "pad": None,
        "crop": None,
        "target_size": list(target_size)
    }
    c, d, h, w = data.shape
    target_d, target_h, target_w = target_size

    # Step 1: Pad dimensions that are smaller than target
    pad_d = max(0, target_d - d)
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    data_tensor = torch.from_numpy(data).float()

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        pad_front = pad_d // 2
        pad_back = pad_d - pad_front
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        transform_params["pad"] = {
            "front": pad_front, "back": pad_back,
            "top": pad_top, "bottom": pad_bottom,
            "left": pad_left, "right": pad_right
        }

        data_tensor = F.pad(
            data_tensor,
            (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back),
            mode=mode
        )

    # Step 2: Crop dimensions that are larger than target (mask-anchored)
    _, curr_d, curr_h, curr_w = data_tensor.shape

    crop_d = max(0, curr_d - target_d)
    crop_h = max(0, curr_h - target_h)
    crop_w = max(0, curr_w - target_w)

    if crop_d > 0 or crop_h > 0 or crop_w > 0:
        # Find mask center (tumor location) for anchored cropping
        mask_squeezed = mask.squeeze()  # [D, H, W]
        tumor_indices = np.argwhere(mask_squeezed > 0)

        if len(tumor_indices) == 0:
            # No tumor found, use geometric center
            center_d, center_h, center_w = curr_d // 2, curr_h // 2, curr_w // 2
        else:
            # Use tumor centroid
            center_d = int(np.mean(tumor_indices[:, 0]))
            center_h = int(np.mean(tumor_indices[:, 1]))
            center_w = int(np.mean(tumor_indices[:, 2]))

        # Calculate crop boundaries centered on tumor
        start_d = max(0, min(center_d - target_d // 2, curr_d - target_d))
        end_d = min(curr_d, start_d + target_d)
        if end_d - start_d < target_d:
            if start_d == 0:
                end_d = min(curr_d, target_d)
            else:
                start_d = max(0, end_d - target_d)

        start_h = max(0, min(center_h - target_h // 2, curr_h - target_h))
        end_h = min(curr_h, start_h + target_h)
        if end_h - start_h < target_h:
            if start_h == 0:
                end_h = min(curr_h, target_h)
            else:
                start_h = max(0, end_h - target_h)

        start_w = max(0, min(center_w - target_w // 2, curr_w - target_w))
        end_w = min(curr_w, start_w + target_w)
        if end_w - start_w < target_w:
            if start_w == 0:
                end_w = min(curr_w, target_w)
            else:
                start_w = max(0, end_w - target_w)

        data_tensor = data_tensor[
            :,
            start_d:end_d,
            start_h:end_h,
            start_w:end_w
        ]

        # Record crop parameters for potential reverse transformation
        transform_params["crop"] = {
            "start_d": start_d, "end_d": end_d,
            "start_h": start_h, "end_h": end_h,
            "start_w": start_w, "end_w": end_w,
            "original_size": [curr_d, curr_h, curr_w]
        }

    return data_tensor.numpy(), transform_params


def process_triplet(
    sub_path: str,
    pre_path: str,
    mask_path: str,
    args: argparse.Namespace,
    autoencoder: torch.nn.Module,
    device: torch.device,
    base_transform: Compose,
    logger: logging.Logger
) -> None:
    """
    Process a single triplet (sub, pre, mask) to create training data.

    Strategy (Plan B - Fixed 256³ with mask-anchored cropping):
    1. Load all three files with base transform (orientation only)
    2. Normalize ALL samples to EXACTLY 256³:
       - Pad dimensions < 256
       - Crop dimensions > 256 using mask-anchored center (preserves tumor)
    3. All three files get SAME spatial transforms for alignment
    4. Only sub is VAE-encoded; pre and mask are saved as aligned files

    Args:
        sub_path (str): Path to subtraction map file.
        pre_path (str): Path to pre-contrast file.
        mask_path (str): Path to mask file.
        args (argparse.Namespace): Configuration arguments.
        autoencoder (torch.nn.Module): Autoencoder model.
        device (torch.device): Device to process on.
        base_transform (Compose): Base transforms for loading data.
        logger (logging.Logger): Logger for logging information.
    """
    base_id = os.path.basename(sub_path).replace("_sub.nii.gz", "")

    # Build output paths
    out_sub_emb = os.path.join(args.embedding_base_dir, f"{base_id}_sub_emb.nii.gz")
    out_pre_aligned = os.path.join(args.pre_output_dir, f"{base_id}_pre_aligned.nii.gz")
    out_mask_aligned = os.path.join(args.mask_output_dir, f"{base_id}_mask_aligned.nii.gz")
    out_metadata = os.path.join(args.embedding_base_dir, f"{base_id}_metadata.json")

    # Skip if all outputs (including metadata) already exist
    if os.path.isfile(out_sub_emb) and os.path.isfile(out_pre_aligned) and os.path.isfile(out_mask_aligned) and os.path.isfile(out_metadata):
        logger.info(f"Skipping {base_id}: outputs already exist")
        return

    # Build full paths
    def resolve_path(path, base_dir):
        if os.path.isabs(path):
            return path
        base_dir_normalized = base_dir.lstrip("./")
        if path.startswith(base_dir_normalized + os.sep) or path.startswith(base_dir_normalized):
            return path
        return os.path.join(base_dir, path)

    full_sub_path = resolve_path(sub_path, args.data_base_dir)
    full_pre_path = resolve_path(pre_path, args.data_base_dir)
    full_mask_path = resolve_path(mask_path, args.data_base_dir)

    try:
        # Step 1: Load all three files with base transform (orientation only)
        sub_data_base = base_transform({"image": full_sub_path})["image"]
        pre_data_base = base_transform({"image": full_pre_path})["image"]
        mask_data_base = base_transform({"image": full_mask_path})["image"]

        # Get original dimensions and spacing
        orig_dim = [int(sub_data_base.meta["dim"][_i]) for _i in range(1, 4)]
        spacing = [float(sub_data_base.meta["pixdim"][_i]) for _i in range(1, 4)]
        affine = sub_data_base.meta["affine"].numpy()

        logger.info(f"{base_id}: orig dim: {orig_dim}")

        # Step 2: Convert to numpy arrays
        sub_numpy = sub_data_base.numpy()
        pre_numpy = pre_data_base.numpy()
        mask_numpy = mask_data_base.numpy()

        # Step 3: Normalize all to EXACTLY 256³ (pad + mask-anchored crop)
        sub_final, sub_params = normalize_to_fixed_size(
            sub_numpy, mask_numpy, target_size=(256, 256, 256), mode='constant'
        )
        pre_final, pre_params = normalize_to_fixed_size(
            pre_numpy, mask_numpy, target_size=(256, 256, 256), mode='constant'
        )
        mask_final, mask_params = normalize_to_fixed_size(
            mask_numpy, mask_numpy, target_size=(256, 256, 256), mode='constant'
        )

        logger.info(f"{base_id}: final dim: (256, 256, 256)")

        # Ensure output directories exist
        Path(out_sub_emb).parent.mkdir(parents=True, exist_ok=True)
        Path(out_pre_aligned).parent.mkdir(parents=True, exist_ok=True)
        Path(out_mask_aligned).parent.mkdir(parents=True, exist_ok=True)

        # === Process Sub: VAE encoding ===
        with torch.amp.autocast("cuda"):
            pt_sub = torch.from_numpy(sub_final).float().to(device).unsqueeze(0)
            # Keep sub in [0, 1] range (MAISI default)

            inferer = SlidingWindowInferer(
                roi_size=[320, 320, 160],
                sw_batch_size=1,
                progress=True,
                mode="gaussian",
                overlap=0.4,
                sw_device=device,
                device=device,
            )
            z = dynamic_infer(inferer, autoencoder.encode_stage_2_inputs, pt_sub)
            logger.info(f"{base_id}: z latent shape: {z.size()}")

        # Save Sub_emb (64³, 4 channels)
        out_nda = z.squeeze().cpu().detach().numpy().transpose(1, 2, 3, 0)
        out_img = nib.Nifti1Image(np.float32(out_nda), affine=affine)
        nib.save(out_img, out_sub_emb)
        logger.info(f"{base_id}: Saved Sub_emb to {out_sub_emb}")

        # === Save Pre_aligned (256³, 1 channel) ===
        pre_nda = pre_final.squeeze()
        pre_img = nib.Nifti1Image(np.float32(pre_nda), affine=affine)
        nib.save(pre_img, out_pre_aligned)
        logger.info(f"{base_id}: Saved Pre_aligned to {out_pre_aligned}")

        # === Save Mask_aligned (256³, 1 channel) ===
        mask_nda = mask_final.squeeze()
        mask_img = nib.Nifti1Image(np.float32(mask_nda), affine=affine)
        nib.save(mask_img, out_mask_aligned)
        logger.info(f"{base_id}: Saved Mask_aligned to {out_mask_aligned}")

        # === Save Metadata (for potential reverse lookup) ===
        metadata = {
            "base_id": base_id,
            "original_dim": orig_dim,
            "target_size": [256, 256, 256],
            "sub_range": "[0, 1]",
            "pre_range": "[0, 1]",
            "mask_range": "[0, 1]",
            "spacing": spacing,
            "normalize_method": "pad_and_mask_anchor_crop",
            "vae_encoding": True,
            "latent_shape": [64, 64, 64, 4],
            "input_files": {
                "sub": full_sub_path,
                "pre": full_pre_path,
                "mask": full_mask_path
            },
            "output_files": {
                "sub_emb": out_sub_emb,
                "pre_aligned": out_pre_aligned,
                "mask_aligned": out_mask_aligned
            }
        }
        with open(out_metadata, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"{base_id}: Saved Metadata to {out_metadata}")

    except Exception as e:
        logger.error(f"Error processing {base_id}: {e}")
        import traceback
        traceback.print_exc()


@torch.inference_mode()
def diff_model_create_training_data_triplet(
    env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int
) -> None:
    """Create training data for triplet-based diffusion model training."""
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus=num_gpus)
    logger = setup_logging("creating triplet training data")
    logger.info(f"Using device {device}")

    # Instantiate autoencoder
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=False)
    if "unet_state_dict" in checkpoint_autoencoder.keys():
        checkpoint_autoencoder = checkpoint_autoencoder["unet_state_dict"]
    autoencoder.load_state_dict(checkpoint_autoencoder)
    logger.info("Loaded autoencoder checkpoint")

    # Ensure output directories exist
    Path(args.embedding_base_dir).mkdir(parents=True, exist_ok=True)
    if hasattr(args, 'pre_output_dir'):
        Path(args.pre_output_dir).mkdir(parents=True, exist_ok=True)
    else:
        args.pre_output_dir = "./processed_pre"
        Path(args.pre_output_dir).mkdir(parents=True, exist_ok=True)
    if hasattr(args, 'mask_output_dir'):
        Path(args.mask_output_dir).mkdir(parents=True, exist_ok=True)
    else:
        args.mask_output_dir = "./processed_mask"
        Path(args.mask_output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset JSON
    with open(args.json_data_list, "r") as file:
        json_data = json.load(file)
    files_raw = json_data["training"]
    logger.info(f"Total samples in dataset: {len(files_raw)}")

    # Set default modality if not specified
    modality = getattr(args, 'modality', "mri")

    # Static work partitioning over files
    for _iter in range(len(files_raw)):
        if _iter % world_size != local_rank:
            continue

        sample = files_raw[_iter]
        sub_path = sample["sub"]
        pre_path = sample["pre"]
        mask_path = sample["mask"]
        sample_modality = sample.get("modality", modality)

        logger.info(f"Processing sample {_iter + 1}/{len(files_raw)}: {sub_path}")

        # Create base transform
        base_transform = create_triplet_transforms(modality=sample_modality, skip_intensity=True)

        # Process the triplet with fixed 256³ strategy
        process_triplet(
            sub_path, pre_path, mask_path,
            args, autoencoder, device,
            base_transform,
            logger
        )

    if dist.is_initialized():
        dist.destroy_process_group()

    logger.info("Triplet training data creation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triplet Diffusion Model Training Data Creation - Plan B")
    parser.add_argument("-e", "--env_config", type=str, default="./configs/environment_breast_sub.json")
    parser.add_argument("-c", "--model_config", type=str, default="./configs/config_breast_sub_train.json")
    parser.add_argument("-t", "--model_def", type=str, default="./configs/config_network_rflow.json")
    parser.add_argument("-g", "--num_gpus", type=int, default=1)

    args = parser.parse_args()
    diff_model_create_training_data_triplet(args.env_config, args.model_config, args.model_def, args.num_gpus)
