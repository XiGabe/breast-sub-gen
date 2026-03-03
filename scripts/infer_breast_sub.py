#!/usr/bin/env python3
"""
Inference script for breast subtraction synthesis using trained ControlNet.
Uses pre-contrast MRI as condition to generate subtraction map.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from tqdm import tqdm

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.diff_model_setting import load_config
from scripts.utils import define_instance
from scripts.sample import initialize_noise_latents
from monai.networks.schedulers import RFlowScheduler

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("breast_sub.infer")


@torch.inference_mode()
def infer_breast_sub(
    env_config_path: str,
    model_config_path: str,
    model_def_path: str,
    num_samples: int = 5,
    device: str = "cuda:0"
):
    """Run inference on validation set samples."""

    # Load configurations
    env_args = load_config(env_config_path, model_config_path, model_def_path)
    logger.info(f"Loaded environment config from {env_config_path}")

    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load models
    logger.info("Loading models...")

    # 1. Autoencoder
    autoencoder = define_instance(env_args, "autoencoder_def").to(device)
    ckpt_ae = torch.load(env_args.trained_autoencoder_path, weights_only=False)
    if "unet_state_dict" in ckpt_ae:
        ckpt_ae = ckpt_ae["unet_state_dict"]
    autoencoder.load_state_dict(ckpt_ae)
    autoencoder.eval()
    logger.info(f"  VAE: {env_args.trained_autoencoder_path}")

    # 2. Diffusion UNet (backbone, frozen)
    unet = define_instance(env_args, "diffusion_unet_def").to(device)
    ckpt_unet = torch.load(env_args.trained_diffusion_path, weights_only=False)
    unet.load_state_dict(ckpt_unet["unet_state_dict"], strict=False)
    scale_factor = ckpt_unet["scale_factor"].to(device)
    unet.eval()
    logger.info(f"  Diffusion UNet (base): {env_args.trained_diffusion_path}")
    logger.info(f"  Scale factor: {scale_factor.item():.6f}")

    # 3. ControlNet
    controlnet = define_instance(env_args, "controlnet_def").to(device)
    from monai.networks.utils import copy_model_state
    copy_model_state(controlnet, unet.state_dict())
    ckpt_controlnet = torch.load(env_args.existing_ckpt_filepath, weights_only=False)
    controlnet.load_state_dict(ckpt_controlnet["controlnet_state_dict"], strict=False)
    controlnet.eval()
    logger.info(f"  ControlNet: {env_args.existing_ckpt_filepath}")

    # 3.5. Load finetuned UNet state if available (important for Stage 3+!)
    if "unet_finetuned_state_dict" in ckpt_controlnet:
        unet_finetuned = ckpt_controlnet["unet_finetuned_state_dict"]
        # Load with strict=False to only update matching keys (finetuned layers)
        missing, unexpected = unet.load_state_dict(unet_finetuned, strict=False)
        logger.info(f"  Loaded finetuned UNet state: {len(unet_finetuned)} parameters")
        if missing:
            logger.info(f"    Missing keys: {len(missing)}")
        if unexpected:
            logger.info(f"    Unexpected keys: {len(unexpected)}")
        finetune_stage = ckpt_controlnet.get("finetune_stage", "unknown")
        finetune_layers = ckpt_controlnet.get("finetune_unfreeze_layers", [])
        logger.info(f"  Finetune Stage: {finetune_stage}, Unfrozen: {finetune_layers}")

    # 4. Scheduler
    noise_scheduler = define_instance(env_args, "noise_scheduler")

    # Load validation data
    with open(env_args.json_data_list, 'r') as f:
        dataset = json.load(f)
    val_samples = dataset['validation'][:num_samples]
    logger.info(f"Running inference on {len(val_samples)} validation samples...")

    # Create output directory
    os.makedirs(env_args.output_dir, exist_ok=True)

    # Get inference parameters
    num_inference_steps = env_args.controlnet_infer.get("num_inference_steps", 20)
    output_size = tuple(env_args.controlnet_infer.get("output_size", [256, 256, 256]))
    latent_shape = (4, output_size[0]//4, output_size[1]//4, output_size[2]//4)

    include_modality = unet.num_class_embeds is not None
    modality = env_args.controlnet_infer.get("modality", 9)

    # Process each sample
    for idx, sample in enumerate(tqdm(val_samples, desc="Inference")):
        sample_id = sample['pre'].split('/')[-1].replace('_pre_aligned.nii.gz', '')

        # Load pre image (conditioning)
        # Paths in JSON are already relative to project root
        pre_path = sample['pre'] if os.path.exists(sample['pre']) else os.path.join(env_args.data_base_dir, sample['pre'])
        pre_nii = nib.load(pre_path)
        pre_data = pre_nii.get_fdata()

        # Convert to tensor and normalize to [-1, 1] for VAE
        # Note: pre data is [0, 1] range from preprocessing
        pre_tensor = torch.from_numpy(pre_data).float().unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        pre_tensor = pre_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
        pre_tensor = pre_tensor.to(device)

        # Load ground truth subtraction (for comparison)
        # Paths in JSON are already relative to project root
        gt_sub_path = sample['image'] if os.path.exists(sample['image']) else os.path.join(env_args.embedding_base_dir, sample['image'])
        gt_sub_nii = nib.load(gt_sub_path)
        gt_sub_data = gt_sub_nii.get_fdata()

        # Prepare spacing tensor
        spacing = sample['spacing']  # [Z, Y, X]
        spacing_tensor = torch.tensor(spacing).unsqueeze(0).float().to(device) * 100  # Scale factor

        # Prepare modality tensor
        modality_tensor = torch.tensor([modality], dtype=torch.long).to(device)

        # Initialize noise
        latents = initialize_noise_latents(latent_shape, device)

        # Set timesteps
        if isinstance(noise_scheduler, RFlowScheduler):
            noise_scheduler.set_timesteps(
                num_inference_steps=num_inference_steps,
                input_img_size_numel=torch.prod(torch.tensor(latent_shape[-3:])),
            )
        else:
            noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        all_timesteps = noise_scheduler.timesteps
        all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype, device=all_timesteps.device)))

        # Diffusion sampling loop with autocast for mixed precision
        for t, next_t in tqdm(zip(all_timesteps, all_next_timesteps), total=len(all_timesteps), leave=False, desc=f"Sample {idx+1}"):
            # Extract scalar values and create properly typed tensors
            t_val = t.item() if hasattr(t, 'item') else float(t)
            t_tensor = torch.tensor([t_val], dtype=torch.float32, device=device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                # ControlNet forward (pre image as condition)
                controlnet_inputs = {
                    "x": latents,
                    "timesteps": t_tensor,
                    "controlnet_cond": pre_tensor,  # Use pre image as condition!
                }
                if include_modality:
                    controlnet_inputs["class_labels"] = modality_tensor

                down_block_res_samples, mid_block_res_sample = controlnet(**controlnet_inputs)

                # UNet forward
                unet_inputs = {
                    "x": latents,
                    "timesteps": t_tensor,
                    "spacing_tensor": spacing_tensor,
                    "down_block_additional_residuals": down_block_res_samples,
                    "mid_block_additional_residual": mid_block_res_sample,
                }
                if include_modality:
                    unet_inputs["class_labels"] = modality_tensor

                model_output = unet(**unet_inputs)

            # Update latents using noise_scheduler.step (handles both RF and DDPM correctly)
            latents, _ = noise_scheduler.step(model_output, t, latents, next_t)

        # Decode latents to physical space
        logger.info(f"Decoding sample {idx+1}...")

        # Check for NaN/Inf in latents before decoding
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            logger.warning(f"Latents contain NaN/Inf! Replacing with zeros...")
            latents = torch.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)

        # Use autocast for VAE decoding (model expects float16)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            # Scale and decode
            latents_scaled = latents / scale_factor
            pred_sub = autoencoder.decode(latents_scaled).squeeze()

        # Check for NaN/Inf in output (convert back to float32 first)
        pred_sub = pred_sub.float()
        if torch.isnan(pred_sub).any() or torch.isinf(pred_sub).any():
            logger.warning(f"VAE output contains NaN/Inf! Replacing with zeros...")
            pred_sub = torch.nan_to_num(pred_sub, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert to numpy and denormalize [-1,1] -> [0,1]
        pred_sub_np = pred_sub.cpu().numpy().astype(np.float32)
        pred_sub_np = (pred_sub_np + 1.0) / 2.0
        pred_sub_np = np.clip(pred_sub_np, 0, 1)

        # Invert: model predicts inverted subtraction
        # Verified: removing inversion results in negative correlation (-0.40)
        pred_sub_np = 1.0 - pred_sub_np

        # Save output
        output_path = os.path.join(env_args.output_dir, f"{sample_id}_pred_sub.nii.gz")
        pred_sub_nii = nib.Nifti1Image(pred_sub_np, affine=pre_nii.affine)
        nib.save(pred_sub_nii, output_path)

        # Also save ground truth for comparison
        gt_output_path = os.path.join(env_args.output_dir, f"{sample_id}_gt_sub.nii.gz")
        # Decode GT from latent (since GT is in latent space)
        gt_sub_tensor = torch.from_numpy(gt_sub_data).float().permute(3,0,1,2).unsqueeze(0).to(device)  # [1,4,D,H,W]
        gt_sub_tensor = gt_sub_tensor / scale_factor
        with torch.amp.autocast("cuda", dtype=torch.float16):
            gt_decoded = autoencoder.decode(gt_sub_tensor).squeeze()
        gt_decoded = gt_decoded.float()  # Convert to float32
        gt_decoded_np = gt_decoded.cpu().numpy().astype(np.float32)
        gt_decoded_np = (gt_decoded_np + 1.0) / 2.0
        gt_decoded_np = np.clip(gt_decoded_np, 0, 1)
        # No inversion needed - same as prediction
        gt_sub_nii_decoded = nib.Nifti1Image(gt_decoded_np, affine=pre_nii.affine)
        nib.save(gt_sub_nii_decoded, gt_output_path)

        logger.info(f"  Saved: {output_path}")

    logger.info(f"Inference complete! Results saved to {env_args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breast Subtraction Synthesis Inference")
    parser.add_argument("-e", "--env_config", type=str, required=True,
                        help="Path to environment config JSON")
    parser.add_argument("-c", "--model_config", type=str, required=True,
                        help="Path to model config JSON")
    parser.add_argument("-t", "--model_def", type=str, required=True,
                        help="Path to model definition JSON")
    parser.add_argument("-n", "--num_samples", type=int, default=5,
                        help="Number of samples to process")
    parser.add_argument("-d", "--device", type=str, default="cuda:0",
                        help="Device to use")

    args = parser.parse_args()

    infer_breast_sub(
        env_config_path=args.env_config,
        model_config_path=args.model_config,
        model_def_path=args.model_def,
        num_samples=args.num_samples,
        device=args.device
    )
