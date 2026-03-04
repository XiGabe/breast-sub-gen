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

import argparse
import json
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai.networks.utils import copy_model_state
from monai.utils import RankFilter
# Removed MONAI morphology ops - using pure PyTorch implementations for stability
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from .utils import define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp
from .diff_model_setting import initialize_distributed, load_config, setup_logging
# Removed: from .augmentation import remove_tumors (no longer needed)

def apply_random_morphological_perturbation(masks):
    """
    Apply random morphological perturbation to tumor masks for data augmentation.

    Randomly applies dilation or erosion with kernel size 1-3 voxels to simulate
    variations in tumor segmentation boundaries.

    Uses pure PyTorch implementations (F.max_pool3d) for stability and compatibility
    with gradient computation, avoiding MONAI's native morphology ops which can cause
    dimension/type errors in forward loops.

    Args:
        masks (torch.Tensor): Binary tumor masks. Shape [B, 1, X, Y, Z].

    Returns:
        torch.Tensor: Perturbed masks with same shape and device as input.
    """
    masks_perturbed = []
    for b in range(masks.shape[0]):
        mask = masks[b:b+1]  # Keep [1, 1, X, Y, Z] shape

        # Randomly choose operation: 0=dilation, 1=erosion, 2=none
        op_choice = torch.randint(0, 3, (1,)).item()

        # Random kernel size between 1-3 (use odd sizes: 1, 3)
        kernel_size = torch.randint(0, 2, (1,)).item() * 2 + 1  # 1 or 3

        if kernel_size == 1:
            # Kernel size 1: no effect, just return original
            mask_pert = mask
        elif op_choice == 0:
            # Dilation: max pooling with proper padding
            pad_size = kernel_size // 2
            # Use padding_mode='replicate' for better edge handling
            mask_padded = F.pad(mask, pad=(pad_size, pad_size, pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
            mask_pert = F.max_pool3d(mask_padded, kernel_size=kernel_size, stride=1, padding=0)
        elif op_choice == 1:
            # Erosion: min pooling via max pool on inverted values
            pad_size = kernel_size // 2
            # Pad with 0s (min value for inverted mask)
            mask_padded = F.pad(1 - mask, pad=(pad_size, pad_size, pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
            mask_pert = 1 - F.max_pool3d(mask_padded, kernel_size=kernel_size, stride=1, padding=0)
        else:
            # No operation
            mask_pert = mask

        # Ensure output shape matches input shape
        if mask_pert.shape != mask.shape:
            # Resize to match original shape
            mask_pert = F.interpolate(mask_pert, size=mask.shape[2:], mode='nearest')

        masks_perturbed.append(mask_pert)

    return torch.cat(masks_perturbed, dim=0)

def compute_model_output(
    images,labels,noise,timesteps,noise_scheduler,
    controlnet,unet,
    spacing_tensor,
    pre_images=None,
    apply_morphological_perturb=False,
    modality_tensor=None,
    top_region_index_tensor=None,
    bottom_region_index_tensor=None,
    return_controlnet_blocks=False
):
    """
    Run ControlNet + U-Net to obtain the denoising network output (and optionally
    the ControlNet intermediate blocks) for a given noisy latent and conditions.

    Pipeline:
      1) Construct dual-channel condition from pre-contrast MRI and tumor masks.
      2) Optionally apply morphological perturbation to masks (30% probability).
      3) Add noise to `images` at `timesteps` via the scheduler.
      4) Pass noisy latent and conditions to ControlNet to get down/mid features.
      5) Pass everything to U-Net (with spacing, optional modality & body-region
         tokens) to produce `model_output`.

    Args:
        images (torch.Tensor):
            Input latent/image tensor to be noised. Shape [B, C, X, Y, Z].
        labels (torch.Tensor or monai.data.MetaTensor):
            Segmentation labels (tumor masks) used to create ControlNet condition.
        noise (torch.Tensor):
            Noise tensor aligned with `images`.
        timesteps (torch.Tensor or Any):
            Diffusion timesteps for the scheduler and networks.
        noise_scheduler:
            Object exposing `add_noise(original_samples, noise, timesteps)`.
        controlnet (torch.nn.Module):
            Control network returning `(down_block_res_samples, mid_block_res_sample)`.
        unet (torch.nn.Module):
            Denoising network that accepts additional residuals from ControlNet.
        spacing_tensor (torch.Tensor):
            Per-sample spacing or resolution encoding; passed into U-Net.
        pre_images (torch.Tensor, optional):
            Pre-contrast MRI images at physical resolution [B, 1, 256, 256, 256].
            If provided, used as first channel of dual-channel ControlNet condition.
        apply_morphological_perturb (bool, optional):
            Whether to apply morphological perturbation to masks. Defaults to False.
        modality_tensor (torch.Tensor, optional):
            Class labels or modality codes for conditional generation (e.g., MRI/CT).
        top_region_index_tensor (torch.Tensor, optional):
            Region index tensor (top bound) for body-region-aware conditioning.
        bottom_region_index_tensor (torch.Tensor, optional):
            Region index tensor (bottom bound) for body-region-aware conditioning.
        return_controlnet_blocks (bool, optional):
            If True, also return `(down_block_res_samples, mid_block_res_sample)`.
            Defaults to False.

    Returns:
        Tuple[torch.Tensor, Optional[Any], Optional[Any]]:
            - model_output (torch.Tensor): U-Net output with shape [B, C, X, Y, Z].
            - down_block_res_samples (optional): ControlNet down-block features if requested, else None.
            - mid_block_res_sample (optional): ControlNet mid-block feature if requested, else None.
    """
    # generate random noise
    include_modality = ( modality_tensor is not None )
    include_body_region = ( top_region_index_tensor is not None) and (bottom_region_index_tensor is not None)

    # Construct dual-channel condition: [Pre-contrast MRI, Tumor Mask]
    # labels can be either a MetaTensor or torch.Tensor
    if hasattr(labels, 'as_tensor'):
        masks = labels.as_tensor().to(torch.long)
    else:
        masks = labels.to(torch.long)

    masks = masks.float()

    # Apply morphological perturbation with 30% probability
    if apply_morphological_perturb and torch.rand(1).item() < 0.3:
        masks = apply_random_morphological_perturbation(masks)

    # If pre_images provided, use dual-channel [Pre, Mask]
    # Otherwise duplicate mask to 2 channels for compatibility (inference-only paths)
    if pre_images is not None:
        controlnet_cond = torch.cat([pre_images.float(), masks], dim=1)  # [B, 2, H, W, D]
    else:
        # Fallback: duplicate mask to 2 channels for backward compatibility
        controlnet_cond = torch.cat([masks, masks], dim=1)  # [B, 2, H, W, D]

    # create noisy latent
    noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

    # get controlnet output
    # Create a dictionary to store the inputs
    controlnet_inputs = {
        "x": noisy_latent,
        "timesteps": timesteps,
        "controlnet_cond": controlnet_cond,
    }
    if include_modality:
        controlnet_inputs.update(
            {
                "class_labels": modality_tensor,
            }
        )
    down_block_res_samples, mid_block_res_sample = controlnet(**controlnet_inputs)

    # get diffusion network output
    # Create a dictionary to store the inputs
    unet_inputs = {
        "x": noisy_latent,
        "timesteps": timesteps,
        "spacing_tensor": spacing_tensor,
        "down_block_additional_residuals": down_block_res_samples,
        "mid_block_additional_residual": mid_block_res_sample,
    }
    # Add extra arguments if include_body_region is True
    if include_body_region:
        unet_inputs.update(
            {
                "top_region_index_tensor": top_region_index_tensor,
                "bottom_region_index_tensor": bottom_region_index_tensor,
            }
        )
    if include_modality:
        unet_inputs.update(
            {
                "class_labels": modality_tensor,
            }
        )
    model_output = unet(**unet_inputs)
    if return_controlnet_blocks:
        return model_output, down_block_res_samples, mid_block_res_sample
    else:
        return model_output, None, None

def train_controlnet(
    env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int
) -> None:
    # Step 0: configuration
    # whether to use distributed data parallel
    use_ddp = num_gpus > 1
    if use_ddp:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = setup_ddp(rank, world_size)
    else:
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:{rank}")

    torch.cuda.set_device(device)

    args = load_config(env_config_path, model_config_path, model_def_path)

    # Setup logging with file output
    log_file_path = os.path.join(args.output_dir, "logs", args.exp_name, "train.log")
    logger = setup_logging(
        logger_name="maisi.controlnet.training",
        log_file_path=log_file_path,
        rank=rank
    )

    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"World_size: {world_size}")
    logger.info(f"Log file: {log_file_path}")

    # initialize tensorboard writer
    if rank == 0:
        tensorboard_path = os.path.join(args.tfevent_path, args.exp_name)
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)


    # Step 2: define diffusion model and controlnet
    # define diffusion Model
    unet = define_instance(args, "diffusion_unet_def").to(device)
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    # load trained diffusion model
    if args.trained_diffusion_path is not None:
        if not os.path.exists(args.trained_diffusion_path):
            raise ValueError(f"Please download the trained diffusion unet checkpoint to {args.trained_diffusion_path}.")
        diffusion_model_ckpt = torch.load(args.trained_diffusion_path, map_location=device, weights_only=False)
        unet.load_state_dict(diffusion_model_ckpt["unet_state_dict"], strict=False)
        # load scale factor from diffusion model checkpoint
        scale_factor = diffusion_model_ckpt["scale_factor"]
        logger.info(f"Load trained diffusion model from {args.trained_diffusion_path}.")
        logger.info(f"loaded scale_factor from diffusion model ckpt -> {scale_factor}.")
    else:
        raise ValueError(f"'trained_diffusion_path' in {env_config_path} cannot be null.")

    # define ControlNet
    controlnet = define_instance(args, "controlnet_def").to(device)
    # copy weights from the DM to the controlnet
    copy_model_state(controlnet, unet.state_dict())
    # load trained controlnet model if it is provided
    if args.existing_ckpt_filepath is not None:
        if not os.path.exists(args.existing_ckpt_filepath):
            raise ValueError("Please download the trained ControlNet checkpoint.")
        controlnet.load_state_dict(
            torch.load(args.existing_ckpt_filepath, map_location=device, weights_only=False)["controlnet_state_dict"]
        )
        logger.info(f"load trained controlnet model from {args.existing_ckpt_filepath}")
    else:
        logger.info("train controlnet model from scratch.")
    # we freeze the parameters of the diffusion model.
    for p in unet.parameters():
        p.requires_grad = False

    noise_scheduler = define_instance(args, "noise_scheduler")

    if use_ddp:
        controlnet = DDP(controlnet, device_ids=[device], output_device=rank, find_unused_parameters=True)

    # set data loader
    if include_modality:
        if args.modality_mapping_path is not None:
            if not os.path.exists(args.modality_mapping_path):
                raise ValueError(f"Please check if {args.modality_mapping_path} exist.")
        else:
            raise ValueError(f"'modality_mapping_path' in {env_config_path} cannot be null")
        with open(args.modality_mapping_path, "r") as f:
            args.modality_mapping = json.load(f)
    else:
        args.modality_mapping = None

    train_loader, _ = prepare_maisi_controlnet_json_dataloader(
        json_data_list=args.json_data_list,
        data_base_dir=args.data_base_dir,
        rank=rank,
        world_size=world_size,
        batch_size=args.controlnet_train["batch_size"],
        cache_rate=args.controlnet_train["cache_rate"],
        fold=args.controlnet_train["fold"],
        modality_mapping = args.modality_mapping
    )

    # Step 3: training config
    optimizer = torch.optim.AdamW(params=controlnet.parameters(), lr=args.controlnet_train["lr"])
    total_steps = (args.controlnet_train["n_epochs"] * len(train_loader.dataset)) / args.controlnet_train["batch_size"]
    logger.info(f"total number of training steps: {total_steps}.")

    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)

    # Step 4: training
    n_epochs = args.controlnet_train["n_epochs"]
    scaler = GradScaler("cuda")
    total_step = 0
    best_loss = 1e4

    controlnet.train()
    unet.eval()
    prev_time = time.time()
    for epoch in range(n_epochs):
        epoch_loss_ = 0
        valid_batches = 0  # Track number of valid (non-NaN) batches
        for step, batch in enumerate(train_loader):
            # get image embedding and label mask and scale image embedding by the provided scale_factor
            images = batch["image"].to(device) * scale_factor
            labels = batch["label"].to(device)
            if labels.shape[1] != 1:
                raise ValueError(f"We expect labels with shape [B,1,X,Y,Z], yet got {labels.shape}")
            # get corresponding conditions
            spacing_tensor = batch["spacing"].to(device)
            top_region_index_tensor = None
            bottom_region_index_tensor = None
            modality_tensor = None
            if include_body_region:
                top_region_index_tensor = batch["top_region_index"].to(device)
                bottom_region_index_tensor = batch["bottom_region_index"].to(device)
            # We trained with only CT in this version
            if include_modality:
                modality_tensor = batch["modality"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=True):
                # randomly sample noise
                noise_shape = list(images.shape)
                noise = torch.randn(noise_shape, dtype=images.dtype).to(device)
                # randomly sample timesteps
                if isinstance(noise_scheduler, RFlowScheduler):
                    timesteps = noise_scheduler.sample_timesteps(images)
                else:
                    timesteps = torch.randint(
                        0, noise_scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                    ).long()

                # Get pre-contrast images from batch for dual-channel ControlNet condition
                pre_images = batch["pre"].to(device)  # [B, 1, 256, 256, 256]

                (
                    model_output,
                    model_block1_output,
                    model_block2_output
                ) = compute_model_output(
                    images,labels,noise,timesteps,noise_scheduler,
                    controlnet,unet,
                    spacing_tensor,
                    pre_images=pre_images,
                    apply_morphological_perturb=True,
                    modality_tensor=modality_tensor,
                    top_region_index_tensor=top_region_index_tensor,
                    bottom_region_index_tensor=bottom_region_index_tensor,
                    return_controlnet_blocks=False
                )

                if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                    # predict noise
                    model_gt = noise
                elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                    # predict sample
                    model_gt = images
                elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                    # predict velocity
                    model_gt = images - noise
                else:
                    raise ValueError(
                        "noise scheduler prediction type has to be chosen from ",
                        f"[{DDPMPredictionType.EPSILON},{DDPMPredictionType.SAMPLE},{DDPMPredictionType.V_PREDICTION}]",
                    )
    
                # ========== The Great Cleanup: Pure Loss Implementation ==========
                # 1. Precise dimension reduction using Max Pooling (preserves ANY tiny lesion!)
                # labels: [B, 1, 256, 256, 256] -> roi_mask_latent: [B, 1, 64, 64, 64]
                roi_mask_latent = F.max_pool3d(labels.float(), kernel_size=4, stride=4) > 0.0

                # 2. Base Global L1 with moderate ROI weighting
                weights = torch.ones_like(model_output)
                weights[roi_mask_latent.repeat(1, model_output.shape[1], 1, 1, 1)] = 3.0
                l1_loss_raw = F.l1_loss(model_output.float(), model_gt.float(), reduction="none")
                loss = (l1_loss_raw * weights).mean()

                # 3. Absolute Background Penalty (严查假阳性糊团)
                # Penalize false positives in background regions to suppress snow noise
                if roi_mask_latent.sum() > 0:
                    # Has tumor: background = ~roi
                    bg_mask_expanded = (~roi_mask_latent).repeat(1, model_output.shape[1], 1, 1, 1)
                else:
                    # No tumor: entire image is background
                    bg_mask_expanded = torch.ones_like(model_output, dtype=torch.bool)

                pred_bg = model_output.float()[bg_mask_expanded]
                gt_bg = model_gt.float()[bg_mask_expanded]

                # Only penalize false positives (white spots that shouldn't be there)
                # F.relu(pred - gt) = max(0, pred - gt): only positive when pred > gt
                false_positive_bg = F.relu(pred_bg - gt_bg)
                loss = loss + 5.0 * (false_positive_bg ** 2).mean()

            # Check for NaN loss before backward pass
            loss_value = loss.detach().cpu().item()
            if not torch.isfinite(loss):
                # Log warning and skip this batch
                logger.warning(
                    f"NaN or Inf loss detected at [Epoch {epoch + 1}/{n_epochs}] [Batch {step + 1}/{len(train_loader)}]. "
                    f"Skipping this batch."
                )
                # Check if inputs have NaN
                if torch.isnan(images).any():
                    logger.warning(f"  -> images contains NaN!")
                if torch.isnan(labels).any():
                    logger.warning(f"  -> labels contains NaN!")
                if torch.isnan(pre_images).any():
                    logger.warning(f"  -> pre_images contains NaN!")
                continue  # Skip this batch

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            total_step += 1
            valid_batches += 1  # Increment valid batch counter

            if rank == 0:
                # write train loss for each batch into tensorboard
                tensorboard_writer.add_scalar(
                    "train/train_controlnet_loss_iter", loss_value, total_step
                )
                batches_done = step + 1
                batches_left = len(train_loader) - batches_done
                time_left = timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Console output: every batch (for real-time monitoring)
                print(
                    "\r[Epoch %d/%d] [Batch %d/%d] [LR: %.8f] [loss: %.4f] ETA: %s "
                    % (
                        epoch + 1,
                        n_epochs,
                        step + 1,
                        len(train_loader),
                        lr_scheduler.get_last_lr()[0],
                        loss_value,
                        time_left,
                    ),
                    end="", flush=True
                )

                # File output: every 100 batches (to keep log file manageable)
                if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                    logger.info(
                        "[Epoch %d/%d] [Batch %d/%d] [LR: %.8f] [loss: %.4f] ETA: %s"
                        % (
                            epoch + 1,
                            n_epochs,
                            step + 1,
                            len(train_loader),
                            lr_scheduler.get_last_lr()[0],
                            loss_value,
                            time_left,
                        )
                    )
            epoch_loss_ += loss.detach()

        # Calculate epoch average, accounting for skipped batches
        if valid_batches > 0:
            epoch_loss = epoch_loss_ / valid_batches
        else:
            # All batches were skipped (NaN), use previous loss or skip this epoch
            logger.warning(f"[Epoch {epoch + 1}/{n_epochs}] All batches skipped due to NaN loss!")
            epoch_loss = torch.tensor(0.0)  # Placeholder

        # Print newline and summary after epoch
        if rank == 0:
            print()  # newline after progress bar
            if valid_batches < len(train_loader):
                logger.info(
                    f"[Epoch {epoch + 1}/{n_epochs}] Skipped {len(train_loader) - valid_batches}/{len(train_loader)} batches due to NaN loss"
                )

        if use_ddp:
            dist.barrier()
            dist.all_reduce(epoch_loss, op=torch.distributed.ReduceOp.AVG)

        if rank == 0:
            tensorboard_writer.add_scalar("train/train_controlnet_loss_epoch", epoch_loss.cpu().item(), total_step)
            # save controlnet only on master GPU (rank 0)
            controlnet_state_dict = controlnet.module.state_dict() if world_size > 1 else controlnet.state_dict()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "controlnet_state_dict": controlnet_state_dict,
                },
                f"{args.model_dir}/{args.exp_name}_current.pt",
            )
            logger.info(f"Save trained model to {args.model_dir}/{args.exp_name}_current.pt")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                logger.info(f"best loss -> {best_loss}.")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "loss": best_loss,
                        "controlnet_state_dict": controlnet_state_dict,
                    },
                    f"{args.model_dir}/{args.exp_name}_best.pt",
                )
                logger.info(f"Save trained model to {args.model_dir}/{args.exp_name}_best.pt")

        torch.cuda.empty_cache()
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ControlNet Model Training")
    parser.add_argument(
        "-e",
        "--env_config_path",
        type=str,
        default="./configs/environment_maisi_diff_model.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "-c",
        "--model_config_path",
        type=str,
        default="./configs/config_maisi_diff_model.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "-t",
        "--model_def_path", 
        type=str, 
        default="./configs/config_maisi.json", 
        help="Path to model definition file"
    )
    parser.add_argument(
        "-g",
        "--num_gpus", 
        type=int, 
        default=1, 
        help="Number of GPUs to use for training"
    )

    args = parser.parse_args()
    train_controlnet(args.env_config_path, args.model_config_path, args.model_def_path, args.num_gpus)

