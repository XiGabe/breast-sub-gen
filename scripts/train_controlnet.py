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

def set_unet_frozen_state(unet, blocks_to_unfreeze=None, disable_inplace=False):
    """
    Freeze or unfreeze specific U-Net blocks for progressive training.

    This function enables staged fine-tuning of the U-Net by selectively unfreezing
    specific blocks while keeping others frozen. This is critical for the progressive
    co-tuning strategy where ControlNet is trained first, then deep U-Net blocks,
    and finally shallow U-Net blocks.

    Args:
        unet: The U-Net model (DDP wrapped or unwrapped).
        blocks_to_unfreeze: List of block prefixes to unfreeze (e.g., ['down_blocks.2']).
            If None or empty, all blocks are frozen.
        disable_inplace: If True, recursively disable inplace operations on all modules.
            This is CRITICAL when using gradient checkpointing with mixed frozen/unfrozen
            blocks to avoid RuntimeError: "one of the variables needed for gradient
            computation has been modified by an inplace operation."

    Block naming structure:
        Shallow encoder: 'down_blocks.0', 'down_blocks.1'
        Deep encoder: 'down_blocks.2', 'down_blocks.3'
        Bridge: 'middle_block'
        Deep decoder: 'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3'
        Shallow decoder: 'up_blocks.3'

    Permanently frozen blocks (non-spatial features):
        'conv_in', 'class_embedding', 'time_embed'

    Returns:
        tuple: (unfrozen_count, total_count) - Number of unfrozen parameters and total parameters.

    Example:
        # Stage 3.1: ControlNet only (all U-Net frozen)
        set_unet_frozen_state(unet)

        # Stage 3.2: Deep blocks
        set_unet_frozen_state(unet, blocks_to_unfreeze=[
            'down_blocks.2', 'down_blocks.3', 'middle_block',
            'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3'
        ], disable_inplace=True)

        # Stage 3.3: All blocks
        set_unet_frozen_state(unet, blocks_to_unfreeze=[
            'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3',
            'middle_block',
            'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3'
        ], disable_inplace=True)
    """
    if blocks_to_unfreeze is None:
        blocks_to_unfreeze = []

    # Always keep these frozen (non-spatial features that should remain fixed)
    frozen_blocks = ['conv_in', 'class_embedding', 'time_embed']

    # Disable inplace operations if requested (CRITICAL for gradient checkpointing + mixed frozen/unfrozen)
    if disable_inplace:
        for module in unet.modules():
            if hasattr(module, 'inplace') and module.inplace:
                module.inplace = False

    for name, param in unet.named_parameters():
        # Check if this parameter belongs to a block we want to unfreeze
        should_unfreeze = any(name.startswith(block) for block in blocks_to_unfreeze)
        # Check if this parameter belongs to a permanently frozen block
        should_freeze = any(name.startswith(block) for block in frozen_blocks)
        # Set requires_grad: True only if unfreezing AND not in permanently frozen blocks
        param.requires_grad = should_unfreeze and not should_freeze

    unfrozen_count = sum(p.requires_grad for p in unet.parameters())
    total_count = sum(1 for p in unet.parameters())
    return unfrozen_count, total_count

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
            raise ValueError(f"Please check if {args.existing_ckpt_filepath} exists.")
        checkpoint = torch.load(args.existing_ckpt_filepath, map_location=device, weights_only=False)

        # Load ControlNet
        controlnet.load_state_dict(checkpoint["controlnet_state_dict"])
        logger.info(f"Loaded ControlNet from {args.existing_ckpt_filepath}")

        # Load U-Net if available (from Stage 2 or 3)
        if "unet_state_dict" in checkpoint and checkpoint["unet_state_dict"] is not None:
            unet.load_state_dict(checkpoint["unet_state_dict"], strict=False)
            logger.info("Loaded fine-tuned U-Net state from checkpoint")
        else:
            logger.info("No U-Net state in checkpoint (using base pretrained U-Net)")
    else:
        logger.info("train controlnet model from scratch.")

    # Freeze/unfreeze U-Net blocks based on config for progressive training
    unet_blocks_to_unfreeze = args.controlnet_train.get("unet_blocks_to_unfreeze", [])
    disable_inplace = args.controlnet_train.get("disable_inplace_for_checkpointing", False)

    if unet_blocks_to_unfreeze:
        unfrozen_count, total_count = set_unet_frozen_state(
            unet,
            blocks_to_unfreeze=unet_blocks_to_unfreeze,
            disable_inplace=disable_inplace
        )
        logger.info(f"U-Net progressive training: {unfrozen_count}/{total_count} parameters unfrozen")
        logger.info(f"Unfrozen blocks: {unet_blocks_to_unfreeze}")
        if disable_inplace:
            logger.info("Inplace operations disabled for gradient checkpointing compatibility")
    else:
        # Default: freeze all U-Net parameters (Stage 3.1: ControlNet only)
        for p in unet.parameters():
            p.requires_grad = False
        logger.info("U-Net fully frozen (Stage 3.1: ControlNet only)")

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

    train_loader, val_loader = prepare_maisi_controlnet_json_dataloader(
        json_data_list=args.json_data_list,
        data_base_dir=args.data_base_dir,
        rank=rank,
        world_size=world_size,
        batch_size=args.controlnet_train["batch_size"],
        cache_rate=args.controlnet_train["cache_rate"],
        fold=args.controlnet_train["fold"],
        modality_mapping = args.modality_mapping
    )

    # Validation settings
    validation_enabled = val_loader is not None
    validation_freq = args.controlnet_train.get("validation_frequency", 1)  # Validate every N epochs
    if validation_enabled:
        logger.info(f"Validation enabled: every {validation_freq} epoch(s)")
    else:
        logger.info("No validation set - using training loss for checkpoint selection")

    # Step 3: training config with dual learning rate support
    # Create optimizer with optional U-Net parameter group
    params = [{"params": controlnet.parameters(), "lr": args.controlnet_train["lr"]}]

    # Add U-Net parameters if any are unfrozen
    unet_lr = args.controlnet_train.get("unet_lr", None)
    if unet_lr and any(p.requires_grad for p in unet.parameters()):
        # Get unfrozen U-Net parameters
        unet_params = [p for p in unet.parameters() if p.requires_grad]
        params.append({"params": unet_params, "lr": unet_lr})
        logger.info(f"Two-stage optimizer: ControlNet LR={args.controlnet_train['lr']:.2e}, U-Net LR={unet_lr:.2e}")
    else:
        logger.info(f"Single-stage optimizer: ControlNet LR={args.controlnet_train['lr']:.2e}")

    optimizer = torch.optim.AdamW(params)
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
                l1_loss = (l1_loss_raw * weights).mean()

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
                bg_penalty_loss = 5.0 * (false_positive_bg ** 2).mean()

                # Total loss
                loss = l1_loss + bg_penalty_loss

            # Check for NaN loss before backward pass
            loss_value = loss.detach().cpu().item()
            l1_loss_value = l1_loss.detach().cpu().item()
            bg_penalty_value = bg_penalty_loss.detach().cpu().item()

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
                tensorboard_writer.add_scalar(
                    "train/l1_loss_iter", l1_loss_value, total_step
                )
                tensorboard_writer.add_scalar(
                    "train/bg_penalty_loss_iter", bg_penalty_value, total_step
                )
                batches_done = step + 1
                batches_left = len(train_loader) - batches_done
                time_left = timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Console output: every batch (for real-time monitoring)
                print(
                    "\r[Epoch %d/%d] [Batch %d/%d] [LR: %.8f] [loss: %.4f] [L1: %.4f] [BgPen: %.4f] ETA: %s "
                    % (
                        epoch + 1,
                        n_epochs,
                        step + 1,
                        len(train_loader),
                        lr_scheduler.get_last_lr()[0],
                        loss_value,
                        l1_loss_value,
                        bg_penalty_value,
                        time_left,
                    ),
                    end="", flush=True
                )

                # File output: every 100 batches (to keep log file manageable)
                if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                    logger.info(
                        "[Epoch %d/%d] [Batch %d/%d] [LR: %.8f] [Total: %.4f] [L1: %.4f] [BgPen: %.4f] ETA: %s"
                        % (
                            epoch + 1,
                            n_epochs,
                            step + 1,
                            len(train_loader),
                            lr_scheduler.get_last_lr()[0],
                            loss_value,
                            l1_loss_value,
                            bg_penalty_value,
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

            # Validation loop (if enabled)
            val_loss = None
            if validation_enabled and val_loader is not None and (epoch + 1) % validation_freq == 0:
                logger.info(f"Running validation at epoch {epoch + 1}...")
                controlnet.eval()
                val_loss_ = 0
                val_valid_batches = 0

                with torch.no_grad():
                    for val_step, val_batch in enumerate(val_loader):
                        # Check if all required keys exist
                        if "image" not in val_batch:
                            logger.warning(f"[Validation Batch {val_step}] Missing 'image' key in val_batch. Keys: {list(val_batch.keys())}")
                            continue
                        if "label" not in val_batch:
                            logger.warning(f"[Validation Batch {val_step}] Missing 'label' key in val_batch. Keys: {list(val_batch.keys())}")
                            continue
                        if "pre" not in val_batch:
                            logger.warning(f"[Validation Batch {val_step}] Missing 'pre' key in val_batch. Keys: {list(val_batch.keys())}")
                            continue

                        val_images = val_batch["image"].to(device) * scale_factor
                        val_labels = val_batch["label"].to(device)
                        val_spacing_tensor = val_batch["spacing"].to(device)
                        val_pre_images = val_batch["pre"].to(device)
                        val_modality_tensor = val_batch["modality"].to(device) if include_modality else None

                        # Check for NaN in inputs
                        if torch.isnan(val_images).any():
                            logger.warning(f"[Validation Batch {val_step}] val_images contains NaN!")
                            continue
                        if torch.isnan(val_labels).any():
                            logger.warning(f"[Validation Batch {val_step}] val_labels contains NaN!")
                            continue
                        if torch.isnan(val_pre_images).any():
                            logger.warning(f"[Validation Batch {val_step}] val_pre_images contains NaN!")
                            continue

                        val_noise = torch.randn_like(val_images)
                        if isinstance(noise_scheduler, RFlowScheduler):
                            val_timesteps = noise_scheduler.sample_timesteps(val_images)
                        else:
                            val_timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (val_images.shape[0],), device=val_images.device).long()

                        with autocast("cuda", enabled=True):
                            val_model_output, _, _ = compute_model_output(
                                val_images, val_labels, val_noise, val_timesteps, noise_scheduler,
                                controlnet, unet, val_spacing_tensor,
                                pre_images=val_pre_images,
                                apply_morphological_perturb=False,  # No augmentation during validation
                                modality_tensor=val_modality_tensor,
                            )

                            if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                                val_model_gt = val_noise
                            elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                                val_model_gt = val_images
                            elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                                val_model_gt = val_images - val_noise
                            else:
                                raise ValueError(f"Unknown prediction type: {noise_scheduler.prediction_type}")

                            # Same loss computation as training, but separate components
                            val_roi_mask_latent = F.max_pool3d(val_labels.float(), kernel_size=4, stride=4) > 0.0
                            val_weights = torch.ones_like(val_model_output)
                            val_weights[val_roi_mask_latent.repeat(1, val_model_output.shape[1], 1, 1, 1)] = 3.0
                            val_l1_loss_raw = F.l1_loss(val_model_output.float(), val_model_gt.float(), reduction="none")
                            val_l1_loss = (val_l1_loss_raw * val_weights).mean()

                            if val_roi_mask_latent.sum() > 0:
                                val_bg_mask_expanded = (~val_roi_mask_latent).repeat(1, val_model_output.shape[1], 1, 1, 1)
                            else:
                                val_bg_mask_expanded = torch.ones_like(val_model_output, dtype=torch.bool)

                            val_pred_bg = val_model_output.float()[val_bg_mask_expanded]
                            val_gt_bg = val_model_gt.float()[val_bg_mask_expanded]
                            val_false_positive_bg = F.relu(val_pred_bg - val_gt_bg)
                            val_bg_penalty_loss = 5.0 * (val_false_positive_bg ** 2).mean()

                            # Total validation loss
                            val_loss = val_l1_loss + val_bg_penalty_loss

                        # Check for NaN or Inf loss
                        if not torch.isfinite(val_loss):
                            logger.warning(
                                f"[Validation Batch {val_step}] NaN or Inf loss detected! "
                                f"loss={val_loss}. Skipping this batch."
                            )
                            if torch.isnan(val_model_output).any():
                                logger.warning(f"  -> val_model_output contains NaN!")
                            if torch.isnan(val_model_gt).any():
                                logger.warning(f"  -> val_model_gt contains NaN!")
                            continue

                        val_loss_ += val_loss.detach()
                        val_valid_batches += 1

                if val_valid_batches > 0:
                    val_loss = val_loss_ / val_valid_batches
                    logger.info(f"[Epoch {epoch + 1}] Validation loss: {val_loss:.4f}")
                    tensorboard_writer.add_scalar("val/val_controlnet_loss_epoch", val_loss.cpu().item(), total_step)
                else:
                    logger.warning(f"[Epoch {epoch + 1}] All validation batches skipped!")
                    val_loss = None

                controlnet.train()

            # Save checkpoint for this epoch
            controlnet_state_dict = controlnet.module.state_dict() if world_size > 1 else controlnet.state_dict()

            # Save U-Net state if any blocks are unfrozen
            unet_state_dict = None
            if any(p.requires_grad for p in unet.parameters()):
                unet_state_dict = unet.module.state_dict() if world_size > 1 else unet.state_dict()

            # Save per-epoch checkpoint
            epoch_checkpoint_path = f"{args.model_dir}/{args.exp_name}_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "val_loss": val_loss,
                    "controlnet_state_dict": controlnet_state_dict,
                    "unet_state_dict": unet_state_dict,
                    "scale_factor": scale_factor,
                },
                epoch_checkpoint_path,
            )
            logger.info(f"Saved epoch {epoch + 1} checkpoint to {epoch_checkpoint_path}")

            # Update best checkpoint based on validation loss (if available) or training loss
            loss_for_best = val_loss if val_loss is not None else epoch_loss
            if loss_for_best < best_loss:
                best_loss = loss_for_best
                best_checkpoint_path = f"{args.model_dir}/{args.exp_name}_best.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "train_loss": epoch_loss,
                        "val_loss": val_loss,
                        "controlnet_state_dict": controlnet_state_dict,
                        "unet_state_dict": unet_state_dict,
                        "scale_factor": scale_factor,
                    },
                    best_checkpoint_path,
                )
                logger.info(f"New best {'validation' if val_loss is not None else 'training'} loss: {best_loss:.4f} -> {best_checkpoint_path}")

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

