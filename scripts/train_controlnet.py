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

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai.networks.utils import copy_model_state
from monai.utils import RankFilter
from monai.transforms.utils_morphological_ops import dilate
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from .utils import define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp
from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .augmentation import remove_tumors

def remove_roi(labels):
    """
    Remove ROI voxels from a label tensor. 
    Users need to define their own function of remove_roi.
    Here we use scripts.augmentation.remove_tumors as default

    Args:
        labels (torch.Tensor): Segmentation tensor. Shape is
            [B, 1, X, Y, Z]. Dtype is usually integer/long.

    Returns:
        torch.Tensor: Labels with ROI content removed. Same shape and
        device as `labels`.
    """
    labels_roi_free = []
    for b in range(labels.shape[0]):
        labels_roi_free_b = remove_tumors(labels[b,...])
        labels_roi_free.append(labels_roi_free_b)
    labels_roi_free = torch.cat(labels_roi_free,dim=0)
    return labels_roi_free
    
def compute_region_contrasive_loss(
    model_output,model_output_roi_free,model_gt,
    roi_contrastive,roi_contrastive_bg,
    max_region_contrasive_loss=2,
    loss_contrastive = torch.nn.L1Loss(reduction = 'none')
):
    """
    Compute region-wise contrastive losses between the model output with and
    without ROIs, promoting differences inside ROI and similarity outside ROI.

    The loss has two parts:
      1) `loss_region_contrasive`: encourages the model output to differ from
         its ROI-free counterpart *inside* the ROI (foreground). Implemented as
         a (negative) masked L1 reduced by the foreground voxel count and then
         clipped by a ReLU window around `max_region_contrasive_loss`.
      2) `loss_region_bg`: encourages *similarity* in the background
         (outside ROI) between the ROI-free output and the original output,
         implemented as masked L1 reduced by background voxel count.

    Args:
        model_output (torch.Tensor):
            Network output with ROI present. Shape [B, C, X, Y, Z].
        model_output_roi_free (torch.Tensor):
            Network output for ROI-removed labels (same shape/device).
        roi_contrastive (torch.Tensor):
            Foreground ROI mask (1 inside ROI, 0 outside). Can be bool or
            integer; will be resized to `model_output.shape[2:]` using
            nearest-neighbor and multiplied as weights.
            Expected shape broadcastable to [B, C, X, Y, Z].
        roi_contrastive_bg (torch.Tensor):
            Background mask (1 outside ROI, 0 inside). Will be resized to
            `inputs.shape[2:]` (see Notes) and repeated over channels to match
            [B, C, X, Y, Z].
        max_region_contrasive_loss (float, optional):
            Upper-window parameter used to bound the foreground loss via
            `relu(loss + max) - max`. Defaults to 2.
        loss_contrastive (torch.nn.modules.loss._Loss, optional):
            Elementwise regression loss with `reduction='none'` (e.g., L1).
            Defaults to `torch.nn.L1Loss(reduction='none')`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - loss_region_contrasive (scalar tensor): foreground contrastive loss.
            - loss_region_bg (scalar tensor): background similarity loss.
    """
    if roi_contrastive.shape[1]!=1 or roi_contrastive_bg.shape[1]!=1:
        raise ValueError(f"Assert roi_contrastive.shape[1]==1 or roi_contrastive_bg.shape[1]==1, yet got {roi_contrastive.shape} and {roi_contrastive_bg.shape}.")
        
    roi_contrastive = F.interpolate(roi_contrastive, size=model_output.shape[2:], mode="nearest")  
    roi_contrastive = roi_contrastive.repeat(1, model_output.shape[1], 1, 1, 1)    
    loss_region_contrasive = -(loss_contrastive(model_output, model_output_roi_free)*roi_contrastive).sum()/(torch.sum(roi_contrastive>0)+1e-5)
    loss_region_contrasive = F.relu(loss_region_contrasive+max_region_contrasive_loss)-max_region_contrasive_loss # we do not need it to be extreme           
    
    roi_contrastive_bg = F.interpolate(roi_contrastive_bg, size=model_output.shape[2:], mode="nearest").to(torch.long)
    roi_contrastive_bg = roi_contrastive_bg.repeat(1, model_output.shape[1], 1, 1, 1)    
    loss_region_bg = (loss_contrastive(model_output_roi_free, model_gt)*roi_contrastive_bg).sum()/(torch.sum(roi_contrastive_bg>0)+1e-5)
    return loss_region_contrasive, loss_region_bg

def compute_topk_loss(
    pred,
    gt,
    weights,
    topk_ratio=0.3
):
    """
    Top-K Hard Example Mining Loss for sparse signal learning.

    Instead of averaging over ALL pixels (including 95% easy background),
    only compute loss on the hardest 30% pixels. This forces the model to
    focus on difficult tumor boundaries and faint signals.

    Args:
        pred (torch.Tensor): Model prediction. Shape [B, C, X, Y, Z].
        gt (torch.Tensor): Ground truth. Shape [B, C, X, Y, Z].
        weights (torch.Tensor): Per-pixel weights (e.g., tumor region mask).
            Shape [B, C, X, Y, Z].
        topk_ratio (float): Ratio of hardest pixels to use (default 0.3 = 30%).

    Returns:
        torch.Tensor: Scalar loss averaged over top-k hardest pixels.

    Note:
        Loss values will be HIGHER than standard weighted L1 (e.g., ~1.5 vs ~0.86)
        because we remove easy samples from the denominator. This is EXPECTED and
        indicates the model is focusing on hard examples.
    """
    # 1. Compute per-pixel weighted L1 loss (no reduction)
    l1_loss_raw = F.l1_loss(pred.float(), gt.float(), reduction="none") * weights

    # 2. Flatten to [B, C*X*Y*Z]
    B = l1_loss_raw.size(0)
    l1_loss_flat = l1_loss_raw.view(B, -1)

    # 3. Select top-k hardest pixels
    k = max(1, int(l1_loss_flat.size(1) * topk_ratio))  # Ensure at least 1 pixel
    topk_loss, _ = torch.topk(l1_loss_flat, k, dim=1)

    # 4. Average only over hard examples
    return topk_loss.mean()

def compute_model_output(
    images,labels,noise,timesteps,noise_scheduler,
    controlnet,unet,
    spacing_tensor,
    modality_tensor=None,
    top_region_index_tensor=None,
    bottom_region_index_tensor=None,
    pre_images=None,
    return_controlnet_blocks=False
):
    """
    Run ControlNet + U-Net to obtain the denoising network output (and optionally
    the ControlNet intermediate blocks) for a given noisy latent and conditions.

    Pipeline:
      1) Build ControlNet condition (from labels or pre_images).
      2) Add noise to `images` at `timesteps` via the scheduler.
      3) Pass noisy latent and conditions to ControlNet to get down/mid features.
      4) Pass everything to U-Net (with spacing, optional modality & body-region
         tokens) to produce `model_output`.

    Args:
        images (torch.Tensor):
            Input latent/image tensor to be noised. Shape [B, C, X, Y, Z].
        labels (torch.Tensor or monai.data.MetaTensor):
            Segmentation labels used to create ControlNet condition (if pre_images is None).
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
        modality_tensor (torch.Tensor, optional):
            Class labels or modality codes for conditional generation (e.g., MRI/CT).
        top_region_index_tensor (torch.Tensor, optional):
            Region index tensor (top bound) for body-region-aware conditioning.
        bottom_region_index_tensor (torch.Tensor, optional):
            Region index tensor (bottom bound) for body-region-aware conditioning.
        pre_images (torch.Tensor, optional):
            Pre-contrast images used as ControlNet condition. If provided, takes
            priority over labels. Shape [B, 1, X, Y, Z] at physical resolution.
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

    # Build controlnet condition: use pre-contrast images as conditioning
    # Pre images are at physical resolution (256³). The Conditioning Embedding layer
    # (3 strided convolutions) will handle downsampling to 64³ while extracting features.
    controlnet_cond = pre_images.float()
    # Ensure single channel input
    if controlnet_cond.shape[1] != 1:
        controlnet_cond = controlnet_cond[:, :1, ...]

    # create noisy latent
    noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

    # CRITICAL FIX for Stage 4: Clone noisy_latent to prevent memory sharing
    # noise_scheduler.add_noise may create a view that shares memory with 'images'.
    # When down_blocks.2,3 are unfrozen, this shared memory causes inplace operation errors.
    noisy_latent = noisy_latent.clone()

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

    # Clone residuals to avoid inplace operation issues when deep encoder is unfrozen
    # This prevents MAISI UNet's internal inplace ops (e.g., ReLU) from breaking the grad graph
    down_block_res_samples = [r.clone() if r is not None else None for r in down_block_res_samples]
    if mid_block_res_sample is not None:
        mid_block_res_sample = mid_block_res_sample.clone()

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

    # UNet forward pass with gradient checkpointing for Stage 4
    # When down_blocks.2,3 are unfrozen, UNet's internal inplace ops corrupt the gradient graph.
    # Gradient checkpointing recomputes forward pass during backward, avoiding saving problematic intermediates.
    def create_unet_forward_func(unet_inputs_dict):
        """Closure to capture unet_inputs for checkpoint"""
        def forward_func():
            return unet(**unet_inputs_dict)
        return forward_func

    # Use checkpointing to avoid saving intermediate activations that get modified
    model_output = torch.utils.checkpoint.checkpoint(
        create_unet_forward_func(unet_inputs),
        use_reentrant=False,
    )
    if return_controlnet_blocks:
        return model_output, down_block_res_samples, mid_block_res_sample
    else:
        return model_output, None, None

def train_controlnet(
    env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int
) -> None:
    # Step 0: configuration
    # Setup logging to stdout and file
    log_file = "training_latest.log"
    logger = setup_logging("maisi.controlnet.training", log_file=log_file)
    logger.info(f"Logging configured: stdout + {log_file}")

    # whether to use distributed data parallel
    use_ddp = num_gpus > 1
    if use_ddp:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = setup_ddp(rank, world_size)
        logger.addFilter(RankFilter())
    else:
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:{rank}")

    torch.cuda.set_device(device)
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"World_size: {world_size}")

    args = load_config(env_config_path, model_config_path, model_def_path)
    if "use_region_contrasive_loss" not in args.controlnet_train.keys():
        args.use_region_contrasive_loss = False
    else:
        args.use_region_contrasive_loss = args.controlnet_train["use_region_contrasive_loss"]
        for k in ["region_contrasive_loss_delta", "region_contrasive_loss_weight"]:
            if k not in args.controlnet_train.keys():
                raise ValueError(f"Since 'use_region_contrasive_loss' is in 'controlnet_train' of {model_config_path}, we need 'region_contrasive_loss_delta' and 'region_contrasive_loss_weight' also be in it.")
    
    logger.info(f"use_region_contrasive_loss: {args.use_region_contrasive_loss}")
    if args.use_region_contrasive_loss:
        logger.warning(f"User sets 'use_region_contrasive_loss' as true in {model_config_path}.")
        logger.warning("********************")
        logger.warning(
            "Please check remove_roi() in train_controlnet.py to ensure ROI is removed as intended; "
            "default logic will not match your requirement."
        )
        logger.warning("********************")


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
    # Log ControlNet structure to verify Conditioning Embedding is included
    logger.info(f"ControlNet type: {type(controlnet).__name__}")
    # Check if Conditioning Embedding exists
    has_cond_emb = hasattr(controlnet, "conditioning_embedding")
    logger.info(f"Has Conditioning Embedding: {has_cond_emb}")
    if has_cond_emb:
        cond_emb = controlnet.conditioning_embedding
        logger.info(f"Conditioning Embedding: {type(cond_emb).__name__}")
        logger.info(f"  - Input channels: {getattr(cond_emb, 'in_channels', 'N/A')}")
        if hasattr(cond_emb, 'out_channels'):
            logger.info(f"  - Output channels: {cond_emb.out_channels}")

    # copy weights from the DM to the controlnet
    copy_model_state(controlnet, unet.state_dict())
    # load trained controlnet model if it is provided
    start_epoch = 0
    if args.existing_ckpt_filepath is not None:
        if not os.path.exists(args.existing_ckpt_filepath):
            raise ValueError("Please download the trained ControlNet checkpoint.")
        checkpoint = torch.load(args.existing_ckpt_filepath, map_location=device, weights_only=False)
        controlnet.load_state_dict(checkpoint["controlnet_state_dict"])

        # Load fine-tuned UNet if available (for Stage 2/3 continuation)
        if "unet_finetuned_state_dict" in checkpoint:
            unet.load_state_dict(checkpoint["unet_finetuned_state_dict"])
            logger.info(f"Loaded fine-tuned UNet from {args.existing_ckpt_filepath}")

        # Resume from checkpoint epoch
        start_epoch = checkpoint.get("epoch", 0)
        logger.info(f"load trained controlnet model from {args.existing_ckpt_filepath}")
        logger.info(f"resuming from epoch {start_epoch}")
    else:
        logger.info("train controlnet model from scratch.")
    # we freeze the parameters of the diffusion model.
    for p in unet.parameters():
        p.requires_grad = False

    # Verify freezing and log trainable parameter count
    unet_trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    controlnet_trainable = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    total_trainable = controlnet_trainable + unet_trainable

    logger.info(f"=== Parameter Freezing Status ===")
    logger.info(f"DiT/U-Net trainable parameters: {unet_trainable:,} (should be 0)")
    logger.info(f"ControlNet trainable parameters: {controlnet_trainable:,}")
    logger.info(f"Total trainable parameters: {total_trainable:,}")

    if unet_trainable > 0:
        raise ValueError("ERROR: DiT/U-Net has trainable parameters! Check freezing logic.")

    # Critical fix for Stage 4: Disable inplace operations when deep encoder is unfrozen
    def disable_inplace_recursive(module, name="root"):
        """Recursively disable inplace operations to prevent gradient graph corruption."""
        count = 0
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}"
            if hasattr(child, 'inplace') and child.inplace:
                child.inplace = False
                logger.info(f"  Disabled inplace: {full_name} ({type(child).__name__})")
                count += 1
            count += disable_inplace_recursive(child, full_name)
        return count

    # Check if we're unfreezing deep encoder (Stage 4)
    unfreeze_layers = args.controlnet_train.get('finetune_unfreeze_layers', [])
    has_deep_encoder = any('down_blocks.2' in name or 'down_blocks.3' in name for name in unfreeze_layers)

    if has_deep_encoder:
        logger.info("=== Stage 4 Detected: Disabling inplace operations ===")
        unet_count = disable_inplace_recursive(unet, "unet")
        controlnet_count = disable_inplace_recursive(controlnet, "controlnet")
        logger.info(f"Total inplace ops disabled: UNet={unet_count}, ControlNet={controlnet_count}")

    # Fine-tuning mode: selectively unfreeze U-Net layers if configured
    def unfreeze_unet_layers(unet, layer_names):
        """
        Unfreeze specific U-Net layers for fine-tuning.

        Args:
            unet: DiT/U-Net model
            layer_names: List of layer prefixes to unfreeze (e.g., ['up_blocks.2', 'up_blocks.3'])

        Returns:
            Number of newly trainable parameters
        """
        newly_trainable = 0
        for name, param in unet.named_parameters():
            if param.requires_grad:
                continue  # Already trainable
            for layer_name in layer_names:
                if name.startswith(layer_name):
                    param.requires_grad = True
                    newly_trainable += param.numel()
                    logger.info(f"  Unfrozen: {name} ({param.numel():,} params)")

        return newly_trainable

    # Check if fine-tuning mode is enabled
    if 'finetune_unfreeze_layers' in args.controlnet_train:
        unfreeze_layers = args.controlnet_train['finetune_unfreeze_layers']
        new_params = unfreeze_unet_layers(unet, unfreeze_layers)

        # Re-count trainable parameters
        unet_trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        controlnet_trainable = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
        total_trainable = controlnet_trainable + unet_trainable

        logger.info(f"=== Fine-Tuning Mode: Stage {args.controlnet_train.get('finetune_stage', 1)} ===")
        logger.info(f"Newly unfrozen: {new_params:,} parameters")
        logger.info(f"Total trainable: {total_trainable:,} parameters")
        logger.info(f"  ControlNet: {controlnet_trainable:,}")
        logger.info(f"  DiT/U-Net: {unet_trainable:,}")

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
    logger.info(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")

    # Step 3: training config
    weighted_loss = args.controlnet_train["weighted_loss"]
    weighted_loss_label = args.controlnet_train["weighted_loss_label"]

    # Top-K Loss config (hard example mining)
    use_topk = args.controlnet_train.get("use_topk_loss", False)
    topk_ratio = args.controlnet_train.get("topk_ratio", 0.3)
    if use_topk:
        logger.info(f"=== Top-K Hard Example Mining Loss ENABLED ===")
        logger.info(f"  Top-K ratio: {topk_ratio*100:.1f}% (hardest pixels only)")
        logger.info(f"  Expected loss range: ~1.5-2.5 (higher than standard L1!)")

    # Build optimizer with differential learning rates if in fine-tuning mode
    if 'finetune_stage' in args.controlnet_train:
        # Fine-tuning mode: separate LRs for ControlNet and UNet
        controlnet_lr = args.controlnet_train.get('finetune_controlnet_lr', 1e-4)
        unet_lr = args.controlnet_train.get('finetune_unet_lr', 1e-5)

        # Collect parameters by group
        if world_size > 1:
            controlnet_params = list(controlnet.module.parameters())
        else:
            controlnet_params = list(controlnet.parameters())
        unet_params = [p for p in unet.parameters() if p.requires_grad]

        param_groups = [
            {'params': controlnet_params, 'lr': controlnet_lr, 'name': 'controlnet'},
            {'params': unet_params, 'lr': unet_lr, 'name': 'unet_finetune'}
        ]

        optimizer = torch.optim.AdamW(param_groups)
        logger.info(f"Differential learning rates:")
        logger.info(f"  ControlNet: {controlnet_lr:.2e}")
        logger.info(f"  UNet finetune: {unet_lr:.2e}")
    else:
        # Original: ControlNet only
        if world_size > 1:
            optimizer = torch.optim.AdamW(params=controlnet.module.parameters(), lr=args.controlnet_train["lr"])
        else:
            optimizer = torch.optim.AdamW(params=controlnet.parameters(), lr=args.controlnet_train["lr"])

    total_steps = (args.controlnet_train["n_epochs"] * len(train_loader.dataset)) / args.controlnet_train["batch_size"]
    logger.info(f"total number of training steps: {total_steps}.")

    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)

    def should_validate(epoch, val_schedule):
        """Determine if validation should run this epoch based on schedule."""
        for epoch_range_str, frequency in val_schedule.items():
            start_end = epoch_range_str.split('-')
            start, end = int(start_end[0]), int(start_end[1])
            if start <= epoch < end:
                return (epoch - start) % frequency == 0
        return True  # Default: validate every epoch

    # Step 4: training
    n_epochs = args.controlnet_train["n_epochs"]
    scaler = GradScaler("cuda")
    total_step = 0
    best_val_loss = 1e4

    if weighted_loss > 1.0:
        logger.info(f"apply weighted loss = {weighted_loss} on labels: {weighted_loss_label}")

    controlnet.train()
    unet.eval()
    prev_time = time.time()
    for epoch in range(start_epoch, n_epochs):
        epoch_loss_ = 0
        epoch_loss_history = []
        for step, batch in enumerate(train_loader):
            # get image embedding and label mask and scale image embedding by the provided scale_factor
            images = batch["image"].to(device) * scale_factor
            labels = batch["label"].to(device)
            if labels.shape[1] != 1:
                raise ValueError(f"We expect labels with shape [B,1,X,Y,Z], yet got {labels.shape}")

            # Get pre images for controlnet conditioning (if available)
            pre_images = batch.get("pre", None)
            if pre_images is not None:
                pre_images = pre_images.to(device)

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

            if args.use_region_contrasive_loss:
                labels_roi_free = remove_roi(labels)

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
                (
                    model_output,
                    model_block1_output,
                    model_block2_output
                ) = compute_model_output(
                    images,labels,noise,timesteps,noise_scheduler,
                    controlnet,unet,
                    spacing_tensor,
                    modality_tensor,
                    top_region_index_tensor,
                    bottom_region_index_tensor,
                    pre_images=pre_images,  # Pass pre images as conditioning
                    return_controlnet_blocks=False
                )
                if args.use_region_contrasive_loss:
                    (
                        model_output_roi_free,
                        model_block1_output_roi_free,
                        model_block2_output_roi_free,
                    ) = compute_model_output(
                        images,labels_roi_free,noise,timesteps,noise_scheduler,
                        controlnet,unet,
                        spacing_tensor,
                        modality_tensor,
                        top_region_index_tensor,
                        bottom_region_index_tensor,
                        pre_images=pre_images,  # Pass pre images as conditioning
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
    
                if weighted_loss > 1.0:
                    weights = torch.ones_like(images).to(images.device)
                    roi = torch.zeros([noise_shape[0]] + [1] + noise_shape[2:]).to(images.device)
                    # Convert labels to float before interpolation (Long not supported by F.interpolate)
                    interpolate_label = F.interpolate(labels.float(), size=images.shape[2:], mode="nearest")
                    # Dilation to compensate VAE receptive field edge effects
                    # kernel_size=5 provides 2-voxel dilation (~8mm physical space per side)
                    interpolate_label = F.max_pool3d(
                        interpolate_label.float(),
                        kernel_size=5,
                        stride=1,
                        padding=2
                    )
                    interpolate_label = (interpolate_label > 0.5).float()  # Re-binarize
                    # assign larger weights for ROI (tumor)
                    for label in weighted_loss_label:
                        mask = (interpolate_label.squeeze(1) == label).unsqueeze(1)  # Add back channel dim
                        roi[mask] = 1
                    weights[roi.repeat(1, images.shape[1], 1, 1, 1) == 1] = weighted_loss

                    # Check if to use ROI Intensity Loss (Ibarra et al. strategy)
                    use_roi_intensity = args.controlnet_train.get("use_roi_intensity_loss", False)
                    roi_intensity_weight = args.controlnet_train.get("roi_intensity_weight", 1.0)

                    # Check if to use Background False Positive Penalty (Hard Negative Mining)
                    use_bg_penalty = args.controlnet_train.get("use_bg_penalty", False)
                    bg_penalty_weight = args.controlnet_train.get("bg_penalty_weight", 5.0)

                    # Global weighted L1 Loss (crucial: keeps background pure black!)
                    l1_loss_raw = F.l1_loss(model_output.float(), model_gt.float(), reduction="none")
                    loss_global = (l1_loss_raw * weights).mean()

                    if use_roi_intensity:
                        # ROI Intensity Loss from Ibarra et al. with Asymmetric Contrast
                        # Use dilated mask as ROI (already computed above)
                        roi_mask = (interpolate_label > 0.5)  # Boolean mask for tumor region

                        # Ensure batch has tumor pixels to avoid NaN
                        if roi_mask.sum() > 0:
                            # Expand mask to match latent channels
                            roi_mask_expanded = roi_mask.repeat(1, images.shape[1], 1, 1, 1)

                            # Extract predictions and GT in ROI
                            pred_roi = model_output.float()[roi_mask_expanded]
                            gt_roi = model_gt.float()[roi_mask_expanded]

                            # 1. Mean intensity constraint (macro-level regulation)
                            # Keep L1 for mean matching (stable gradient for global brightness)
                            loss_intensity = F.l1_loss(pred_roi.mean(), gt_roi.mean())

                            # 2. MSE Contrast Loss - squared penalty on spatial distribution!
                            # MSE heavily penalizes large errors: (1.0 - 0.1)^2 = 0.81 vs L1 = 0.9
                            # Gradient: MSE gives 2*(pred-gt), stronger correction for outliers
                            # This prevents model from "cheating" by spreading signal thinly
                            loss_contrast = F.mse_loss(pred_roi, gt_roi)

                            # Combined ROI loss with balanced weight on contrast
                            # MSE already gives stronger gradients, reduce weight to 0.1
                            loss_roi_total = loss_intensity + 0.1 * loss_contrast
                        else:
                            loss_roi_total = torch.tensor(0.0, device=model_output.device)

                        loss = loss_global + roi_intensity_weight * loss_roi_total

                        # Background False Positive Penalty (Hard Negative Mining)
                        # Asymmetric penalty: ONLY punish background pixels that are "too white" (pred > gt)
                        if use_bg_penalty and roi_mask.sum() > 0:
                            # Background region is where GT label is 0
                            bg_mask_expanded = (~roi_mask).repeat(1, images.shape[1], 1, 1, 1)
                            pred_bg = model_output.float()[bg_mask_expanded]
                            gt_bg = model_gt.float()[bg_mask_expanded]

                            # F.relu ensures we ONLY punish pred > gt (false positives), not pred < gt
                            # This is asymmetric: under-prediction in background is OK, over-prediction is BAD
                            false_positive_bg = F.relu(pred_bg - gt_bg)

                            # Squared penalty: larger errors get exponentially harsher punishment
                            # pred=0.01 when gt=0: penalty=0.0001 (negligible)
                            # pred=0.10 when gt=0: penalty=0.0100 (10x worse)
                            # pred=0.30 when gt=0: penalty=0.0900 (900x worse!)
                            loss_bg_penalty = (false_positive_bg ** 2).mean()

                            loss = loss + bg_penalty_weight * loss_bg_penalty

                        # Log individual loss components every 50 steps
                        if total_step % 50 == 0:
                            if isinstance(loss_roi_total, torch.Tensor):
                                log_msg = f"  Loss Components - Global L1: {loss_global.item():.4f}, ROI Total: {loss_roi_total.item():.4f}"
                                if use_bg_penalty and 'loss_bg_penalty' in locals():
                                    log_msg += f", BG Penalty: {loss_bg_penalty.item():.4f}"
                                log_msg += f", Combined: {loss.item():.4f}"
                                logger.info(log_msg)
                                tensorboard_writer.add_scalar("train/loss_roi_total", loss_roi_total.item(), total_step)
                                if use_bg_penalty and 'loss_bg_penalty' in locals():
                                    tensorboard_writer.add_scalar("train/loss_bg_penalty", loss_bg_penalty.item(), total_step)
                            else:
                                logger.info(f"  Loss Components - Global L1: {loss_global.item():.4f}, ROI Total: 0.0000, Combined: {loss.item():.4f}")
                            tensorboard_writer.add_scalar("train/loss_global", loss_global.item(), total_step)
                    else:
                        # Standard weighted L1 loss only
                        loss = loss_global
                else:
                    loss = F.l1_loss(model_output.float(), model_gt.float())
    
    
                if args.use_region_contrasive_loss:
                    roi_contrastive = (labels_roi_free != labels).to(torch.uint8)  # 0/1 mask
                    roi_contrastive_bg = 1 - dilate(roi_contrastive, filter_size=3).to(torch.uint8)
                    loss_region_contrasive, loss_region_bg = compute_region_contrasive_loss(
                        model_output,model_output_roi_free,model_gt,
                        roi_contrastive,roi_contrastive_bg,
                        max_region_contrasive_loss=args.controlnet_train["region_contrasive_loss_delta"],
                        loss_contrastive = torch.nn.L1Loss(reduction = 'none')
                    )
                    final_loss_region_contrasive = loss_region_contrasive + loss_region_bg
                    logger.info(f"loss_region_contrasive: {loss_region_contrasive}")
                    logger.info(f"loss_region_bg: {loss_region_bg}")
                    loss += args.controlnet_train["region_contrasive_loss_weight"]*final_loss_region_contrasive

            scaler.scale(loss).backward()

            # Gradient clipping to prevent numerical instability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_norm=1.0)

            # Verify gradient flow (first few steps only)
            if total_step < 5:
                unet_has_grad = any(p.grad is not None for p in unet.parameters())
                if unet_has_grad:
                    logger.warning("WARNING: DiT/U-Net has gradients! Check freezing logic.")

            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            total_step += 1

            if rank == 0:
                # write train loss for each batch into tensorboard
                tensorboard_writer.add_scalar(
                    "train/train_controlnet_loss_iter", loss.detach().cpu().item(), total_step
                )

            # Track loss history
            epoch_loss_history.append(loss.detach().cpu().item())

            # Log every 10 batches with rolling average
            if rank == 0 and (step + 1) % 10 == 0:
                recent_mean = np.mean(epoch_loss_history[-10:])
                recent_std = np.std(epoch_loss_history[-10:])
                logger.info(
                    f"[Epoch {epoch+1:3d}/{n_epochs}] "
                    f"[Batch {step+1:3d}/{len(train_loader)}] "
                    f"[LR: {lr_scheduler.get_last_lr()[0]:.2e}] "
                    f"[Loss: {recent_mean:.4f} ±{recent_std:.2f}]"
                )
            epoch_loss_ += loss.detach()

        epoch_loss = epoch_loss_ / (step + 1)

        if use_ddp:
            dist.barrier()
            dist.all_reduce(epoch_loss, op=torch.distributed.ReduceOp.AVG)

        if rank == 0:
            tensorboard_writer.add_scalar("train/train_controlnet_loss_epoch", epoch_loss.cpu().item(), total_step)

            # Get state dict once for all saves
            controlnet_state_dict = controlnet.module.state_dict() if world_size > 1 else controlnet.state_dict()

            # Save epoch checkpoint (each epoch separately)
            # Include UNet state dict if in fine-tuning mode
            save_dict = {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "controlnet_state_dict": controlnet_state_dict,
            }

            # Add fine-tuning metadata
            if 'finetune_stage' in args.controlnet_train:
                save_dict["unet_finetuned_state_dict"] = unet.state_dict()
                save_dict["finetune_stage"] = args.controlnet_train.get("finetune_stage", 0)
                save_dict["finetune_unfreeze_layers"] = args.controlnet_train.get("finetune_unfreeze_layers", [])
                logger.info(f"Saving fine-tuned UNet ({len(save_dict['finetune_unfreeze_layers'])} layers)")

            torch.save(save_dict, f"{args.model_dir}/{args.exp_name}_epoch_{epoch+1}.pt")
            logger.info(f"Save epoch {epoch+1} model to {args.model_dir}/{args.exp_name}_epoch_{epoch+1}.pt")

        # ==================== Validation Loop ====================
        # Get validation schedule (default to every epoch)
        val_schedule = args.controlnet_train.get('val_frequency_schedule', {'0-999': 1})

        # Only validate if scheduled
        if should_validate(epoch, val_schedule):
            controlnet.eval()
            val_loss_sum = 0.0
            val_steps = 0

            logger.info(f"Running validation on {len(val_loader)} batches...")

            with torch.no_grad():
                for val_step, val_batch in enumerate(val_loader):
                    # Get validation data
                    val_images = val_batch["image"].to(device) * scale_factor
                    val_labels = val_batch["label"].to(device)
                    val_pre_images = val_batch.get("pre", None)
                    if val_pre_images is not None:
                        val_pre_images = val_pre_images.to(device)

                    val_spacing_tensor = val_batch["spacing"].to(device)
                    val_top_region_index_tensor = None
                    val_bottom_region_index_tensor = None
                    val_modality_tensor = None

                    if include_body_region:
                        val_top_region_index_tensor = val_batch["top_region_index"].to(device)
                        val_bottom_region_index_tensor = val_batch["bottom_region_index"].to(device)
                    if include_modality:
                        val_modality_tensor = val_batch["modality"].to(device)

                    # Sample timesteps for validation (use fixed timesteps for consistency)
                    if isinstance(noise_scheduler, RFlowScheduler):
                        val_timesteps = noise_scheduler.sample_timesteps(val_images)
                    else:
                        val_timesteps = torch.randint(
                            0, noise_scheduler.num_train_timesteps, (val_images.shape[0],), device=val_images.device
                        ).long()

                    # Sample noise
                    val_noise = torch.randn(list(val_images.shape), dtype=val_images.dtype).to(device)

                    # Compute model output
                    with autocast("cuda", enabled=True):
                        (val_model_output, _, _) = compute_model_output(
                            val_images, val_labels, val_noise, val_timesteps, noise_scheduler,
                            controlnet, unet,
                            val_spacing_tensor,
                            val_modality_tensor,
                            val_top_region_index_tensor,
                            val_bottom_region_index_tensor,
                            pre_images=val_pre_images,
                            return_controlnet_blocks=False
                        )

                        # Compute ground truth based on prediction type
                        if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                            val_model_gt = val_noise
                        elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                            val_model_gt = val_images
                        elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                            val_model_gt = val_images - val_noise
                        else:
                            raise ValueError("Unknown prediction type")

                        # Compute weighted loss (same as training)
                        if weighted_loss > 1.0:
                            val_weights = torch.ones_like(val_images).to(val_images.device)
                            val_roi = torch.zeros([val_images.shape[0], 1, *val_images.shape[2:]], device=val_images.device)
                            val_interpolate_label = F.interpolate(val_labels.float(), size=val_images.shape[2:], mode="nearest")
                            val_interpolate_label = F.max_pool3d(
                                val_interpolate_label.float(),
                                kernel_size=5,
                                stride=1,
                                padding=2
                            )
                            val_interpolate_label = (val_interpolate_label > 0.5).float()
                            for label in weighted_loss_label:
                                val_mask = (val_interpolate_label.squeeze(1) == label).unsqueeze(1)
                                val_roi[val_mask] = 1
                            val_weights[val_roi.repeat(1, val_images.shape[1], 1, 1, 1) == 1] = weighted_loss
                            val_loss = (F.l1_loss(val_model_output.float(), val_model_gt.float(), reduction="none") * val_weights).mean()
                        else:
                            val_loss = F.l1_loss(val_model_output.float(), val_model_gt.float())

                    # Check for NaN loss and skip this batch if detected
                    loss_item = val_loss.detach().item()
                    if not np.isnan(loss_item):
                        val_loss_sum += loss_item
                        val_steps += 1
                    else:
                        logger.warning(f"WARNING: NaN detected in validation batch {val_step}, skipping...")

            # Compute average validation loss
            if val_steps > 0:
                avg_val_loss = val_loss_sum / val_steps
            else:
                avg_val_loss = 0.0

            if use_ddp:
                dist.barrier()
                # Aggregate validation loss across GPUs
                val_loss_tensor = torch.tensor([avg_val_loss], device=device)
                dist.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.AVG)
                avg_val_loss = val_loss_tensor.item()

            if rank == 0:
                tensorboard_writer.add_scalar("val/val_controlnet_loss_epoch", avg_val_loss, total_step)
                logger.info(f"Epoch {epoch + 1} - Train Loss: {epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

                # Update best validation loss checkpoint
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    logger.info(f"*** New best validation loss -> {best_val_loss:.4f} ***")

                    # Save best checkpoint with UNet if in fine-tuning mode
                    save_dict = {
                        "epoch": epoch + 1,
                        "train_loss": epoch_loss,
                        "val_loss": best_val_loss,
                        "controlnet_state_dict": controlnet_state_dict,
                    }

                    if 'finetune_stage' in args.controlnet_train:
                        save_dict["unet_finetuned_state_dict"] = unet.state_dict()
                        save_dict["finetune_stage"] = args.controlnet_train.get("finetune_stage", 0)
                        save_dict["finetune_unfreeze_layers"] = args.controlnet_train.get("finetune_unfreeze_layers", [])

                    torch.save(save_dict, f"{args.model_dir}/{args.exp_name}_best.pt")
                    logger.info(f"Save best validation model to {args.model_dir}/{args.exp_name}_best.pt")

            # Set back to training mode
            controlnet.train()
            # Aggressive cache clearing after validation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        else:
            logger.info(f"Skipping validation this epoch (scheduled)")
            controlnet.train()
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

