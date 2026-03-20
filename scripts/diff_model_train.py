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

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import monai
import torch
import torch.distributed as dist
from monai.data import DataLoader, partition_dataset
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from monai.transforms import Compose
from monai.utils import first
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance

def augment_modality_label(modality_tensor, prob=0.1):
    """
    Augments the modality tensor by randomly modifying certain elements based on a given probability.

    - A proportion of elements (determined by `prob`) are randomly set to 0.
    - Elements equal to 2 or 3 are randomly set to 1 with a probability defined by `prob`.
    - Elements between 9 and 12 are randomly set to 8 with a probability defined by `prob`.

    Parameters:
    modality_tensor (torch.Tensor): A tensor containing modality labels.
    prob (float): The probability of modifying certain elements (should be between 0 and 1).
                  For example, if `prob` is 0.3, there's a 30% chance of modification.

    Returns:
    torch.Tensor: The modified modality tensor with the applied augmentations.
    """
    # Randomly set elements that are smaller than 8 with probability `prob`
    mask_ct = (modality_tensor < 8) & (modality_tensor >= 2)
    prob_ct = torch.rand(modality_tensor.size(),device=modality_tensor.device) < prob
    modality_tensor[mask_ct & prob_ct] = 1
    
    # Randomly set elements larger than 9 with probability `prob`
    mask_mri = (modality_tensor >= 9)
    prob_mri = torch.rand(modality_tensor.size(),device=modality_tensor.device) < prob
    modality_tensor[mask_mri & prob_mri] = 8

    # Randomly set a proportion (prob) of the elements to 0
    mask_zero = torch.rand(modality_tensor.size(),device=modality_tensor.device) > prob
    modality_tensor = modality_tensor * mask_zero.long()
    
    return modality_tensor



def load_filenames(data_list_path: str, split: str = "training") -> list:
    """
    Load filenames from the JSON data list.

    Args:
        data_list_path (str): Path to the JSON data list file.
        split (str): Which split to load - "training" or "validation".

    Returns:
        list: List of filenames.
    """
    with open(data_list_path, "r") as file:
        json_data = json.load(file)
    if split not in json_data:
        return []
    filenames = json_data[split]
    return [_item["image"] for _item in filenames]


def prepare_data(
    train_files: list,
    device: torch.device,
    cache_rate: float,
    num_workers: int = 2,
    batch_size: int = 1,
    include_body_region: bool = False,
    include_modality: bool = True,
    modality_mapping: dict = None
) -> DataLoader:
    """
    Prepare training data.

    Args:
        train_files (list): List of training files.
        device (torch.device): Device to use for training.
        cache_rate (float): Cache rate for dataset.
        num_workers (int): Number of workers for data loading.
        batch_size (int): Mini-batch size.
        include_body_region (bool): Whether to include body region in data

    Returns:
        DataLoader: Data loader for training.
    """

    def _load_spacing(x):
        """Load spacing from direct value or file path."""
        if isinstance(x, list):
            return torch.FloatTensor(x) * 1e2
        elif isinstance(x, str):
            with open(x) as f:
                return torch.FloatTensor(json.load(f)["spacing"]) * 1e2
        return x

    def _load_modality(x, modality_mapping):
        """Load modality from direct value or file path."""
        if isinstance(x, str):
            # Direct string value like "mri"
            return modality_mapping.get(x, x)
        elif isinstance(x, int):
            return x
        return modality_mapping.get(x, x)

    train_transforms_list = [
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
        monai.transforms.Lambdad(keys="spacing", func=lambda x: _load_spacing(x)),
    ]
    if include_body_region:
        train_transforms_list += [
            monai.transforms.Lambdad(keys="top_region_index", func=lambda x: torch.FloatTensor(x) * 1e2 if isinstance(x, list) else x),
            monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: torch.FloatTensor(x) * 1e2 if isinstance(x, list) else x),
        ]
    if include_modality:
         train_transforms_list += [
             monai.transforms.Lambdad(
                keys="modality", func=lambda x: _load_modality(x, modality_mapping)
             ),
             monai.transforms.EnsureTyped(keys=['modality'], dtype=torch.long),
         ]
    train_transforms = Compose(train_transforms_list)

    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers
    )

    return DataLoader(train_ds, num_workers=6, batch_size=batch_size, shuffle=True)


def load_unet(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> torch.nn.Module:
    """
    Load the UNet model.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load the model on.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.nn.Module: Loaded UNet model.
    """
    unet = define_instance(args, "diffusion_unet_def").to(device)
    unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)

    if dist.is_initialized():
        unet = DistributedDataParallel(unet, device_ids=[device], find_unused_parameters=True)

    if args.existing_ckpt_filepath is None:
        logger.info("Training from scratch.")
    else:
        checkpoint_unet = torch.load(f"{args.existing_ckpt_filepath}", map_location=device, weights_only=False)
        
        if dist.is_initialized():
            unet.module.load_state_dict(checkpoint_unet["unet_state_dict"], strict=False)
        else:
            unet.load_state_dict(checkpoint_unet["unet_state_dict"], strict=False)
        logger.info(f"Pretrained checkpoint {args.existing_ckpt_filepath} loaded.")

    return unet


def calculate_scale_factor(train_loader: DataLoader, device: torch.device, logger: logging.Logger) -> torch.Tensor:
    """
    Calculate the scaling factor for the dataset.

    Args:
        train_loader (DataLoader): Data loader for training.
        device (torch.device): Device to use for calculation.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.Tensor: Calculated scaling factor.
    """
    check_data = first(train_loader)
    z = check_data["image"].to(device)
    scale_factor = 1 / torch.std(z)
    logger.info(f"Scaling factor set to {scale_factor}.")

    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    logger.info(f"scale_factor -> {scale_factor}.")
    return scale_factor


def create_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    Create optimizer for training.

    Args:
        model (torch.nn.Module): Model to optimize.
        lr (float): Learning rate.

    Returns:
        torch.optim.Optimizer: Created optimizer.
    """
    return torch.optim.Adam(params=model.parameters(), lr=lr)


def create_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.PolynomialLR:
    """
    Create learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        total_steps (int): Total number of training steps.

    Returns:
        torch.optim.lr_scheduler.PolynomialLR: Created learning rate scheduler.
    """
    return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)


def train_one_epoch(
    epoch: int,
    unet: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.PolynomialLR,
    loss_pt: torch.nn.L1Loss,
    scaler: GradScaler,
    scale_factor: torch.Tensor,
    noise_scheduler: torch.nn.Module,
    num_images_per_batch: int,
    num_train_timesteps: int,
    device: torch.device,
    logger: logging.Logger,
    local_rank: int,
    amp: bool = True,
) -> torch.Tensor:
    """
    Train the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        unet (torch.nn.Module): UNet model.
        train_loader (DataLoader): Data loader for training.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler (torch.optim.lr_scheduler.PolynomialLR): Learning rate scheduler.
        loss_pt (torch.nn.L1Loss): Loss function.
        scaler (GradScaler): Gradient scaler for mixed precision training.
        scale_factor (torch.Tensor): Scaling factor.
        noise_scheduler (torch.nn.Module): Noise scheduler.
        num_images_per_batch (int): Number of images per batch.
        num_train_timesteps (int): Number of training timesteps.
        device (torch.device): Device to use for training.
        logger (logging.Logger): Logger for logging information.
        local_rank (int): Local rank for distributed training.
        amp (bool): Use automatic mixed precision training.

    Returns:
        torch.Tensor: Training loss for the epoch.
    """
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    if local_rank == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch + 1}, lr {current_lr}.")

    _iter = 0
    loss_torch = torch.zeros(2, dtype=torch.float, device=device)

    unet.train()
    for train_data in train_loader:
        current_lr = optimizer.param_groups[0]["lr"]

        _iter += 1
        images = train_data["image"].to(device)
        images = images * scale_factor

        if include_body_region:
            top_region_index_tensor = train_data["top_region_index"].to(device)
            bottom_region_index_tensor = train_data["bottom_region_index"].to(device)
        if include_modality:
            modality_tensor = train_data["modality"].to(device)     
            modality_tensor = augment_modality_label(modality_tensor).to(device)

        spacing_tensor = train_data["spacing"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=amp):
            noise = torch.randn_like(images)

            if isinstance(noise_scheduler, RFlowScheduler):
                timesteps = noise_scheduler.sample_timesteps(images)
            else:
                timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()

            noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

            # Create a dictionary to store the inputs
            unet_inputs = {
                "x": noisy_latent,
                "timesteps": timesteps,
                "spacing_tensor": spacing_tensor,
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

            loss = loss_pt(model_output.float(), model_gt.float())

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        loss_torch[0] += loss.item()
        loss_torch[1] += 1.0

        if local_rank == 0:
            logger.info(
                "[{0}] epoch {1}, iter {2}/{3}, loss: {4:.4f}, lr: {5:.12f}.".format(
                    str(datetime.now())[:19], epoch + 1, _iter, len(train_loader), loss.item(), current_lr
                )
            )

    if dist.is_initialized():
        dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

    return loss_torch


def validate(
    epoch: int,
    unet: torch.nn.Module,
    val_loader: DataLoader,
    loss_pt: torch.nn.L1Loss,
    scale_factor: torch.Tensor,
    noise_scheduler: torch.nn.Module,
    num_train_timesteps: int,
    device: torch.device,
    logger: logging.Logger,
    local_rank: int,
) -> torch.Tensor:
    """
    Validate the model on validation set.

    Args:
        epoch (int): Current epoch number.
        unet (torch.nn.Module): UNet model.
        val_loader (DataLoader): Data loader for validation.
        loss_pt (torch.nn.L1Loss): Loss function.
        scale_factor (torch.Tensor): Scaling factor.
        noise_scheduler (torch.nn.Module): Noise scheduler.
        num_train_timesteps (int): Number of training timesteps.
        device (torch.device): Device to use for validation.
        logger (logging.Logger): Logger for logging information.
        local_rank (int): Local rank for distributed training.

    Returns:
        torch.Tensor: Validation loss.
    """
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    _iter = 0
    loss_torch = torch.zeros(2, dtype=torch.float, device=device)

    unet.eval()
    for val_data in val_loader:
        _iter += 1
        images = val_data["image"].to(device)
        images = images * scale_factor

        if include_body_region:
            top_region_index_tensor = val_data["top_region_index"].to(device)
            bottom_region_index_tensor = val_data["bottom_region_index"].to(device)
        if include_modality:
            modality_tensor = val_data["modality"].to(device)

        spacing_tensor = val_data["spacing"].to(device)

        with torch.no_grad():
            noise = torch.randn_like(images)

            if isinstance(noise_scheduler, RFlowScheduler):
                timesteps = noise_scheduler.sample_timesteps(images)
            else:
                timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()

            noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

            unet_inputs = {
                "x": noisy_latent,
                "timesteps": timesteps,
                "spacing_tensor": spacing_tensor,
            }
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

            if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                model_gt = noise
            elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                model_gt = images
            elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                model_gt = images - noise
            else:
                raise ValueError(
                    "noise scheduler prediction type has to be chosen from ",
                    f"[{DDPMPredictionType.EPSILON},{DDPMPredictionType.SAMPLE},{DDPMPredictionType.V_PREDICTION}]",
                )

            loss = loss_pt(model_output.float(), model_gt.float())

        loss_torch[0] += loss.item()
        loss_torch[1] += 1.0

    if dist.is_initialized():
        dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

    return loss_torch


def save_checkpoint(
    epoch: int,
    unet: torch.nn.Module,
    loss_torch_epoch: float,
    num_train_timesteps: int,
    scale_factor: torch.Tensor,
    ckpt_folder: str,
    args: argparse.Namespace,
    is_best: bool = False,
    val_loss: float = None,
) -> None:
    """
    Save checkpoint.

    Args:
        epoch (int): Current epoch number.
        unet (torch.nn.Module): UNet model.
        loss_torch_epoch (float): Training loss for the epoch.
        num_train_timesteps (int): Number of training timesteps.
        scale_factor (torch.Tensor): Scaling factor.
        ckpt_folder (str): Checkpoint folder path.
        args (argparse.Namespace): Configuration arguments.
        is_best (bool): Whether this is the best model based on validation loss.
        val_loss (float): Validation loss (if available).
    """
    unet_state_dict = unet.module.state_dict() if dist.is_initialized() else unet.state_dict()
    save_dict = {
        "epoch": epoch + 1,
        "loss": loss_torch_epoch,
        "num_train_timesteps": num_train_timesteps,
        "scale_factor": scale_factor,
        "unet_state_dict": unet_state_dict,
    }
    if val_loss is not None:
        save_dict["val_loss"] = val_loss

    # Save each epoch checkpoint with epoch number (e.g., epoch_001.pt)
    epoch_filename = args.model_filename.replace(".pt", f"_epoch_{epoch + 1:03d}.pt")
    torch.save(
        save_dict,
        f"{ckpt_folder}/{epoch_filename}",
    )

    # Save best model if validation loss improved
    if is_best:
        best_filename = args.model_filename.replace(".pt", "_best.pt")
        torch.save(
            save_dict,
            f"{ckpt_folder}/{best_filename}",
        )


def diff_model_train(
    env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int, amp: bool = True
) -> None:
    """
    Main function to train a diffusion model.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
        num_gpus (int): Number of GPUs to use for training.
        amp (bool): Use automatic mixed precision training.
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("training")

    logger.info(f"Using {device} of {world_size}")

    if local_rank == 0:
        logger.info(f"[config] ckpt_folder -> {args.model_dir}.")
        logger.info(f"[config] data_root -> {args.embedding_base_dir}.")
        logger.info(f"[config] data_list -> {args.json_data_list}.")
        logger.info(f"[config] lr -> {args.diffusion_unet_train['lr']}.")
        logger.info(f"[config] num_epochs -> {args.diffusion_unet_train['n_epochs']}.")
        logger.info(f"[config] num_train_timesteps -> {args.noise_scheduler['num_train_timesteps']}.")

        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    unet = load_unet(args, device, logger)
    noise_scheduler = define_instance(args, "noise_scheduler")
    include_body_region = unet.include_top_region_index_input
    include_modality = (unet.num_class_embeds is not None)
    if include_modality:
        with open(args.modality_mapping_path, "r") as f:
            args.modality_mapping = json.load(f)
    else:
        args.modality_mapping = None

    # Load full dataset JSON to get metadata (spacing, modality)
    with open(args.json_data_list, "r") as f:
        dataset_json = json.load(f)

    train_data_list = dataset_json.get("training", [])
    val_data_list = dataset_json.get("validation", [])

    if local_rank == 0:
        logger.info(f"num_files_train: {len(train_data_list)}")
        logger.info(f"num_files_val: {len(val_data_list)}")

    train_files = []
    for item in train_data_list:
        str_img = os.path.join(args.embedding_base_dir, item["image"])
        if not os.path.exists(str_img):
            continue

        train_files_i = {
            "image": str_img,
            "spacing": item["spacing"],
            "modality": item["modality"]
        }
        train_files.append(train_files_i)

    val_files = []
    for item in val_data_list:
        str_img = os.path.join(args.embedding_base_dir, item["image"])
        if not os.path.exists(str_img):
            continue

        val_files_i = {
            "image": str_img,
            "spacing": item["spacing"],
            "modality": item["modality"]
        }
        val_files.append(val_files_i)

    if dist.is_initialized():
        train_files = partition_dataset(
            data=train_files, shuffle=True, num_partitions=dist.get_world_size(), even_divisible=True
        )[local_rank]
        # For validation, we don't need distributed partitioning for single GPU

    train_loader = prepare_data(
        train_files,
        device,
        args.diffusion_unet_train["cache_rate"],
        batch_size=args.diffusion_unet_train["batch_size"],
        include_body_region=include_body_region,
        include_modality=include_modality,
        modality_mapping = args.modality_mapping
    )

    # Create validation DataLoader (only on rank 0 for single GPU or for validation)
    val_loader = None
    if len(val_files) > 0:
        val_loader = prepare_data(
            val_files,
            device,
            args.diffusion_unet_train["cache_rate"],
            batch_size=args.diffusion_unet_train["batch_size"],
            include_body_region=include_body_region,
            include_modality=include_modality,
            modality_mapping=args.modality_mapping
        )

    scale_factor = calculate_scale_factor(train_loader, device, logger)
    optimizer = create_optimizer(unet, args.diffusion_unet_train["lr"])

    total_steps = (args.diffusion_unet_train["n_epochs"] * len(train_loader.dataset)) / args.diffusion_unet_train[
        "batch_size"
    ]
    lr_scheduler = create_lr_scheduler(optimizer, total_steps)
    loss_pt = torch.nn.L1Loss()
    scaler = GradScaler("cuda")

    torch.set_float32_matmul_precision("highest")
    logger.info("torch.set_float32_matmul_precision -> highest.")

    # Track best validation loss
    best_val_loss = float("inf")
    is_best = False

    for epoch in range(args.diffusion_unet_train["n_epochs"]):
        loss_torch = train_one_epoch(
            epoch,
            unet,
            train_loader,
            optimizer,
            lr_scheduler,
            loss_pt,
            scaler,
            scale_factor,
            noise_scheduler,
            args.diffusion_unet_train["batch_size"],
            args.noise_scheduler["num_train_timesteps"],
            device,
            logger,
            local_rank,
            amp=amp,
        )

        loss_torch = loss_torch.tolist()
        train_loss = loss_torch[0] / loss_torch[1] if loss_torch[1] > 0 else 0

        # Validation
        val_loss = None
        if val_loader is not None:
            val_loss_torch = validate(
                epoch,
                unet,
                val_loader,
                loss_pt,
                scale_factor,
                noise_scheduler,
                args.noise_scheduler["num_train_timesteps"],
                device,
                logger,
                local_rank,
            )
            val_loss_torch = val_loss_torch.tolist()
            val_loss = val_loss_torch[0] / val_loss_torch[1] if val_loss_torch[1] > 0 else 0

            # Check if this is the best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                logger.info(f"epoch {epoch + 1} - New best model! val_loss: {val_loss:.4f}")

        if torch.cuda.device_count() == 1 or local_rank == 0:
            log_msg = f"epoch {epoch + 1} train loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f", val loss: {val_loss:.4f}"
            logger.info(log_msg)

            save_checkpoint(
                epoch,
                unet,
                train_loss,
                args.noise_scheduler["num_train_timesteps"],
                scale_factor,
                args.model_dir,
                args,
                is_best=is_best,
                val_loss=val_loss,
            )

    if local_rank == 0:
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
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
    parser.add_argument("--no_amp", dest="amp", action="store_false", help="Disable automatic mixed precision training")

    args = parser.parse_args()
    diff_model_train(args.env_config_path, args.model_config_path, args.model_def_path, args.num_gpus, args.amp)
