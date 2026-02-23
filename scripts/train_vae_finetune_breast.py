import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from monai.utils import set_determinism
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from monai.losses import PatchAdversarialLoss

# 引入修正后的变换逻辑
from transforms_breast import VAE_Transform  
# 修正 1: 适配最新 MONAI 的感知损失路径
from monai.losses.perceptual import PerceptualLoss

def main():
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 VAE 微调任务，使用设备: {device}")

    env_config_path = "breast_project/configs/environment_maisi_vae_train.json"
    hyper_config_path = "breast_project/configs/config_maisi_vae_train.json"

    with open(env_config_path, "r") as f:
        env_dict = json.load(f)
    with open(hyper_config_path, "r") as f:
        hyper_dict = json.load(f)

    opt = hyper_dict["autoencoder_train"]
    data_opt = hyper_dict["data_option"]

    os.makedirs(env_dict["model_dir"], exist_ok=True)
    os.makedirs(env_dict["tfevent_path"], exist_ok=True)
    tensorboard_writer = SummaryWriter(env_dict["tfevent_path"])

    print("📦 正在加载并缓存数据集...")
    train_datalist = load_decathlon_datalist(env_dict["dataset_json"], True, "training")
    val_datalist = load_decathlon_datalist(env_dict["dataset_json"], True, "validation")

    train_transforms = VAE_Transform(
        is_train=True, 
        random_aug=data_opt["random_aug"], 
        patch_size=opt["patch_size"],
        spacing_type=data_opt["spacing_type"],
        spacing=data_opt["spacing"]
    )
    val_transforms = VAE_Transform(
        is_train=False, 
        random_aug=False, 
        val_patch_size=opt["val_patch_size"],
        spacing_type=data_opt["spacing_type"],
        spacing=data_opt["spacing"]
    )

    train_ds = CacheDataset(data=train_datalist, transform=train_transforms, cache_rate=opt["cache"])
    val_ds = CacheDataset(data=val_datalist, transform=val_transforms, cache_rate=opt["cache"])

    train_loader = DataLoader(train_ds, batch_size=opt["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=opt["val_batch_size"], shuffle=False, num_workers=2)

    print("🧠 正在初始化 MAISI VAE (v2 架构)...")
    model = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        channels=(64, 128, 256),
        num_res_blocks=2,
        norm_num_groups=32,
        attention_levels=(False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    ).to(device)

    # 修正 2: 跨版本权重翻译官逻辑
    if env_dict.get("finetune", False):
        ckpt_path = env_dict["trained_autoencoder_path"]
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            raw_state_dict = checkpoint.get('unet_state_dict', checkpoint.get('state_dict', checkpoint))
            adapted_state_dict = {}
            for k, v in raw_state_dict.items():
                new_k = k
                if "decoder.blocks.3.conv.conv.conv" in new_k:
                    new_k = new_k.replace("decoder.blocks.3.conv.conv.conv", "decoder.blocks.3.postconv.conv")
                elif "decoder.blocks.6.conv.conv.conv" in new_k:
                    new_k = new_k.replace("decoder.blocks.6.conv.conv.conv", "decoder.blocks.6.postconv.conv")
                new_k = new_k.replace(".conv.conv.", ".conv.")
                adapted_state_dict[new_k] = v
            model.load_state_dict(adapted_state_dict, strict=True)
            print(f"✅ 成功跨版本注入预训练权重: {ckpt_path}")

    # 修正 3: num_channels -> channels
    discriminator = PatchDiscriminator(
        spatial_dims=3, 
        num_layers_d=3, 
        channels=32, 
        in_channels=1, 
        out_channels=1
    ).to(device)

    l1_loss = torch.nn.L1Loss()
    perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).to(device)
    adv_loss = PatchAdversarialLoss(criterion="least_squares")

    optimizer_g = torch.optim.Adam(model.parameters(), lr=opt["lr"])
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=opt["lr"])

    # 适配最新 GradScaler API
    scaler_g = torch.amp.GradScaler('cuda', enabled=(opt["amp"] and device.type == 'cuda'))
    scaler_d = torch.amp.GradScaler('cuda', enabled=(opt["amp"] and device.type == 'cuda'))

    # 初始化最佳 Loss
    best_val_loss = float("inf")
    
    print(f"🔥 开始微调训练，总 Epochs: {opt['n_epochs']}")
    for epoch in range(1, opt["n_epochs"] + 1):
        model.train()
        discriminator.train()
        epoch_loss = 0
        
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            optimizer_g.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(opt["amp"] and device.type == 'cuda')):
                reconstruction, z_mu, z_sigma = model(images)
                recon_loss = l1_loss(reconstruction, images)
                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4]).mean()
                p_loss = perceptual_loss(reconstruction.float(), images.float())
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = recon_loss + opt["kl_weight"] * kl_loss + opt["perceptual_weight"] * p_loss + opt["adv_weight"] * generator_loss

            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            optimizer_d.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(opt["amp"] and device.type == 'cuda')):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                loss_d = (loss_d_fake + loss_d_real) * 0.5

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            epoch_loss += recon_loss.item()
            if step % 50 == 0:
                print(f"Epoch {epoch} Step {step} - Train Recon Loss: {recon_loss.item():.4f}")

        # 计算并写入平均 Train Loss
        epoch_loss /= len(train_loader)
        tensorboard_writer.add_scalar("train_recon_loss_epoch", epoch_loss, epoch)

        # ==========================================================
        # 完整的 Validation 与模型保存逻辑
        # ==========================================================
        if epoch % opt["val_interval"] == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_step, val_batch in enumerate(val_loader):
                    val_images = val_batch["image"].to(device)
                    with torch.amp.autocast('cuda', enabled=(opt["amp"] and device.type == 'cuda')):
                        val_recon, _, _ = model(val_images)
                        val_loss += l1_loss(val_recon, val_images).item()
                    
                    # 取验证集第一张图的中心切片发送到 TensorBoard 供明早肉眼查看
                    if val_step == 0:
                        mid_z = val_images.shape[4] // 2
                        # 提取单通道切片 [1, H, W]
                        orig_slice = val_images[0, 0, :, :, mid_z].unsqueeze(0).cpu().numpy()
                        recon_slice = val_recon[0, 0, :, :, mid_z].unsqueeze(0).cpu().numpy()
                        tensorboard_writer.add_image("val_orig_img_Zcenter", orig_slice, epoch)
                        tensorboard_writer.add_image("val_recon_img_Zcenter", recon_slice, epoch)
                        
            val_loss /= len(val_loader)
            tensorboard_writer.add_scalar("val_recon_loss_epoch", val_loss, epoch)
            print(f"⭐️ Epoch {epoch} 验证结束 - 平均 Val Recon Loss: {val_loss:.4f}")

            # 核心：保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(env_dict["model_dir"], "best_autoencoder.pt")
                torch.save(model.state_dict(), save_path)
                print(f"💾 [突破！] 新的最佳模型已保存至: {save_path} (Val Loss: {best_val_loss:.4f})")
            
            # 顺手保存一个最新状态，防止最后几个 epoch 虽不是最佳但你想用
            latest_path = os.path.join(env_dict["model_dir"], "latest_autoencoder.pt")
            torch.save(model.state_dict(), latest_path)

    print("🎉 训练完成！")

if __name__ == "__main__":
    main()