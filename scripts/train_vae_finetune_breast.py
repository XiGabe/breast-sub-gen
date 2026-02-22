import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from monai.utils import set_determinism
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from monai.losses import PatchAdversarialLoss

# 导入你之前写好的、专门处理乳腺 256^3 补齐与裁剪的 Transform
from transforms_breast import VAE_Transform  

# (如果环境里没有 lpips，需运行 pip install lpips)
from generative.losses import PerceptualLoss

def main():
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 VAE 微调任务，使用设备: {device}")

    # ==========================================
    # 1. 加载配置 (Configs)
    # ==========================================
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

    # ==========================================
    # 2. 数据流水线 (Data Pipeline)
    # ==========================================
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

    # ==========================================
    # 3. 初始化网络与预训练权重注入 (Networks & Weights)
    # ==========================================
    print("🧠 正在初始化 MAISI VAE (v2 架构)...")
    model = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=4,        # MAISI 必须是 4
        channels=(64, 128, 256),  # MAISI 架构通道数
        num_res_blocks=2,
        norm_num_groups=32,
        attention_levels=(False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    ).to(device)

    # 载入官方权重进行 Fine-tune
    if env_dict.get("finetune", False):
        ckpt_path = env_dict["trained_autoencoder_path"]
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            # 兼容 NVIDIA MAISI 特殊的键名 'unet_state_dict'
            state_dict = checkpoint.get('unet_state_dict', checkpoint.get('state_dict', checkpoint))
            model.load_state_dict(state_dict, strict=True)
            print(f"✅ 成功注入预训练权重: {ckpt_path}")
        else:
            raise FileNotFoundError(f"❌ 找不到权重文件，请检查路径: {ckpt_path}")

    discriminator = PatchDiscriminator(
        spatial_dims=3, 
        num_layers_d=3, 
        num_channels=32, 
        in_channels=1, 
        out_channels=1
    ).to(device)

    # ==========================================
    # 4. 损失函数与优化器 (Losses & Optimizers)
    # ==========================================
    l1_loss = torch.nn.L1Loss()
    perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).to(device)
    adv_loss = PatchAdversarialLoss(criterion="least_squares")

    optimizer_g = torch.optim.Adam(model.parameters(), lr=opt["lr"])
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=opt["lr"])

    scaler_g = torch.cuda.amp.GradScaler(enabled=opt["amp"])
    scaler_d = torch.cuda.amp.GradScaler(enabled=opt["amp"])

    # ==========================================
    # 5. 训练循环 (Training Loop)
    # ==========================================
    n_epochs = opt["n_epochs"]
    val_interval = opt["val_interval"]
    best_val_loss = float("inf")
    global_step = 0

    print(f"🔥 开始微调训练，总 Epochs: {n_epochs}")
    for epoch in range(1, n_epochs + 1):
        model.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            optimizer_g.zero_grad(set_to_none=True)

            # --- 生成器 (Generator/VAE) 更新 ---
            with torch.cuda.amp.autocast(enabled=opt["amp"]):
                reconstruction, z_mu, z_sigma = model(images)
                
                recon_loss = l1_loss(reconstruction, images)
                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                p_loss = perceptual_loss(reconstruction.float(), images.float())

                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

                # 综合 Loss (严格遵循官方权重)
                loss_g = recon_loss + opt["kl_weight"] * kl_loss + opt["perceptual_weight"] * p_loss + opt["adv_weight"] * generator_loss

            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            # --- 判别器 (Discriminator) 更新 ---
            optimizer_d.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=opt["amp"]):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                loss_d = (loss_d_fake + loss_d_real) * 0.5

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            epoch_loss += recon_loss.item()
            gen_epoch_loss += generator_loss.item()
            global_step += 1

            tensorboard_writer.add_scalar("train_recon_loss", recon_loss.item(), global_step)

        print(f"Epoch {epoch}/{n_epochs} - Train Recon Loss: {epoch_loss / len(train_loader):.4f}")

        # --- 验证循环 (Validation) ---
        if epoch % val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_step, val_batch in enumerate(val_loader):
                    val_images = val_batch["image"].to(device)
                    with torch.cuda.amp.autocast(enabled=opt["amp"]):
                        val_recon, _, _ = model(val_images)
                        val_loss += l1_loss(val_recon, val_images).item()
            
            val_loss /= len(val_loader)
            tensorboard_writer.add_scalar("val_recon_loss", val_loss, epoch)
            print(f"⭐️ Epoch {epoch} - Val Recon Loss: {val_loss:.4f}")

            # 保存当前最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(env_dict["model_dir"], "best_autoencoder.pt")
                torch.save(model.state_dict(), save_path)
                print(f"💾 新的最佳模型已保存至: {save_path}")

    print("🎉 训练完成！")

if __name__ == "__main__":
    main()