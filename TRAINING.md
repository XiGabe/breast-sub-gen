# Training Documentation - 3D Breast MRI Subtraction Generation

## 项目目标 (Project Goal)

训练双通道 ControlNet 生成高保真 3D 乳腺 MRI 减影图像

**架构**: 3D Locator (nnU-Net) + MAISI ControlNet (双通道: Pre+Mask) + Progressive Unfreezing

**Last Updated**: 2026-03-04

---

## 当前状态 (Current Status)

- [x] 数据预处理完成 (data/step_4)
- [x] 离线缓存完成 (embeddings_breast_sub, processed_mask, processed_pre)
- [x] 数据集JSON生成 (dataset_breast_cached.json)
- [x] VAE权重加载 (weights/autoencoder_v2.pt)
- [x] Diffusion U-Net权重加载 (weights/diff_unet_3d_rflow-mr.pt)
- [x] **阶段零完成**：基础设施与安全防雷
- [x] **阶段二完成**：网络重构与极简 Loss 大清洗
- [x] **阶段三完成**：MAISI 高保真渲染器渐进式微调 (Shell Script 分阶段方案)

---

## 阶段划分 (Training Stages)

### ✅ 阶段零：基础设施与安全防雷 (已完成)

#### 0.1 确认 VAE 归一化与离线缓存
- Target 减影图 (Sub) 经过 VAE 编码为 64³ 潜变量
- Raw VAE latents (range ~[-8, 9]) 仅需乘以 scale_factor (~0.18) 即可直接送入模型
- Sub 存为 `sub_emb.nii.gz` (64³, 4通道)
- 条件图 Pre 和 Mask 不经过 VAE，保留原物理分辨率 256³

#### 0.2 阉割空间数据增强 (Data Augmentation)
- 删除 Transform 中所有空间形变（RandomCrop, RandomAffine, RandomFlip）
- 仅对 Pre 施加强度增强（RandGaussianNoise 15%, RandAdjustContrast 20%）

#### 0.3 生成按病例划分的 Dataset JSON
- 结构：`{"image": "sub_emb.nii.gz", "pre": "...", "label": "mask.nii.gz"}`
- 严格执行三层分层抽样（数据集 → 肿瘤状态 → 患者ID）
- **数据分布统计**:
  - 总样本: 1,943 | 总患者: 1,504
  - Train: 1,553 样本 (80%), 1,203 患者, 有肿瘤 77.4%, 无肿瘤 22.6%
  - Val: 390 样本 (20%), 301 患者, 有肿瘤 77.2%, 无肿瘤 22.8%
  - ✓ 无患者级别数据泄漏

### 阶段一：级联前置网络 - 3D 肿瘤定位器训练

- [ ] **1.1 选择并配置定位器网络**
  - 工具：nnU-Net v2 或 MONAI DynUNet
  - 输入/输出：1ch 256³ Pre-contrast → 1ch 256³ Predicted Mask

- [ ] **1.2 设置极度不平衡分割 Loss**
  - 使用 `Dice Loss + Focal Loss` 组合

- [ ] **1.3 评估预测结果**

### ✅ 阶段二：网络重构与极简 Loss 大清洗 (已完成)

#### ✅ 2.1 修改网络配置 (`configs/config_network_rflow.json`)
- `conditioning_embedding_in_channels` 从 8 改为 **2** (双通道: Pre+Mask)

#### ✅ 2.2 重构 ControlNet 输入端与扰动增强 (`scripts/train_controlnet.py`)
```python
# 双通道输入构造
pre_images = batch["pre"].to(device)  # [B, 1, 256, 256, 256]
masks = batch["label"].to(device)      # [B, 1, 256, 256, 256]

# 30%概率进行Mask形态学扰动（纯PyTorch实现）
if torch.rand(1).item() < 0.3:
    masks = apply_random_morphological_perturbation(masks)

# 拼接为 [B, 2, 256, 256, 256]
controlnet_cond = torch.cat([pre_images.float(), masks.float()], dim=1)
```

#### ✅ 2.3 执行 The Great Cleanup (纯净版 Loss 代码植入)

**已删除**:
- 原有的 `F.interpolate` 标签降维（会丢失微小病灶）
- Region Contrastive Loss（2x 慢 + OOM 风险）
- MONAI 原生形态学操作（前向循环不稳定）

**已实现**:
```python
# 1. 精准降维 (使用 Max Pooling 保证任何极小病灶像素都不被丢失！)
roi_mask_latent = F.max_pool3d(labels.float(), kernel_size=4, stride=4) > 0.0

# 2. 基础 Global L1 (局部适度加权 3.0)
weights = torch.ones_like(model_output)
weights[roi_mask_latent.repeat(1, model_output.shape[1], 1, 1, 1)] = 3.0
l1_loss_raw = F.l1_loss(model_output.float(), model_gt.float(), reduction="none")
loss = (l1_loss_raw * weights).mean()

# 3. 绝对背景惩罚 (Background Penalty - 严查假阳性糊团)
if roi_mask_latent.sum() > 0:
    bg_mask_expanded = (~roi_mask_latent).repeat(1, model_output.shape[1], 1, 1, 1)
else:
    bg_mask_expanded = torch.ones_like(model_output, dtype=torch.bool)

pred_bg = model_output.float()[bg_mask_expanded]
gt_bg = model_gt.float()[bg_mask_expanded]
false_positive_bg = F.relu(pred_bg - gt_bg)  # 仅惩罚 pred > gt
loss = loss + 5.0 * (false_positive_bg ** 2).mean()
```

### ✅ 阶段三：MAISI 高保真渲染器渐进式微调 (已完成)

**设计决策**: Shell 脚本分阶段（避免优化器重建导致的动量丢失）

#### Stage 3.1: ControlNet 独立对齐期 (50 epochs)
- [x] 目标：U-Net 全冻结。ControlNet 专心学双通道映射。
- [x] 配置：`configs/config_maisi_controlnet_train_stage1.json`
- [x] 脚本：`scripts/submit_stage1.sh`
- [x] 超参：ControlNet LR `1e-4`, U-Net 全冻结, 每 2 epoch 验证

#### Stage 3.2: 深层语义放行期 (50 epochs)
- [x] 目标：解决亮度不足。解冻深层特征路由，释放高光异常值。
- [x] 配置：`configs/config_maisi_controlnet_train_stage2.json`
- [x] 脚本：`scripts/submit_stage2.sh`
- [x] 超参：U-Net LR `5e-5`, 解冻 `down_blocks.2`, `down_blocks.3`, `middle_block`, `up_blocks`

#### Stage 3.3: 浅层边缘精雕期 (50 epochs)
- [x] 目标：解决边缘模糊，生成逼真异质性强化。
- [x] 配置：`configs/config_maisi_controlnet_train_stage3.json`
- [x] 脚本：`scripts/submit_stage3.sh`
- [x] 超参：U-Net LR `1e-5`, 全部 U-Net 块解冻

**Checkpoint 策略**:
- 每个 epoch 保存独立文件: `{exp_name}_epoch_{N}.pt`
- `best.pt` 跟踪最佳 validation loss (无验证时用 training loss)

**使用方式**:
```bash
sbatch scripts/submit_stage1.sh  # Stage 1
sbatch scripts/submit_stage2.sh  # Stage 2 (依赖 stage1_best.pt)
sbatch scripts/submit_stage3.sh  # Stage 3 (依赖 stage2_best.pt)
```

### 阶段四：全自动临床推理闭环

- [ ] **4.1 无 CFG 纯净推理**
  - 推理时强制 `cfg_guidance_scale = 1.0` (模型未训练 mask dropout)

- [ ] **4.2 级联推理流水线**
  - Step 1: 未知 Pre-contrast 输入 Locator → `Predicted Mask`
  - Step 2: Pre + Predicted Mask 输入 Renderer → `Predicted Subtraction`
  - Step 3: `Post_syn = Pre_raw + Sub_pred`

---

## Network Architecture

### ControlNet Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `conditioning_embedding_in_channels` | **2** | Dual-channel input: [Pre-contrast MRI, Tumor Mask] |
| `conditioning_embedding_num_channels` | [8, 32, 64] | Hidden channels in conditioning embedding |
| `latent_channels` | 4 | VAE latent channels |
| `num_channels` | [64, 128, 256, 512] | U-Net channel progression |
| `num_class_embeds` | 128 | Modality embedding classes |

### Input/Output Shapes

| Component | Shape | Description |
|-----------|-------|-------------|
| Pre-contrast MRI | [B, 1, 256, 256, 256] | Physical resolution input |
| Tumor Mask | [B, 1, 256, 256, 256] | Binary segmentation mask |
| ControlNet Condition | [B, 2, 256, 256, 256] | Concatenated [Pre, Mask] |
| VAE Latent (target) | [B, 4, 64, 64, 64] | Pre-encoded subtraction |
| Model Output | [B, 4, 64, 64, 64] | Predicted subtraction latent |

---

## Loss Function

### Implemented: The Great Cleanup (Pure L1 + Background Penalty)

**Why this works**:
- `F.max_pool3d`: Any single tumor voxel → ROI preserved (no tiny lesion loss)
- ROI weight 3.0: Moderate focus on tumor regions
- Background penalty: Squared ReLU on false positives → suppresses snow noise

**Removed**: Region Contrastive Loss (was 2x slower + caused OOM)

---

## Training Configuration

### Current Setup

```json
{
  "controlnet_train": {
    "batch_size": 4,
    "cache_rate": 0.0,
    "fold": 0,
    "lr": 1e-4,
    "n_epochs": 150
  }
}
```

### Files

- **Network**: `configs/config_network_rflow.json`
- **Training**: `configs/config_maisi_controlnet_train_rflow-mr_breast.json`
- **Environment**: `configs/environment_maisi_controlnet_train_rflow-mr_breast.json`
- **Modality**: `configs/modality_mapping_breast.json`

---

## Data Augmentation

### Morphological Perturbation (30% probability)

Pure PyTorch implementation (no MONAI native ops):
- **Dilation**: `F.max_pool3d` with padding
- **Erosion**: `1 - F.max_pool3d(1 - mask)`
- **Kernel sizes**: 1 or 3 (odd only)
- **Applied to**: Tumor masks only (not pre-images)

**Why pure PyTorch?** MONAI's morphology ops are designed for DataLoader transforms and can cause C++ operator errors or memory leaks in forward loops with gradients.

### What NOT to do

- ❌ **NO spatial augmentation**: No crop, flip, rotate, zoom
- ❌ **NO intensity augmentation**: Pre-images already normalized
- ❌ **NO mask dropout**: Model learns from natural 22% no-tumor samples

---

## Running Training

### Activate Environment

```bash
conda activate breast_gen
```

### Single GPU

```bash
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast.json \
    --model_config_path configs/config_maisi_controlnet_train_rflow-mr_breast.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1
```

### Multi-GPU (DDP)

```bash
python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast.json \
    --model_config_path configs/config_maisi_controlnet_train_rflow-mr_breast.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 4
```

### Monitoring

**TensorBoard logs**: `outputs/tfevent/{exp_name}/`
```bash
tensorboard --logdir outputs/tfevent
```

**Training logs**: `outputs/logs/{exp_name}/train.log`
- Records every 100 batches and epoch-level summaries
- Useful for long-term training history and debugging

**Note**: When using `sbatch`, additional logs will be created by Slurm (e.g., `slurm-*.out`)

---

## 修改日志 (Modification Log)

### 2026-03-04 - Checkpoint 策略与 Validation 支持

**Completed Tasks**:
- ✅ 每个 epoch 保存独立 checkpoint (`_epoch_{N}.pt`)
- ✅ 移除 `_current.pt`（不再覆盖式保存）
- ✅ 添加 validation loop（每 N epoch）
- ✅ `best.pt` 基于 validation loss（无验证时 fallback 到 training loss）
- ✅ 配置参数：`validation_frequency` (默认: 2)

**Files Modified**:
- `scripts/train_controlnet.py`: Checkpoint 保存逻辑, validation loop
- `configs/config_maisi_controlnet_train_stage{1,2,3}.json`: 添加 `validation_frequency: 2`

### 2026-03-04 - Stage 3 渐进式微调实现 (Shell Script 分阶段方案)

**Completed Tasks**:
- ✅ `set_unet_frozen_state()` 函数：选择性冻结/解冻 U-Net 块
- ✅ 配置驱动：`unet_blocks_to_unfreeze`, `unet_lr`, `disable_inplace_for_checkpointing`
- ✅ 双学习率优化器：ControlNet 和 U-Net 独立 LR
- ✅ Checkpoint 保存/加载：支持 U-Net state_dict
- ✅ 推理脚本更新：`infer_controlnet.py` 加载微调 U-Net
- ✅ 3 个 Stage 配置文件 + 3 个提交脚本

**Files Created**:
- `configs/config_maisi_controlnet_train_stage{1,2,3}.json`
- `configs/environment_maisi_controlnet_train_rflow-mr_breast_stage{1,2,3}.json`
- `scripts/submit_stage{1,2,3}.sh`

**Files Modified**:
- `scripts/train_controlnet.py`: U-Net 冻结/解冻, 双 LR, Checkpoint 管理
- `scripts/infer_controlnet.py`: 加载微调 U-Net

### 2026-03-04 - NaN Loss 自动跳过机制

**Completed Tasks**:
- ✅ Added NaN/Inf loss detection before backward pass
- ✅ Skip problematic batches automatically without crashing training
- ✅ Diagnostic logging: checks which input (images/labels/pre_images) contains NaN
- ✅ Epoch summary reports skipped batch count
- ✅ Epoch average loss computed from valid batches only

**Behavior**:
- When NaN loss detected: batch skipped, warning logged with input diagnostics
- Training continues uninterrupted
- Epoch end: logs number of skipped batches

**Files Modified**:
- `scripts/train_controlnet.py`: Added NaN detection and skip logic

### 2026-03-04 - 日志输出系统升级

**Completed Tasks**:
- ✅ Added file logging with `setup_logging()` in `diff_model_setting.py`
- ✅ Log files organized by experiment name: `outputs/logs/{exp_name}/train.log`
- ✅ Every 100 batches logged to file (manageable file size)
- ✅ Console output: every batch (real-time monitoring)
- ✅ Epoch summaries and model saving events logged

**Files Modified**:
- `scripts/diff_model_setting.py`: Enhanced `setup_logging()` with file handler support
- `scripts/train_controlnet.py`: Added 100-batch file logging, separated console/file output

### 2026-03-04 - Stage 2 Complete (网络重构与Loss大清洗)

**Completed Tasks**:
- ✅ Dual-channel input implementation (2 channels: Pre + Mask)
- ✅ Pure PyTorch morphology ops (no MONAI native APIs)
- ✅ Max pooling loss (tiny lesion preservation)
- ✅ Background penalty (false positive suppression)
- ✅ Removed Region Contrastive Loss (2x faster, no OOM)
- ✅ Inference code updated for dual-channel
- ✅ CFG guidance validation warnings added

**Files Modified**:
- `configs/config_network_rflow.json`: `conditioning_embedding_in_channels: 2`
- `scripts/train_controlnet.py`: Dual-channel input, pure PyTorch morphology, new loss
- `scripts/sample.py`: Dual-channel inference, CFG warnings
- `scripts/infer_controlnet.py`: Dual-channel inference
- `scripts/utils.py`: Fixed fold handling for datasets without fold field

### 2026-03-04 - Data Verification

- Confirmed embeddings_breast_sub contains raw VAE latents ([-8, 9])
- Corrected documentation: only scale_factor needed, NOT * 2.0 - 1.0
- Data status: 1,943 files in each directory (embeddings, masks, pre)
- All processing: [0, 1] normalized; masks binary

---

## Important Configuration Values

### 数据规格 (Data Specs)
- 输入分辨率: 256×256×256
- 潜空间分辨率: 64×64×64
- Spacing: [0.7, 0.7, 1.2] mm³
- VAE scale_factor: 需从 checkpoint 加载
- 朝向: RAS

### 训练超参数 (Training Hyperparameters)
- Batch size: 2-4 (根据 GPU 显存调整)
- ControlNet LR: 1e-4 → 5e-5 → 1e-5 (三阶段)
- U-Net LR: 0 → 5e-5 → 1e-5 (渐进解冻)
- 推理步数: 30
- CFG guidance_scale: 0 或 1.0 (不要 > 1.0!)

---

## Common Issues

### Warnings (Safe to Ignore)

- `Orientationd` default value changed: INFO only, no action needed
- `DataLoader will create 8 worker processes`: INFO only, training works fine
- `lr_scheduler.step() before optimizer.step()`: First LR value skipped, negligible impact

### Errors to Fix

- **CUDA OOM**: Reduce `batch_size` or `cache_rate`
- **Shape mismatch in morphological perturbation**: Already fixed with fallback resize
- **KeyError: 'fold'**: Already fixed in `utils.py` add_data_dir2path()

---

## Checkpoint Management

### Saved Files

- `models/{exp_name}_epoch_{N}.pt`: Per-epoch checkpoint (independent, not overwritten)
- `models/{exp_name}_best.pt`: Best checkpoint (lowest val loss, or train loss if no val)

### Example (Stage 1, 50 epochs)

```
models/
├── breast_controlnet_stage1_epoch_1.pt
├── breast_controlnet_stage1_epoch_2.pt
├── ...
├── breast_controlnet_stage1_epoch_50.pt
└── breast_controlnet_stage1_best.pt
```

### Loading Checkpoints

```python
checkpoint = torch.load("models/breast_controlnet_stage1_best.pt")
controlnet.load_state_dict(checkpoint["controlnet_state_dict"])

# Load fine-tuned U-Net if available (Stage 2+)
if checkpoint["unet_state_dict"] is not None:
    unet.load_state_dict(checkpoint["unet_state_dict"], strict=False)
```

---

## References

- CLAUDE.md: Project overview and code guidance
- configs/config_network_rflow.json: Network architecture
- scripts/train_controlnet.py: Main training loop
