#!/bin/bash
#SBATCH --job-name=breast_ft_s4_5_bg
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm_finetune_s4_5_bg_penalty_%j.log
#SBATCH --error=logs/slurm_finetune_s4_5_bg_penalty_%j.err

mkdir -p logs

echo "=========================================="
echo "Stage 4.5 BG Penalty - 背景橡皮擦策略"
echo "=========================================="
echo "Problem: Stage 4.5 预测过于扩散 (5.4x GT)"
echo "         Outside signal 76% (背景太亮)"
echo ""
echo "Solution: 非对称背景假阳性惩罚"
echo "  - 正常 weighted_loss: 3.0 (保持tumor focus)"
echo "  - ROI weight: 1.0 (保持MSE contrast)"
echo "  - NEW: bg_penalty_weight: 5.0"
echo ""
echo "Mechanism:"
echo "  loss_bg = relu(pred_bg - gt_bg) ^ 2"
echo "  - 只惩罚 pred > gt (过度预测)"
echo "  - 不惩罚 pred < gt (允许不足)"
echo "  - 平方惩罚 (越白罚越狠)"
echo ""
echo "Expected effect:"
echo "  - 背景杂讯被强行擦除 (橡皮擦效应)"
echo "  - Outside signal: 76% → <60%"
echo "  - Spatial spread: 5.4x → <4.0x"
echo "  - 保持 tumor core 改善 (top 5% > 45%)"
echo ""
echo "Resume from: Stage 4.5 Epoch 147"
echo "Target: Epoch 160 (12 more epochs)"
echo "=========================================="
echo "Started at: $(date)"
echo "Node: $SLURM_NODELIST"
echo ""

source $(conda info --base)/etc/profile.d/conda.sh
conda activate breast_gen
export PYTHONUNBUFFERED=1

# Stage 4.5 with Background False Positive Penalty
python -m scripts.train_controlnet \
    -e configs/environment_breast_sub_finetune_stage4_5_bg_penalty.json \
    -c configs/config_breast_controlnet_finetune_stage4_5_bg_penalty.json \
    -t configs/config_network_rflow.json

echo ""
echo "Finished at: $(date)"
exit $?
