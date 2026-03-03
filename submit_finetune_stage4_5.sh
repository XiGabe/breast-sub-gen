#!/bin/bash
#SBATCH --job-name=breast_ft_s4_5
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_finetune_s4_5_%j.log
#SBATCH --error=logs/slurm_finetune_s4_5_%j.err

mkdir -p logs

echo "=========================================="
echo "Stage 4.5 - 激进破局阶段 (Aggressive Breakthrough)"
echo "=========================================="
echo "Features:"
echo "  - Unfreeze: down_blocks.2,3 + middle + ALL up_blocks"
echo "  - UNet LR: 1e-5 → 1e-4 (同频 ControlNet)"
echo "  - Tumor Weight: 10.0 → 3.0 (释放高强度预测)"
echo "  - ROI Weight: 0.5 → 1.0 (MSE Contrast 接管)"
echo "  - Batch Size: 4 (梯度更稳定)"
echo "  - Resume from: Stage 4 Epoch 142"
echo "  - Target: Epoch 160 (18 more epochs)"
echo "=========================================="
echo "Started at: $(date)"
echo "Node: $SLURM_NODELIST"
echo ""

source $(conda info --base)/etc/profile.d/conda.sh
conda activate breast_gen
export PYTHONUNBUFFERED=1

# Stage 4.5: 激进破局 - 高LR + 低TumorWeight
python -m scripts.train_controlnet \
    -e configs/environment_breast_sub_finetune_stage4_5.json \
    -c configs/config_breast_controlnet_finetune_stage4_5.json \
    -t configs/config_network_rflow.json \
    -g 1

echo ""
echo "Finished at: $(date)"
exit $?
