#!/bin/bash
#SBATCH --job-name=breast_ft_s4
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_finetune_s4_%j.log
#SBATCH --error=logs/slurm_finetune_s4_%j.err

mkdir -p logs

echo "=========================================="
echo "Breast Fine-Tuning Stage 4 - Deep Encoder"
echo "=========================================="
echo "Features:"
echo "  - Unfreeze: down_blocks.2,3 + middle + ALL up_blocks (FULL decoder + deep encoder)"
echo "  - Fix: Clone residuals to avoid inplace operation errors"
echo "  - Loss: MSE Contrast + Global L1 (tumor 10x)"
echo "  - Batch size: 2 (gradient checkpointing disabled)"
echo "  - UNet LR: 1e-5 | ControlNet LR: 1e-4"
echo "  - Resume from: Stage 3 Epoch 130"
echo "  - Target: Epoch 150 (20 more epochs)"
echo "=========================================="
echo "Started at: $(date)"
echo "Node: $SLURM_NODELIST"
echo ""

source $(conda info --base)/etc/profile.d/conda.sh
conda activate breast_gen
export PYTHONUNBUFFERED=1

# Stage 4: Unfreeze deep encoder (down_blocks.2,3) to preserve sparse high-intensity features
python -m scripts.train_controlnet \
    -e configs/environment_breast_sub_finetune_stage4.json \
    -c configs/config_breast_controlnet_finetune_stage4.json \
    -t configs/config_network_rflow.json \
    -g 1

echo ""
echo "Finished at: $(date)"
exit $?
