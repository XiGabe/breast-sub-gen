#!/bin/bash
#SBATCH --job-name=breast_ft_s3
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_finetune_s3_%j.log
#SBATCH --error=logs/slurm_finetune_s3_%j.err

mkdir -p logs

echo "=========================================="
echo "Breast Fine-Tuning Stage 3 - Full Unfreeze"
echo "=========================================="
echo "Features:"
echo "  - Unfreeze: ALL up_blocks + middle_block"
echo "  - Loss: Top-K (30%) + Std Matching (1.0)"
echo "  - Batch size: 2 (gradient checkpointing enabled)"
echo "  - UNet LR: 1e-5 (conservative for up_blocks.0)"
echo "  - Resume from: Stage 2.5 Epoch 107"
echo "=========================================="
echo "Started at: $(date)"
echo "Node: $SLURM_NODELIST"
echo ""

source $(conda info --base)/etc/profile.d/conda.sh
conda activate breast_gen
export PYTHONUNBUFFERED=1

# Stage 3: Full model fine-tuning with enhanced loss
python -m scripts.train_controlnet \
    -e configs/environment_breast_sub_finetune_stage3.json \
    -c configs/config_breast_controlnet_finetune_stage3.json \
    -t configs/config_network_rflow.json \
    -g 1

echo ""
echo "Finished at: $(date)"
exit $?
