#!/bin/bash
#SBATCH --partition=sablab-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=breast_stage2
#SBATCH --output=slurm_stage2_%j.out
#SBATCH --error=slurm_stage2_%j.err

# Stage 3.2: Deep Semantic Release (Epoch 50-100)
# Unfreeze: down_blocks.2, down_blocks.3, middle_block, up_blocks
# U-Net LR: 5e-5
# Goal: Fix brightness issues
# Note: Requires stage1_best.pt to exist!

set -e  # Exit on error

# Change to project directory
cd /midtier/sablab/scratch/hoc4008/breast-sub-gen

echo "=========================================="
echo "Stage 3.2: Deep Semantic Release"
echo "=========================================="
echo "Starting time: $(date)"

# Check if stage1 checkpoint exists
if [ ! -f "models/breast_controlnet_stage1_best.pt" ]; then
    echo "ERROR: models/breast_controlnet_stage1_best.pt not found!"
    echo "Please run Stage 1 first (sbatch scripts/submit_stage1.sh)"
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate breast_gen

python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage2.json \
    --model_config_path configs/config_maisi_controlnet_train_stage2.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1

echo "=========================================="
echo "Stage 3.2 completed!"
echo "End time: $(date)"
echo "=========================================="
echo "Checkpoint saved to: models/breast_controlnet_stage2_best.pt"
