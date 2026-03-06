#!/bin/bash
#SBATCH --partition=sablab-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=breast_stage1
#SBATCH --output=slurm_stage1_%j.out
#SBATCH --error=slurm_stage1_%j.err

# Stage 3.1: ControlNet Alignment (Epoch 0-50)
# U-Net: Frozen
# ControlNet LR: 1e-4
# Goal: Learn dual-channel conditioning

set -e  # Exit on error

# Change to project directory
cd /midtier/sablab/scratch/hoc4008/breast-sub-gen

echo "=========================================="
echo "Stage 3.1: ControlNet Alignment"
echo "=========================================="
echo "Starting time: $(date)"

eval "$(conda shell.bash hook)"
conda activate breast_gen

python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage1.json \
    --model_config_path configs/config_maisi_controlnet_train_stage1.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1

echo "=========================================="
echo "Stage 3.1 completed!"
echo "End time: $(date)"
echo "=========================================="
echo "Checkpoint saved to: models/breast_controlnet_stage1_best.pt"
