#!/bin/bash
#SBATCH --partition=sablab-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=breast_stage1
#SBATCH --output=slurm_stage1_%j.out
#SBATCH --error=slurm_stage1_%j.err

# Stage 3.1: ControlNet Anchor Phase (Epoch 0-50)
# U-Net: Fully Frozen
# ControlNet LR: 1e-4
# U-Net LR: 0.0 (frozen)
# Goal: Learn dual-channel conditioning mapping

set -e  # Exit on error

# Change to project directory
cd /midtier/sablab/scratch/hoc4008/breast-sub-gen

echo "=========================================="
echo "Stage 3.1: ControlNet Anchor Phase"
echo "=========================================="
echo "Starting time: $(date)"
echo ""
echo "Configuration Summary:"
echo "  - Epochs: 0-50 (50 epochs)"
echo "  - ControlNet LR: 1e-4"
echo "  - U-Net LR: 0.0 (frozen)"
echo "  - ROI Weight: 100"
echo "  - Validation: Every epoch"
echo ""

eval "$(conda shell.bash hook)"
conda activate breast_gen

python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage3_1.json \
    --model_config_path configs/config_maisi_controlnet_train_stage3_1.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1

echo "=========================================="
echo "Stage 3.1 completed!"
echo "End time: $(date)"
echo "=========================================="
echo "Checkpoint saved to: models/breast_controlnet_stage3_1_best.pt"
