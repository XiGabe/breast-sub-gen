#!/bin/bash
#SBATCH --partition=sablab-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=breast_stage3
#SBATCH --output=slurm_stage3_%j.out
#SBATCH --error=slurm_stage3_%j.err

# Stage 3.3: Shallow Edge Refinement (Epoch 100-150)
# Unfreeze: All U-Net blocks
# U-Net LR: 1e-5
# Goal: Generate realistic heterogeneity
# Note: Requires stage2_best.pt to exist!

set -e  # Exit on error

# Change to project directory
cd /midtier/sablab/scratch/hoc4008/breast-sub-gen

echo "=========================================="
echo "Stage 3.3: Shallow Edge Refinement"
echo "=========================================="
echo "Starting time: $(date)"

# Check if stage2 checkpoint exists
if [ ! -f "models/breast_controlnet_stage2_best.pt" ]; then
    echo "ERROR: models/breast_controlnet_stage2_best.pt not found!"
    echo "Please run Stage 2 first (sbatch scripts/submit_stage2.sh)"
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate breast_gen

python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage3.json \
    --model_config_path configs/config_maisi_controlnet_train_stage3.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1

echo "=========================================="
echo "Stage 3.3 completed!"
echo "End time: $(date)"
echo "=========================================="
echo "Checkpoint saved to: models/breast_controlnet_stage3_best.pt"
