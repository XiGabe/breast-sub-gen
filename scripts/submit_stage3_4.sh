#!/bin/bash
#SBATCH --partition=sablab-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=breast_stage4
#SBATCH --output=slurm_stage4_%j.out
#SBATCH --error=slurm_stage4_%j.err

# Stage 3.4: Full Refinement (Epoch 150-200)
# Unfreeze: All U-Net blocks (Full Unfreeze)
# ControlNet LR: 5e-6
# U-Net LR: 1e-6 to 5e-6 (conservative fine-tuning)
# Goal: Generate sharp micro-vessels and realistic spiculated margins

set -e  # Exit on error

# Change to project directory
cd /midtier/sablab/scratch/hoc4008/breast-sub-gen

echo "=========================================="
echo "Stage 3.4: Full Refinement"
echo "=========================================="
echo "Starting time: $(date)"
echo ""
echo "Configuration Summary:"
echo "  - Epochs: 150-200 (50 epochs)"
echo "  - ControlNet LR: 5e-6"
echo "  - U-Net LR: 1e-6 (extremely conservative)"
echo "  - Validation: Every epoch"
echo "  - Unfrozen Blocks: ALL"
echo "  - Gradient Checkpointing: Enabled"
echo ""

# Check if stage3_3 checkpoint exists
if [ ! -f "models/breast_controlnet_stage3_3_best.pt" ]; then
    echo "ERROR: models/breast_controlnet_stage3_3_best.pt not found!"
    echo "Please run Stage 3.3 first (sbatch scripts/submit_stage3.sh)"
    exit 1
fi

echo "Stage 3.3 checkpoint found: models/breast_controlnet_stage3_3_best.pt"
echo ""

eval "$(conda shell.bash hook)"
conda activate breast_gen

python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage3_4.json \
    --model_config_path configs/config_maisi_controlnet_train_stage3_4.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1

echo "=========================================="
echo "Stage 3.4 completed!"
echo "End time: $(date)"
echo "=========================================="
echo "Checkpoint saved to: models/breast_controlnet_stage3_4_best.pt"
