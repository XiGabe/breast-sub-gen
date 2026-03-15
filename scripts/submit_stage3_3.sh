#!/bin/bash
#SBATCH --partition=sablab-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=breast_stage3
#SBATCH --output=slurm_stage3_%j.out
#SBATCH --error=slurm_stage3_%j.err

# Stage 3.3: Mid-level Texture Release (Epoch 100-150)
# Unfreeze: down_blocks.2, down_blocks.3, middle_block, up_blocks.0, up_blocks.1
# ControlNet LR: 1e-5
# U-Net LR: 5e-6
# Goal: Learn heterogeneous contrast agent distribution (ring enhancement, uneven nodules)

set -e  # Exit on error

# Change to project directory
cd /midtier/sablab/scratch/hoc4008/breast-sub-gen

echo "=========================================="
echo "Stage 3.3: Mid-level Texture Release"
echo "=========================================="
echo "Starting time: $(date)"
echo ""
echo "Configuration Summary:"
echo "  - Epochs: 100-150 (50 epochs)"
echo "  - ControlNet LR: 1e-5"
echo "  - U-Net LR: 5e-6"
echo "  - Validation: Every epoch"
echo "  - Unfrozen Blocks:"
echo "    * down_blocks.2"
echo "    * down_blocks.3"
echo "    * middle_block"
echo "    * up_blocks.0"
echo "    * up_blocks.1"
echo "  - Gradient Checkpointing: Enabled"
echo ""

# Check if stage3_2 checkpoint exists
if [ ! -f "models/breast_controlnet_stage3_2_best.pt" ]; then
    echo "ERROR: models/breast_controlnet_stage3_2_best.pt not found!"
    echo "Please run Stage 3.2 first (sbatch scripts/submit_stage2.sh)"
    exit 1
fi

echo "Stage 3.2 checkpoint found: models/breast_controlnet_stage3_2_best.pt"
echo ""

eval "$(conda shell.bash hook)"
conda activate breast_gen

python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage3_3.json \
    --model_config_path configs/config_maisi_controlnet_train_stage3_3.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1

echo "=========================================="
echo "Stage 3.3 completed!"
echo "End time: $(date)"
echo "=========================================="
echo "Checkpoint saved to: models/breast_controlnet_stage3_3_best.pt"
