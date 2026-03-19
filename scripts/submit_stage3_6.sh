#!/bin/bash
#SBATCH --partition=sablab-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=breast_stage3_6
#SBATCH --output=outputs/slurm_stage3_6_%j.out
#SBATCH --error=outputs/slurm_stage3_6_%j.err

# Stage 3.6: Full Unfreeze (Epoch 111-210)
# Resume from Stage 3.1 epoch 50 checkpoint
# Loss: Weighted L1 (ROI weight=3)
# ControlNet LR: 5e-6
# U-Net LR: 2e-6 (fully unfrozen)

set -e  # Exit on error

# Change to project directory
cd /midtier/sablab/scratch/hoc4008/breast-sub-gen

echo "=========================================="
echo "Stage 3.6: Full Unfreeze"
echo "=========================================="
echo "Starting time: $(date)"
echo ""
echo "Configuration Summary:"
echo "  - Start Epoch: 111"
echo "  - End Epoch: 210 (100 epochs)"
echo "  - ControlNet LR: 5e-6"
echo "  - U-Net LR: 2e-6"
echo "  - Validation: Every epoch"
echo "  - Unfrozen Blocks: ALL"
echo "  - Gradient Checkpointing: Enabled"
echo ""
echo "Loss Function:"
echo "  - Weighted L1 Loss (ROI weight=1)"
echo ""

# Check if stage3_1 checkpoint exists
if [ ! -f "models/breast_controlnet_stage3_1_epoch_50.pt" ]; then
    echo "ERROR: models/breast_controlnet_stage3_1_epoch_50.pt not found!"
    echo "Please ensure Stage 3.1 epoch 50 checkpoint exists"
    exit 1
fi

echo "Stage 3.1 checkpoint found: models/breast_controlnet_stage3_1_epoch_50.pt"
echo ""

eval "$(conda shell.bash hook)"
conda activate breast_gen

python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage3_6.json \
    --model_config_path configs/config_maisi_controlnet_train_stage3_6.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1

echo "=========================================="
echo "Stage 3.6 completed!"
echo "End time: $(date)"
echo "=========================================="
echo "Checkpoint will be saved to: models/breast_controlnet_stage3_6_*.pt"
