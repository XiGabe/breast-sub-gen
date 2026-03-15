#!/bin/bash
#SBATCH --partition=sablab-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=breast_stage2
#SBATCH --output=slurm_stage2_%j.out
#SBATCH --error=slurm_stage2_%j.err

# Stage 3.2: Bottleneck Adaptation (Epoch 50-100)
# Unfreeze: down_blocks.3, middle_block, up_blocks.0
# ControlNet LR: 5e-5
# U-Net LR: 1e-5 (10x smaller than ControlNet)
# Goal: Adapt deep semantic features to sparse distribution

set -e  # Exit on error

# Change to project directory
cd /midtier/sablab/scratch/hoc4008/breast-sub-gen

echo "=========================================="
echo "Stage 3.2: Bottleneck Adaptation"
echo "=========================================="
echo "Starting time: $(date)"
echo ""
echo "Configuration Summary:"
echo "  - Epochs: 50-100 (50 epochs)"
echo "  - ControlNet LR: 5e-5"
echo "  - U-Net LR: 1e-5"
echo "  - LR Schedule: Warmup (5%) + Cosine Annealing"
echo "  - Validation: Every epoch"
echo "  - Unfrozen Blocks:"
echo "    * down_blocks.3"
echo "    * middle_block"
echo "    * up_blocks.0"
echo "  - Gradient Checkpointing: Enabled"
echo ""

# Check if stage3_1 checkpoint exists
if [ ! -f "models/breast_controlnet_stage3_1_best.pt" ]; then
    echo "ERROR: models/breast_controlnet_stage3_1_best.pt not found!"
    echo "Please run Stage 3.1 first (sbatch scripts/submit_stage1.sh)"
    exit 1
fi

echo "Stage 3.1 checkpoint found: models/breast_controlnet_stage3_1_best.pt"
echo ""

eval "$(conda shell.bash hook)"
conda activate breast_gen

python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage3_2.json \
    --model_config_path configs/config_maisi_controlnet_train_stage3_2.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1

echo ""
echo "=========================================="
echo "Stage 3.2 completed!"
echo "End time: $(date)"
echo "=========================================="
echo "Outputs:"
echo "  - Checkpoint: models/breast_controlnet_stage3_2_best.pt"
echo "  - Logs: outputs/logs/breast_controlnet_stage3_2/train.log"
echo "  - TensorBoard: outputs/tfevent/breast_controlnet_stage3_2/"
