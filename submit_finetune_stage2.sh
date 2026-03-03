#!/bin/bash
#SBATCH --job-name=breast_ft_s2
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_finetune_s2_%j.log
#SBATCH --error=logs/slurm_finetune_s2_%j.err

mkdir -p logs

echo "=========================================="
echo "Breast Fine-Tuning Stage 2"
echo "=========================================="
echo "Started at: $(date)"
echo "Node: $SLURM_NODELIST"
echo ""

source $(conda info --base)/etc/profile.d/conda.sh
conda activate breast_gen
export PYTHONUNBUFFERED=1

# Stage 2: Resume from Stage 1 checkpoint
python -m scripts.train_controlnet \
    -e configs/environment_breast_sub_finetune_stage2.json \
    -c configs/config_breast_controlnet_finetune_stage2.json \
    -t configs/config_network_rflow.json \
    -g 1

echo ""
echo "Finished at: $(date)"
exit $?
