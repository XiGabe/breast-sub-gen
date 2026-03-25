#!/bin/bash
#SBATCH --partition=sablab-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dit_train
#SBATCH --output=logs/dit_train_%j.out
#SBATCH --error=logs/dit_train_%j.err

# Activate conda environment
source /home/hoc4008/miniconda3/etc/profile.d/conda.sh
conda activate breast_gen

# Change to project directory
cd /midtier/sablab/scratch/hoc4008/breast-sub-gen

# Run DiT training
python -m scripts.diff_model_train \
  -e configs/environment_maisi_diff_model_rflow-mr_breast.json \
  -c configs/config_maisi_diff_model_rflow-mr_breast.json \
  -t configs/config_network_rflow.json \
  -g 1

echo "Training completed"