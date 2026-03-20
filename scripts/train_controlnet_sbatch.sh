#!/bin/bash
#SBATCH --partition=sablab-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=controlnet_train
#SBATCH --output=logs/controlnet_train_%j.out
#SBATCH --error=logs/controlnet_train_%j.err

# Activate conda environment
source /home/hoc4008/miniconda3/etc/profile.d/conda.sh
conda activate breast_gen

# Change to project directory
cd /midtier/sablab/scratch/hoc4008/breast-sub-gen

# Run ControlNet training
python -m scripts.train_controlnet \
  -e configs/environment_maisi_controlnet_rflow-mr_breast.json \
  -c configs/config_maisi_controlnet_train_rflow-mr_breast.json \
  -t configs/config_network_rflow.json \
  -g 1

echo "Training completed"
