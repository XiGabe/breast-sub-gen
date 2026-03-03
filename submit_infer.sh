#!/bin/bash
#SBATCH --job-name=breast_infer
#SBATCH --partition=preempt_gpu
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --output=logs/infer_%j.log
#SBATCH --error=logs/infer_%j.err

# Ensure log directory exists
mkdir -p logs

# Change to project directory
cd /midtier/sablab/scratch/hoc4008/breast-sub-gen

# Setup environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate breast_gen
export PYTHONUNBUFFERED=1

# Run inference
echo "Starting inference..."
python -m scripts.infer_breast_sub \
    -e configs/environment_breast_sub_infer.json \
    -c configs/config_breast_infer.json \
    -t configs/config_network_rflow.json \
    -n 5 \
    -d cuda:0

echo "Inference complete!"
