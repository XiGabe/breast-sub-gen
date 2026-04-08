#!/bin/bash
#SBATCH --partition=sablab-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=nnunet_train
#SBATCH --output=logs/nnunet_train_%j.out
#SBATCH --error=logs/nnunet_train_%j.err

# Activate conda environment
source /home/hoc4008/miniconda3/etc/profile.d/conda.sh
conda activate breast_gen

# Change to project directory
cd /midtier/sablab/scratch/hoc4008/breast-sub-gen

# Set nnUNet environment variables
export nnUNet_raw="./nnUNet_train/data_nnunet"
export nnUNet_preprocessed="./nnUNet_train/data_nnunet_preprocessed"
export nnUNet_results="./nnUNet_train/training_results"

# Run training for fold 1 (use -c to continue if interrupted)
# - num_gpus: number of GPUs to use
# - npz: save COMPOSITE (loss and dice) as npz files for later analysis
# - tr: use nnUNetTrainer_250epochs (250 epochs)
echo "Starting fold 3 training..."
nnUNetv2_train 800 3d_fullres 3 -num_gpus 1 --npz -tr nnUNetTrainer_250epochs

echo "Fold 3 completed, starting fold 4..."
nnUNetv2_train 800 3d_fullres 4 -num_gpus 1 --npz -tr nnUNetTrainer_250epochs

echo "Training completed"
