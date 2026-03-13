#!/bin/bash
#SBATCH --partition=sablab-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=breast_stage3
#SBATCH --output=outputs/logs/stage3_%j.log
#SBATCH --error=outputs/logs/stage3_%j.err

conda activate breast_gen

python -m scripts.train_controlnet \
    --env_config_path configs/environment_maisi_controlnet_train_rflow-mr_breast_stage3.json \
    --model_config_path configs/config_maisi_controlnet_train_stage3.json \
    --model_def_path configs/config_network_rflow.json \
    --num_gpus 1
