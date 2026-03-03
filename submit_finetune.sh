#!/bin/bash
#SBATCH --job-name=breast_ft_sab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_finetune_%j.log
#SBATCH --error=logs/slurm_finetune_%j.err

# 确保日志目录存在
mkdir -p logs

# 打印执行信息
echo "=========================================="
echo "Breast Subtraction Fine-Tuning"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "GPU: $SLURM_JOB_GRES"
echo "=========================================="
echo ""

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate breast_gen
export PYTHONUNBUFFERED=1

# 打印环境信息
echo "Python: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Working dir: $(pwd)"
echo ""

# 运行fine-tuning训练
echo "Starting fine-tuning training..."
echo "=========================================="

python -m scripts.train_controlnet \
    -e configs/environment_breast_sub_finetune.json \
    -c configs/config_breast_controlnet_finetune_stage1.json \
    -t configs/config_network_rflow.json \
    -g 1

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
echo "Job finished at: $(date)"
echo "=========================================="

exit $EXIT_CODE
