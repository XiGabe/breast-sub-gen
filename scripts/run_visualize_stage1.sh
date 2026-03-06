#!/bin/bash
# Shell script to run inference and visualization for Stage 1 ControlNet

# Activate conda environment
echo "Activating breast_gen conda environment..."
source /midtier/sablab/scratch/hoc4008/anaconda3/etc/profile.d/conda.sh
conda activate breast_gen

# Set paths
PROJECT_DIR="/midtier/sablab/scratch/hoc4008/breast-sub-gen"
cd "$PROJECT_DIR"

# Default parameters
NUM_SAMPLES=5
OUTPUT_DIR="./outputs/inference_stage1_vis"
NUM_GPUS=1
NUM_INFERENCE_STEPS=30

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num_steps)
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "======================================"
echo "Stage 1 ControlNet Visualization"
echo "======================================"
echo "Number of samples: $NUM_SAMPLES"
echo "Output directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Inference steps: $NUM_INFERENCE_STEPS"
echo "======================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run visualization script
python -m scripts.visualize_inference \
    --env_config configs/environment_maisi_controlnet_infer_stage1.json \
    --model_config configs/config_maisi_controlnet_train_stage1.json \
    --model_def configs/config_network_rflow.json \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --num_gpus "$NUM_GPUS" \
    --num_inference_steps "$NUM_INFERENCE_STEPS"

echo ""
echo "======================================"
echo "Visualization complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "======================================"
