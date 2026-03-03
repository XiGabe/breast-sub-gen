#!/bin/bash
# Start TensorBoard server for monitoring training

# Default port
PORT=6006

# Allow custom port
if [ ! -z "$1" ]; then
    PORT=$1
fi

echo "Starting TensorBoard on port ${PORT}..."
echo "Logs will be loaded from: tensorboard_logs/"
echo ""
echo "Available experiments:"
echo "  - breast_sub_controlnet (original training)"
echo "  - breast_sub_controlnet_finetune (Stage 1)"
echo "  - breast_sub_controlnet_finetune_s2 (Stage 2 - current)"
echo ""
echo "Access URL: http://localhost:${PORT}"
echo "Or from remote: http://$(hostname):${PORT}"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate breast_gen

# Start TensorBoard
tensorboard --logdir=tensorboard_logs --port=${PORT} --bind_all
