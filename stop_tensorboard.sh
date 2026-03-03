#!/bin/bash
# Stop TensorBoard server

echo "Stopping TensorBoard..."
pkill -f "tensorboard" && echo "✓ TensorBoard stopped" || echo "✗ No TensorBoard process found"

# Verify
if pgrep -f "tensorboard" > /dev/null; then
    echo "⚠ Warning: TensorBoard still running. Try:"
    echo "   ps aux | grep tensorboard"
    echo "   kill <PID>"
else
    echo "✓ Confirmed: TensorBoard is not running"
fi
