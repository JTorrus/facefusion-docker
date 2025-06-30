#!/usr/bin/env bash

echo "=== FaceFusion RunPod Worker Starting ==="

# Enable error handling and verbose output
set -euo pipefail

# Create network volume directories if they don't exist
mkdir -p /runpod-volume/.cache
mkdir -p /runpod-volume/.ifnude  
mkdir -p /runpod-volume/.insightface
mkdir -p /runpod-volume/facefusion-cache

echo "Setting up symlinks to network volume..."

# Clean and create symlinks
rm -rf /root/.cache /root/.ifnude /root/.insightface
ln -s /runpod-volume/.cache /root/.cache
ln -s /runpod-volume/.ifnude /root/.ifnude  
ln -s /runpod-volume/.insightface /root/.insightface

# Symlink workspace to volume for persistence
if [ ! -L /workspace ]; then
    ln -s /runpod-volume /workspace
fi

echo "Checking FaceFusion model cache..."

echo "=== Starting RunPod Handler ==="

# Set environment for optimal performance
export PYTHONUNBUFFERED=1
export PYTHONPATH="/facefusion:/workspace/runpod-worker-facefusion:$PYTHONPATH"
export CUDA_VISIBLE_devices=0

# Change to handler directory and start
cd /workspace/runpod-worker-facefusion

# Execute handler with proper logging[1]
python3 -u handler.py
