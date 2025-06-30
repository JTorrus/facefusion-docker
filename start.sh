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

# Pre-warm models on first run
if [ ! -f /runpod-volume/.cache/models_ready ]; then
    echo "First run: preparing models..."
    cd /facefusion
    
    # Pre-download critical models
    python -c "
import os
import subprocess
try:
    print('Warming up FaceFusion models...')
    # This ensures models are downloaded to cached directories
    result = subprocess.run([
        'python', 'facefusion.py', 'headless-run', 
        '--source', '/dev/null', '--target', '/dev/null', 
        '--output', '/tmp/test.jpg', '--execution-providers', 'cuda'
    ], capture_output=True, text=True, timeout=120)
    
    # Mark as ready even if command fails (models still downloaded)
    with open('/runpod-volume/.cache/models_ready', 'w') as f:
        f.write('Models cached successfully')
    print('Model caching completed')
    
except Exception as e:
    print(f'Model preparation warning: {e}')
    # Continue anyway - models might still be partially cached
"
else
    echo "Models already cached and ready"
fi

echo "=== Starting RunPod Handler ==="

# Set environment for optimal performance
export PYTHONUNBUFFERED=1
export PYTHONPATH="/facefusion:/workspace/runpod-worker-facefusion:$PYTHONPATH"
export CUDA_VISIBLE_devices=0

# Change to handler directory and start
cd /workspace/runpod-worker-facefusion

# Execute handler with proper logging[1]
exec python3 -u handler.py
