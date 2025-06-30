FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ARG FACEFUSION_VERSION=3.3.0
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Create workspace directory structure[2]
WORKDIR /workspace/runpod-worker-facefusion

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python-is-python3 \
    pip \
    git \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install FaceFusion in separate directory
WORKDIR /facefusion
RUN git clone https://github.com/facefusion/facefusion.git --branch ${FACEFUSION_VERSION} --single-branch .
RUN python install.py --onnxruntime cuda --skip-conda

# Install RunPod SDK
RUN pip install runpod

# Switch back to workspace
WORKDIR /workspace/runpod-worker-facefusion

# Copy files with proper permissions in one step[3]
COPY --chmod=755 handler.py /workspace/runpod-worker-facefusion/handler.py
COPY --chmod=755 start.sh /start.sh

# Use ENTRYPOINT for guaranteed execution[5]
ENTRYPOINT /start.sh
