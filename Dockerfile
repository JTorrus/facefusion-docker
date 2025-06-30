FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ARG FACEFUSION_VERSION=3.3.0
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /facefusion

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python-is-python3 \
    pip \
    git \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install FaceFusion
RUN git clone https://github.com/facefusion/facefusion.git --branch ${FACEFUSION_VERSION} --single-branch .
RUN python install.py --onnxruntime cuda --skip-conda

# Install RunPod SDK
RUN pip install runpod

# Copy your handler
COPY handler.py /facefusion/handler.py

# CRITICAL: Start the handler, NOT FaceFusion directly[1]
CMD ["python", "-u", "/facefusion/handler.py"]
