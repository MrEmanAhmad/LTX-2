# Unified Video Generator: FLUX + LTX-Video + Wan2.1
# Requires: 48GB+ VRAM (A6000, A100-40GB, or A100-80GB)

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    runpod \
    diffusers>=0.27.0 \
    transformers \
    accelerate \
    safetensors \
    sentencepiece \
    Pillow \
    opencv-python-headless \
    imageio[ffmpeg]

# Copy handler code
COPY handler.py model.py ./

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/models
ENV TRANSFORMERS_CACHE=/runpod-volume/models

CMD ["python", "-u", "handler.py"]
