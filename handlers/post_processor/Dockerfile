# Unified Post-Processor: CodeFormer + Real-ESRGAN + RIFE
# Requires: 24GB VRAM (RTX 4090, A5000, or A6000)

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# Install basicsr and realesrgan (includes CodeFormer dependencies)
RUN pip install --no-cache-dir \
    runpod \
    basicsr>=1.4.2 \
    realesrgan \
    facexlib \
    gfpgan \
    opencv-python-headless \
    Pillow \
    numpy

# Copy handler code
COPY handler.py model.py ./

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/models
ENV TRANSFORMERS_CACHE=/runpod-volume/models

CMD ["python", "-u", "handler.py"]
