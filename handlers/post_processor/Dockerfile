# Unified Post-Processor: CodeFormer + Real-ESRGAN + RIFE
# Requires: 24GB VRAM (RTX 4090, A5000, or A6000)

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install additional system dependencies
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Python packages
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
