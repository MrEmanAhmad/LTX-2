# Wan2.1 I2V Handler for RunPod Serverless (requires 48GB VRAM)
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py model.py ./
COPY shared/ /app/shared/

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/models/huggingface

CMD ["python", "-u", "handler.py"]
