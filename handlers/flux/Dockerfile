# FLUX.1 Dev Handler for RunPod Serverless
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler code
COPY handler.py model.py ./

# Copy shared utilities
COPY shared/ /app/shared/

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/models/huggingface

CMD ["python", "-u", "handler.py"]
