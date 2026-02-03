# CodeFormer Handler for RunPod Serverless
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git ffmpeg libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Clone CodeFormer repo for dependencies
RUN git clone https://github.com/sczhou/CodeFormer.git /app/CodeFormer

# Install dependencies
WORKDIR /app/CodeFormer
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install -r requirements.txt
RUN python basicsr/setup.py develop

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py model.py ./
COPY shared/ /app/shared/

ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/runpod-volume/models/codeformer

CMD ["python", "-u", "handler.py"]
