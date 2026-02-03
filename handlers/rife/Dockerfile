# RIFE Handler for RunPod Serverless
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git ffmpeg libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Clone RIFE repo
RUN git clone https://github.com/hzwer/ECCV2022-RIFE.git /app/rife_repo
RUN cp -r /app/rife_repo/model /app/model

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py model.py ./
COPY shared/ /app/shared/

ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/runpod-volume/models/rife

CMD ["python", "-u", "handler.py"]
