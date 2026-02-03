# RunPod Setup Guide

This guide walks through deploying the video pipeline to RunPod.

## Prerequisites

1. RunPod account with credits
2. Docker Hub account (or other container registry)
3. Docker installed locally

## Step 1: Create Network Volume

1. Go to RunPod Dashboard → Storage → Network Volumes
2. Click "Create Network Volume"
3. Configure:
   - Name: `vidpipe-volume`
   - Size: 100GB+ (recommended)
   - Datacenter: Choose one close to you
4. Note the volume ID for later

## Step 2: Build and Push Docker Images

```bash
# Set your Docker Hub username
export DOCKER_USERNAME=your-username

# Login to Docker Hub
docker login

# Build and push all handlers
./scripts/deploy.sh all
```

This will create images:
- `your-username/vidpipe-flux:latest`
- `your-username/vidpipe-ltx_video:latest`
- `your-username/vidpipe-wan21:latest`
- `your-username/vidpipe-codeformer:latest`
- `your-username/vidpipe-rife:latest`
- `your-username/vidpipe-realesrgan:latest`
- `your-username/vidpipe-orchestrator:latest`

## Step 3: Create Serverless Endpoints

For each handler, create a serverless endpoint:

### FLUX.1 Endpoint

1. Go to RunPod Dashboard → Serverless → Create Endpoint
2. Configure:
   - Name: `vidpipe-flux`
   - Container Image: `your-username/vidpipe-flux:latest`
   - GPU: RTX 4090 (24GB) or A5000
   - Container Disk: 20GB
   - Volume: Mount `vidpipe-volume` at `/runpod-volume`
   - Min Workers: 0
   - Max Workers: 3
   - Idle Timeout: 5 seconds
   - FlashBoot: Enable (recommended)
3. Save and note the Endpoint ID

### LTX-Video Endpoint

- Same as FLUX, but use `vidpipe-ltx_video` image
- GPU: RTX 4090 (24GB)

### Wan2.1 Endpoint

- Use `vidpipe-wan21` image
- GPU: A6000 (48GB) - **Important: requires 48GB!**

### CodeFormer Endpoint

- Use `vidpipe-codeformer` image
- GPU: RTX 3090 or 4090 (24GB)

### RIFE Endpoint

- Use `vidpipe-rife` image
- GPU: RTX 3090 (can use smaller GPU, 8GB+)

### Real-ESRGAN Endpoint

- Use `vidpipe-realesrgan` image
- GPU: RTX 3090 (can use smaller GPU, 8GB+)

## Step 4: Configure Environment

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env` with your endpoint IDs:

```env
RUNPOD_API_KEY=your_api_key_here
FLUX_ENDPOINT_ID=abc123...
LTX_VIDEO_ENDPOINT_ID=def456...
WAN21_ENDPOINT_ID=ghi789...
CODEFORMER_ENDPOINT_ID=jkl012...
RIFE_ENDPOINT_ID=mno345...
REALESRGAN_ENDPOINT_ID=pqr678...
WEBHOOK_BASE_URL=https://your-orchestrator.com
```

## Step 5: Deploy Orchestrator

### Option A: Run on RunPod (Recommended)

1. Create a new Pod (not serverless)
2. Use `vidpipe-orchestrator` image
3. Configure:
   - GPU: CPU only (no GPU needed)
   - Expose port 8000
   - Mount the same volume at `/runpod-volume`
   - Set environment variables from `.env`

### Option B: Run Externally

Deploy to Railway, Fly.io, Render, or your own server:

```bash
cd orchestrator
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Make sure the orchestrator is publicly accessible for webhooks!

## Step 6: Test the Pipeline

```bash
# Test health
curl https://your-orchestrator.com/health

# Test single video generation
curl -X POST https://your-orchestrator.com/generate \
  -H "Content-Type: application/json" \
  -d '{
    "image_prompt": "A woman with red hair",
    "motion_prompt": "She smiles",
    "video_model": "ltx",
    "duration_seconds": 3
  }'
```

## Cost Optimization Tips

1. **Use Spot Instances** - 3x cheaper but can be interrupted
2. **Enable FlashBoot** - Reduces cold start from 5min to 30s
3. **Set Min Workers to 0** - Only pay when processing
4. **Use smaller models for testing** - Wan2.1-T2V-1.3B instead of 14B
5. **Run warm_workers.py** - Keep 1 worker warm during active hours (~$0.50/hr)

## Troubleshooting

### Cold starts are too long

- Enable FlashBoot
- Keep min workers at 1 (costs money when idle)
- Pre-download models to network volume

### Out of memory errors

- Check GPU has enough VRAM
- Wan2.1 needs 48GB, others need 24GB
- Enable model CPU offloading in handler

### Webhook not receiving callbacks

- Ensure orchestrator URL is publicly accessible
- Check firewall/security group rules
- Verify WEBHOOK_BASE_URL is correct

### Models not loading

- Check HuggingFace token is set (for gated models like FLUX)
- Verify network volume is mounted
- Check container logs in RunPod dashboard
