# RunPod Video Generation Pipeline

A **2-endpoint** serverless video pipeline on RunPod. No cold starts between pipeline steps.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     FastAPI Orchestrator                      │
│                   (Webhook-based pipeline)                    │
└──────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│    VIDEO GENERATOR       │    │     POST-PROCESSOR       │
│      (48GB GPU)          │    │       (24GB GPU)         │
│                          │    │                          │
│  • FLUX.1 Dev (image)    │───▶│  • CodeFormer (faces)    │
│  • LTX-Video 2 (video)   │    │  • Real-ESRGAN (4x)      │
│  • Wan2.1-I2V (video)    │    │  • RIFE (60fps)          │
└──────────────────────────┘    └──────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
                    ┌──────────────────┐
                    │  Network Volume  │
                    │  /runpod-volume  │
                    └──────────────────┘
```

## Why 2 Endpoints?

| Benefit | Description |
|---------|-------------|
| **No cold starts** | All models in each endpoint stay loaded |
| **Simpler** | Only 2 endpoints to manage |
| **Cheaper** | Less idle time, fewer workers |
| **Faster** | No network hops between models |

## Quick Start

### 1. Deploy to RunPod

Create 2 serverless endpoints:

**Endpoint 1: Video Generator**
- Dockerfile: `handlers/video_generator/Dockerfile`
- GPU: A6000 48GB or A100 40GB
- Mount network volume to `/runpod-volume`

**Endpoint 2: Post-Processor**
- Dockerfile: `handlers/post_processor/Dockerfile`
- GPU: RTX 4090 24GB or A5000
- Mount network volume to `/runpod-volume`

### 2. Configure

```bash
cp .env.example .env
# Add your endpoint IDs:
# VIDEO_GENERATOR_ENDPOINT_ID=xxx
# POST_PROCESSOR_ENDPOINT_ID=xxx
```

### 3. Run Orchestrator

```bash
cd orchestrator
pip install -r requirements.txt
python main.py
```

## API Usage

### Generate Video

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A woman with red hair smiles at the camera",
    "video_model": "ltx",
    "post_process": true,
    "restore_faces": true,
    "upscale": true,
    "interpolate": true
  }'
```

### Check Status

```bash
curl http://localhost:8000/jobs/{job_id}
```

## Handler Actions

### Video Generator (`full_pipeline` default)

| Action | Description |
|--------|-------------|
| `generate_image` | FLUX image only |
| `generate_video_ltx` | LTX-Video from image |
| `generate_video_wan` | Wan2.1 from image |
| `full_pipeline` | Image → Video (no cold start) |

### Post-Processor (`full_post_process` default)

| Action | Description |
|--------|-------------|
| `restore_faces` | CodeFormer only |
| `upscale` | Real-ESRGAN 4x only |
| `interpolate` | RIFE 24→60fps only |
| `full_post_process` | All three (no cold start) |

## Project Structure

```
├── handlers/
│   ├── video_generator/    # FLUX + LTX + Wan2.1
│   └── post_processor/     # CodeFormer + ESRGAN + RIFE
├── orchestrator/           # FastAPI service
└── .env                    # Your endpoint IDs
```

## Estimated Costs

| Endpoint | GPU | Cost/hr | Typical Job |
|----------|-----|---------|-------------|
| Video Generator | A6000 48GB | $0.79 | ~2 min |
| Post-Processor | RTX 4090 24GB | $0.44 | ~3 min |

**Total per video:** ~$0.10-0.20

## License

MIT
