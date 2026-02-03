# RunPod Video Generation Pipeline

A serverless video generation pipeline running on RunPod, featuring multiple AI models for image generation, video synthesis, face restoration, frame interpolation, and upscaling.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Orchestrator                            │
│                    (Webhook-based, multi-clip pipeline)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   FLUX.1 Dev    │         │  LTX-Video 2    │         │  Wan2.1-I2V     │
│   (24GB VRAM)   │         │   (24GB VRAM)   │         │  (48GB VRAM)    │
│   First Frames  │         │   Fast Video    │         │  Better Faces   │
└─────────────────┘         └─────────────────┘         └─────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   CodeFormer    │         │      RIFE       │         │  Real-ESRGAN    │
│   (16GB VRAM)   │         │   (8GB VRAM)    │         │   (8GB VRAM)    │
│   Face Restore  │         │   24→60fps      │         │  480p→1080p     │
└─────────────────┘         └─────────────────┘         └─────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │   RunPod Network Volume │
                        │     /runpod-volume      │
                        └─────────────────────────┘
```

## Pipeline Flow

1. **FLUX.1** → Generate photorealistic first frame from text prompt
2. **LTX-Video 2 / Wan2.1** → Animate image to video clip (with last-frame extraction for continuity)
3. **CodeFormer** → Frame-by-frame face restoration
4. **RIFE** → Frame interpolation (24fps → 60fps)
5. **Real-ESRGAN** → Upscale to 1080p
6. **FFmpeg** → Stitch clips + add audio

## Project Structure

```
runpod-video-pipeline/
├── orchestrator/           # FastAPI orchestrator service
│   ├── main.py            # API endpoints + webhook handlers
│   ├── pipeline.py        # Multi-clip pipeline logic
│   ├── runpod_client.py   # RunPod API wrapper
│   ├── job_store.py       # Job state management
│   └── config.py          # Configuration
│
├── handlers/              # RunPod serverless handlers
│   ├── flux/              # FLUX.1 Dev image generation
│   ├── ltx_video/         # LTX-Video 2 video synthesis
│   ├── wan21/             # Wan2.1-I2V video synthesis
│   ├── codeformer/        # Face restoration
│   ├── rife/              # Frame interpolation
│   └── realesrgan/        # Video upscaling
│
├── shared/                # Shared utilities
│   ├── storage.py         # RunPod volume operations
│   ├── video_utils.py     # FFmpeg operations
│   └── schemas.py         # Pydantic models
│
└── scripts/               # Deployment & utilities
    ├── deploy.sh          # Deploy to RunPod
    ├── warm_workers.py    # Keep workers warm
    └── test_pipeline.py   # End-to-end tests
```

## Quick Start

### 1. Prerequisites

- RunPod account with API key
- Docker installed locally
- Python 3.10+

### 2. Environment Setup

```bash
cp .env.example .env
# Edit .env with your RunPod API key and endpoint IDs
```

### 3. Deploy Handlers to RunPod

```bash
# Build and push Docker images
./scripts/deploy.sh build

# Create serverless endpoints (do this in RunPod dashboard)
# Then update .env with endpoint IDs
```

### 4. Run Orchestrator

```bash
cd orchestrator
pip install -r requirements.txt
uvicorn main:app --reload
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Full pipeline (single clip) |
| `/generate/multi-clip` | POST | Multi-scene video with frame chaining |
| `/generate/image` | POST | FLUX image generation only |
| `/generate/video` | POST | Video from existing image |
| `/callback/{job_id}` | POST | Webhook receiver |
| `/jobs/{id}` | GET | Check job status |
| `/jobs/{id}/result` | GET | Download final video |

## Example Request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "image_prompt": "A photorealistic portrait of a woman with flowing red hair",
    "motion_prompt": "She turns her head slowly and smiles",
    "video_model": "ltx",
    "duration_seconds": 5,
    "apply_face_restore": true,
    "upscale": true,
    "interpolate": true
  }'
```

## Multi-Clip Example

```bash
curl -X POST http://localhost:8000/generate/multi-clip \
  -H "Content-Type: application/json" \
  -d '{
    "clips": [
      {
        "image_prompt": "A man standing in a forest at sunset",
        "motion_prompt": "He looks around curiously"
      },
      {
        "motion_prompt": "He starts walking forward through the trees"
      },
      {
        "motion_prompt": "He stops and looks up at the sky in wonder"
      }
    ],
    "video_model": "wan21",
    "apply_face_restore": true
  }'
```

## Model Specifications

| Handler | Model | VRAM | HuggingFace/GitHub |
|---------|-------|------|-------------------|
| flux | FLUX.1-dev | 24GB | `black-forest-labs/FLUX.1-dev` |
| ltx_video | LTX-Video 2 | 24GB | `Lightricks/LTX-Video` |
| wan21 | Wan2.1-I2V-14B-480P | 48GB | `Wan-AI/Wan2.1-I2V-14B-480P` |
| codeformer | CodeFormer | 16GB | `sczhou/CodeFormer` |
| rife | RIFE v4.6 | 8GB | `hzwer/ECCV2022-RIFE` |
| realesrgan | Real-ESRGAN x4 | 8GB | `xinntao/Real-ESRGAN` |

## Estimated Costs (RunPod)

| GPU | Cost/hr | Used For |
|-----|---------|----------|
| RTX 4090 (24GB) | ~$0.44 | FLUX, LTX-Video |
| A6000 (48GB) | ~$0.79 | Wan2.1-I2V |
| RTX 3090 (24GB) | ~$0.34 | CodeFormer, RIFE, ESRGAN |

**Typical 30s video (full pipeline):** ~$0.15-0.30

## Cold Start Strategy

Large models have 2-5 minute cold starts. Options:

1. **Keep workers warm** - Run `scripts/warm_workers.py` (~$0.50/hr idle cost)
2. **FlashBoot volumes** - Pre-load model weights on network storage
3. **Smaller variants** - Use Wan2.1-T2V-1.3B for testing (fits 24GB)

## License

MIT
