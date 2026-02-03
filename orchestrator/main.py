"""FastAPI orchestrator for the video generation pipeline."""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Optional
import logging
import sys

# Add parent directory for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.schemas import (
    GenerateRequest,
    MultiClipRequest,
    ImageGenerateRequest,
    VideoFromImageRequest,
    JobStatus,
    JobResult,
    JobState,
)
from config import get_settings
from pipeline import get_pipeline
from job_store import get_job_store
from runpod_client import get_runpod_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    logger.info("Starting Video Pipeline Orchestrator")
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(cleanup_loop())
    
    yield
    
    # Cleanup on shutdown
    cleanup_task.cancel()
    logger.info("Shutting down Video Pipeline Orchestrator")


app = FastAPI(
    title="Video Generation Pipeline",
    description="Orchestrator for serverless video generation on RunPod",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def cleanup_loop():
    """Background task to clean up old jobs."""
    settings = get_settings()
    store = get_job_store()
    
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            await store.cleanup_old_jobs(settings.job_retention_hours)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# ============== Generation Endpoints ==============

@app.post("/generate", response_model=dict)
async def generate_video(request: GenerateRequest):
    """
    Generate a video from a text prompt.
    
    Full pipeline: FLUX → Video → CodeFormer → RIFE → Real-ESRGAN
    """
    try:
        pipeline = get_pipeline()
        result = await pipeline.start_single_clip(request)
        return result
    except Exception as e:
        logger.error(f"Error starting generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/multi-clip", response_model=dict)
async def generate_multi_clip_video(request: MultiClipRequest):
    """
    Generate a multi-scene video with frame chaining.
    
    Each clip uses the last frame of the previous clip as input
    for visual continuity.
    """
    try:
        pipeline = get_pipeline()
        result = await pipeline.start_multi_clip(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting multi-clip: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/image", response_model=dict)
async def generate_image(request: ImageGenerateRequest):
    """Generate just an image with FLUX (no video)."""
    try:
        client = get_runpod_client()
        store = get_job_store()
        
        from uuid import uuid4
        job_id = str(uuid4())
        settings = get_settings()
        
        await store.create(job_id, {
            "status": JobState.GENERATING_IMAGE,
            "current_step": "flux",
            "request": request.model_dump()
        })
        
        await client.run_async(
            "flux",
            {
                "prompt": request.prompt,
                "output_path": f"{settings.runpod_volume_path}/jobs/{job_id}/images",
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed
            },
            job_id
        )
        
        return {"job_id": job_id, "status": "processing"}
    except Exception as e:
        logger.error(f"Error generating image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/video", response_model=dict)
async def generate_video_from_image(request: VideoFromImageRequest):
    """Generate video from an existing image."""
    try:
        pipeline = get_pipeline()
        store = get_job_store()
        client = get_runpod_client()
        
        from uuid import uuid4
        job_id = str(uuid4())
        settings = get_settings()
        base_path = f"{settings.runpod_volume_path}/jobs/{job_id}"
        
        await store.create(job_id, {
            "status": JobState.GENERATING_VIDEO,
            "current_step": "video",
            "request": request.model_dump(),
            "base_path": base_path
        })
        
        video_model = request.video_model.value
        handler = "ltx_video" if video_model == "ltx" else "wan21"
        num_frames = int(request.duration_seconds * 24) + 1
        
        await client.run_async(
            handler,
            {
                "image_path": request.image_path,
                "prompt": request.motion_prompt,
                "output_path": f"{base_path}/video",
                "num_frames": num_frames
            },
            job_id
        )
        
        return {"job_id": job_id, "status": "processing"}
    except Exception as e:
        logger.error(f"Error generating video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============== Webhook Endpoint ==============

@app.post("/callback/{job_id}")
async def webhook_callback(job_id: str, payload: dict):
    """
    Receive webhook callbacks from RunPod handlers.
    
    This endpoint is called by RunPod when a handler completes.
    It progresses the pipeline to the next step.
    """
    logger.info(f"Webhook received for job {job_id}")
    logger.debug(f"Payload: {payload}")
    
    try:
        pipeline = get_pipeline()
        await pipeline.handle_webhook(job_id, payload)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Webhook error for {job_id}: {e}", exc_info=True)
        # Don't raise - RunPod might retry
        return {"status": "error", "message": str(e)}


# ============== Job Management Endpoints ==============

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a job."""
    store = get_job_store()
    job = await store.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(
        job_id=job_id,
        status=JobState(job.get("status", "pending")),
        current_step=job.get("current_step"),
        progress=None,  # TODO: Calculate progress
        error=job.get("error"),
        created_at=job.get("created_at", ""),
        updated_at=job.get("updated_at", "")
    )


@app.get("/jobs/{job_id}/result", response_model=JobResult)
async def get_job_result(job_id: str):
    """Get the result of a completed job."""
    store = get_job_store()
    job = await store.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = job.get("status")
    if status != JobState.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {status}"
        )
    
    results = job.get("results", {})
    final_video = results.get("final_video_path")
    
    return JobResult(
        job_id=job_id,
        status=JobState.COMPLETED,
        video_path=final_video,
        video_url=None,  # TODO: Generate download URL
        duration_seconds=None,
        resolution=None,
        processing_time_seconds=None
    )


@app.get("/jobs/{job_id}/download")
async def download_job_result(job_id: str):
    """Download the final video file."""
    store = get_job_store()
    job = await store.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.get("status") != JobState.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")
    
    results = job.get("results", {})
    video_path = results.get("final_video_path")
    
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4"
    )


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a job and clean up resources."""
    store = get_job_store()
    job = await store.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # TODO: Cancel RunPod jobs if running
    
    await store.delete(job_id)
    
    # TODO: Clean up files on volume
    
    return {"status": "cancelled", "job_id": job_id}


@app.get("/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 100):
    """List jobs, optionally filtered by status."""
    store = get_job_store()
    jobs = await store.list_jobs(status=status, limit=limit)
    
    return {
        "jobs": [
            {
                "job_id": j["job_id"],
                "status": j.get("status"),
                "created_at": j.get("created_at"),
                "updated_at": j.get("updated_at")
            }
            for j in jobs
        ],
        "total": len(jobs)
    }


# ============== Health & Utility Endpoints ==============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/endpoints")
async def list_endpoints():
    """List configured RunPod endpoints."""
    settings = get_settings()
    
    return {
        "endpoints": {
            "flux": settings.flux_endpoint_id or "not configured",
            "ltx_video": settings.ltx_video_endpoint_id or "not configured",
            "wan21": settings.wan21_endpoint_id or "not configured",
            "codeformer": settings.codeformer_endpoint_id or "not configured",
            "rife": settings.rife_endpoint_id or "not configured",
            "realesrgan": settings.realesrgan_endpoint_id or "not configured",
        }
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
