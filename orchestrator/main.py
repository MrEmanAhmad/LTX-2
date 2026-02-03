"""
Simplified FastAPI orchestrator for 2-endpoint video pipeline
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from pipeline import pipeline, PipelineStage
from job_store import job_store

app = FastAPI(title="Video Pipeline API", version="2.0")


class GenerateRequest(BaseModel):
    prompt: str
    video_model: str = "ltx"  # "ltx" or "wan"
    num_frames: int = 49
    width: int = 768
    height: int = 512
    seed: Optional[int] = None
    post_process: bool = True
    restore_faces: bool = True
    upscale: bool = True
    interpolate: bool = True
    target_fps: int = 60


class WebhookPayload(BaseModel):
    status: str
    output: Optional[dict] = None
    error: Optional[str] = None


@app.post("/generate")
async def generate_video(req: GenerateRequest):
    """Generate video with full pipeline"""
    job_id = await pipeline.generate_video(
        prompt=req.prompt,
        video_model=req.video_model,
        num_frames=req.num_frames,
        width=req.width,
        height=req.height,
        seed=req.seed,
        post_process=req.post_process,
        restore_faces=req.restore_faces,
        upscale=req.upscale,
        interpolate=req.interpolate,
        target_fps=req.target_fps
    )
    return {"job_id": job_id, "status": "started"}


@app.post("/callback/{job_id}")
async def webhook_callback(job_id: str, payload: WebhookPayload):
    """Receive RunPod webhook callbacks"""
    result = payload.output or {}
    if payload.error:
        result["status"] = "error"
        result["message"] = payload.error
    
    await pipeline.handle_webhook(job_id, result)
    return {"received": True}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status and results"""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "status": job.status,
        "created_at": job.created_at.isoformat(),
        "image_path": job.image_path,
        "video_path": job.video_path,
        "final_video_path": job.final_video_path,
        "error": job.error
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
