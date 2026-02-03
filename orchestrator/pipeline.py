"""Pipeline orchestration logic with webhook-driven state machine."""

import asyncio
from uuid import uuid4
from pathlib import Path
from typing import Optional
from config import get_settings
from runpod_client import get_runpod_client
from job_store import get_job_store
import logging
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.schemas import (
    GenerateRequest, MultiClipRequest, JobState, VideoModel
)

logger = logging.getLogger(__name__)


class VideoPipeline:
    """
    Orchestrates the video generation pipeline.
    
    Uses webhook callbacks to progress through pipeline stages.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client = get_runpod_client()
        self.store = get_job_store()
        self.volume_path = self.settings.runpod_volume_path
    
    async def start_single_clip(self, request: GenerateRequest) -> dict:
        """
        Start a single-clip video generation pipeline.
        
        Args:
            request: Generation request
        
        Returns:
            Dict with job_id and initial status
        """
        job_id = str(uuid4())
        base_path = f"{self.volume_path}/jobs/{job_id}"
        
        # Create job state
        await self.store.create(job_id, {
            "status": JobState.GENERATING_IMAGE,
            "request": request.model_dump(),
            "base_path": base_path,
            "current_step": "flux",
            "results": {}
        })
        
        # Start FLUX image generation
        await self.client.run_async(
            "flux",
            {
                "prompt": request.image_prompt,
                "output_path": f"{base_path}/images",
                "seed": request.seed
            },
            job_id
        )
        
        logger.info(f"Started single-clip pipeline: {job_id}")
        
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Generating first frame with FLUX"
        }
    
    async def start_multi_clip(self, request: MultiClipRequest) -> dict:
        """
        Start a multi-clip video generation pipeline.
        
        Args:
            request: Multi-clip generation request
        
        Returns:
            Dict with job_id and initial status
        """
        job_id = str(uuid4())
        base_path = f"{self.volume_path}/jobs/{job_id}"
        
        # Validate first clip has image prompt
        if not request.clips[0].image_prompt:
            raise ValueError("First clip must have an image_prompt")
        
        # Create job state
        await self.store.create(job_id, {
            "status": JobState.GENERATING_IMAGE,
            "request": request.model_dump(),
            "base_path": base_path,
            "current_clip": 0,
            "total_clips": len(request.clips),
            "current_step": "flux",
            "clip_videos": [],
            "results": {}
        })
        
        # Start FLUX for first clip
        await self.client.run_async(
            "flux",
            {
                "prompt": request.clips[0].image_prompt,
                "output_path": f"{base_path}/clip_0/images",
                "seed": request.seed
            },
            job_id
        )
        
        logger.info(f"Started multi-clip pipeline: {job_id}, clips={len(request.clips)}")
        
        return {
            "job_id": job_id,
            "status": "processing",
            "message": f"Generating first frame for clip 1/{len(request.clips)}"
        }
    
    async def handle_webhook(self, job_id: str, payload: dict) -> None:
        """
        Handle webhook callback from RunPod.
        
        Progresses the pipeline based on current state.
        
        Args:
            job_id: Our internal job ID
            payload: Webhook payload from RunPod
        """
        job = await self.store.get(job_id)
        if not job:
            logger.error(f"Job not found for webhook: {job_id}")
            return
        
        # Check for errors
        if payload.get("status") == "FAILED":
            error_msg = payload.get("error", "Unknown error")
            await self.store.update(job_id, {
                "status": JobState.FAILED,
                "error": error_msg
            })
            logger.error(f"Job {job_id} failed: {error_msg}")
            return
        
        output = payload.get("output", {})
        current_step = job.get("current_step")
        
        logger.info(f"Webhook received for {job_id}, step={current_step}")
        
        # Route to appropriate handler
        if current_step == "flux":
            await self._handle_flux_complete(job_id, job, output)
        elif current_step == "video":
            await self._handle_video_complete(job_id, job, output)
        elif current_step == "codeformer":
            await self._handle_codeformer_complete(job_id, job, output)
        elif current_step == "rife":
            await self._handle_rife_complete(job_id, job, output)
        elif current_step == "realesrgan":
            await self._handle_realesrgan_complete(job_id, job, output)
    
    async def _handle_flux_complete(
        self, job_id: str, job: dict, output: dict
    ) -> None:
        """Handle FLUX completion - start video generation."""
        image_path = output["image_path"]
        request = job["request"]
        base_path = job["base_path"]
        current_clip = job.get("current_clip", 0)
        
        # Determine video model
        video_model = request.get("video_model", "ltx")
        handler = "ltx_video" if video_model == "ltx" else "wan21"
        
        # Get motion prompt
        if "clips" in request:
            # Multi-clip
            motion_prompt = request["clips"][current_clip]["motion_prompt"]
            duration = request["clips"][current_clip].get("duration_seconds", 5)
        else:
            # Single clip
            motion_prompt = request["motion_prompt"]
            duration = request.get("duration_seconds", 5)
        
        # Calculate frames (24fps)
        num_frames = int(duration * 24) + 1
        
        await self.store.update(job_id, {
            "status": JobState.GENERATING_VIDEO,
            "current_step": "video",
            "results": {**job.get("results", {}), "image_path": image_path}
        })
        
        await self.client.run_async(
            handler,
            {
                "image_path": image_path,
                "prompt": motion_prompt,
                "output_path": f"{base_path}/clip_{current_clip}/video",
                "num_frames": num_frames,
                "seed": request.get("seed")
            },
            job_id
        )
        
        logger.info(f"Job {job_id}: Started video generation with {handler}")
    
    async def _handle_video_complete(
        self, job_id: str, job: dict, output: dict
    ) -> None:
        """Handle video completion - check for more clips or proceed to post-processing."""
        video_path = output["video_path"]
        last_frame_path = output.get("last_frame_path")
        request = job["request"]
        
        # Store video result
        clip_videos = job.get("clip_videos", [])
        clip_videos.append(video_path)
        
        if "clips" in request:
            # Multi-clip mode
            current_clip = job.get("current_clip", 0)
            total_clips = job.get("total_clips", 1)
            
            if current_clip < total_clips - 1:
                # More clips to generate - use last frame as input
                next_clip = current_clip + 1
                motion_prompt = request["clips"][next_clip]["motion_prompt"]
                duration = request["clips"][next_clip].get("duration_seconds", 5)
                num_frames = int(duration * 24) + 1
                
                video_model = request.get("video_model", "ltx")
                handler = "ltx_video" if video_model == "ltx" else "wan21"
                
                await self.store.update(job_id, {
                    "current_clip": next_clip,
                    "current_step": "video",
                    "clip_videos": clip_videos
                })
                
                await self.client.run_async(
                    handler,
                    {
                        "image_path": last_frame_path,
                        "prompt": motion_prompt,
                        "output_path": f"{job['base_path']}/clip_{next_clip}/video",
                        "num_frames": num_frames,
                        "seed": request.get("seed")
                    },
                    job_id
                )
                
                logger.info(f"Job {job_id}: Started clip {next_clip + 1}/{total_clips}")
                return
        
        # All clips done - proceed to post-processing
        await self.store.update(job_id, {
            "clip_videos": clip_videos,
            "results": {**job.get("results", {}), "raw_video_path": video_path}
        })
        
        # Stitch clips if multiple
        if len(clip_videos) > 1:
            # TODO: Call FFmpeg to stitch clips
            # For now, use the last clip as the video path
            video_path = clip_videos[-1]
        
        # Start face restoration if requested
        if request.get("apply_face_restore", True):
            await self._start_codeformer(job_id, job, video_path)
        elif request.get("interpolate", False):
            await self._start_rife(job_id, job, video_path)
        elif request.get("upscale", False):
            await self._start_realesrgan(job_id, job, video_path)
        else:
            await self._complete_job(job_id, job, video_path)
    
    async def _start_codeformer(
        self, job_id: str, job: dict, video_path: str
    ) -> None:
        """Start CodeFormer face restoration."""
        await self.store.update(job_id, {
            "status": JobState.FACE_RESTORATION,
            "current_step": "codeformer"
        })
        
        await self.client.run_async(
            "codeformer",
            {
                "video_path": video_path,
                "output_path": f"{job['base_path']}/codeformer"
            },
            job_id
        )
        
        logger.info(f"Job {job_id}: Started face restoration")
    
    async def _handle_codeformer_complete(
        self, job_id: str, job: dict, output: dict
    ) -> None:
        """Handle CodeFormer completion."""
        video_path = output["video_path"]
        request = job["request"]
        
        await self.store.update(job_id, {
            "results": {**job.get("results", {}), "restored_video_path": video_path}
        })
        
        if request.get("interpolate", False):
            await self._start_rife(job_id, job, video_path)
        elif request.get("upscale", False):
            await self._start_realesrgan(job_id, job, video_path)
        else:
            await self._complete_job(job_id, job, video_path)
    
    async def _start_rife(
        self, job_id: str, job: dict, video_path: str
    ) -> None:
        """Start RIFE frame interpolation."""
        await self.store.update(job_id, {
            "status": JobState.FRAME_INTERPOLATION,
            "current_step": "rife"
        })
        
        await self.client.run_async(
            "rife",
            {
                "video_path": video_path,
                "output_path": f"{job['base_path']}/rife",
                "multiplier": 2  # 24fps -> 48fps (or 2.5 for 60fps)
            },
            job_id
        )
        
        logger.info(f"Job {job_id}: Started frame interpolation")
    
    async def _handle_rife_complete(
        self, job_id: str, job: dict, output: dict
    ) -> None:
        """Handle RIFE completion."""
        video_path = output["video_path"]
        request = job["request"]
        
        await self.store.update(job_id, {
            "results": {**job.get("results", {}), "interpolated_video_path": video_path}
        })
        
        if request.get("upscale", False):
            await self._start_realesrgan(job_id, job, video_path)
        else:
            await self._complete_job(job_id, job, video_path)
    
    async def _start_realesrgan(
        self, job_id: str, job: dict, video_path: str
    ) -> None:
        """Start Real-ESRGAN upscaling."""
        await self.store.update(job_id, {
            "status": JobState.UPSCALING,
            "current_step": "realesrgan"
        })
        
        await self.client.run_async(
            "realesrgan",
            {
                "video_path": video_path,
                "output_path": f"{job['base_path']}/upscaled",
                "scale": 2  # 2x upscale (480p -> 960p or 720p -> 1440p)
            },
            job_id
        )
        
        logger.info(f"Job {job_id}: Started upscaling")
    
    async def _handle_realesrgan_complete(
        self, job_id: str, job: dict, output: dict
    ) -> None:
        """Handle Real-ESRGAN completion."""
        video_path = output["video_path"]
        
        await self.store.update(job_id, {
            "results": {**job.get("results", {}), "upscaled_video_path": video_path}
        })
        
        await self._complete_job(job_id, job, video_path)
    
    async def _complete_job(
        self, job_id: str, job: dict, final_video_path: str
    ) -> None:
        """Mark job as completed."""
        await self.store.update(job_id, {
            "status": JobState.COMPLETED,
            "current_step": None,
            "results": {
                **job.get("results", {}),
                "final_video_path": final_video_path
            }
        })
        
        logger.info(f"Job {job_id}: Completed! Final video: {final_video_path}")


# Singleton instance
_pipeline: Optional[VideoPipeline] = None


def get_pipeline() -> VideoPipeline:
    """Get the pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = VideoPipeline()
    return _pipeline
