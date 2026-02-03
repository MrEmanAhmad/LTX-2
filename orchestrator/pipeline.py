"""
Simplified pipeline for 2-endpoint architecture
"""

import uuid
from enum import Enum
from job_store import job_store, JobState
from runpod_client import runpod_client


class PipelineStage(str, Enum):
    GENERATING_VIDEO = "generating_video"
    POST_PROCESSING = "post_processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Pipeline:
    
    async def generate_video(
        self,
        prompt: str,
        video_model: str = "ltx",
        num_frames: int = 49,
        width: int = 768,
        height: int = 512,
        seed: int = None,
        post_process: bool = True,
        restore_faces: bool = True,
        upscale: bool = True,
        interpolate: bool = True,
        target_fps: int = 60
    ) -> str:
        """
        Full pipeline: Generate video, then optionally post-process
        Returns job_id to track progress
        """
        job_id = str(uuid.uuid4())
        
        # Store job config for webhook handler
        job_store.create(job_id, JobState(
            status=PipelineStage.GENERATING_VIDEO,
            config={
                "prompt": prompt,
                "video_model": video_model,
                "post_process": post_process,
                "restore_faces": restore_faces,
                "upscale": upscale,
                "interpolate": interpolate,
                "target_fps": target_fps
            }
        ))
        
        # Start video generation (FLUX + LTX/Wan in one call)
        await runpod_client.run_video_generator(
            action="full_pipeline",
            params={
                "prompt": prompt,
                "video_model": video_model,
                "num_frames": num_frames,
                "width": width,
                "height": height,
                "seed": seed
            },
            job_id=job_id
        )
        
        return job_id
    
    async def handle_webhook(self, job_id: str, result: dict):
        """Handle RunPod webhook callback"""
        job = job_store.get(job_id)
        if not job:
            return
        
        if result.get("status") == "error":
            job.status = PipelineStage.FAILED
            job.error = result.get("message", "Unknown error")
            job_store.update(job_id, job)
            return
        
        if job.status == PipelineStage.GENERATING_VIDEO:
            # Video generation complete
            video_path = result.get("video_path")
            job.video_path = video_path
            job.last_frame_path = result.get("last_frame_path")
            job.image_path = result.get("image_path")
            
            if job.config.get("post_process", True):
                # Start post-processing
                job.status = PipelineStage.POST_PROCESSING
                job_store.update(job_id, job)
                
                await runpod_client.run_post_processor(
                    action="full_post_process",
                    params={
                        "video_path": video_path,
                        "restore_faces": job.config.get("restore_faces", True),
                        "upscale": job.config.get("upscale", True),
                        "interpolate": job.config.get("interpolate", True),
                        "target_fps": job.config.get("target_fps", 60)
                    },
                    job_id=job_id
                )
            else:
                # No post-processing, we're done
                job.status = PipelineStage.COMPLETED
                job.final_video_path = video_path
                job_store.update(job_id, job)
        
        elif job.status == PipelineStage.POST_PROCESSING:
            # Post-processing complete
            job.status = PipelineStage.COMPLETED
            job.final_video_path = result.get("final_path")
            job.restored_path = result.get("restored_path")
            job.upscaled_path = result.get("upscaled_path")
            job.interpolated_path = result.get("interpolated_path")
            job_store.update(job_id, job)


pipeline = Pipeline()
