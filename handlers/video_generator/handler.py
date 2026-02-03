"""
Unified Video Generator Handler for RunPod
Supports: generate_image, generate_video, full_pipeline
"""

# Fix torch.xpu issue before any imports
import torch
if not hasattr(torch, 'xpu'):
    class FakeXPU:
        def is_available(self): return False
        def empty_cache(self): pass
        def synchronize(self): pass
        def device_count(self): return 0
    torch.xpu = FakeXPU()

import runpod

print("=" * 50)
print("VIDEO GENERATOR HANDLER STARTING")
print("=" * 50)

from model import get_model

print("Handler module loaded successfully")


def handler(job):
    print(f"Received job: {job.get('id', 'unknown')}")
    """
    RunPod handler for unified video generation
    
    Actions:
    - generate_image: Generate image with FLUX
    - generate_video_ltx: Generate video with LTX-Video
    - generate_video_wan: Generate video with Wan2.1 (requires image)
    - full_pipeline: Image + Video in one call (no cold start between)
    """
    try:
        job_input = job["input"]
        action = job_input.get("action", "full_pipeline")
        
        model = get_model()
        
        if action == "generate_image":
            result = model.generate_image(
                prompt=job_input["prompt"],
                width=job_input.get("width", 1024),
                height=job_input.get("height", 1024),
                num_inference_steps=job_input.get("num_inference_steps", 30),
                guidance_scale=job_input.get("guidance_scale", 7.5),
                seed=job_input.get("seed")
            )
            return {"status": "success", "image_path": result}
        
        elif action == "generate_video_ltx":
            result = model.generate_video_ltx(
                prompt=job_input["prompt"],
                image_path=job_input.get("image_path"),
                num_frames=job_input.get("num_frames", 49),
                width=job_input.get("width", 768),
                height=job_input.get("height", 512),
                num_inference_steps=job_input.get("num_inference_steps", 50),
                guidance_scale=job_input.get("guidance_scale", 7.5),
                seed=job_input.get("seed")
            )
            return {"status": "success", **result}
        
        elif action == "generate_video_wan":
            result = model.generate_video_wan(
                prompt=job_input["prompt"],
                image_path=job_input["image_path"],  # Required for Wan
                num_frames=job_input.get("num_frames", 49),
                width=job_input.get("width", 848),
                height=job_input.get("height", 480),
                num_inference_steps=job_input.get("num_inference_steps", 50),
                guidance_scale=job_input.get("guidance_scale", 5.0),
                seed=job_input.get("seed")
            )
            return {"status": "success", **result}
        
        elif action == "full_pipeline":
            result = model.full_pipeline(
                prompt=job_input["prompt"],
                video_model=job_input.get("video_model", "ltx"),
                num_frames=job_input.get("num_frames", 49),
                width=job_input.get("width", 768),
                height=job_input.get("height", 512),
                num_inference_steps=job_input.get("num_inference_steps", 50),
                guidance_scale=job_input.get("guidance_scale", 7.5),
                seed=job_input.get("seed")
            )
            return {"status": "success", **result}
        
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}


runpod.serverless.start({"handler": handler})
