"""
Unified Post-Processor Handler for RunPod
Supports: restore_faces, upscale, interpolate, full_post_process
"""

import runpod
from model import get_model


def handler(job):
    """
    RunPod handler for unified post-processing
    
    Actions:
    - restore_faces: Apply CodeFormer face restoration to video
    - upscale: Upscale video with Real-ESRGAN (4x by default)
    - interpolate: Interpolate frames with RIFE (24fps -> 60fps)
    - full_post_process: All three in sequence (no cold starts!)
    """
    try:
        job_input = job["input"]
        action = job_input.get("action", "full_post_process")
        video_path = job_input.get("video_path")
        
        if not video_path:
            return {"status": "error", "message": "video_path is required"}
        
        model = get_model()
        
        if action == "restore_faces":
            result = model.restore_faces_video(
                video_path=video_path,
                fidelity_weight=job_input.get("fidelity_weight", 0.7)
            )
            return {"status": "success", "video_path": result}
        
        elif action == "upscale":
            result = model.upscale_video(
                video_path=video_path,
                scale=job_input.get("scale", 4)
            )
            return {"status": "success", "video_path": result}
        
        elif action == "interpolate":
            result = model.interpolate_video(
                video_path=video_path,
                target_fps=job_input.get("target_fps", 60)
            )
            return {"status": "success", "video_path": result}
        
        elif action == "full_post_process":
            result = model.full_post_process(
                video_path=video_path,
                restore_faces=job_input.get("restore_faces", True),
                upscale=job_input.get("upscale", True),
                interpolate=job_input.get("interpolate", True),
                target_fps=job_input.get("target_fps", 60),
                upscale_factor=job_input.get("upscale_factor", 4),
                fidelity_weight=job_input.get("fidelity_weight", 0.7)
            )
            return {"status": "success", **result}
        
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}


runpod.serverless.start({"handler": handler})
