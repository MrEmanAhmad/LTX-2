"""RunPod serverless handler for CodeFormer face restoration."""

import runpod
import logging
from model import CodeFormerModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None


def get_model() -> CodeFormerModel:
    """Lazy load the model."""
    global model
    if model is None:
        logger.info("Loading CodeFormer model...")
        model = CodeFormerModel()
        logger.info("CodeFormer model loaded successfully")
    return model


def handler(job: dict) -> dict:
    """
    RunPod serverless handler for CodeFormer face restoration.
    
    Processes video by extracting frames, restoring each frame,
    then reassembling into video.
    
    Input schema:
    {
        "video_path": str,          # Input video path
        "output_path": str,         # Output directory
        "fidelity_weight": float = 0.5,  # 0=quality, 1=fidelity
        "upscale": int = 1          # Upscale factor
    }
    
    Output schema:
    {
        "video_path": str,
        "frames_processed": int
    }
    """
    try:
        input_data = job["input"]
        
        video_path = input_data["video_path"]
        output_path = input_data["output_path"]
        fidelity_weight = input_data.get("fidelity_weight", 0.5)
        upscale = input_data.get("upscale", 1)
        
        logger.info(f"Restoring faces in video: {video_path}")
        
        cf = get_model()
        result = cf.restore_video(
            video_path=video_path,
            output_path=output_path,
            fidelity_weight=fidelity_weight,
            upscale=upscale
        )
        
        logger.info(f"Video restored: {result['video_path']}")
        return result
        
    except Exception as e:
        logger.error(f"Error restoring video: {e}", exc_info=True)
        raise


runpod.serverless.start({"handler": handler})
