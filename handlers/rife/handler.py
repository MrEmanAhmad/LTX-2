"""RunPod serverless handler for RIFE frame interpolation."""

import runpod
import logging
from model import RIFEModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None


def get_model() -> RIFEModel:
    """Lazy load the model."""
    global model
    if model is None:
        logger.info("Loading RIFE model...")
        model = RIFEModel()
        logger.info("RIFE model loaded successfully")
    return model


def handler(job: dict) -> dict:
    """
    RunPod serverless handler for RIFE frame interpolation.
    
    Input schema:
    {
        "video_path": str,
        "output_path": str,
        "multiplier": int = 2      # 2 = double fps, 2.5 = 2.5x fps
    }
    
    Output schema:
    {
        "video_path": str,
        "original_fps": int,
        "new_fps": int
    }
    """
    try:
        input_data = job["input"]
        
        video_path = input_data["video_path"]
        output_path = input_data["output_path"]
        multiplier = input_data.get("multiplier", 2)
        
        logger.info(f"Interpolating video: {video_path}, multiplier={multiplier}")
        
        rife = get_model()
        result = rife.interpolate(
            video_path=video_path,
            output_path=output_path,
            multiplier=multiplier
        )
        
        logger.info(f"Video interpolated: {result['video_path']}")
        return result
        
    except Exception as e:
        logger.error(f"Error interpolating video: {e}", exc_info=True)
        raise


runpod.serverless.start({"handler": handler})
