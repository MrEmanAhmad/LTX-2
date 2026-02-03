"""RunPod serverless handler for Real-ESRGAN video upscaling."""

import runpod
import logging
from model import RealESRGANModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None


def get_model() -> RealESRGANModel:
    """Lazy load the model."""
    global model
    if model is None:
        logger.info("Loading Real-ESRGAN model...")
        model = RealESRGANModel()
        logger.info("Real-ESRGAN model loaded successfully")
    return model


def handler(job: dict) -> dict:
    """
    RunPod serverless handler for Real-ESRGAN video upscaling.
    
    Input schema:
    {
        "video_path": str,
        "output_path": str,
        "scale": int = 2           # Upscale factor (2 or 4)
    }
    
    Output schema:
    {
        "video_path": str,
        "original_resolution": str,
        "new_resolution": str
    }
    """
    try:
        input_data = job["input"]
        
        video_path = input_data["video_path"]
        output_path = input_data["output_path"]
        scale = input_data.get("scale", 2)
        
        logger.info(f"Upscaling video: {video_path}, scale={scale}x")
        
        esrgan = get_model()
        result = esrgan.upscale_video(
            video_path=video_path,
            output_path=output_path,
            scale=scale
        )
        
        logger.info(f"Video upscaled: {result['video_path']}")
        return result
        
    except Exception as e:
        logger.error(f"Error upscaling video: {e}", exc_info=True)
        raise


runpod.serverless.start({"handler": handler})
