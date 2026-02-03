"""RunPod serverless handler for Wan2.1 I2V video generation."""

import runpod
import logging
from model import Wan21Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None


def get_model() -> Wan21Model:
    """Lazy load the model."""
    global model
    if model is None:
        logger.info("Loading Wan2.1-I2V model...")
        model = Wan21Model()
        logger.info("Wan2.1-I2V model loaded successfully")
    return model


def handler(job: dict) -> dict:
    """
    RunPod serverless handler for Wan2.1 I2V video generation.
    
    Input schema:
    {
        "image_path": str,
        "prompt": str,
        "output_path": str,
        "num_frames": int = 81,     # ~3 seconds at 24fps
        "fps": int = 24,
        "seed": int | None = None
    }
    
    Output schema:
    {
        "video_path": str,
        "last_frame_path": str,
        "num_frames": int,
        "fps": int,
        "seed_used": int
    }
    """
    try:
        input_data = job["input"]
        
        image_path = input_data["image_path"]
        prompt = input_data["prompt"]
        output_path = input_data["output_path"]
        num_frames = input_data.get("num_frames", 81)
        fps = input_data.get("fps", 24)
        seed = input_data.get("seed")
        
        logger.info(f"Generating video: prompt='{prompt[:50]}...', frames={num_frames}")
        
        wan = get_model()
        result = wan.generate(
            image_path=image_path,
            prompt=prompt,
            output_path=output_path,
            num_frames=num_frames,
            fps=fps,
            seed=seed
        )
        
        logger.info(f"Video generated: {result['video_path']}")
        return result
        
    except Exception as e:
        logger.error(f"Error generating video: {e}", exc_info=True)
        raise


runpod.serverless.start({"handler": handler})
