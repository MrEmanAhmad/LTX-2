"""RunPod serverless handler for LTX-Video 2 video generation."""

import runpod
import logging
from model import LTXVideoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None


def get_model() -> LTXVideoModel:
    """Lazy load the model."""
    global model
    if model is None:
        logger.info("Loading LTX-Video model...")
        model = LTXVideoModel()
        logger.info("LTX-Video model loaded successfully")
    return model


def handler(job: dict) -> dict:
    """
    RunPod serverless handler for LTX-Video generation.
    
    Input schema:
    {
        "image_path": str,          # Path to input image
        "prompt": str,              # Motion/action prompt
        "output_path": str,         # Path on RunPod volume
        "num_frames": int = 121,    # ~5 seconds at 24fps
        "fps": int = 24,
        "seed": int | None = None
    }
    
    Output schema:
    {
        "video_path": str,          # Path to generated video
        "last_frame_path": str,     # Path to last frame (for chaining)
        "num_frames": int,
        "fps": int,
        "seed_used": int
    }
    """
    try:
        input_data = job["input"]
        
        # Extract parameters
        image_path = input_data["image_path"]
        prompt = input_data["prompt"]
        output_path = input_data["output_path"]
        num_frames = input_data.get("num_frames", 121)
        fps = input_data.get("fps", 24)
        seed = input_data.get("seed")
        
        logger.info(f"Generating video: prompt='{prompt[:50]}...', frames={num_frames}")
        
        # Get model and generate
        ltx = get_model()
        result = ltx.generate(
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


# Start the serverless handler
runpod.serverless.start({"handler": handler})
