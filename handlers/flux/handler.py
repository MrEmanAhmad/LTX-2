"""RunPod serverless handler for FLUX.1 Dev image generation."""

import runpod
import logging
from model import FluxModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance (loaded once, reused across requests)
model = None


def get_model() -> FluxModel:
    """Lazy load the model."""
    global model
    if model is None:
        logger.info("Loading FLUX.1 model...")
        model = FluxModel()
        logger.info("FLUX.1 model loaded successfully")
    return model


def handler(job: dict) -> dict:
    """
    RunPod serverless handler for FLUX.1 image generation.
    
    Input schema:
    {
        "prompt": str,              # Image generation prompt
        "output_path": str,         # Path on RunPod volume
        "width": int = 1024,        # Image width
        "height": int = 1024,       # Image height
        "num_inference_steps": int = 28,
        "guidance_scale": float = 3.5,
        "seed": int | None = None
    }
    
    Output schema:
    {
        "image_path": str,          # Path to generated image
        "seed_used": int            # Seed used for generation
    }
    """
    try:
        input_data = job["input"]
        
        # Extract parameters
        prompt = input_data["prompt"]
        output_path = input_data["output_path"]
        width = input_data.get("width", 1024)
        height = input_data.get("height", 1024)
        num_inference_steps = input_data.get("num_inference_steps", 28)
        guidance_scale = input_data.get("guidance_scale", 3.5)
        seed = input_data.get("seed")
        
        logger.info(f"Generating image: prompt='{prompt[:50]}...', size={width}x{height}")
        
        # Get model and generate
        flux = get_model()
        result = flux.generate(
            prompt=prompt,
            output_path=output_path,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        logger.info(f"Image generated: {result['image_path']}")
        
        return {
            "image_path": result["image_path"],
            "seed_used": result["seed_used"]
        }
        
    except Exception as e:
        logger.error(f"Error generating image: {e}", exc_info=True)
        raise


# Start the serverless handler
runpod.serverless.start({"handler": handler})
