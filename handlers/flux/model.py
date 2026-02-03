"""FLUX.1 Dev model implementation."""

import os
import torch
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FluxModel:
    """FLUX.1 Dev image generation model."""
    
    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-dev"):
        """
        Initialize FLUX.1 model.
        
        Args:
            model_id: HuggingFace model ID
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        self._load_model()
    
    def _load_model(self):
        """Load the FLUX.1 pipeline."""
        from diffusers import FluxPipeline
        
        logger.info(f"Loading FLUX.1 from {self.model_id}")
        
        # Check for local cache first
        cache_dir = os.environ.get("HF_HOME", "/runpod-volume/models/huggingface")
        
        self.pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            cache_dir=cache_dir,
        )
        
        # Move to GPU
        self.pipe = self.pipe.to(self.device)
        
        # Enable memory optimizations
        self.pipe.enable_model_cpu_offload()
        
        logger.info("FLUX.1 model loaded and ready")
    
    def generate(
        self,
        prompt: str,
        output_path: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None
    ) -> dict:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image
            output_path: Directory to save the image
            width: Image width (default 1024)
            height: Image height (default 1024)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
        
        Returns:
            Dict with image_path and seed_used
        """
        # Set up generator with seed
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        logger.info(f"Generating with seed {seed}, steps {num_inference_steps}")
        
        # Generate image
        result = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        image = result.images[0]
        
        # Save image
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_path = output_dir / f"flux_image_{seed}.png"
        image.save(image_path, format="PNG")
        
        logger.info(f"Saved image to {image_path}")
        
        return {
            "image_path": str(image_path),
            "seed_used": seed
        }
    
    def unload(self):
        """Unload model from GPU to free memory."""
        if hasattr(self, 'pipe'):
            del self.pipe
            torch.cuda.empty_cache()
            logger.info("FLUX.1 model unloaded")
