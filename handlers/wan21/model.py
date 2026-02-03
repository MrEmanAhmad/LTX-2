"""Wan2.1 I2V-14B-480P model implementation."""

import os
import subprocess
import torch
from pathlib import Path
from typing import Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class Wan21Model:
    """Wan2.1 Image-to-Video generation model (480P variant for 48GB VRAM)."""
    
    def __init__(self, model_id: str = "Wan-AI/Wan2.1-I2V-14B-480P"):
        """
        Initialize Wan2.1 model.
        
        Args:
            model_id: HuggingFace model ID
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        
        self._load_model()
    
    def _load_model(self):
        """Load the Wan2.1 pipeline."""
        from diffusers import WanImageToVideoPipeline
        
        logger.info(f"Loading Wan2.1-I2V from {self.model_id}")
        
        cache_dir = os.environ.get("HF_HOME", "/runpod-volume/models/huggingface")
        
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            cache_dir=cache_dir,
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Memory optimizations for 48GB
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_tiling()
        
        logger.info("Wan2.1-I2V model loaded and ready")
    
    def generate(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        num_frames: int = 81,
        fps: int = 24,
        seed: Optional[int] = None
    ) -> dict:
        """
        Generate video from an image with Wan2.1.
        
        Args:
            image_path: Path to input image
            prompt: Motion/action description
            output_path: Directory to save output
            num_frames: Number of frames (Wan2.1 supports up to ~81)
            fps: Frames per second
            seed: Random seed
        
        Returns:
            Dict with video_path, last_frame_path, etc.
        """
        # Load and prepare image
        image = Image.open(image_path).convert("RGB")
        
        # Wan2.1-480P dimensions
        width, height = 832, 480
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        logger.info(f"Generating {num_frames} frames with seed {seed}")
        
        result = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted face, deformed, bad anatomy",
            num_frames=num_frames,
            guidance_scale=5.0,
            num_inference_steps=30,
            generator=generator,
        )
        
        frames = result.frames[0]
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = output_dir / f"wan21_video_{seed}.mp4"
        
        import imageio
        
        imageio.mimwrite(
            str(video_path),
            frames,
            fps=fps,
            codec='libx264',
            quality=8,
            pixelformat='yuv420p'
        )
        
        # Extract last frame for chaining
        last_frame_path = output_dir / "last_frame.png"
        frames[-1].save(last_frame_path, format="PNG")
        
        logger.info(f"Saved video to {video_path}")
        
        return {
            "video_path": str(video_path),
            "last_frame_path": str(last_frame_path),
            "num_frames": len(frames),
            "fps": fps,
            "seed_used": seed
        }
    
    def unload(self):
        """Unload model from GPU."""
        if hasattr(self, 'pipe'):
            del self.pipe
            torch.cuda.empty_cache()
