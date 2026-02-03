"""LTX-Video 2 model implementation with frame chaining support."""

import os
import subprocess
import torch
from pathlib import Path
from typing import Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class LTXVideoModel:
    """LTX-Video 2 image-to-video generation model."""
    
    def __init__(self, model_id: str = "Lightricks/LTX-Video"):
        """
        Initialize LTX-Video model.
        
        Args:
            model_id: HuggingFace model ID
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        self._load_model()
    
    def _load_model(self):
        """Load the LTX-Video pipeline."""
        from diffusers import LTXImageToVideoPipeline
        
        logger.info(f"Loading LTX-Video from {self.model_id}")
        
        cache_dir = os.environ.get("HF_HOME", "/runpod-volume/models/huggingface")
        
        self.pipe = LTXImageToVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            cache_dir=cache_dir,
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Enable memory optimizations
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()
        
        logger.info("LTX-Video model loaded and ready")
    
    def generate(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        num_frames: int = 121,
        fps: int = 24,
        seed: Optional[int] = None
    ) -> dict:
        """
        Generate video from an image.
        
        Args:
            image_path: Path to input image
            prompt: Motion/action description
            output_path: Directory to save output
            num_frames: Number of frames to generate
            fps: Frames per second
            seed: Random seed for reproducibility
        
        Returns:
            Dict with video_path, last_frame_path, num_frames, fps, seed_used
        """
        # Load input image
        image = Image.open(image_path).convert("RGB")
        
        # Resize to model's expected dimensions
        width, height = 768, 512  # LTX-Video standard
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Set up generator with seed
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        logger.info(f"Generating {num_frames} frames with seed {seed}")
        
        # Generate video
        result = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted, deformed",
            num_frames=num_frames,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator,
        )
        
        frames = result.frames[0]  # List of PIL images
        
        # Save video
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = output_dir / f"ltx_video_{seed}.mp4"
        
        # Export frames to video using imageio
        import imageio
        
        # Convert PIL images to numpy arrays
        frame_arrays = [frame for frame in frames]
        
        imageio.mimwrite(
            str(video_path),
            frame_arrays,
            fps=fps,
            codec='libx264',
            quality=8,
            pixelformat='yuv420p'
        )
        
        # Extract last frame for continuity chaining
        last_frame_path = output_dir / "last_frame.png"
        frames[-1].save(last_frame_path, format="PNG")
        
        logger.info(f"Saved video to {video_path}")
        logger.info(f"Extracted last frame to {last_frame_path}")
        
        return {
            "video_path": str(video_path),
            "last_frame_path": str(last_frame_path),
            "num_frames": len(frames),
            "fps": fps,
            "seed_used": seed
        }
    
    def extract_last_frame(self, video_path: str, output_path: str) -> str:
        """
        Extract the last frame from a video using FFmpeg.
        
        Args:
            video_path: Input video path
            output_path: Output image path
        
        Returns:
            Path to extracted frame
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y",
            "-sseof", "-0.1",
            "-i", str(video_path),
            "-update", "1",
            "-q:v", "2",
            str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Extracted last frame to {output_path}")
        
        return str(output_path)
    
    def unload(self):
        """Unload model from GPU."""
        if hasattr(self, 'pipe'):
            del self.pipe
            torch.cuda.empty_cache()
            logger.info("LTX-Video model unloaded")
