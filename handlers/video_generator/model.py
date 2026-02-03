"""
Unified Video Generator Model
Combines: FLUX.1 Dev (image) + LTX-Video 2 (video) + Wan2.1 (video)
"""

import os
import subprocess
from pathlib import Path

# Fix torch.xpu issue before importing diffusers
import torch
if not hasattr(torch, 'xpu'):
    class FakeXPU:
        def is_available(self): return False
        def empty_cache(self): pass
        def synchronize(self): pass
        def device_count(self): return 0
    torch.xpu = FakeXPU()

# Model cache directory
MODEL_CACHE = os.environ.get("HF_HOME", "/runpod-volume/models")
os.makedirs(MODEL_CACHE, exist_ok=True)

class UnifiedVideoGenerator:
    def __init__(self):
        self.flux_pipe = None
        self.ltx_pipe = None
        self.wan_pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def load_flux(self):
        """Load FLUX.1 Dev model for image generation"""
        if self.flux_pipe is None:
            print("Loading FLUX.1 Dev model...")
            from diffusers import FluxPipeline
            self.flux_pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16,
                cache_dir=MODEL_CACHE
            ).to(self.device)
            self.flux_pipe.enable_model_cpu_offload()
            print("FLUX.1 Dev loaded successfully")
        return self.flux_pipe
    
    def load_ltx(self):
        """Load LTX-Video 2 model for video generation"""
        if self.ltx_pipe is None:
            print("Loading LTX-Video 2 model...")
            from diffusers import LTXPipeline
            self.ltx_pipe = LTXPipeline.from_pretrained(
                "Lightricks/LTX-Video",
                torch_dtype=torch.bfloat16,
                cache_dir=MODEL_CACHE
            ).to(self.device)
            self.ltx_pipe.enable_model_cpu_offload()
            print("LTX-Video 2 loaded successfully")
        return self.ltx_pipe
    
    def load_wan(self):
        """Load Wan2.1 model for video generation with better faces"""
        if self.wan_pipe is None:
            print("Loading Wan2.1-I2V-14B model...")
            from diffusers import WanPipeline
            self.wan_pipe = WanPipeline.from_pretrained(
                "Wan-AI/Wan2.1-I2V-14B-480P",
                torch_dtype=torch.bfloat16,
                cache_dir=MODEL_CACHE
            ).to(self.device)
            self.wan_pipe.enable_model_cpu_offload()
            print("Wan2.1 loaded successfully")
        return self.wan_pipe
    
    def generate_image(self, prompt: str, width: int = 1024, height: int = 1024,
                       num_inference_steps: int = 30, guidance_scale: float = 7.5,
                       seed: int = None) -> str:
        """Generate image with FLUX"""
        pipe = self.load_flux()
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        # Save to network volume
        output_path = f"/runpod-volume/outputs/flux_{os.urandom(8).hex()}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        
        return output_path
    
    def generate_video_ltx(self, prompt: str, image_path: str = None,
                           num_frames: int = 49, width: int = 768, height: int = 512,
                           num_inference_steps: int = 50, guidance_scale: float = 7.5,
                           seed: int = None) -> dict:
        """Generate video with LTX-Video"""
        from diffusers.utils import export_to_video
        from PIL import Image
        
        pipe = self.load_ltx()
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        kwargs = {
            "prompt": prompt,
            "num_frames": num_frames,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator
        }
        
        if image_path and os.path.exists(image_path):
            kwargs["image"] = Image.open(image_path).convert("RGB")
        
        video_frames = pipe(**kwargs).frames[0]
        
        # Save video
        output_path = f"/runpod-volume/outputs/ltx_{os.urandom(8).hex()}.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        export_to_video(video_frames, output_path, fps=24)
        
        # Extract last frame for chaining
        last_frame_path = output_path.replace(".mp4", "_last_frame.png")
        self._extract_last_frame(output_path, last_frame_path)
        
        return {
            "video_path": output_path,
            "last_frame_path": last_frame_path,
            "num_frames": len(video_frames)
        }
    
    def generate_video_wan(self, prompt: str, image_path: str,
                           num_frames: int = 49, width: int = 848, height: int = 480,
                           num_inference_steps: int = 50, guidance_scale: float = 5.0,
                           seed: int = None) -> dict:
        """Generate video with Wan2.1 (requires input image)"""
        from diffusers.utils import export_to_video
        from PIL import Image
        
        if not image_path or not os.path.exists(image_path):
            raise ValueError("Wan2.1 requires an input image")
        
        pipe = self.load_wan()
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        image = Image.open(image_path).convert("RGB").resize((width, height))
        
        video_frames = pipe(
            prompt=prompt,
            image=image,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).frames[0]
        
        # Save video
        output_path = f"/runpod-volume/outputs/wan_{os.urandom(8).hex()}.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        export_to_video(video_frames, output_path, fps=24)
        
        # Extract last frame for chaining
        last_frame_path = output_path.replace(".mp4", "_last_frame.png")
        self._extract_last_frame(output_path, last_frame_path)
        
        return {
            "video_path": output_path,
            "last_frame_path": last_frame_path,
            "num_frames": len(video_frames)
        }
    
    def _extract_last_frame(self, video_path: str, output_path: str):
        """Extract the last frame from a video using FFmpeg"""
        cmd = [
            "ffmpeg", "-y", "-sseof", "-1", "-i", video_path,
            "-update", "1", "-q:v", "2", output_path
        ]
        subprocess.run(cmd, capture_output=True)
    
    def full_pipeline(self, prompt: str, video_model: str = "ltx",
                      num_frames: int = 49, width: int = 768, height: int = 512,
                      num_inference_steps: int = 50, guidance_scale: float = 7.5,
                      seed: int = None) -> dict:
        """
        Full pipeline: Generate image with FLUX, then video with LTX or Wan
        """
        # Step 1: Generate image
        print(f"Step 1: Generating image with FLUX...")
        image_path = self.generate_image(
            prompt=prompt,
            width=width,
            height=height,
            seed=seed
        )
        print(f"Image generated: {image_path}")
        
        # Step 2: Generate video
        print(f"Step 2: Generating video with {video_model}...")
        if video_model.lower() == "wan":
            result = self.generate_video_wan(
                prompt=prompt,
                image_path=image_path,
                num_frames=num_frames,
                width=848,
                height=480,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed
            )
        else:  # default to ltx
            result = self.generate_video_ltx(
                prompt=prompt,
                image_path=image_path,
                num_frames=num_frames,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed
            )
        
        result["image_path"] = image_path
        return result


# Global instance for warm starts
_model = None

def get_model():
    global _model
    if _model is None:
        _model = UnifiedVideoGenerator()
    return _model
