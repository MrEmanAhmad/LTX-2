"""Real-ESRGAN video upscaling model."""

import os
import subprocess
import json
import torch
import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RealESRGANModel:
    """Real-ESRGAN for video upscaling."""
    
    def __init__(self, model_name: str = "RealESRGAN_x4plus"):
        """
        Initialize Real-ESRGAN model.
        
        Args:
            model_name: Model variant to use
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load Real-ESRGAN model."""
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        
        logger.info(f"Loading Real-ESRGAN model: {self.model_name}")
        
        model_dir = os.environ.get("MODEL_DIR", "/runpod-volume/models/realesrgan")
        os.makedirs(model_dir, exist_ok=True)
        
        # Define model architecture
        if self.model_name == "RealESRGAN_x4plus":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )
            netscale = 4
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        elif self.model_name == "RealESRGAN_x2plus":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2
            )
            netscale = 2
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        model_path = os.path.join(model_dir, f"{self.model_name}.pth")
        
        # Download if needed
        if not os.path.exists(model_path):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(model_url, model_dir=model_dir)
        
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=0,  # No tiling for faster processing
            tile_pad=10,
            pre_pad=0,
            half=True if torch.cuda.is_available() else False,
            device=self.device
        )
        
        logger.info("Real-ESRGAN model loaded")
    
    def upscale_frame(self, frame: np.ndarray, outscale: int = 4) -> np.ndarray:
        """
        Upscale a single frame.
        
        Args:
            frame: Input frame (BGR numpy array)
            outscale: Output scale factor
        
        Returns:
            Upscaled frame
        """
        output, _ = self.upsampler.enhance(frame, outscale=outscale)
        return output
    
    def upscale_video(
        self,
        video_path: str,
        output_path: str,
        scale: int = 2
    ) -> dict:
        """
        Upscale a video frame by frame.
        
        Args:
            video_path: Input video path
            output_path: Output directory
            scale: Upscale factor
        
        Returns:
            Dict with video_path, original_resolution, new_resolution
        """
        output_dir = Path(output_path)
        frames_dir = output_dir / "frames"
        upscaled_dir = output_dir / "upscaled_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        upscaled_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video info
        probe_cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(video_path)
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_info = json.loads(probe_result.stdout)
        stream = video_info['streams'][0]
        
        original_width = int(stream['width'])
        original_height = int(stream['height'])
        fps_str = stream.get('r_frame_rate', '24/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den else 24
        else:
            fps = float(fps_str)
        
        new_width = original_width * scale
        new_height = original_height * scale
        
        logger.info(f"Upscaling from {original_width}x{original_height} to {new_width}x{new_height}")
        
        # Extract frames
        extract_cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-qscale:v", "2",
            str(frames_dir / "frame_%05d.png")
        ]
        subprocess.run(extract_cmd, check=True, capture_output=True)
        
        # Upscale each frame
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        logger.info(f"Upscaling {len(frame_files)} frames...")
        
        for i, frame_path in enumerate(frame_files):
            if i % 10 == 0:
                logger.info(f"Processing frame {i+1}/{len(frame_files)}")
            
            frame = cv2.imread(str(frame_path))
            upscaled = self.upscale_frame(frame, outscale=scale)
            
            output_frame_path = upscaled_dir / frame_path.name
            cv2.imwrite(str(output_frame_path), upscaled)
        
        # Reassemble video
        output_video = output_dir / "upscaled.mp4"
        reassemble_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(upscaled_dir / "frame_%05d.png"),
            "-c:v", "libx264",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(output_video)
        ]
        subprocess.run(reassemble_cmd, check=True, capture_output=True)
        
        logger.info(f"Upscaled video saved: {output_video}")
        
        return {
            "video_path": str(output_video),
            "original_resolution": f"{original_width}x{original_height}",
            "new_resolution": f"{new_width}x{new_height}"
        }
    
    def unload(self):
        """Unload model from GPU."""
        if hasattr(self, 'upsampler'):
            del self.upsampler
            torch.cuda.empty_cache()
