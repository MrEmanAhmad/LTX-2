"""RIFE frame interpolation model."""

import os
import subprocess
import json
import torch
import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RIFEModel:
    """RIFE (Real-Time Intermediate Flow Estimation) for frame interpolation."""
    
    def __init__(self):
        """Initialize RIFE model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load RIFE model."""
        from model.RIFE import Model
        
        logger.info("Loading RIFE model...")
        
        model_dir = os.environ.get("MODEL_DIR", "/runpod-volume/models/rife")
        
        self.model = Model()
        self.model.load_model(model_dir, -1)
        self.model.eval()
        self.model.device()
        
        logger.info("RIFE model loaded")
    
    def interpolate_frames(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_intermediate: int = 1
    ) -> list[np.ndarray]:
        """
        Generate intermediate frames between two frames.
        
        Args:
            frame1: First frame (BGR numpy array)
            frame2: Second frame (BGR numpy array)
            num_intermediate: Number of frames to generate
        
        Returns:
            List of intermediate frames
        """
        # Convert to tensor
        img0 = torch.from_numpy(frame1.transpose(2, 0, 1)).float() / 255.0
        img1 = torch.from_numpy(frame2.transpose(2, 0, 1)).float() / 255.0
        
        img0 = img0.unsqueeze(0).to(self.device)
        img1 = img1.unsqueeze(0).to(self.device)
        
        # Pad to multiple of 32
        h, w = frame1.shape[:2]
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        
        img0 = torch.nn.functional.pad(img0, padding)
        img1 = torch.nn.functional.pad(img1, padding)
        
        intermediate_frames = []
        
        for i in range(num_intermediate):
            timestep = (i + 1) / (num_intermediate + 1)
            
            with torch.no_grad():
                mid = self.model.inference(img0, img1, timestep)
            
            # Remove padding and convert back
            mid = mid[:, :, :h, :w]
            mid = (mid[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            intermediate_frames.append(mid)
        
        return intermediate_frames
    
    def interpolate(
        self,
        video_path: str,
        output_path: str,
        multiplier: int = 2
    ) -> dict:
        """
        Interpolate a video to increase frame rate.
        
        Args:
            video_path: Input video path
            output_path: Output directory
            multiplier: FPS multiplier (2 = double, etc.)
        
        Returns:
            Dict with video_path, original_fps, new_fps
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video info
        probe_cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(video_path)
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_info = json.loads(probe_result.stdout)
        
        fps_str = video_info['streams'][0].get('r_frame_rate', '24/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            original_fps = num / den if den else 24
        else:
            original_fps = float(fps_str)
        
        new_fps = int(original_fps * multiplier)
        num_intermediate = multiplier - 1
        
        logger.info(f"Interpolating from {original_fps}fps to {new_fps}fps")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output video
        output_video = output_dir / "interpolated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video), fourcc, new_fps, (width, height))
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Could not read video")
        
        out.write(prev_frame)
        frame_count = 1
        
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            # Generate intermediate frames
            intermediates = self.interpolate_frames(
                prev_frame, curr_frame, num_intermediate
            )
            
            for inter_frame in intermediates:
                out.write(inter_frame)
                frame_count += 1
            
            out.write(curr_frame)
            frame_count += 1
            prev_frame = curr_frame
        
        cap.release()
        out.release()
        
        # Re-encode with FFmpeg for better compression
        final_output = output_dir / "interpolated_final.mp4"
        encode_cmd = [
            "ffmpeg", "-y",
            "-i", str(output_video),
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            str(final_output)
        ]
        subprocess.run(encode_cmd, check=True, capture_output=True)
        
        # Clean up temp file
        output_video.unlink()
        
        logger.info(f"Interpolated video saved: {final_output}")
        
        return {
            "video_path": str(final_output),
            "original_fps": int(original_fps),
            "new_fps": new_fps
        }
    
    def unload(self):
        """Unload model from GPU."""
        if hasattr(self, 'model'):
            del self.model
            torch.cuda.empty_cache()
