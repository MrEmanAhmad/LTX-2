"""CodeFormer face restoration model with video frame loop."""

import os
import subprocess
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class CodeFormerModel:
    """CodeFormer face restoration - processes video frame by frame."""
    
    def __init__(self):
        """Initialize CodeFormer model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load CodeFormer and face detection models."""
        # Import CodeFormer components
        from basicsr.archs.codeformer_arch import CodeFormer
        from basicsr.utils import img2tensor, tensor2img
        from basicsr.utils.download_util import load_file_from_url
        from facelib.utils.face_restoration_helper import FaceRestoreHelper
        from facelib.detection.retinaface import retinaface
        
        logger.info("Loading CodeFormer model...")
        
        # Model paths
        model_dir = os.environ.get("MODEL_DIR", "/runpod-volume/models/codeformer")
        os.makedirs(model_dir, exist_ok=True)
        
        # Download model if needed
        model_path = os.path.join(model_dir, "codeformer.pth")
        if not os.path.exists(model_path):
            model_url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
            load_file_from_url(model_url, model_dir=model_dir)
        
        # Initialize CodeFormer
        self.net = CodeFormer(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=['32', '64', '128', '256']
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['params_ema'])
        self.net.eval()
        
        # Face detection helper
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=self.device
        )
        
        logger.info("CodeFormer model loaded")
    
    def restore_frame(
        self,
        image: np.ndarray,
        fidelity_weight: float = 0.5
    ) -> np.ndarray:
        """
        Restore faces in a single frame.
        
        Args:
            image: Input image as numpy array (BGR)
            fidelity_weight: 0 = quality, 1 = fidelity
        
        Returns:
            Restored image as numpy array
        """
        self.face_helper.clean_all()
        self.face_helper.read_image(image)
        
        # Detect faces
        num_faces = self.face_helper.get_face_landmarks_5(
            only_center_face=False,
            resize=640,
            eye_dist_threshold=5
        )
        
        if num_faces == 0:
            # No faces detected, return original
            return image
        
        # Align and warp faces
        self.face_helper.align_warp_face()
        
        # Restore each face
        for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
            # Prepare input
            cropped_face_t = img2tensor(
                cropped_face / 255.,
                bgr2rgb=True,
                float32=True
            )
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.net(
                    cropped_face_t,
                    w=fidelity_weight,
                    adain=True
                )[0]
                restored_face = tensor2img(
                    output,
                    rgb2bgr=True,
                    min_max=(-1, 1)
                )
            
            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)
        
        # Paste faces back
        self.face_helper.get_inverse_affine(None)
        restored_img = self.face_helper.paste_faces_to_input_image()
        
        return restored_img
    
    def restore_video(
        self,
        video_path: str,
        output_path: str,
        fidelity_weight: float = 0.5,
        upscale: int = 1
    ) -> dict:
        """
        Restore faces in a video frame by frame.
        
        Args:
            video_path: Input video path
            output_path: Output directory
            fidelity_weight: CodeFormer fidelity weight
            upscale: Upscale factor
        
        Returns:
            Dict with video_path and frames_processed
        """
        output_dir = Path(output_path)
        frames_dir = output_dir / "frames"
        restored_dir = output_dir / "restored_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        restored_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Extract frames
        logger.info("Extracting frames...")
        extract_cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-qscale:v", "2",
            str(frames_dir / "frame_%05d.png")
        ]
        subprocess.run(extract_cmd, check=True, capture_output=True)
        
        # Get video info for reassembly
        probe_cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(video_path)
        ]
        import json
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_info = json.loads(probe_result.stdout)
        fps = eval(video_info['streams'][0].get('r_frame_rate', '24/1'))
        
        # Step 2: Restore each frame
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        logger.info(f"Restoring {len(frame_files)} frames...")
        
        for i, frame_path in enumerate(frame_files):
            if i % 10 == 0:
                logger.info(f"Processing frame {i+1}/{len(frame_files)}")
            
            # Read frame
            frame = cv2.imread(str(frame_path))
            
            # Restore faces
            restored = self.restore_frame(frame, fidelity_weight)
            
            # Optionally upscale
            if upscale > 1:
                h, w = restored.shape[:2]
                restored = cv2.resize(
                    restored,
                    (w * upscale, h * upscale),
                    interpolation=cv2.INTER_LANCZOS4
                )
            
            # Save restored frame
            output_frame_path = restored_dir / frame_path.name
            cv2.imwrite(str(output_frame_path), restored)
        
        # Step 3: Reassemble video
        logger.info("Reassembling video...")
        output_video = output_dir / "restored.mp4"
        
        reassemble_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(restored_dir / "frame_%05d.png"),
            "-c:v", "libx264",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(output_video)
        ]
        subprocess.run(reassemble_cmd, check=True, capture_output=True)
        
        logger.info(f"Restored video saved to {output_video}")
        
        return {
            "video_path": str(output_video),
            "frames_processed": len(frame_files)
        }
    
    def unload(self):
        """Unload model from GPU."""
        if hasattr(self, 'net'):
            del self.net
            del self.face_helper
            torch.cuda.empty_cache()


def normalize(tensor, mean, std, inplace=False):
    """Normalize tensor with mean and std."""
    if not inplace:
        tensor = tensor.clone()
    
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor
