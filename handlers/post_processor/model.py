"""
Unified Post-Processor Model
Combines: CodeFormer (face restore) + Real-ESRGAN (upscale) + RIFE (interpolation)
"""

import os
import subprocess
import shutil
from pathlib import Path

# Fix torch.xpu issue before importing other libraries
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


class UnifiedPostProcessor:
    def __init__(self):
        self.codeformer = None
        self.esrgan = None
        self.rife = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def load_codeformer(self):
        """Load CodeFormer model for face restoration"""
        if self.codeformer is None:
            print("Loading CodeFormer model...")
            try:
                from basicsr.archs.codeformer_arch import CodeFormer
                from basicsr.utils.download_util import load_file_from_url
                
                model_path = os.path.join(MODEL_CACHE, "codeformer.pth")
                if not os.path.exists(model_path):
                    load_file_from_url(
                        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
                        model_dir=MODEL_CACHE,
                        file_name="codeformer.pth"
                    )
                
                self.codeformer = CodeFormer(
                    dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                    connect_list=['32', '64', '128', '256']
                ).to(self.device)
                self.codeformer.load_state_dict(
                    torch.load(model_path, map_location=self.device)['params_ema']
                )
                self.codeformer.eval()
                print("CodeFormer loaded successfully")
            except ImportError:
                print("CodeFormer not available, using fallback")
                self.codeformer = "unavailable"
        return self.codeformer
    
    def load_esrgan(self):
        """Load Real-ESRGAN model for upscaling"""
        if self.esrgan is None:
            print("Loading Real-ESRGAN model...")
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                
                model_path = os.path.join(MODEL_CACHE, "RealESRGAN_x4plus.pth")
                if not os.path.exists(model_path):
                    from basicsr.utils.download_util import load_file_from_url
                    load_file_from_url(
                        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                        model_dir=MODEL_CACHE,
                        file_name="RealESRGAN_x4plus.pth"
                    )
                
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                               num_block=23, num_grow_ch=32, scale=4)
                self.esrgan = RealESRGANer(
                    scale=4, model_path=model_path, model=model,
                    tile=400, tile_pad=10, pre_pad=0, half=True,
                    device=self.device
                )
                print("Real-ESRGAN loaded successfully")
            except ImportError:
                print("Real-ESRGAN not available, using fallback")
                self.esrgan = "unavailable"
        return self.esrgan
    
    def load_rife(self):
        """Load RIFE model for frame interpolation"""
        if self.rife is None:
            print("Loading RIFE model...")
            try:
                # RIFE uses a custom model loader
                import sys
                rife_path = os.path.join(MODEL_CACHE, "rife")
                if not os.path.exists(rife_path):
                    subprocess.run([
                        "git", "clone", 
                        "https://github.com/hzwer/Practical-RIFE.git",
                        rife_path
                    ], check=True)
                sys.path.insert(0, rife_path)
                from model.RIFE import Model
                self.rife = Model()
                self.rife.load_model(os.path.join(rife_path, "train_log"), -1)
                self.rife.eval()
                self.rife.device()
                print("RIFE loaded successfully")
            except Exception as e:
                print(f"RIFE not available: {e}")
                self.rife = "unavailable"
        return self.rife
    
    def restore_faces_video(self, video_path: str, fidelity_weight: float = 0.7) -> str:
        """Apply face restoration to all frames in a video"""
        import cv2
        import numpy as np
        from PIL import Image
        
        codeformer = self.load_codeformer()
        if codeformer == "unavailable":
            print("CodeFormer unavailable, returning original video")
            return video_path
        
        # Create temp directories
        job_id = os.urandom(8).hex()
        frames_dir = f"/tmp/codeformer_{job_id}/frames"
        restored_dir = f"/tmp/codeformer_{job_id}/restored"
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(restored_dir, exist_ok=True)
        
        # Extract frames
        subprocess.run([
            "ffmpeg", "-i", video_path, "-qscale:v", "2",
            f"{frames_dir}/frame_%05d.png"
        ], capture_output=True)
        
        # Get video info for reassembly
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ], capture_output=True, text=True)
        fps = eval(probe.stdout.strip()) if probe.stdout.strip() else 24
        
        # Process each frame
        frame_files = sorted(Path(frames_dir).glob("*.png"))
        for frame_path in frame_files:
            img = cv2.imread(str(frame_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize and process
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                output = codeformer(img_tensor, w=fidelity_weight)[0]
            
            # Convert back
            output = output.squeeze().permute(1, 2, 0).cpu().numpy() * 255
            output = output.clip(0, 255).astype(np.uint8)
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            # Save restored frame
            cv2.imwrite(str(Path(restored_dir) / frame_path.name), output_bgr)
        
        # Reassemble video
        output_path = f"/runpod-volume/outputs/restored_{job_id}.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", f"{restored_dir}/frame_%05d.png",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", output_path
        ], capture_output=True)
        
        # Cleanup
        shutil.rmtree(f"/tmp/codeformer_{job_id}", ignore_errors=True)
        
        return output_path
    
    def upscale_video(self, video_path: str, scale: int = 4) -> str:
        """Upscale video using Real-ESRGAN (frame by frame)"""
        import cv2
        
        esrgan = self.load_esrgan()
        if esrgan == "unavailable":
            print("Real-ESRGAN unavailable, returning original video")
            return video_path
        
        # Create temp directories
        job_id = os.urandom(8).hex()
        frames_dir = f"/tmp/esrgan_{job_id}/frames"
        upscaled_dir = f"/tmp/esrgan_{job_id}/upscaled"
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(upscaled_dir, exist_ok=True)
        
        # Extract frames
        subprocess.run([
            "ffmpeg", "-i", video_path, "-qscale:v", "2",
            f"{frames_dir}/frame_%05d.png"
        ], capture_output=True)
        
        # Get video info
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ], capture_output=True, text=True)
        fps = eval(probe.stdout.strip()) if probe.stdout.strip() else 24
        
        # Process each frame
        frame_files = sorted(Path(frames_dir).glob("*.png"))
        for frame_path in frame_files:
            img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
            output, _ = esrgan.enhance(img, outscale=scale)
            cv2.imwrite(str(Path(upscaled_dir) / frame_path.name), output)
        
        # Reassemble video
        output_path = f"/runpod-volume/outputs/upscaled_{job_id}.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", f"{upscaled_dir}/frame_%05d.png",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", output_path
        ], capture_output=True)
        
        # Cleanup
        shutil.rmtree(f"/tmp/esrgan_{job_id}", ignore_errors=True)
        
        return output_path
    
    def interpolate_video(self, video_path: str, target_fps: int = 60) -> str:
        """Interpolate video frames using RIFE to increase FPS"""
        import cv2
        import numpy as np
        
        rife = self.load_rife()
        if rife == "unavailable":
            # Fallback to FFmpeg minterpolate
            print("RIFE unavailable, using FFmpeg minterpolate")
            return self._ffmpeg_interpolate(video_path, target_fps)
        
        # Create temp directories
        job_id = os.urandom(8).hex()
        frames_dir = f"/tmp/rife_{job_id}/frames"
        interp_dir = f"/tmp/rife_{job_id}/interpolated"
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(interp_dir, exist_ok=True)
        
        # Extract frames
        subprocess.run([
            "ffmpeg", "-i", video_path, "-qscale:v", "2",
            f"{frames_dir}/frame_%05d.png"
        ], capture_output=True)
        
        # Get original FPS
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ], capture_output=True, text=True)
        original_fps = eval(probe.stdout.strip()) if probe.stdout.strip() else 24
        
        # Calculate multiplier (how many intermediate frames to generate)
        multiplier = int(target_fps / original_fps)
        if multiplier < 2:
            multiplier = 2
        
        # Load frames
        frame_files = sorted(Path(frames_dir).glob("*.png"))
        frames = [cv2.imread(str(f)) for f in frame_files]
        
        # Interpolate
        output_idx = 0
        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            
            # Save first frame
            cv2.imwrite(f"{interp_dir}/frame_{output_idx:05d}.png", frame1)
            output_idx += 1
            
            # Generate intermediate frames
            for j in range(1, multiplier):
                t = j / multiplier
                # Convert to tensor
                img1 = torch.from_numpy(frame1.transpose(2, 0, 1)).float() / 255.0
                img2 = torch.from_numpy(frame2.transpose(2, 0, 1)).float() / 255.0
                img1 = img1.unsqueeze(0).to(self.device)
                img2 = img2.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    mid = rife.inference(img1, img2, t)
                
                mid = (mid[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                cv2.imwrite(f"{interp_dir}/frame_{output_idx:05d}.png", mid)
                output_idx += 1
        
        # Save last frame
        cv2.imwrite(f"{interp_dir}/frame_{output_idx:05d}.png", frames[-1])
        
        # Reassemble video
        output_path = f"/runpod-volume/outputs/interpolated_{job_id}.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(target_fps),
            "-i", f"{interp_dir}/frame_%05d.png",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", output_path
        ], capture_output=True)
        
        # Cleanup
        shutil.rmtree(f"/tmp/rife_{job_id}", ignore_errors=True)
        
        return output_path
    
    def _ffmpeg_interpolate(self, video_path: str, target_fps: int) -> str:
        """Fallback interpolation using FFmpeg's minterpolate filter"""
        job_id = os.urandom(8).hex()
        output_path = f"/runpod-volume/outputs/interpolated_{job_id}.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", output_path
        ], capture_output=True)
        
        return output_path
    
    def full_post_process(self, video_path: str, 
                          restore_faces: bool = True,
                          upscale: bool = True,
                          interpolate: bool = True,
                          target_fps: int = 60,
                          upscale_factor: int = 4,
                          fidelity_weight: float = 0.7) -> dict:
        """
        Full post-processing pipeline:
        1. Restore faces (CodeFormer)
        2. Upscale (Real-ESRGAN) 
        3. Interpolate frames (RIFE)
        
        All models stay loaded - no cold starts between steps!
        """
        current_video = video_path
        results = {"original_path": video_path}
        
        if restore_faces:
            print("Step 1: Restoring faces with CodeFormer...")
            current_video = self.restore_faces_video(current_video, fidelity_weight)
            results["restored_path"] = current_video
            print(f"Faces restored: {current_video}")
        
        if upscale:
            print("Step 2: Upscaling with Real-ESRGAN...")
            current_video = self.upscale_video(current_video, upscale_factor)
            results["upscaled_path"] = current_video
            print(f"Upscaled: {current_video}")
        
        if interpolate:
            print("Step 3: Interpolating frames with RIFE...")
            current_video = self.interpolate_video(current_video, target_fps)
            results["interpolated_path"] = current_video
            print(f"Interpolated: {current_video}")
        
        results["final_path"] = current_video
        return results


# Global instance for warm starts
_model = None

def get_model():
    global _model
    if _model is None:
        _model = UnifiedPostProcessor()
    return _model
