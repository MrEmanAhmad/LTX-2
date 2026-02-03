"""FFmpeg video utilities for frame extraction, reassembly, and stitching."""

import subprocess
import json
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class VideoUtils:
    """FFmpeg-based video processing utilities."""
    
    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path
    
    def get_video_info(self, video_path: str | Path) -> dict:
        """Get video metadata using ffprobe.
        
        Returns:
            Dict with width, height, fps, duration, num_frames, codec
        """
        cmd = [
            self.ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        video_stream = next(
            (s for s in data.get("streams", []) if s["codec_type"] == "video"),
            {}
        )
        
        # Parse frame rate (can be "24/1" or "23.976")
        fps_str = video_stream.get("r_frame_rate", "24/1")
        if "/" in fps_str:
            num, den = map(int, fps_str.split("/"))
            fps = num / den if den else 24
        else:
            fps = float(fps_str)
        
        duration = float(data.get("format", {}).get("duration", 0))
        
        return {
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": fps,
            "duration": duration,
            "num_frames": int(video_stream.get("nb_frames", duration * fps)),
            "codec": video_stream.get("codec_name", "unknown"),
        }
    
    def extract_frames(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        pattern: str = "frame_%05d.png",
        fps: Optional[float] = None,
        quality: int = 2
    ) -> list[Path]:
        """Extract frames from video.
        
        Args:
            video_path: Input video file
            output_dir: Directory to save frames
            pattern: Output filename pattern
            fps: Extract at specific FPS (None = original)
            quality: JPEG/PNG quality (1-31, lower is better)
        
        Returns:
            List of extracted frame paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [self.ffmpeg, "-y", "-i", str(video_path)]
        
        if fps:
            cmd.extend(["-vf", f"fps={fps}"])
        
        cmd.extend([
            "-qscale:v", str(quality),
            str(output_dir / pattern)
        ])
        
        logger.info(f"Extracting frames: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        
        frames = sorted(output_dir.glob("frame_*.png"))
        logger.info(f"Extracted {len(frames)} frames to {output_dir}")
        return frames
    
    def extract_last_frame(
        self,
        video_path: str | Path,
        output_path: str | Path,
        quality: int = 2
    ) -> Path:
        """Extract the last frame of a video for continuity chaining.
        
        Args:
            video_path: Input video file
            output_path: Output image path
            quality: Image quality
        
        Returns:
            Path to extracted frame
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.ffmpeg, "-y",
            "-sseof", "-0.1",  # Seek to 0.1s before end
            "-i", str(video_path),
            "-update", "1",
            "-q:v", str(quality),
            str(output_path)
        ]
        
        logger.info(f"Extracting last frame: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path
    
    def reassemble_video(
        self,
        frames_dir: str | Path,
        output_path: str | Path,
        fps: int = 24,
        pattern: str = "frame_%05d.png",
        codec: str = "libx264",
        crf: int = 18,
        preset: str = "medium",
        pix_fmt: str = "yuv420p"
    ) -> Path:
        """Reassemble frames into a video.
        
        Args:
            frames_dir: Directory containing frames
            output_path: Output video path
            fps: Output frame rate
            pattern: Frame filename pattern
            codec: Video codec
            crf: Constant rate factor (quality, lower is better)
            preset: Encoding preset (ultrafast to veryslow)
            pix_fmt: Pixel format
        
        Returns:
            Path to output video
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", str(Path(frames_dir) / pattern),
            "-c:v", codec,
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", pix_fmt,
            str(output_path)
        ]
        
        logger.info(f"Reassembling video: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path
    
    def stitch_videos(
        self,
        video_paths: list[str | Path],
        output_path: str | Path,
        transition: Optional[str] = None,
        transition_duration: float = 0.5
    ) -> Path:
        """Stitch multiple video clips together.
        
        Args:
            video_paths: List of video files to concatenate
            output_path: Output video path
            transition: Transition type (None, 'fade', 'dissolve')
            transition_duration: Duration of transition in seconds
        
        Returns:
            Path to output video
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if transition is None:
            # Simple concatenation
            concat_file = output_path.parent / "concat_list.txt"
            with open(concat_file, "w") as f:
                for vp in video_paths:
                    f.write(f"file '{Path(vp).absolute()}'\n")
            
            cmd = [
                self.ffmpeg, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                str(output_path)
            ]
        else:
            # Concatenation with transitions using filter_complex
            inputs = []
            for vp in video_paths:
                inputs.extend(["-i", str(vp)])
            
            n = len(video_paths)
            filter_parts = []
            
            # Build xfade filter chain
            if n == 2:
                filter_parts.append(
                    f"[0:v][1:v]xfade=transition={transition}:duration={transition_duration}:offset=auto[v]"
                )
            else:
                # Chain multiple videos
                prev = "0:v"
                for i in range(1, n):
                    out = "v" if i == n - 1 else f"v{i}"
                    filter_parts.append(
                        f"[{prev}][{i}:v]xfade=transition={transition}:duration={transition_duration}:offset=auto[{out}]"
                    )
                    prev = out
            
            filter_complex = ";".join(filter_parts)
            
            cmd = [
                self.ffmpeg, "-y",
                *inputs,
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-c:v", "libx264",
                "-crf", "18",
                str(output_path)
            ]
        
        logger.info(f"Stitching videos: {len(video_paths)} clips")
        subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path
    
    def add_audio(
        self,
        video_path: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
        loop_audio: bool = True,
        audio_volume: float = 1.0
    ) -> Path:
        """Add audio track to video.
        
        Args:
            video_path: Input video file
            audio_path: Audio file to add
            output_path: Output video path
            loop_audio: Loop audio if shorter than video
            audio_volume: Audio volume multiplier
        
        Returns:
            Path to output video
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get video duration
        info = self.get_video_info(video_path)
        duration = info["duration"]
        
        audio_filter = f"volume={audio_volume}"
        if loop_audio:
            audio_filter = f"aloop=loop=-1:size=2e+09,{audio_filter}"
        
        cmd = [
            self.ffmpeg, "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-filter_complex", f"[1:a]{audio_filter},atrim=0:{duration}[a]",
            "-map", "0:v",
            "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(output_path)
        ]
        
        logger.info(f"Adding audio to video")
        subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path
    
    def resize_video(
        self,
        video_path: str | Path,
        output_path: str | Path,
        width: int,
        height: int,
        maintain_aspect: bool = True
    ) -> Path:
        """Resize video to specific dimensions.
        
        Args:
            video_path: Input video
            output_path: Output video path
            width: Target width
            height: Target height
            maintain_aspect: Pad to maintain aspect ratio
        
        Returns:
            Path to output video
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if maintain_aspect:
            scale_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        else:
            scale_filter = f"scale={width}:{height}"
        
        cmd = [
            self.ffmpeg, "-y",
            "-i", str(video_path),
            "-vf", scale_filter,
            "-c:v", "libx264",
            "-crf", "18",
            "-c:a", "copy",
            str(output_path)
        ]
        
        logger.info(f"Resizing video to {width}x{height}")
        subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path
    
    def change_fps(
        self,
        video_path: str | Path,
        output_path: str | Path,
        target_fps: int
    ) -> Path:
        """Change video frame rate (simple method, may drop/duplicate frames).
        
        Args:
            video_path: Input video
            output_path: Output video path
            target_fps: Target frame rate
        
        Returns:
            Path to output video
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.ffmpeg, "-y",
            "-i", str(video_path),
            "-vf", f"fps={target_fps}",
            "-c:v", "libx264",
            "-crf", "18",
            "-c:a", "copy",
            str(output_path)
        ]
        
        logger.info(f"Changing FPS to {target_fps}")
        subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path


# Singleton instance
_video_utils: Optional[VideoUtils] = None


def get_video_utils() -> VideoUtils:
    """Get the video utils singleton."""
    global _video_utils
    if _video_utils is None:
        _video_utils = VideoUtils()
    return _video_utils
