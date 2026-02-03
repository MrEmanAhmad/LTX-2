"""Pydantic schemas for the video generation pipeline."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class VideoModel(str, Enum):
    """Available video generation models."""
    LTX = "ltx"
    WAN21 = "wan21"


class JobState(str, Enum):
    """Job processing states."""
    PENDING = "pending"
    GENERATING_IMAGE = "generating_image"
    GENERATING_VIDEO = "generating_video"
    FACE_RESTORATION = "face_restoration"
    FRAME_INTERPOLATION = "frame_interpolation"
    UPSCALING = "upscaling"
    STITCHING = "stitching"
    COMPLETED = "completed"
    FAILED = "failed"


# ============== Request Schemas ==============

class ClipConfig(BaseModel):
    """Configuration for a single video clip in a multi-clip sequence."""
    image_prompt: Optional[str] = Field(
        None,
        description="Text prompt for FLUX image generation. Only needed for first clip."
    )
    motion_prompt: str = Field(
        ...,
        description="Motion/action prompt for video generation."
    )
    duration_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=10.0,
        description="Clip duration in seconds."
    )


class GenerateRequest(BaseModel):
    """Request for single-clip video generation."""
    image_prompt: str = Field(
        ...,
        description="Text prompt for FLUX image generation."
    )
    motion_prompt: str = Field(
        ...,
        description="Motion/action prompt for video generation."
    )
    video_model: VideoModel = Field(
        default=VideoModel.LTX,
        description="Video generation model to use."
    )
    duration_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=10.0,
        description="Video duration in seconds."
    )
    apply_face_restore: bool = Field(
        default=True,
        description="Apply CodeFormer face restoration."
    )
    interpolate: bool = Field(
        default=False,
        description="Apply RIFE frame interpolation (24→60fps)."
    )
    upscale: bool = Field(
        default=False,
        description="Apply Real-ESRGAN upscaling (480p→1080p)."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility."
    )


class MultiClipRequest(BaseModel):
    """Request for multi-clip video generation with frame chaining."""
    clips: list[ClipConfig] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of clip configurations. First clip must have image_prompt."
    )
    video_model: VideoModel = Field(
        default=VideoModel.LTX,
        description="Video generation model to use for all clips."
    )
    apply_face_restore: bool = Field(
        default=True,
        description="Apply CodeFormer face restoration."
    )
    interpolate: bool = Field(
        default=False,
        description="Apply RIFE frame interpolation."
    )
    upscale: bool = Field(
        default=False,
        description="Apply Real-ESRGAN upscaling."
    )
    audio_url: Optional[str] = Field(
        default=None,
        description="URL to audio file to overlay on final video."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility."
    )


class ImageGenerateRequest(BaseModel):
    """Request for FLUX image generation only."""
    prompt: str = Field(..., description="Image generation prompt.")
    width: int = Field(default=1024, ge=512, le=2048)
    height: int = Field(default=1024, ge=512, le=2048)
    num_inference_steps: int = Field(default=28, ge=1, le=50)
    guidance_scale: float = Field(default=3.5, ge=1.0, le=20.0)
    seed: Optional[int] = None


class VideoFromImageRequest(BaseModel):
    """Request to generate video from an existing image."""
    image_path: str = Field(..., description="Path to input image on RunPod volume.")
    motion_prompt: str = Field(..., description="Motion/action prompt.")
    video_model: VideoModel = Field(default=VideoModel.LTX)
    duration_seconds: float = Field(default=5.0, ge=1.0, le=10.0)
    apply_face_restore: bool = Field(default=True)
    interpolate: bool = Field(default=False)
    upscale: bool = Field(default=False)


# ============== Response Schemas ==============

class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: JobState
    current_step: Optional[str] = None
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    error: Optional[str] = None
    created_at: str
    updated_at: str


class JobResult(BaseModel):
    """Job completion result."""
    job_id: str
    status: JobState
    video_path: Optional[str] = None
    video_url: Optional[str] = None
    duration_seconds: Optional[float] = None
    resolution: Optional[str] = None
    processing_time_seconds: Optional[float] = None


# ============== Handler Input/Output Schemas ==============

class FluxInput(BaseModel):
    """Input schema for FLUX handler."""
    prompt: str
    output_path: str
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    seed: Optional[int] = None


class FluxOutput(BaseModel):
    """Output schema for FLUX handler."""
    image_path: str
    seed_used: int


class VideoInput(BaseModel):
    """Input schema for video generation handlers (LTX/Wan21)."""
    image_path: str
    prompt: str
    output_path: str
    num_frames: int = 121  # ~5 seconds at 24fps
    fps: int = 24
    seed: Optional[int] = None


class VideoOutput(BaseModel):
    """Output schema for video generation handlers."""
    video_path: str
    last_frame_path: str  # For frame chaining
    num_frames: int
    fps: int
    seed_used: int


class CodeFormerInput(BaseModel):
    """Input schema for CodeFormer handler."""
    video_path: str
    output_path: str
    fidelity_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    upscale: int = Field(default=1, ge=1, le=4)


class CodeFormerOutput(BaseModel):
    """Output schema for CodeFormer handler."""
    video_path: str
    frames_processed: int


class RIFEInput(BaseModel):
    """Input schema for RIFE handler."""
    video_path: str
    output_path: str
    multiplier: int = Field(default=2, description="Frame multiplier (2=48fps, 2.5=60fps)")


class RIFEOutput(BaseModel):
    """Output schema for RIFE handler."""
    video_path: str
    original_fps: int
    new_fps: int


class RealESRGANInput(BaseModel):
    """Input schema for Real-ESRGAN handler."""
    video_path: str
    output_path: str
    scale: int = Field(default=2, ge=1, le=4)


class RealESRGANOutput(BaseModel):
    """Output schema for Real-ESRGAN handler."""
    video_path: str
    original_resolution: str
    new_resolution: str
