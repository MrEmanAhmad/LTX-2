"""Configuration for the video pipeline orchestrator."""

import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # RunPod Configuration
    runpod_api_key: str = ""
    
    # Serverless Endpoint IDs
    flux_endpoint_id: str = ""
    ltx_video_endpoint_id: str = ""
    wan21_endpoint_id: str = ""
    codeformer_endpoint_id: str = ""
    rife_endpoint_id: str = ""
    realesrgan_endpoint_id: str = ""
    
    # Orchestrator Configuration
    webhook_base_url: str = "http://localhost:8000"
    runpod_volume_path: str = "/runpod-volume"
    
    # Job Management
    job_retention_hours: int = 24
    
    # Redis (optional, for production)
    redis_url: str | None = None
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_endpoint_id(self, handler_name: str) -> str:
        """Get endpoint ID for a handler."""
        mapping = {
            "flux": self.flux_endpoint_id,
            "ltx_video": self.ltx_video_endpoint_id,
            "wan21": self.wan21_endpoint_id,
            "codeformer": self.codeformer_endpoint_id,
            "rife": self.rife_endpoint_id,
            "realesrgan": self.realesrgan_endpoint_id,
        }
        endpoint_id = mapping.get(handler_name)
        if not endpoint_id:
            raise ValueError(f"No endpoint configured for handler: {handler_name}")
        return endpoint_id


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
