"""
Simplified configuration for 2-endpoint architecture
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # RunPod API
    RUNPOD_API_KEY: str = ""
    
    # Two unified endpoints
    VIDEO_GENERATOR_ENDPOINT_ID: str = ""  # FLUX + LTX + Wan2.1
    POST_PROCESSOR_ENDPOINT_ID: str = ""   # CodeFormer + ESRGAN + RIFE
    
    # Orchestrator webhook URL (for RunPod callbacks)
    WEBHOOK_BASE_URL: str = "http://localhost:8000"
    
    # Job cleanup
    JOB_RETENTION_HOURS: int = 24
    
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
