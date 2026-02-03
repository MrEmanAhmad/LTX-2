"""
Simple in-memory job store
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class JobState:
    status: str = "pending"
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Paths
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    last_frame_path: Optional[str] = None
    restored_path: Optional[str] = None
    upscaled_path: Optional[str] = None
    interpolated_path: Optional[str] = None
    final_video_path: Optional[str] = None
    
    error: Optional[str] = None


class JobStore:
    def __init__(self):
        self._jobs: Dict[str, JobState] = {}
    
    def create(self, job_id: str, state: JobState) -> None:
        self._jobs[job_id] = state
    
    def get(self, job_id: str) -> Optional[JobState]:
        return self._jobs.get(job_id)
    
    def update(self, job_id: str, state: JobState) -> None:
        self._jobs[job_id] = state
    
    def delete(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)
    
    def list_all(self) -> Dict[str, JobState]:
        return self._jobs.copy()


job_store = JobStore()
