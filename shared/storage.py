"""RunPod Network Storage utilities."""

import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages file operations on RunPod Network Storage volume."""
    
    def __init__(self, base_path: str = "/runpod-volume"):
        self.base_path = Path(base_path)
        self.jobs_path = self.base_path / "jobs"
        self.models_path = self.base_path / "models"
        self.temp_path = self.base_path / "temp"
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create required directories if they don't exist."""
        for path in [self.jobs_path, self.models_path, self.temp_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_job_path(self, job_id: str) -> Path:
        """Get the base path for a job's files."""
        job_path = self.jobs_path / job_id
        job_path.mkdir(parents=True, exist_ok=True)
        return job_path
    
    def get_clip_path(self, job_id: str, clip_index: int) -> Path:
        """Get the path for a specific clip within a job."""
        clip_path = self.get_job_path(job_id) / f"clip_{clip_index}"
        clip_path.mkdir(parents=True, exist_ok=True)
        return clip_path
    
    def get_frames_path(self, job_id: str, stage: str = "raw") -> Path:
        """Get path for extracted frames.
        
        Args:
            job_id: Job identifier
            stage: Processing stage (raw, restored, interpolated, etc.)
        """
        frames_path = self.get_job_path(job_id) / "frames" / stage
        frames_path.mkdir(parents=True, exist_ok=True)
        return frames_path
    
    def get_output_path(self, job_id: str, filename: str) -> Path:
        """Get path for final output file."""
        output_path = self.get_job_path(job_id) / "output"
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / filename
    
    def save_file(self, content: bytes, job_id: str, filename: str, subdir: Optional[str] = None) -> Path:
        """Save a file to job storage.
        
        Args:
            content: File content as bytes
            job_id: Job identifier
            filename: Name for the file
            subdir: Optional subdirectory within job folder
        
        Returns:
            Path to saved file
        """
        job_path = self.get_job_path(job_id)
        if subdir:
            job_path = job_path / subdir
            job_path.mkdir(parents=True, exist_ok=True)
        
        file_path = job_path / filename
        file_path.write_bytes(content)
        logger.info(f"Saved file: {file_path}")
        return file_path
    
    def read_file(self, path: str | Path) -> bytes:
        """Read a file from storage."""
        return Path(path).read_bytes()
    
    def file_exists(self, path: str | Path) -> bool:
        """Check if a file exists."""
        return Path(path).exists()
    
    def list_files(self, directory: str | Path, pattern: str = "*") -> list[Path]:
        """List files in a directory matching a pattern."""
        return sorted(Path(directory).glob(pattern))
    
    def copy_file(self, src: str | Path, dst: str | Path) -> Path:
        """Copy a file within storage."""
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst_path
    
    def move_file(self, src: str | Path, dst: str | Path) -> Path:
        """Move a file within storage."""
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)
        return dst_path
    
    def delete_file(self, path: str | Path) -> bool:
        """Delete a file."""
        try:
            Path(path).unlink()
            return True
        except FileNotFoundError:
            return False
    
    def delete_job(self, job_id: str) -> bool:
        """Delete all files for a job."""
        job_path = self.jobs_path / job_id
        if job_path.exists():
            shutil.rmtree(job_path)
            logger.info(f"Deleted job directory: {job_path}")
            return True
        return False
    
    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Delete jobs older than max_age_hours.
        
        Returns:
            Number of jobs deleted
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        deleted_count = 0
        
        for job_dir in self.jobs_path.iterdir():
            if job_dir.is_dir():
                # Use directory modification time
                mtime = datetime.fromtimestamp(job_dir.stat().st_mtime)
                if mtime < cutoff:
                    shutil.rmtree(job_dir)
                    deleted_count += 1
                    logger.info(f"Cleaned up old job: {job_dir.name}")
        
        return deleted_count
    
    def get_disk_usage(self) -> dict:
        """Get disk usage statistics for the volume."""
        total, used, free = shutil.disk_usage(self.base_path)
        return {
            "total_gb": total / (1024**3),
            "used_gb": used / (1024**3),
            "free_gb": free / (1024**3),
            "percent_used": (used / total) * 100
        }
    
    def get_job_size(self, job_id: str) -> int:
        """Get total size of a job's files in bytes."""
        job_path = self.jobs_path / job_id
        if not job_path.exists():
            return 0
        
        total_size = 0
        for file_path in job_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size


# Singleton instance for handlers
_storage_manager: Optional[StorageManager] = None


def get_storage() -> StorageManager:
    """Get the storage manager singleton."""
    global _storage_manager
    if _storage_manager is None:
        base_path = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
        _storage_manager = StorageManager(base_path)
    return _storage_manager
