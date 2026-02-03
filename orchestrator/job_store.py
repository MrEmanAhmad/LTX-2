"""Job state management for pipeline orchestration."""

import asyncio
from datetime import datetime
from typing import Optional, Any
from config import get_settings
import logging
import json

logger = logging.getLogger(__name__)


class JobStore:
    """
    In-memory job state store.
    
    For production, replace with Redis or a database.
    """
    
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = asyncio.Lock()
    
    async def create(self, job_id: str, initial_state: dict) -> dict:
        """
        Create a new job entry.
        
        Args:
            job_id: Unique job identifier
            initial_state: Initial job state
        
        Returns:
            The created job state
        """
        async with self._lock:
            now = datetime.utcnow().isoformat()
            job = {
                "job_id": job_id,
                "created_at": now,
                "updated_at": now,
                **initial_state
            }
            self._jobs[job_id] = job
            logger.info(f"Created job: {job_id}")
            return job
    
    async def get(self, job_id: str) -> Optional[dict]:
        """Get a job by ID."""
        return self._jobs.get(job_id)
    
    async def update(self, job_id: str, updates: dict) -> Optional[dict]:
        """
        Update a job's state.
        
        Args:
            job_id: Job identifier
            updates: Fields to update
        
        Returns:
            Updated job state or None if not found
        """
        async with self._lock:
            if job_id not in self._jobs:
                logger.warning(f"Job not found for update: {job_id}")
                return None
            
            self._jobs[job_id].update(updates)
            self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
            
            logger.debug(f"Updated job {job_id}: {list(updates.keys())}")
            return self._jobs[job_id]
    
    async def delete(self, job_id: str) -> bool:
        """Delete a job."""
        async with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                logger.info(f"Deleted job: {job_id}")
                return True
            return False
    
    async def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> list[dict]:
        """
        List jobs, optionally filtered by status.
        
        Args:
            status: Filter by status
            limit: Maximum number of jobs to return
        
        Returns:
            List of job states
        """
        jobs = list(self._jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.get("status") == status]
        
        # Sort by created_at descending
        jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return jobs[:limit]
    
    async def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """
        Remove jobs older than max_age_hours.
        
        Returns:
            Number of jobs removed
        """
        from datetime import timedelta
        
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        async with self._lock:
            to_delete = []
            
            for job_id, job in self._jobs.items():
                created_at = datetime.fromisoformat(job["created_at"])
                if created_at < cutoff:
                    to_delete.append(job_id)
            
            for job_id in to_delete:
                del self._jobs[job_id]
            
            if to_delete:
                logger.info(f"Cleaned up {len(to_delete)} old jobs")
            
            return len(to_delete)


class RedisJobStore:
    """
    Redis-backed job store for production use.
    
    Install with: pip install redis
    """
    
    def __init__(self, redis_url: str):
        import redis.asyncio as redis
        self._redis = redis.from_url(redis_url)
        self._prefix = "vidpipe:job:"
    
    def _key(self, job_id: str) -> str:
        return f"{self._prefix}{job_id}"
    
    async def create(self, job_id: str, initial_state: dict) -> dict:
        now = datetime.utcnow().isoformat()
        job = {
            "job_id": job_id,
            "created_at": now,
            "updated_at": now,
            **initial_state
        }
        
        await self._redis.set(
            self._key(job_id),
            json.dumps(job),
            ex=86400 * 7  # 7 day expiry
        )
        
        return job
    
    async def get(self, job_id: str) -> Optional[dict]:
        data = await self._redis.get(self._key(job_id))
        if data:
            return json.loads(data)
        return None
    
    async def update(self, job_id: str, updates: dict) -> Optional[dict]:
        job = await self.get(job_id)
        if not job:
            return None
        
        job.update(updates)
        job["updated_at"] = datetime.utcnow().isoformat()
        
        await self._redis.set(
            self._key(job_id),
            json.dumps(job),
            ex=86400 * 7
        )
        
        return job
    
    async def delete(self, job_id: str) -> bool:
        result = await self._redis.delete(self._key(job_id))
        return result > 0


# Factory function to get appropriate store
_store: Optional[JobStore] = None


def get_job_store() -> JobStore:
    """Get the job store instance."""
    global _store
    if _store is None:
        settings = get_settings()
        if settings.redis_url:
            _store = RedisJobStore(settings.redis_url)
            logger.info("Using Redis job store")
        else:
            _store = JobStore()
            logger.info("Using in-memory job store")
    return _store
