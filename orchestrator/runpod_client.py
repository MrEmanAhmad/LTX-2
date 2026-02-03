"""
Simplified RunPod API client for 2-endpoint architecture
"""

import httpx
from config import settings


class RunPodClient:
    BASE_URL = "https://api.runpod.ai/v2"
    
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {settings.RUNPOD_API_KEY}"}
    
    async def run_video_generator(self, action: str, params: dict, job_id: str = None) -> dict:
        """Run job on Video Generator endpoint (FLUX + LTX + Wan)"""
        return await self._run_async(
            settings.VIDEO_GENERATOR_ENDPOINT_ID,
            {"action": action, **params},
            job_id
        )
    
    async def run_post_processor(self, action: str, params: dict, job_id: str = None) -> dict:
        """Run job on Post-Processor endpoint (CodeFormer + ESRGAN + RIFE)"""
        return await self._run_async(
            settings.POST_PROCESSOR_ENDPOINT_ID,
            {"action": action, **params},
            job_id
        )
    
    async def _run_async(self, endpoint_id: str, input_data: dict, job_id: str = None) -> dict:
        """Submit async job with webhook callback"""
        url = f"{self.BASE_URL}/{endpoint_id}/run"
        
        payload = {"input": input_data}
        if job_id:
            payload["webhook"] = f"{settings.WEBHOOK_BASE_URL}/callback/{job_id}"
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=self.headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
    
    async def get_status(self, endpoint_id: str, runpod_job_id: str) -> dict:
        """Check job status"""
        url = f"{self.BASE_URL}/{endpoint_id}/status/{runpod_job_id}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=self.headers, timeout=30)
            resp.raise_for_status()
            return resp.json()


runpod_client = RunPodClient()
