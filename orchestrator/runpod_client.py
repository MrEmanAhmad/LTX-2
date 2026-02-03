"""RunPod API client with webhook support."""

import httpx
import asyncio
from typing import Optional, Any
from config import get_settings
import logging

logger = logging.getLogger(__name__)


class RunPodClient:
    """Client for interacting with RunPod serverless endpoints."""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = "https://api.runpod.ai/v2"
        self.headers = {
            "Authorization": f"Bearer {self.settings.runpod_api_key}",
            "Content-Type": "application/json"
        }
    
    async def run_async(
        self,
        handler_name: str,
        input_data: dict,
        job_id: str,
        webhook_path: str = "/callback"
    ) -> str:
        """
        Submit a job to RunPod with webhook callback.
        
        Args:
            handler_name: Name of the handler (flux, ltx_video, etc.)
            input_data: Input payload for the handler
            job_id: Our internal job ID for webhook routing
            webhook_path: Path for webhook callback
        
        Returns:
            RunPod job ID
        """
        endpoint_id = self.settings.get_endpoint_id(handler_name)
        webhook_url = f"{self.settings.webhook_base_url}{webhook_path}/{job_id}"
        
        payload = {
            "input": input_data,
            "webhook": webhook_url
        }
        
        logger.info(f"Submitting job to {handler_name} (endpoint: {endpoint_id})")
        logger.debug(f"Webhook URL: {webhook_url}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/{endpoint_id}/run",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
        result = response.json()
        runpod_job_id = result["id"]
        
        logger.info(f"Job submitted: runpod_id={runpod_job_id}")
        return runpod_job_id
    
    async def run_sync(
        self,
        handler_name: str,
        input_data: dict,
        timeout: int = 300
    ) -> dict:
        """
        Submit a job and wait for result (polling mode).
        Use for testing or when webhooks aren't available.
        
        Args:
            handler_name: Name of the handler
            input_data: Input payload
            timeout: Maximum wait time in seconds
        
        Returns:
            Job output data
        """
        endpoint_id = self.settings.get_endpoint_id(handler_name)
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Submit job
            response = await client.post(
                f"{self.base_url}/{endpoint_id}/runsync",
                headers=self.headers,
                json={"input": input_data}
            )
            response.raise_for_status()
            
        result = response.json()
        
        if result.get("status") == "COMPLETED":
            return result.get("output", {})
        else:
            raise RuntimeError(f"Job failed: {result}")
    
    async def get_job_status(
        self,
        handler_name: str,
        runpod_job_id: str
    ) -> dict:
        """
        Get the status of a RunPod job.
        
        Args:
            handler_name: Name of the handler
            runpod_job_id: RunPod job ID
        
        Returns:
            Job status dict
        """
        endpoint_id = self.settings.get_endpoint_id(handler_name)
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{self.base_url}/{endpoint_id}/status/{runpod_job_id}",
                headers=self.headers
            )
            response.raise_for_status()
            
        return response.json()
    
    async def cancel_job(
        self,
        handler_name: str,
        runpod_job_id: str
    ) -> bool:
        """
        Cancel a running job.
        
        Args:
            handler_name: Name of the handler
            runpod_job_id: RunPod job ID
        
        Returns:
            True if cancelled successfully
        """
        endpoint_id = self.settings.get_endpoint_id(handler_name)
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{self.base_url}/{endpoint_id}/cancel/{runpod_job_id}",
                headers=self.headers
            )
            
        return response.status_code == 200
    
    async def health_check(self, handler_name: str) -> dict:
        """
        Check the health/status of an endpoint.
        
        Args:
            handler_name: Name of the handler
        
        Returns:
            Endpoint health info
        """
        endpoint_id = self.settings.get_endpoint_id(handler_name)
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{self.base_url}/{endpoint_id}/health",
                headers=self.headers
            )
            response.raise_for_status()
            
        return response.json()


# Singleton instance
_client: Optional[RunPodClient] = None


def get_runpod_client() -> RunPodClient:
    """Get the RunPod client singleton."""
    global _client
    if _client is None:
        _client = RunPodClient()
    return _client
