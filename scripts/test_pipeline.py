#!/usr/bin/env python3
"""
End-to-end test for the video generation pipeline.

Usage:
    python test_pipeline.py --endpoint http://localhost:8000

Environment variables:
    ORCHESTRATOR_URL - URL of the orchestrator API
"""

import os
import sys
import asyncio
import httpx
import argparse
import logging
from datetime import datetime
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PipelineTest:
    """Test the video generation pipeline."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=300.0)
    
    async def close(self):
        await self.client.aclose()
    
    async def health_check(self) -> bool:
        """Check if the orchestrator is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def check_endpoints(self) -> dict:
        """Check configured endpoints."""
        response = await self.client.get(f"{self.base_url}/endpoints")
        return response.json()
    
    async def submit_job(self, payload: dict, endpoint: str = "/generate") -> str:
        """Submit a job and return the job ID."""
        response = await self.client.post(
            f"{self.base_url}{endpoint}",
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data["job_id"]
    
    async def get_job_status(self, job_id: str) -> dict:
        """Get job status."""
        response = await self.client.get(f"{self.base_url}/jobs/{job_id}")
        return response.json()
    
    async def wait_for_completion(
        self,
        job_id: str,
        timeout: int = 600,
        poll_interval: int = 5
    ) -> dict:
        """Wait for a job to complete."""
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout}s")
            
            status = await self.get_job_status(job_id)
            current_status = status.get("status")
            current_step = status.get("current_step")
            
            logger.info(f"Job {job_id}: {current_status} (step: {current_step})")
            
            if current_status == "completed":
                return status
            elif current_status == "failed":
                raise RuntimeError(f"Job failed: {status.get('error')}")
            
            await asyncio.sleep(poll_interval)
    
    async def test_single_clip(self):
        """Test single-clip video generation."""
        logger.info("=" * 50)
        logger.info("TEST: Single-clip video generation")
        logger.info("=" * 50)
        
        payload = {
            "image_prompt": "A young woman with flowing auburn hair, studio lighting, photorealistic portrait",
            "motion_prompt": "She turns her head slowly to the left and smiles softly",
            "video_model": "ltx",
            "duration_seconds": 3,
            "apply_face_restore": True,
            "interpolate": False,
            "upscale": False
        }
        
        logger.info(f"Submitting job...")
        job_id = await self.submit_job(payload)
        logger.info(f"Job submitted: {job_id}")
        
        result = await self.wait_for_completion(job_id)
        logger.info(f"Job completed!")
        logger.info(f"Result: {result}")
        
        return job_id
    
    async def test_multi_clip(self):
        """Test multi-clip video generation with frame chaining."""
        logger.info("=" * 50)
        logger.info("TEST: Multi-clip video generation")
        logger.info("=" * 50)
        
        payload = {
            "clips": [
                {
                    "image_prompt": "A man in a business suit standing in a modern office, professional lighting",
                    "motion_prompt": "He looks up from his desk",
                    "duration_seconds": 3
                },
                {
                    "motion_prompt": "He stands up and walks towards the window",
                    "duration_seconds": 4
                },
                {
                    "motion_prompt": "He gazes out the window thoughtfully",
                    "duration_seconds": 3
                }
            ],
            "video_model": "ltx",
            "apply_face_restore": True,
            "interpolate": False,
            "upscale": False
        }
        
        logger.info(f"Submitting multi-clip job...")
        job_id = await self.submit_job(payload, "/generate/multi-clip")
        logger.info(f"Job submitted: {job_id}")
        
        result = await self.wait_for_completion(job_id, timeout=1200)
        logger.info(f"Job completed!")
        logger.info(f"Result: {result}")
        
        return job_id
    
    async def test_image_only(self):
        """Test image-only generation."""
        logger.info("=" * 50)
        logger.info("TEST: Image-only generation")
        logger.info("=" * 50)
        
        payload = {
            "prompt": "A majestic lion standing on a rocky outcrop at sunset, photorealistic, 8K",
            "width": 1024,
            "height": 1024
        }
        
        logger.info(f"Submitting image job...")
        job_id = await self.submit_job(payload, "/generate/image")
        logger.info(f"Job submitted: {job_id}")
        
        result = await self.wait_for_completion(job_id, timeout=120)
        logger.info(f"Job completed!")
        logger.info(f"Result: {result}")
        
        return job_id


async def main():
    parser = argparse.ArgumentParser(description="Test the video pipeline")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000"),
        help="Orchestrator URL"
    )
    parser.add_argument(
        "--test",
        choices=["all", "single", "multi", "image"],
        default="all",
        help="Which test to run"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Testing pipeline at: {args.endpoint}")
    
    tester = PipelineTest(args.endpoint)
    
    try:
        # Health check
        if not await tester.health_check():
            logger.error("Orchestrator is not healthy!")
            sys.exit(1)
        
        logger.info("Orchestrator is healthy")
        
        # Check endpoints
        endpoints = await tester.check_endpoints()
        logger.info(f"Configured endpoints: {endpoints}")
        
        # Run tests
        if args.test in ["all", "image"]:
            await tester.test_image_only()
        
        if args.test in ["all", "single"]:
            await tester.test_single_clip()
        
        if args.test in ["all", "multi"]:
            await tester.test_multi_clip()
        
        logger.info("=" * 50)
        logger.info("All tests completed successfully!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())
