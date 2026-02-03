#!/usr/bin/env python3
"""
Keep RunPod workers warm to avoid cold starts.

Run this script in the background to periodically ping endpoints,
keeping at least one worker warm for each handler.

Usage:
    python warm_workers.py

Environment variables:
    RUNPOD_API_KEY - RunPod API key
    WARM_INTERVAL - Seconds between pings (default: 300)
    ENDPOINTS - Comma-separated list of handlers to keep warm
"""

import os
import sys
import asyncio
import httpx
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
WARM_INTERVAL = int(os.environ.get("WARM_INTERVAL", 300))  # 5 minutes

# Endpoint IDs - load from environment
ENDPOINTS = {
    "flux": os.environ.get("FLUX_ENDPOINT_ID", ""),
    "ltx_video": os.environ.get("LTX_VIDEO_ENDPOINT_ID", ""),
    "wan21": os.environ.get("WAN21_ENDPOINT_ID", ""),
    "codeformer": os.environ.get("CODEFORMER_ENDPOINT_ID", ""),
    "rife": os.environ.get("RIFE_ENDPOINT_ID", ""),
    "realesrgan": os.environ.get("REALESRGAN_ENDPOINT_ID", ""),
}

# Which endpoints to keep warm (by default, just the heavy ones)
WARM_ENDPOINTS = os.environ.get("ENDPOINTS", "flux,ltx_video,wan21").split(",")


async def ping_endpoint(name: str, endpoint_id: str) -> bool:
    """
    Ping an endpoint to keep it warm.
    
    Args:
        name: Handler name
        endpoint_id: RunPod endpoint ID
    
    Returns:
        True if successful
    """
    if not endpoint_id:
        logger.warning(f"{name}: No endpoint ID configured")
        return False
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/health"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                workers = data.get("workers", {})
                ready = workers.get("ready", 0)
                running = workers.get("running", 0)
                
                logger.info(f"{name}: ready={ready}, running={running}")
                return True
            else:
                logger.warning(f"{name}: HTTP {response.status_code}")
                return False
                
    except Exception as e:
        logger.error(f"{name}: {e}")
        return False


async def warm_cycle():
    """Run a single warm cycle for all configured endpoints."""
    logger.info(f"Starting warm cycle at {datetime.now().isoformat()}")
    
    tasks = []
    for name in WARM_ENDPOINTS:
        name = name.strip()
        if name in ENDPOINTS:
            tasks.append(ping_endpoint(name, ENDPOINTS[name]))
    
    results = await asyncio.gather(*tasks)
    
    success_count = sum(1 for r in results if r)
    logger.info(f"Warm cycle complete: {success_count}/{len(tasks)} endpoints OK")


async def main():
    """Main loop - keep workers warm indefinitely."""
    if not RUNPOD_API_KEY:
        logger.error("RUNPOD_API_KEY not set!")
        sys.exit(1)
    
    logger.info(f"Starting worker warming service")
    logger.info(f"Endpoints to warm: {WARM_ENDPOINTS}")
    logger.info(f"Warm interval: {WARM_INTERVAL} seconds")
    
    while True:
        try:
            await warm_cycle()
        except Exception as e:
            logger.error(f"Error in warm cycle: {e}")
        
        await asyncio.sleep(WARM_INTERVAL)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
