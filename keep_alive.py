import httpx
import asyncio
import logging
import os

logger = logging.getLogger("keep-alive")
SERVICE_URL = os.getenv("SERVICE_URL", "https://image-generation-api-64k8.onrender.com")

async def keep_alive():
    while True:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{SERVICE_URL}/health", timeout=10)
                logger.info("Keep-alive ping: %s", r.status_code)
        except Exception as e:
            logger.warning("Keep-alive failed: %s", e)
        await asyncio.sleep(300)