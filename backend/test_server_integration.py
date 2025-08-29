#!/usr/bin/env python3
"""
🧪 SERVER INTEGRATION TEST FOR PERFORMANCE MONITORING
Test the integration of performance monitoring with the enhanced server
"""

import asyncio
import aiohttp
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_server_endpoints():
    """Test server endpoints including performance monitoring"""
    
    base_url = "http://localhost:8001"
    
    async with aiohttp.ClientSession() as session:
        # Test basic API
        logger.info("🧪 Testing basic API endpoint...")
        async with session.get(f"{base_url}/api/") as response:
            if response.status == 200:
                data = await response.json()
                logger.info(f"✅ Basic API: {data.get('message', 'Unknown')}")
            else:
                logger.error(f"❌ Basic API failed: {response.status}")
        
        # Test health endpoint
        logger.info("🧪 Testing health endpoint...")
        async with session.get(f"{base_url}/api/health") as response:
            if response.status == 200:
                data = await response.json()
                logger.info(f"✅ Health check: {data.get('status', 'Unknown')}")
            else:
                logger.error(f"❌ Health check failed: {response.status}")
        
        # Test performance endpoints
        performance_endpoints = [
            "/api/performance/health",
            "/api/performance/dashboard", 
            "/api/performance/metrics"
        ]
        
        for endpoint in performance_endpoints:
            logger.info(f"🧪 Testing {endpoint}...")
            async with session.get(f"{base_url}{endpoint}") as response:
                if response.status == 200:
                    logger.info(f"✅ {endpoint} working")
                else:
                    logger.error(f"❌ {endpoint} failed: {response.status}")
                    if response.status == 404:
                        logger.error("   Performance monitoring not integrated properly")

if __name__ == "__main__":
    asyncio.run(test_server_endpoints())