"""
API Client for RuneScape Wiki Prices API
Handles all API interactions with rate limiting and error handling
"""

import asyncio
import aiohttp
import requests
import time
import logging
from typing import Dict, List, Optional, Any
import json

from config.settings import ENDPOINTS, REQUEST_DELAY, MAX_CONCURRENT_REQUESTS, TIMEOUT, USER_AGENT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleThrottler:
    """Simple rate limiter for API requests"""
    
    def __init__(self, rate_limit: int, period: float):
        self.rate_limit = rate_limit
        self.period = period
        self.semaphore = asyncio.Semaphore(rate_limit)
        self.last_request_time = 0
    
    async def __aenter__(self):
        await self.semaphore.acquire()
        
        # Ensure minimum delay between requests
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.period:
            await asyncio.sleep(self.period - time_since_last)
        
        self.last_request_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release()


class OSRSApiClient:
    """Client for interacting with OSRS Wiki Prices API"""
    
    def __init__(self):
        self.session = None
        self.throttler = SimpleThrottler(rate_limit=MAX_CONCURRENT_REQUESTS, period=REQUEST_DELAY)
        self.headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'application/json'
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=TIMEOUT),
            headers=self.headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def get_item_mapping(self) -> Dict[str, Any]:
        """
        Fetch item mapping data (synchronous)
        Returns: Dictionary with item IDs as keys and item data as values
        """
        try:
            logger.info("Fetching item mapping data...")
            response = requests.get(ENDPOINTS["mapping"], headers=self.headers, timeout=TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert list of items to dictionary with item IDs as keys
            if isinstance(data, list):
                item_dict = {}
                for item in data:
                    if 'id' in item:
                        item_dict[str(item['id'])] = item
                data = item_dict
            
            logger.info(f"Successfully fetched mapping for {len(data)} items")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching item mapping: {e}")
            raise
    
    def get_latest_prices(self) -> Dict[str, Any]:
        """
        Fetch latest prices for all items (synchronous)
        Returns: Dictionary with item IDs as keys and price data as values
        """
        try:
            logger.info("Fetching latest prices...")
            response = requests.get(ENDPOINTS["latest"], headers=self.headers, timeout=TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully fetched latest prices for {len(data.get('data', {}))} items")
            return data.get('data', {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching latest prices: {e}")
            raise
    
    async def get_item_timeseries(self, item_id: int, timestep: str = "24h") -> Optional[Dict[str, Any]]:
        """
        Fetch historical price data for a specific item (async)
        
        Args:
            item_id: The item ID to fetch data for
            timestep: Time interval (5m, 1h, 6h, 24h)
        
        Returns: Dictionary with timestamp and price data
        """
        if not self.session:
            raise RuntimeError("Client must be used as async context manager")
        
        try:
            async with self.throttler:
                url = f"{ENDPOINTS['timeseries']}?id={item_id}&timestep={timestep}"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"Fetched timeseries for item {item_id}")
                        return data
                    elif response.status == 404:
                        logger.warning(f"No timeseries data available for item {item_id}")
                        return None
                    else:
                        logger.error(f"Error fetching timeseries for item {item_id}: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Exception fetching timeseries for item {item_id}: {e}")
            return None
    
    async def get_multiple_timeseries(self, item_ids: List[int], timestep: str = "24h") -> Dict[int, Optional[Dict[str, Any]]]:
        """
        Fetch historical data for multiple items concurrently
        
        Args:
            item_ids: List of item IDs to fetch
            timestep: Time interval for data
        
        Returns: Dictionary mapping item IDs to their timeseries data
        """
        total_items = len(item_ids)
        batch_size = 50  # Process in smaller batches for better progress reporting
        
        logger.info(f"📈 Fetching historical data for {total_items} items...")
        print(f"📊 Processing {total_items} items in batches of {batch_size}...")
        
        all_timeseries_data = {}
        
        for i in range(0, total_items, batch_size):
            batch = item_ids[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_items + batch_size - 1) // batch_size
            
            print(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} items)...")
            
            tasks = [self.get_item_timeseries(item_id, timestep) for item_id in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results with item IDs
            batch_successful = 0
            for item_id, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.debug(f"Exception for item {item_id}: {result}")
                    all_timeseries_data[item_id] = None
                else:
                    all_timeseries_data[item_id] = result
                    if result is not None:
                        batch_successful += 1
            
            # Progress update
            completed_items = min(i + batch_size, total_items)
            total_successful = sum(1 for result in all_timeseries_data.values() if result is not None)
            progress_percent = (completed_items / total_items) * 100
            
            print(f"✅ Batch {batch_num} complete: {batch_successful}/{len(batch)} successful")
            print(f"📊 Overall progress: {completed_items}/{total_items} ({progress_percent:.1f}%) - {total_successful} successful fetches")
            print()
        
        successful_fetches = sum(1 for result in all_timeseries_data.values() if result is not None)
        logger.info(f"🎉 Historical data fetch complete: {successful_fetches}/{total_items} items successful")
        
        return all_timeseries_data


# Convenience functions for synchronous usage
def fetch_item_mapping() -> Dict[str, Any]:
    """Convenience function to fetch item mapping synchronously"""
    client = OSRSApiClient()
    return client.get_item_mapping()


def fetch_latest_prices() -> Dict[str, Any]:
    """Convenience function to fetch latest prices synchronously"""
    client = OSRSApiClient()
    return client.get_latest_prices()


async def fetch_historical_data(item_ids: List[int]) -> Dict[int, Optional[Dict[str, Any]]]:
    """Convenience function to fetch historical data asynchronously"""
    async with OSRSApiClient() as client:
        return await client.get_multiple_timeseries(item_ids)
