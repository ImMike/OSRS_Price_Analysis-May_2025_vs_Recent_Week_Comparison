"""
Data Manager for RuneScape Price Tracker
Handles data fetching, processing, and caching
"""

import sqlite3
import json
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import os

from src.api_client import OSRSApiClient, fetch_item_mapping, fetch_latest_prices, fetch_historical_data
from config.settings import DATABASE_PATH, CACHE_DURATION_HOURS, HISTORICAL_DAYS, MIN_PRICE_THRESHOLD

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data fetching, caching, and processing for OSRS price data"""
    
    def __init__(self):
        self.db_path = DATABASE_PATH
        self._ensure_directories()
        self._init_database()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs("data/exports", exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Items mapping table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    examine TEXT,
                    members BOOLEAN,
                    lowalch INTEGER,
                    highalch INTEGER,
                    [limit] INTEGER,
                    value INTEGER,
                    icon TEXT,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Current prices table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS current_prices (
                    item_id INTEGER PRIMARY KEY,
                    high INTEGER,
                    high_time TIMESTAMP,
                    low INTEGER,
                    low_time TIMESTAMP,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (item_id) REFERENCES items (id)
                )
            ''')
            
            # Historical prices table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_prices (
                    item_id INTEGER,
                    timestamp TIMESTAMP,
                    avg_high_price INTEGER,
                    avg_low_price INTEGER,
                    high_price_volume INTEGER,
                    low_price_volume INTEGER,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (item_id, timestamp),
                    FOREIGN KEY (item_id) REFERENCES items (id)
                )
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def _is_cache_valid(self, table_name: str, hours: int = CACHE_DURATION_HOURS) -> bool:
        """Check if cached data is still valid"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT COUNT(*) FROM {table_name} 
                WHERE cached_at > datetime('now', '-{hours} hours')
            ''')
            count = cursor.fetchone()[0]
            return count > 0
    
    def cache_item_mapping(self, mapping_data: Dict[str, Any]) -> None:
        """Cache item mapping data to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clear existing data
            cursor.execute('DELETE FROM items')
            
            # Insert new data
            for item_id, item_data in mapping_data.items():
                cursor.execute('''
                    INSERT INTO items (id, name, examine, members, lowalch, highalch, [limit], value, icon)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    int(item_id),
                    item_data.get('name', ''),
                    item_data.get('examine', ''),
                    item_data.get('members', False),
                    item_data.get('lowalch'),
                    item_data.get('highalch'),
                    item_data.get('limit'),
                    item_data.get('value'),
                    item_data.get('icon', '')
                ))
            
            conn.commit()
            logger.info(f"Cached {len(mapping_data)} items to database")
    
    def cache_current_prices(self, price_data: Dict[str, Any]) -> None:
        """Cache current price data to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clear existing data
            cursor.execute('DELETE FROM current_prices')
            
            # Insert new data
            for item_id, prices in price_data.items():
                cursor.execute('''
                    INSERT INTO current_prices (item_id, high, high_time, low, low_time)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    int(item_id),
                    prices.get('high'),
                    datetime.fromtimestamp(prices.get('highTime', 0)) if prices.get('highTime') else None,
                    prices.get('low'),
                    datetime.fromtimestamp(prices.get('lowTime', 0)) if prices.get('lowTime') else None
                ))
            
            conn.commit()
            logger.info(f"Cached current prices for {len(price_data)} items")
    
    def cache_historical_prices(self, item_id: int, timeseries_data: Dict[str, Any]) -> None:
        """Cache historical price data for a specific item"""
        if not timeseries_data or 'data' not in timeseries_data:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clear existing data for this item
            cursor.execute('DELETE FROM historical_prices WHERE item_id = ?', (item_id,))
            
            # Insert new data
            for data_point in timeseries_data['data']:
                cursor.execute('''
                    INSERT INTO historical_prices 
                    (item_id, timestamp, avg_high_price, avg_low_price, high_price_volume, low_price_volume)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    item_id,
                    datetime.fromtimestamp(data_point['timestamp']),
                    data_point.get('avgHighPrice'),
                    data_point.get('avgLowPrice'),
                    data_point.get('highPriceVolume'),
                    data_point.get('lowPriceVolume')
                ))
            
            conn.commit()
            logger.debug(f"Cached {len(timeseries_data['data'])} historical data points for item {item_id}")
    
    def get_item_mapping(self, force_refresh: bool = False) -> pd.DataFrame:
        """Get item mapping data, either from cache or API"""
        if not force_refresh and self._is_cache_valid('items'):
            logger.info("Loading item mapping from cache")
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query('SELECT * FROM items', conn)
        
        logger.info("Fetching fresh item mapping from API")
        mapping_data = fetch_item_mapping()
        self.cache_item_mapping(mapping_data)
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query('SELECT * FROM items', conn)
    
    def get_current_prices(self, force_refresh: bool = False) -> pd.DataFrame:
        """Get current price data, either from cache or API"""
        if not force_refresh and self._is_cache_valid('current_prices'):
            logger.info("Loading current prices from cache")
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query('''
                    SELECT cp.*, i.name 
                    FROM current_prices cp 
                    JOIN items i ON cp.item_id = i.id
                ''', conn)
        
        logger.info("Fetching fresh current prices from API")
        price_data = fetch_latest_prices()
        self.cache_current_prices(price_data)
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query('''
                SELECT cp.*, i.name 
                FROM current_prices cp 
                JOIN items i ON cp.item_id = i.id
            ''', conn)
    
    async def get_historical_data(self, item_ids: List[int], force_refresh: bool = False) -> Dict[int, pd.DataFrame]:
        """Get historical price data for specified items"""
        historical_data = {}
        items_to_fetch = []
        
        # Check cache for each item
        if not force_refresh:
            with sqlite3.connect(self.db_path) as conn:
                for item_id in item_ids:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT COUNT(*) FROM historical_prices 
                        WHERE item_id = ? AND cached_at > datetime('now', '-24 hours')
                    ''', (item_id,))
                    
                    if cursor.fetchone()[0] > 0:
                        # Load from cache
                        df = pd.read_sql_query('''
                            SELECT * FROM historical_prices 
                            WHERE item_id = ? 
                            ORDER BY timestamp DESC
                        ''', conn, params=(item_id,))
                        historical_data[item_id] = df
                    else:
                        items_to_fetch.append(item_id)
        else:
            items_to_fetch = item_ids
        
        # Fetch missing data from API
        if items_to_fetch:
            logger.info(f"Fetching historical data for {len(items_to_fetch)} items from API")
            api_data = await fetch_historical_data(items_to_fetch)
            
            for item_id, timeseries_data in api_data.items():
                if timeseries_data:
                    self.cache_historical_prices(item_id, timeseries_data)
                    
                    # Convert to DataFrame
                    if 'data' in timeseries_data:
                        df_data = []
                        for point in timeseries_data['data']:
                            df_data.append({
                                'item_id': item_id,
                                'timestamp': datetime.fromtimestamp(point['timestamp']),
                                'avg_high_price': point.get('avgHighPrice'),
                                'avg_low_price': point.get('avgLowPrice'),
                                'high_price_volume': point.get('highPriceVolume'),
                                'low_price_volume': point.get('lowPriceVolume')
                            })
                        
                        if df_data:
                            historical_data[item_id] = pd.DataFrame(df_data)
        
        return historical_data
    
    def get_tradeable_items(self) -> List[int]:
        """Get list of tradeable item IDs (items with current prices)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT cp.item_id 
                FROM current_prices cp 
                JOIN items i ON cp.item_id = i.id
                WHERE (cp.high IS NOT NULL OR cp.low IS NOT NULL)
                AND (cp.high >= ? OR cp.low >= ?)
            ''', (MIN_PRICE_THRESHOLD, MIN_PRICE_THRESHOLD))
            
            return [row[0] for row in cursor.fetchall()]
