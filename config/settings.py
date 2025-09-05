"""
Configuration settings for RuneScape Price Tracker
"""

# API Configuration
BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"
ENDPOINTS = {
    "mapping": f"{BASE_URL}/mapping",
    "latest": f"{BASE_URL}/latest", 
    "timeseries": f"{BASE_URL}/timeseries"
}

# Rate limiting settings
REQUEST_DELAY = 1.5  # seconds between requests
MAX_CONCURRENT_REQUESTS = 5
TIMEOUT = 30  # request timeout in seconds

# Data settings
CACHE_DURATION_HOURS = 24  # How long to cache mapping data
HISTORICAL_DAYS = 365  # Days of historical data to fetch
MIN_PRICE_THRESHOLD = 1  # Minimum price to consider for calculations

# Liquidity and trading settings
DEFAULT_TARGET_VOLUME = 1000  # Default volume for slippage calculations
MIN_DAILY_VOLUME = 100  # Minimum daily volume for liquidity scoring
MIN_WEEKLY_VOLUME = 500  # Minimum weekly volume for liquidity scoring
MAX_SPREAD_PERCENTAGE = 10.0  # Maximum acceptable bid-ask spread %
MIN_MARKET_CAP = 10000  # Minimum market cap for liquidity consideration

# Database settings
DATABASE_PATH = "data/cache.db"
EXPORTS_PATH = "data/exports"

# Visualization settings
TOP_DECLINING_ITEMS = 50  # Number of top declining items to show
CHART_WIDTH = 1200
CHART_HEIGHT = 800

# User agent for API requests
USER_AGENT = "RuneScape-Price-Tracker/1.0 (Educational Project)"
