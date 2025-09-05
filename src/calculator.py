"""
Price Decline Calculator for RuneScape Price Tracker
Handles all price decline calculations and analysis including liquidity and economic viability
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

from config.settings import HISTORICAL_DAYS, MIN_PRICE_THRESHOLD

logger = logging.getLogger(__name__)


class PriceCalculator:
    """Calculates price declines and related metrics including liquidity and economic viability"""
    
    def __init__(self):
        # Define the specific date ranges for May 2025 vs recent comparison
        self.may_2025_start = datetime(2025, 5, 1)  # May 1st, 2025
        self.may_2025_end = datetime(2025, 5, 31)   # May 31st, 2025
        self.recent_days = 7  # Last 7 days for recent price
        
        # Legacy target date for fallback
        self.target_date = datetime.now() - timedelta(days=HISTORICAL_DAYS)
        
        # Liquidity thresholds for economic viability
        self.min_daily_volume = 100  # Minimum daily trading volume
        self.min_weekly_volume = 500  # Minimum weekly trading volume  
        self.max_spread_percentage = 10.0  # Maximum acceptable bid-ask spread %
        self.min_market_cap = 10000  # Minimum market cap (price * volume) for liquidity
    
    def calculate_price_decline(self, current_price: float, historical_price: float) -> Dict[str, float]:
        """
        Calculate price decline metrics
        
        Args:
            current_price: Current item price
            historical_price: Price from target historical date
        
        Returns:
            Dictionary with decline metrics
        """
        if not current_price or not historical_price or historical_price <= 0:
            return {
                'decline_amount': 0,
                'decline_percentage': 0,
                'is_valid': False
            }
        
        decline_amount = historical_price - current_price
        decline_percentage = (decline_amount / historical_price) * 100
        
        return {
            'decline_amount': decline_amount,
            'decline_percentage': decline_percentage,
            'is_valid': True
        }
    
    def get_historical_price_at_date(self, historical_df: pd.DataFrame, target_date: datetime) -> Optional[float]:
        """
        Get the historical price closest to the target date
        
        Args:
            historical_df: DataFrame with historical price data
            target_date: Target date to find price for
        
        Returns:
            Price value or None if not found
        """
        if historical_df.empty:
            return None
        
        # Convert timestamp column to datetime if it's not already
        if 'timestamp' in historical_df.columns:
            historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
            
            # Find the closest date to our target
            historical_df['date_diff'] = abs(historical_df['timestamp'] - target_date)
            closest_row = historical_df.loc[historical_df['date_diff'].idxmin()]
            
            # Prefer avg_high_price, fall back to avg_low_price
            price = closest_row.get('avg_high_price') or closest_row.get('avg_low_price')
            
            # Check if the closest date is within reasonable range (30 days)
            if closest_row['date_diff'] <= timedelta(days=30):
                return price
        
        return None
    
    def get_may_2025_average_price(self, historical_df: pd.DataFrame) -> Optional[float]:
        """
        Get average price specifically from May 2025 data
        
        Args:
            historical_df: DataFrame with historical price data
        
        Returns:
            Average price from May 2025 or None if insufficient data
        """
        if historical_df.empty:
            return None
        
        # Ensure timestamp column is datetime
        if 'timestamp' in historical_df.columns:
            historical_df = historical_df.copy()
            historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
            
            # Filter to May 2025 data only
            may_2025_data = historical_df[
                (historical_df['timestamp'] >= self.may_2025_start) & 
                (historical_df['timestamp'] <= self.may_2025_end)
            ]
            
            if may_2025_data.empty:
                return None
            
            # Calculate average price
            may_2025_data['avg_price'] = (
                may_2025_data['avg_high_price'].fillna(0) + 
                may_2025_data['avg_low_price'].fillna(0)
            ) / 2
            
            # Remove zero prices
            valid_prices = may_2025_data[may_2025_data['avg_price'] > 0]['avg_price']
            
            if len(valid_prices) >= 3:  # Need at least 3 data points from May 2025
                return valid_prices.mean()
        
        return None
    
    def get_recent_average_price(self, historical_df: pd.DataFrame) -> Optional[float]:
        """
        Get average price from the last 7 days of available data
        
        Args:
            historical_df: DataFrame with historical price data
        
        Returns:
            Average price from recent days or None if insufficient data
        """
        if historical_df.empty:
            return None
        
        # Ensure timestamp column is datetime
        if 'timestamp' in historical_df.columns:
            historical_df = historical_df.copy()
            historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
            
            # Get the most recent date and go back 7 days
            max_date = historical_df['timestamp'].max()
            recent_cutoff = max_date - timedelta(days=self.recent_days)
            
            # Filter to recent data
            recent_data = historical_df[historical_df['timestamp'] >= recent_cutoff]
            
            if recent_data.empty:
                return None
            
            # Calculate average price
            recent_data['avg_price'] = (
                recent_data['avg_high_price'].fillna(0) + 
                recent_data['avg_low_price'].fillna(0)
            ) / 2
            
            # Remove zero prices
            valid_prices = recent_data[recent_data['avg_price'] > 0]['avg_price']
            
            if len(valid_prices) >= 2:  # Need at least 2 data points
                return valid_prices.mean()
        
        return None
    
    def calculate_may_2025_decline(self, recent_price: float, may_2025_price: float) -> Dict[str, float]:
        """
        Calculate price decline from May 2025 average to recent average
        
        Args:
            recent_price: Recent average price (last 7 days)
            may_2025_price: May 2025 average price
        
        Returns:
            Dictionary with decline metrics
        """
        if not recent_price or not may_2025_price or may_2025_price <= 0:
            return {
                'decline_amount': 0,
                'decline_percentage': 0,
                'is_valid': False,
                'method': 'may_2025_comparison'
            }
        
        decline_amount = may_2025_price - recent_price
        decline_percentage = (decline_amount / may_2025_price) * 100
        
        return {
            'decline_amount': decline_amount,
            'decline_percentage': decline_percentage,
            'is_valid': True,
            'method': 'may_2025_comparison',
            'may_2025_price': may_2025_price,
            'recent_price': recent_price
        }
    
    def calculate_price_stability_metrics(self, historical_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate price stability and trend consistency metrics
        
        Args:
            historical_df: DataFrame with historical price data
        
        Returns:
            Dictionary with stability metrics
        """
        if historical_df.empty or len(historical_df) < 7:
            return {
                'price_volatility': 100,  # Very high volatility if no data
                'trend_consistency': 0,
                'outlier_ratio': 1,
                'stability_score': 0,
                'is_stable_market': False,
                'coefficient_of_variation': 100
            }
        
        # Use average of high and low prices for analysis
        historical_df = historical_df.copy()
        historical_df['avg_price'] = (
            historical_df['avg_high_price'].fillna(0) + 
            historical_df['avg_low_price'].fillna(0)
        ) / 2
        
        # Remove zero prices
        prices = historical_df[historical_df['avg_price'] > 0]['avg_price']
        
        if len(prices) < 7:
            return {
                'price_volatility': 100,
                'trend_consistency': 0,
                'outlier_ratio': 1,
                'stability_score': 0,
                'is_stable_market': False,
                'coefficient_of_variation': 100
            }
        
        # Calculate basic volatility metrics
        mean_price = prices.mean()
        std_price = prices.std()
        coefficient_of_variation = (std_price / mean_price) * 100 if mean_price > 0 else 100
        
        # Detect outliers using IQR method
        Q1 = prices.quantile(0.25)
        Q3 = prices.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = prices[(prices < lower_bound) | (prices > upper_bound)]
        outlier_ratio = len(outliers) / len(prices)
        
        # Calculate trend consistency (how consistent is the price movement direction)
        price_changes = prices.diff().dropna()
        if len(price_changes) > 0:
            # Count direction changes
            direction_changes = 0
            for i in range(1, len(price_changes)):
                if (price_changes.iloc[i] > 0) != (price_changes.iloc[i-1] > 0):
                    direction_changes += 1
            
            trend_consistency = max(0, 100 - (direction_changes / len(price_changes) * 100))
        else:
            trend_consistency = 0
        
        # Calculate rolling volatility to check for recent stability
        if len(prices) >= 14:
            recent_prices = prices.tail(14)  # Last 2 weeks
            recent_volatility = recent_prices.std() / recent_prices.mean() * 100
        else:
            recent_volatility = coefficient_of_variation
        
        # Overall stability score (0-100, higher is more stable)
        volatility_score = max(0, 100 - coefficient_of_variation)  # Lower CV = higher score
        outlier_score = max(0, 100 - outlier_ratio * 100)  # Fewer outliers = higher score
        consistency_score = trend_consistency  # Already 0-100
        recent_stability_score = max(0, 100 - recent_volatility)
        
        stability_score = (volatility_score * 0.3 + outlier_score * 0.3 + 
                          consistency_score * 0.2 + recent_stability_score * 0.2)
        
        # Market is considered stable if:
        # - CV < 30% (not too volatile)
        # - Outlier ratio < 20% (not many price spikes)
        # - Stability score > 40
        is_stable_market = (coefficient_of_variation < 30 and 
                           outlier_ratio < 0.2 and 
                           stability_score > 40)
        
        return {
            'price_volatility': coefficient_of_variation,
            'trend_consistency': trend_consistency,
            'outlier_ratio': outlier_ratio,
            'stability_score': stability_score,
            'is_stable_market': is_stable_market,
            'coefficient_of_variation': coefficient_of_variation
        }
    
    def analyze_price_trend(self, historical_df: pd.DataFrame, current_price: float) -> Dict[str, any]:
        """
        Analyze price trend to distinguish genuine declines from outlier corrections
        
        Args:
            historical_df: DataFrame with historical price data
            current_price: Current price
        
        Returns:
            Dictionary with trend analysis
        """
        if historical_df.empty or len(historical_df) < 14:
            return {
                'trend_type': 'insufficient_data',
                'is_genuine_decline': False,
                'trend_strength': 0,
                'support_level': current_price,
                'resistance_level': current_price,
                'trend_duration_days': 0
            }
        
        # Prepare price data
        historical_df = historical_df.copy().sort_values('timestamp')
        historical_df['avg_price'] = (
            historical_df['avg_high_price'].fillna(0) + 
            historical_df['avg_low_price'].fillna(0)
        ) / 2
        
        prices = historical_df[historical_df['avg_price'] > 0]['avg_price']
        
        if len(prices) < 14:
            return {
                'trend_type': 'insufficient_data',
                'is_genuine_decline': False,
                'trend_strength': 0,
                'support_level': current_price,
                'resistance_level': current_price,
                'trend_duration_days': 0
            }
        
        # Calculate moving averages for trend analysis
        short_ma = prices.rolling(window=7).mean()
        long_ma = prices.rolling(window=21).mean() if len(prices) >= 21 else prices.rolling(window=len(prices)//2).mean()
        
        # Determine trend direction
        recent_short = short_ma.tail(3).mean()
        recent_long = long_ma.tail(3).mean()
        
        if recent_short < recent_long * 0.95:  # 5% below long MA
            trend_type = 'declining'
        elif recent_short > recent_long * 1.05:  # 5% above long MA
            trend_type = 'rising'
        else:
            trend_type = 'sideways'
        
        # Calculate trend strength (how consistent is the trend)
        if len(prices) >= 14:
            recent_prices = prices.tail(14)
            if trend_type == 'declining':
                # Count how many recent prices are below the trend
                trend_confirmations = sum(recent_prices < recent_prices.rolling(window=7).mean())
                trend_strength = (trend_confirmations / len(recent_prices)) * 100
            elif trend_type == 'rising':
                trend_confirmations = sum(recent_prices > recent_prices.rolling(window=7).mean())
                trend_strength = (trend_confirmations / len(recent_prices)) * 100
            else:
                trend_strength = 50  # Neutral for sideways
        else:
            trend_strength = 0
        
        # Find support and resistance levels
        recent_prices = prices.tail(30) if len(prices) >= 30 else prices
        support_level = recent_prices.min()
        resistance_level = recent_prices.max()
        
        # Estimate trend duration
        if trend_type != 'sideways':
            # Find when the trend started (when MA crossed)
            ma_diff = short_ma - long_ma
            trend_start_idx = 0
            
            if trend_type == 'declining':
                # Find when short MA went below long MA
                for i in range(len(ma_diff)-1, 0, -1):
                    if ma_diff.iloc[i] > 0:  # Found where it was above
                        trend_start_idx = i
                        break
            else:  # rising
                for i in range(len(ma_diff)-1, 0, -1):
                    if ma_diff.iloc[i] < 0:  # Found where it was below
                        trend_start_idx = i
                        break
            
            trend_duration_days = len(ma_diff) - trend_start_idx
        else:
            trend_duration_days = 0
        
        # Determine if this is a genuine decline (not just outlier correction)
        is_genuine_decline = (
            trend_type == 'declining' and 
            trend_strength > 60 and  # Strong trend
            trend_duration_days >= 7 and  # At least a week
            current_price > support_level * 0.8  # Not crashed below support
        )
        
        return {
            'trend_type': trend_type,
            'is_genuine_decline': is_genuine_decline,
            'trend_strength': trend_strength,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'trend_duration_days': trend_duration_days
        }
    
    def calculate_liquidity_metrics(self, historical_df: pd.DataFrame, current_high: float, current_low: float) -> Dict[str, float]:
        """
        Calculate liquidity metrics from historical volume data
        
        Args:
            historical_df: DataFrame with historical price and volume data
            current_high: Current high price
            current_low: Current low price
        
        Returns:
            Dictionary with liquidity metrics
        """
        if historical_df.empty:
            return {
                'avg_daily_volume': 0,
                'avg_weekly_volume': 0,
                'volume_volatility': 0,
                'bid_ask_spread': 0,
                'bid_ask_spread_percentage': 0,
                'market_depth_score': 0,
                'liquidity_score': 0
            }
        
        # Calculate volume metrics
        high_volumes = historical_df['high_price_volume'].fillna(0)
        low_volumes = historical_df['low_price_volume'].fillna(0)
        total_volumes = high_volumes + low_volumes
        
        avg_daily_volume = total_volumes.mean()
        avg_weekly_volume = avg_daily_volume * 7
        volume_volatility = total_volumes.std() / (avg_daily_volume + 1)  # Coefficient of variation
        
        # Calculate bid-ask spread
        bid_ask_spread = current_high - current_low if current_high and current_low else 0
        mid_price = (current_high + current_low) / 2 if current_high and current_low else current_high or current_low
        bid_ask_spread_percentage = (bid_ask_spread / mid_price * 100) if mid_price > 0 else 0
        
        # Market depth score (combination of volume and price stability)
        recent_volumes = total_volumes.tail(7)  # Last 7 days
        market_depth_score = recent_volumes.mean() * (1 / (volume_volatility + 0.1))
        
        # Overall liquidity score (0-100)
        volume_score = min(avg_daily_volume / self.min_daily_volume * 25, 25)
        spread_score = max(25 - (bid_ask_spread_percentage / self.max_spread_percentage * 25), 0)
        depth_score = min(market_depth_score / 1000 * 25, 25)  # Normalize to 25 points
        consistency_score = max(25 - volume_volatility * 5, 0)  # Lower volatility = higher score
        
        liquidity_score = volume_score + spread_score + depth_score + consistency_score
        
        return {
            'avg_daily_volume': avg_daily_volume,
            'avg_weekly_volume': avg_weekly_volume,
            'volume_volatility': volume_volatility,
            'bid_ask_spread': bid_ask_spread,
            'bid_ask_spread_percentage': bid_ask_spread_percentage,
            'market_depth_score': market_depth_score,
            'liquidity_score': min(liquidity_score, 100)  # Cap at 100
        }
    
    def calculate_slippage_estimate(self, target_volume: int, avg_daily_volume: float, bid_ask_spread_percentage: float) -> Dict[str, float]:
        """
        Estimate trading slippage based on volume and market conditions
        
        Args:
            target_volume: Volume you want to trade
            avg_daily_volume: Average daily trading volume
            bid_ask_spread_percentage: Current bid-ask spread percentage
        
        Returns:
            Dictionary with slippage estimates
        """
        if avg_daily_volume <= 0:
            return {
                'market_impact_percentage': 100,  # Very high impact if no volume
                'time_to_execute_days': 999,
                'total_slippage_percentage': 100,
                'execution_feasibility': 'impossible'
            }
        
        # Market impact based on volume ratio
        volume_ratio = target_volume / avg_daily_volume
        
        # Base market impact (square root model commonly used in finance)
        base_impact = np.sqrt(volume_ratio) * 0.5  # 0.5% per sqrt of daily volume
        
        # Additional impact from spread
        spread_impact = bid_ask_spread_percentage / 2  # Half spread as base cost
        
        # Time to execute (spread across multiple days if large volume)
        time_to_execute = max(1, volume_ratio * 0.5)  # Execute over multiple days for large orders
        
        # Total slippage
        total_slippage = base_impact + spread_impact
        
        # Execution feasibility
        if total_slippage > 20:
            feasibility = 'impossible'
        elif total_slippage > 10:
            feasibility = 'very_difficult'
        elif total_slippage > 5:
            feasibility = 'difficult'
        elif total_slippage > 2:
            feasibility = 'moderate'
        else:
            feasibility = 'easy'
        
        return {
            'market_impact_percentage': base_impact,
            'time_to_execute_days': time_to_execute,
            'total_slippage_percentage': total_slippage,
            'execution_feasibility': feasibility
        }
    
    def calculate_economic_viability(self, decline_percentage: float, liquidity_score: float, 
                                   slippage_percentage: float, current_price: float, 
                                   avg_daily_volume: float) -> Dict[str, any]:
        """
        Calculate overall economic viability for trading this item
        
        Args:
            decline_percentage: Price decline percentage
            liquidity_score: Liquidity score (0-100)
            slippage_percentage: Estimated slippage percentage
            current_price: Current item price
            avg_daily_volume: Average daily trading volume
        
        Returns:
            Dictionary with viability metrics
        """
        # Net profit after slippage
        net_profit_percentage = decline_percentage - slippage_percentage
        
        # Market cap approximation
        market_cap = current_price * avg_daily_volume * 30  # 30-day market cap estimate
        
        # Risk-adjusted return
        risk_factor = 1 - (liquidity_score / 100)  # Higher liquidity = lower risk
        risk_adjusted_return = net_profit_percentage * (1 - risk_factor)
        
        # Opportunity score (0-100)
        profit_score = min(max(net_profit_percentage, 0) * 2, 40)  # Up to 40 points for profit
        liquidity_component = liquidity_score * 0.3  # Up to 30 points for liquidity
        market_size_score = min(market_cap / 100000 * 20, 20)  # Up to 20 points for market size
        execution_score = max(10 - slippage_percentage, 0)  # Up to 10 points for low slippage
        
        opportunity_score = profit_score + liquidity_component + market_size_score + execution_score
        
        # Investment recommendation
        if opportunity_score >= 70 and net_profit_percentage > 5:
            recommendation = 'strong_buy'
        elif opportunity_score >= 50 and net_profit_percentage > 3:
            recommendation = 'buy'
        elif opportunity_score >= 30 and net_profit_percentage > 1:
            recommendation = 'consider'
        else:
            recommendation = 'avoid'
        
        return {
            'net_profit_percentage': net_profit_percentage,
            'market_cap_estimate': market_cap,
            'risk_adjusted_return': risk_adjusted_return,
            'opportunity_score': min(opportunity_score, 100),
            'investment_recommendation': recommendation,
            'is_economically_viable': net_profit_percentage > 2 and liquidity_score > 30
        }
    
    def calculate_enhanced_economic_viability(self, decline_percentage: float, liquidity_score: float, 
                                            slippage_percentage: float, current_price: float, 
                                            avg_daily_volume: float, stability_metrics: Dict[str, any],
                                            trend_analysis: Dict[str, any]) -> Dict[str, any]:
        """
        Enhanced economic viability that filters out outliers and considers market stability
        
        Args:
            decline_percentage: Price decline percentage
            liquidity_score: Liquidity score (0-100)
            slippage_percentage: Estimated slippage percentage
            current_price: Current item price
            avg_daily_volume: Average daily trading volume
            stability_metrics: Price stability metrics
            trend_analysis: Trend analysis results
        
        Returns:
            Dictionary with enhanced viability metrics
        """
        # Start with basic viability calculation
        basic_viability = self.calculate_economic_viability(
            decline_percentage, liquidity_score, slippage_percentage, 
            current_price, avg_daily_volume
        )
        
        # Apply stability and trend filters
        stability_penalty = 0
        trend_penalty = 0
        
        # Penalize unstable markets (outliers, high volatility)
        if not stability_metrics['is_stable_market']:
            stability_penalty += 20
        
        if stability_metrics['outlier_ratio'] > 0.3:  # More than 30% outliers
            stability_penalty += 30
            
        if stability_metrics['coefficient_of_variation'] > 50:  # Very high volatility
            stability_penalty += 25
        
        # Penalize items that aren't in genuine decline trends
        if not trend_analysis['is_genuine_decline']:
            trend_penalty += 40  # Major penalty for non-genuine declines
            
        if trend_analysis['trend_type'] != 'declining':
            trend_penalty += 20
            
        if trend_analysis['trend_strength'] < 50:  # Weak trend
            trend_penalty += 15
        
        # Apply penalties to opportunity score
        adjusted_opportunity_score = max(0, basic_viability['opportunity_score'] - stability_penalty - trend_penalty)
        
        # Recalculate recommendation based on adjusted score and filters
        net_profit = basic_viability['net_profit_percentage']
        
        # Must pass all quality filters for good recommendations
        quality_filters_passed = (
            stability_metrics['is_stable_market'] and
            trend_analysis['is_genuine_decline'] and
            stability_metrics['outlier_ratio'] < 0.2 and
            trend_analysis['trend_strength'] > 60
        )
        
        if quality_filters_passed and adjusted_opportunity_score >= 70 and net_profit > 5:
            enhanced_recommendation = 'strong_buy'
        elif quality_filters_passed and adjusted_opportunity_score >= 50 and net_profit > 3:
            enhanced_recommendation = 'buy'
        elif stability_metrics['is_stable_market'] and adjusted_opportunity_score >= 30 and net_profit > 1:
            enhanced_recommendation = 'consider'
        else:
            enhanced_recommendation = 'avoid'
        
        # Enhanced viability requires both profit and quality
        is_enhanced_viable = (
            net_profit > 2 and 
            liquidity_score > 30 and
            stability_metrics['is_stable_market'] and
            trend_analysis['is_genuine_decline']
        )
        
        # Create quality score (0-100) for ranking
        quality_score = (
            stability_metrics['stability_score'] * 0.4 +
            trend_analysis['trend_strength'] * 0.3 +
            liquidity_score * 0.3
        )
        
        return {
            **basic_viability,  # Include all basic metrics
            'adjusted_opportunity_score': adjusted_opportunity_score,
            'enhanced_recommendation': enhanced_recommendation,
            'is_enhanced_viable': is_enhanced_viable,
            'quality_score': quality_score,
            'stability_penalty': stability_penalty,
            'trend_penalty': trend_penalty,
            'quality_filters_passed': quality_filters_passed
        }
    
    def calculate_fallback_decline(self, current_price: float, high_price: float) -> Dict[str, float]:
        """
        Calculate decline using high price as fallback when historical data is unavailable
        
        Args:
            current_price: Current item price
            high_price: High price from current price data
        
        Returns:
            Dictionary with decline metrics
        """
        if not current_price or not high_price or high_price <= 0:
            return {
                'decline_amount': 0,
                'decline_percentage': 0,
                'is_valid': False,
                'method': 'fallback'
            }
        
        decline_amount = high_price - current_price
        decline_percentage = (decline_amount / high_price) * 100
        
        return {
            'decline_amount': decline_amount,
            'decline_percentage': decline_percentage,
            'is_valid': True,
            'method': 'fallback'
        }
    
    def analyze_item_decline(self, 
                           item_id: int, 
                           item_name: str,
                           current_prices: pd.DataFrame, 
                           historical_data: Optional[pd.DataFrame],
                           target_trade_volume: int = 1000) -> Dict[str, any]:
        """
        Analyze price decline for a single item including liquidity and economic viability
        
        Args:
            item_id: Item ID
            item_name: Item name
            current_prices: DataFrame with current price data
            historical_data: DataFrame with historical price data (optional)
            target_trade_volume: Target volume for slippage calculations
        
        Returns:
            Dictionary with complete analysis results including liquidity metrics
        """
        # Get current price data
        current_row = current_prices[current_prices['item_id'] == item_id]
        if current_row.empty:
            return None
        
        current_row = current_row.iloc[0]
        current_price = current_row.get('high') or current_row.get('low')
        high_price = current_row.get('high')
        low_price = current_row.get('low')
        
        if not current_price or current_price < MIN_PRICE_THRESHOLD:
            return None
        
        # Try to get pre-May vs recent price comparison first
        decline_data = None
        historical_price = None
        calculation_method = 'none'
        
        if historical_data is not None and not historical_data.empty:
            # Try May 2025 comparison first (preferred method)
            may_2025_price = self.get_may_2025_average_price(historical_data)
            recent_price = self.get_recent_average_price(historical_data)
            
            if may_2025_price and recent_price:
                decline_data = self.calculate_may_2025_decline(recent_price, may_2025_price)
                historical_price = may_2025_price
                calculation_method = 'may_2025_comparison'
            else:
                # Fallback to legacy historical comparison
                historical_price = self.get_historical_price_at_date(historical_data, self.target_date)
                if historical_price:
                    decline_data = self.calculate_price_decline(current_price, historical_price)
                    decline_data['method'] = 'historical_legacy'
                    calculation_method = 'historical_legacy'
        
        # If no historical data available, use current high as fallback
        if not decline_data or not decline_data['is_valid']:
            if high_price:
                decline_data = self.calculate_fallback_decline(current_price, high_price)
                historical_price = high_price
                calculation_method = 'fallback'
            else:
                return None
        
        if not decline_data['is_valid']:
            return None
        
        # Calculate price stability metrics
        stability_metrics = self.calculate_price_stability_metrics(
            historical_data if historical_data is not None else pd.DataFrame()
        )
        
        # Calculate trend analysis
        trend_analysis = self.analyze_price_trend(
            historical_data if historical_data is not None else pd.DataFrame(),
            current_price
        )
        
        # Calculate liquidity metrics
        liquidity_metrics = self.calculate_liquidity_metrics(
            historical_data if historical_data is not None else pd.DataFrame(),
            high_price or 0,
            low_price or 0
        )
        
        # Calculate slippage estimates
        slippage_data = self.calculate_slippage_estimate(
            target_trade_volume,
            liquidity_metrics['avg_daily_volume'],
            liquidity_metrics['bid_ask_spread_percentage']
        )
        
        # Enhanced economic viability that considers stability and trend
        viability_data = self.calculate_enhanced_economic_viability(
            decline_data['decline_percentage'],
            liquidity_metrics['liquidity_score'],
            slippage_data['total_slippage_percentage'],
            current_price,
            liquidity_metrics['avg_daily_volume'],
            stability_metrics,
            trend_analysis
        )
        
        # Combine all data
        result = {
            'item_id': item_id,
            'item_name': item_name,
            'current_price': current_price,
            'historical_price': historical_price,
            'decline_amount': decline_data['decline_amount'],
            'decline_percentage': decline_data['decline_percentage'],
            'calculation_method': decline_data.get('method', calculation_method),
            'high_price': high_price,
            'low_price': low_price,
            'high_time': current_row.get('high_time'),
            'low_time': current_row.get('low_time'),
            
            # Add May 2025 comparison specific data
            'may_2025_price': decline_data.get('may_2025_price'),
            'recent_price': decline_data.get('recent_price'),
            
            # Price stability metrics
            'price_volatility': stability_metrics['price_volatility'],
            'trend_consistency': stability_metrics['trend_consistency'],
            'outlier_ratio': stability_metrics['outlier_ratio'],
            'stability_score': stability_metrics['stability_score'],
            'is_stable_market': stability_metrics['is_stable_market'],
            
            # Trend analysis
            'trend_type': trend_analysis['trend_type'],
            'is_genuine_decline': trend_analysis['is_genuine_decline'],
            'trend_strength': trend_analysis['trend_strength'],
            'support_level': trend_analysis['support_level'],
            'resistance_level': trend_analysis['resistance_level'],
            'trend_duration_days': trend_analysis['trend_duration_days'],
            
            # Liquidity metrics
            'avg_daily_volume': liquidity_metrics['avg_daily_volume'],
            'avg_weekly_volume': liquidity_metrics['avg_weekly_volume'],
            'volume_volatility': liquidity_metrics['volume_volatility'],
            'bid_ask_spread': liquidity_metrics['bid_ask_spread'],
            'bid_ask_spread_percentage': liquidity_metrics['bid_ask_spread_percentage'],
            'liquidity_score': liquidity_metrics['liquidity_score'],
            
            # Slippage estimates
            'market_impact_percentage': slippage_data['market_impact_percentage'],
            'time_to_execute_days': slippage_data['time_to_execute_days'],
            'total_slippage_percentage': slippage_data['total_slippage_percentage'],
            'execution_feasibility': slippage_data['execution_feasibility'],
            
            # Enhanced economic viability
            'net_profit_percentage': viability_data['net_profit_percentage'],
            'market_cap_estimate': viability_data['market_cap_estimate'],
            'risk_adjusted_return': viability_data['risk_adjusted_return'],
            'opportunity_score': viability_data['opportunity_score'],
            'adjusted_opportunity_score': viability_data['adjusted_opportunity_score'],
            'investment_recommendation': viability_data['investment_recommendation'],
            'enhanced_recommendation': viability_data['enhanced_recommendation'],
            'is_economically_viable': viability_data['is_economically_viable'],
            'is_enhanced_viable': viability_data['is_enhanced_viable'],
            'quality_score': viability_data['quality_score'],
            'quality_filters_passed': viability_data['quality_filters_passed']
        }
        
        return result
    
    def analyze_all_items(self, 
                         current_prices: pd.DataFrame, 
                         historical_data: Dict[int, pd.DataFrame],
                         items_df: pd.DataFrame,
                         include_all_items: bool = False) -> pd.DataFrame:
        """
        Analyze price declines for all items
        
        Args:
            current_prices: DataFrame with current prices
            historical_data: Dictionary mapping item IDs to historical DataFrames
            items_df: DataFrame with item information
            include_all_items: Include items that might be filtered by quality criteria
        
        Returns:
            DataFrame with analysis results sorted by decline percentage
        """
        logger.info("Analyzing price declines for all items...")
        
        results = []
        processed_count = 0
        
        # Get unique item IDs from current prices
        item_ids = current_prices['item_id'].unique()
        
        for item_id in item_ids:
            # Get item name
            item_row = items_df[items_df['id'] == item_id]
            item_name = item_row.iloc[0]['name'] if not item_row.empty else f"Item {item_id}"
            
            # Get historical data for this item
            item_historical = historical_data.get(item_id)
            
            # Analyze decline
            analysis = self.analyze_item_decline(
                item_id, item_name, current_prices, item_historical
            )
            
            if analysis:
                # If include_all_items is True, include items even if they don't meet enhanced quality criteria
                if include_all_items or analysis.get('is_enhanced_viable', False) or analysis.get('quality_filters_passed', False):
                    results.append(analysis)
                elif analysis.get('decline_percentage', 0) > 0:  # At least show items with some decline
                    results.append(analysis)
            
            processed_count += 1
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count}/{len(item_ids)} items")
        
        if not results:
            logger.warning("No valid price decline data found")
            return pd.DataFrame()
        
        # Convert to DataFrame and sort by adjusted opportunity score (best quality opportunities first)
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('adjusted_opportunity_score', ascending=False)
        
        # Create multiple ranking views
        results_df['decline_rank'] = results_df['decline_percentage'].rank(ascending=False)
        results_df['opportunity_rank'] = results_df['opportunity_score'].rank(ascending=False)
        results_df['adjusted_opportunity_rank'] = results_df['adjusted_opportunity_score'].rank(ascending=False)
        results_df['quality_rank'] = results_df['quality_score'].rank(ascending=False)
        
        logger.info(f"Analysis complete. Found {len(results_df)} items with valid decline data")
        logger.info(f"Methods used: {results_df['calculation_method'].value_counts().to_dict()}")
        
        # Log enhanced insights
        enhanced_viable = len(results_df[results_df['is_enhanced_viable'] == True])
        stable_markets = len(results_df[results_df['is_stable_market'] == True])
        genuine_declines = len(results_df[results_df['is_genuine_decline'] == True])
        quality_passed = len(results_df[results_df['quality_filters_passed'] == True])
        
        logger.info(f"Enhanced viable opportunities: {enhanced_viable}")
        logger.info(f"Stable markets: {stable_markets}")
        logger.info(f"Genuine declines (not outlier corrections): {genuine_declines}")
        logger.info(f"Items passing all quality filters: {quality_passed}")
        
        return results_df
    
    def get_summary_statistics(self, results_df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate comprehensive summary statistics including liquidity and viability metrics
        
        Args:
            results_df: DataFrame with analysis results
        
        Returns:
            Dictionary with summary statistics
        """
        if results_df.empty:
            return {}
        
        # Traditional price decline stats
        basic_stats = {
            'total_items': len(results_df),
            'items_with_decline': len(results_df[results_df['decline_percentage'] > 0]),
            'items_with_increase': len(results_df[results_df['decline_percentage'] < 0]),
            'average_decline_percentage': results_df['decline_percentage'].mean(),
            'median_decline_percentage': results_df['decline_percentage'].median(),
            'max_decline_percentage': results_df['decline_percentage'].max(),
            'min_decline_percentage': results_df['decline_percentage'].min(),
            'total_value_lost': results_df[results_df['decline_amount'] > 0]['decline_amount'].sum(),
            'methods_used': results_df['calculation_method'].value_counts().to_dict()
        }
        
        # Liquidity and viability stats
        liquidity_stats = {
            'economically_viable_items': len(results_df[results_df['is_economically_viable'] == True]),
            'high_liquidity_items': len(results_df[results_df['liquidity_score'] >= 70]),
            'medium_liquidity_items': len(results_df[(results_df['liquidity_score'] >= 40) & (results_df['liquidity_score'] < 70)]),
            'low_liquidity_items': len(results_df[results_df['liquidity_score'] < 40]),
            'average_liquidity_score': results_df['liquidity_score'].mean(),
            'average_daily_volume': results_df['avg_daily_volume'].mean(),
            'average_slippage_percentage': results_df['total_slippage_percentage'].mean(),
            'average_opportunity_score': results_df['opportunity_score'].mean()
        }
        
        # Investment recommendations breakdown
        recommendation_stats = {
            'strong_buy_items': len(results_df[results_df['investment_recommendation'] == 'strong_buy']),
            'buy_items': len(results_df[results_df['investment_recommendation'] == 'buy']),
            'consider_items': len(results_df[results_df['investment_recommendation'] == 'consider']),
            'avoid_items': len(results_df[results_df['investment_recommendation'] == 'avoid'])
        }
        
        # Execution feasibility breakdown
        feasibility_stats = {
            'easy_execution': len(results_df[results_df['execution_feasibility'] == 'easy']),
            'moderate_execution': len(results_df[results_df['execution_feasibility'] == 'moderate']),
            'difficult_execution': len(results_df[results_df['execution_feasibility'] == 'difficult']),
            'very_difficult_execution': len(results_df[results_df['execution_feasibility'] == 'very_difficult']),
            'impossible_execution': len(results_df[results_df['execution_feasibility'] == 'impossible'])
        }
        
        # Enhanced quality stats
        quality_stats = {
            'enhanced_viable_items': len(results_df[results_df['is_enhanced_viable'] == True]),
            'stable_market_items': len(results_df[results_df['is_stable_market'] == True]),
            'genuine_decline_items': len(results_df[results_df['is_genuine_decline'] == True]),
            'quality_filters_passed': len(results_df[results_df['quality_filters_passed'] == True]),
            'average_stability_score': results_df['stability_score'].mean(),
            'average_quality_score': results_df['quality_score'].mean(),
            'average_outlier_ratio': results_df['outlier_ratio'].mean()
        }
        
        # Enhanced recommendations breakdown
        enhanced_recommendation_stats = {
            'enhanced_strong_buy': len(results_df[results_df['enhanced_recommendation'] == 'strong_buy']),
            'enhanced_buy': len(results_df[results_df['enhanced_recommendation'] == 'buy']),
            'enhanced_consider': len(results_df[results_df['enhanced_recommendation'] == 'consider']),
            'enhanced_avoid': len(results_df[results_df['enhanced_recommendation'] == 'avoid'])
        }
        
        # Top quality opportunities (enhanced viable with stable markets and genuine declines)
        top_opportunities = results_df[
            (results_df['is_enhanced_viable'] == True) & 
            (results_df['quality_filters_passed'] == True)
        ].head(10)
        
        opportunity_stats = {
            'top_opportunities_count': len(top_opportunities),
            'top_opportunities': top_opportunities[['item_name', 'decline_percentage', 'net_profit_percentage', 
                                                  'liquidity_score', 'adjusted_opportunity_score', 'quality_score',
                                                  'enhanced_recommendation', 'is_stable_market', 'is_genuine_decline']].to_dict('records') if not top_opportunities.empty else []
        }
        
        # Combine all stats
        return {**basic_stats, **liquidity_stats, **recommendation_stats, **feasibility_stats, 
                **quality_stats, **enhanced_recommendation_stats, **opportunity_stats}
