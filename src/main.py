"""
Main controller for RuneScape Price Tracker
Orchestrates the entire workflow from data fetching to visualization
"""

import asyncio
import logging
import argparse
from datetime import datetime
from typing import Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_manager import DataManager
from src.calculator import PriceCalculator
from src.visualizer import PriceVisualizer
from config.settings import TOP_DECLINING_ITEMS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('price_tracker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RuneScapePriceTracker:
    """Main application class for OSRS price tracking and analysis"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.calculator = PriceCalculator()
        self.visualizer = PriceVisualizer()
    
    async def run_analysis(self, 
                          force_refresh: bool = False,
                          max_items: Optional[int] = None,
                          show_console: bool = True,
                          save_files: bool = True,
                          include_all_items: bool = False) -> None:
        """
        Run complete price decline analysis
        
        Args:
            force_refresh: Force refresh of all cached data
            max_items: Limit analysis to N items (for testing)
            show_console: Display results in console
            save_files: Save results and charts to files
            include_all_items: Include items that might be filtered by quality criteria
        """
        try:
            logger.info("Starting OSRS Price Decline Analysis...")
            start_time = datetime.now()
            
            # Step 1: Get item mapping
            logger.info("Fetching item mapping...")
            items_df = self.data_manager.get_item_mapping(force_refresh=force_refresh)
            logger.info(f"Loaded {len(items_df)} items")
            
            # Step 2: Get current prices
            logger.info("Fetching current prices...")
            current_prices = self.data_manager.get_current_prices(force_refresh=force_refresh)
            logger.info(f"Loaded current prices for {len(current_prices)} items")
            
            # Step 3: Get tradeable items
            tradeable_items = self.data_manager.get_tradeable_items()
            logger.info(f"Found {len(tradeable_items)} tradeable items")
            
            # Limit items for testing if specified
            if max_items:
                tradeable_items = tradeable_items[:max_items]
                logger.info(f"Limited analysis to {len(tradeable_items)} items for testing")
            
            # Step 4: Get historical data
            logger.info("Fetching historical price data...")
            historical_data = await self.data_manager.get_historical_data(
                tradeable_items, force_refresh=force_refresh
            )
            
            successful_fetches = sum(1 for data in historical_data.values() if data is not None)
            logger.info(f"Successfully loaded historical data for {successful_fetches}/{len(tradeable_items)} items")
            
            # Step 4.5: Debug - Look for chinchompas (including specific ID 11959)
            print("🔍 Searching for chinchompa items...")
            
            # Check for the specific chinchompa ID from the wiki
            target_chinchompa_id = 11959
            print(f"🎯 Specifically checking for Chinchompa (ID: {target_chinchompa_id})...")
            
            target_item = items_df[items_df['id'] == target_chinchompa_id]
            if not target_item.empty:
                item = target_item.iloc[0]
                has_current_price = target_chinchompa_id in current_prices['item_id'].values
                is_tradeable = target_chinchompa_id in tradeable_items
                has_historical = target_chinchompa_id in historical_data and historical_data[target_chinchompa_id] is not None
                
                print(f"  ✓ Found: {item['name']} (ID: {target_chinchompa_id})")
                print(f"    Current Price: {'✓' if has_current_price else '✗'}")
                print(f"    Tradeable: {'✓' if is_tradeable else '✗'}")
                print(f"    Historical Data: {'✓' if has_historical else '✗'}")
                
                if has_current_price:
                    price_row = current_prices[current_prices['item_id'] == target_chinchompa_id].iloc[0]
                    print(f"    High: {price_row.get('high', 'N/A')} | Low: {price_row.get('low', 'N/A')}")
                    
                if has_historical and historical_data[target_chinchompa_id] is not None:
                    hist_data = historical_data[target_chinchompa_id]
                    print(f"    Historical records: {len(hist_data)} data points")
                    if not hist_data.empty:
                        avg_volume = (hist_data['high_price_volume'].fillna(0) + hist_data['low_price_volume'].fillna(0)).mean()
                        print(f"    Average daily volume: {avg_volume:.0f}")
            else:
                print(f"  ✗ Chinchompa (ID: {target_chinchompa_id}) not found in item mapping!")
            
            # Also check for other chinchompa variants
            chinchompa_items = items_df[items_df['name'].str.contains('chinchompa', case=False, na=False)]
            if not chinchompa_items.empty:
                print(f"\n📋 Found {len(chinchompa_items)} total chinchompa items:")
                for _, item in chinchompa_items.iterrows():
                    item_id = item['id']
                    has_current_price = item_id in current_prices['item_id'].values
                    is_tradeable = item_id in tradeable_items
                    has_historical = item_id in historical_data and historical_data[item_id] is not None
                    
                    print(f"  - {item['name']} (ID: {item_id})")
                    print(f"    Current Price: {'✓' if has_current_price else '✗'}")
                    print(f"    Tradeable: {'✓' if is_tradeable else '✗'}")
                    print(f"    Historical Data: {'✓' if has_historical else '✗'}")
                    
                    if has_current_price:
                        price_row = current_prices[current_prices['item_id'] == item_id].iloc[0]
                        print(f"    High: {price_row.get('high', 'N/A')} | Low: {price_row.get('low', 'N/A')}")
            else:
                print("  No chinchompa items found in item mapping!")
            
            # Step 5: Calculate price declines
            logger.info("Calculating price declines...")
            print("🧮 Analyzing price changes and calculating declines...")
            print("📅 Using May 2025 vs Recent Week comparison methodology:")
            print("   • May 2025: Average prices from May 1-31, 2025")
            print("   • Recent Week: Average prices from last 7 days")
            print("   • Fallback: Legacy historical comparison if insufficient data")
            if include_all_items:
                print("📋 Including all items (even those that might not meet quality criteria)")
            results_df = self.calculator.analyze_all_items(
                current_prices, historical_data, items_df, include_all_items=include_all_items
            )
            
            if results_df.empty:
                logger.error("No valid price decline data found. Analysis cannot continue.")
                return
            
            # Step 5.5: Debug - Check if chinchompas made it to final results
            print("🔍 Checking if chinchompas are in final results...")
            
            # Check specifically for the target chinchompa
            target_chinchompa_result = results_df[results_df['item_id'] == target_chinchompa_id]
            if not target_chinchompa_result.empty:
                item = target_chinchompa_result.iloc[0]
                print(f"🎯 Target Chinchompa (ID: {target_chinchompa_id}) found in results!")
                print(f"  - {item['item_name']}")
                print(f"    Rank: #{item.name + 1} (by adjusted opportunity score)")
                print(f"    Decline: {item['decline_percentage']:.1f}%")
                print(f"    Net Profit: {item['net_profit_percentage']:.1f}%")
                print(f"    Enhanced Recommendation: {item['enhanced_recommendation']}")
                print(f"    Stable Market: {item['is_stable_market']}")
                print(f"    Genuine Decline: {item['is_genuine_decline']}")
                print(f"    Quality Score: {item['quality_score']:.0f}")
                print(f"    Liquidity Score: {item['liquidity_score']:.0f}")
                print(f"    Opportunity Score: {item['adjusted_opportunity_score']:.0f}")
            else:
                print(f"🎯 Target Chinchompa (ID: {target_chinchompa_id}) NOT found in final results!")
            
            # Check for other chinchompa variants
            chinchompa_results = results_df[results_df['item_name'].str.contains('chinchompa', case=False, na=False)]
            if not chinchompa_results.empty:
                print(f"\n📋 Found {len(chinchompa_results)} other chinchompa items in results:")
                for _, item in chinchompa_results.iterrows():
                    print(f"  - {item['item_name']} (ID: {item['item_id']})")
                    print(f"    Decline: {item['decline_percentage']:.1f}%")
                    print(f"    Enhanced Recommendation: {item['enhanced_recommendation']}")
                    print(f"    Stable Market: {item['is_stable_market']}")
                    print(f"    Genuine Decline: {item['is_genuine_decline']}")
                    print(f"    Quality Score: {item['quality_score']:.0f}")
                    print(f"    Liquidity Score: {item['liquidity_score']:.0f}")
            
            if target_chinchompa_result.empty and chinchompa_results.empty:
                print("  No chinchompa items found in final results!")
                print("  They may have been filtered out by quality criteria.")
                
                # Show why chinchompas might have been filtered
                if not chinchompa_items.empty:
                    print("  Possible reasons for exclusion:")
                    for _, item in chinchompa_items.iterrows():
                        item_id = item['id']
                        if item_id in tradeable_items and item_id in historical_data:
                            print(f"    - {item['name']}: Likely filtered by quality/stability criteria")
                        elif item_id not in tradeable_items:
                            print(f"    - {item['name']}: Not in tradeable items (no current price or below threshold)")
                        elif item_id not in historical_data:
                            print(f"    - {item['name']}: No historical data available")
            
            # Step 6: Generate summary statistics
            summary_stats = self.calculator.get_summary_statistics(results_df)
            
            # Step 7: Display results
            if show_console:
                self._display_console_results(results_df, summary_stats)
            
            # Step 8: Create visualizations
            logger.info("Creating visualizations...")
            print("📊 Generating interactive charts and visualizations...")
            
            # Main decline chart
            print("  📉 Creating price decline bar chart...")
            decline_chart = self.visualizer.create_decline_bar_chart(results_df)
            
            # Distribution chart
            print("  📈 Creating decline distribution chart...")
            distribution_chart = self.visualizer.create_decline_distribution(results_df)
            
            # Method comparison
            print("  🔍 Creating method comparison chart...")
            method_chart = self.visualizer.create_method_comparison(results_df)
            
            # Interactive data table
            print("  📊 Creating interactive data table...")
            data_table = self.visualizer.create_interactive_data_table(results_df)
            
            # Sortable HTML table
            print("  🔢 Creating sortable HTML table...")
            sortable_table_path = self.visualizer.create_sortable_html_table(results_df)
            
            # Comprehensive dashboard
            print("  🎯 Creating comprehensive dashboard...")
            dashboard_path = self.visualizer.create_comprehensive_dashboard(results_df, summary_stats)
            
            # Step 9: Save files if requested
            if save_files:
                logger.info("Saving results...")
                print("💾 Saving results to files...")
                
                # Save CSV
                print("  📄 Saving CSV data file...")
                csv_path = self.visualizer.save_results_csv(results_df)
                
                # Save charts
                print("  🌐 Saving HTML chart files...")
                decline_path = self.visualizer.save_chart_html(decline_chart, "decline_chart.html")
                distribution_path = self.visualizer.save_chart_html(distribution_chart, "distribution_chart.html")
                method_path = self.visualizer.save_chart_html(method_chart, "method_chart.html")
                data_table_path = self.visualizer.save_chart_html(data_table, "data_table.html")
                
                logger.info(f"Files saved:")
                logger.info(f"  - Data: {csv_path}")
                logger.info(f"  - Charts: {decline_path}, {distribution_path}, {method_path}")
                logger.info(f"  - Interactive Table: {data_table_path}")
                logger.info(f"  - Sortable Table: {sortable_table_path}")
                logger.info(f"  - Comprehensive Dashboard: {dashboard_path}")
                
                print(f"\n🎉 ANALYSIS COMPLETE!")
                print(f"📂 Open this file in your browser for the full interactive experience:")
                print(f"   {dashboard_path}")
                print(f"📊 Interactive data table available at: {data_table_path}")
                print(f"🔢 SORTABLE table (click headers to sort): {sortable_table_path}")
                print(f"📄 Raw data exported to: {csv_path}")
            
            # Step 10: Show completion summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("Analysis completed successfully!")
            logger.info(f"Duration: {duration.total_seconds():.1f} seconds")
            logger.info(f"Analyzed {len(results_df)} items with valid decline data")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def _display_console_results(self, results_df, summary_stats):
        """Display results in console"""
        logger.info("Displaying results...")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("OSRS PRICE DECLINE & LIQUIDITY ANALYSIS SUMMARY")
        print("="*80)
        
        if summary_stats:
            # Basic decline stats
            print(f"📊 BASIC STATISTICS")
            print(f"   Total Items Analyzed: {summary_stats.get('total_items', 0):,}")
            print(f"   Items with Price Decline: {summary_stats.get('items_with_decline', 0):,}")
            print(f"   Items with Price Increase: {summary_stats.get('items_with_increase', 0):,}")
            print(f"   Average Decline: {summary_stats.get('average_decline_percentage', 0):.1f}%")
            print(f"   Maximum Decline: {summary_stats.get('max_decline_percentage', 0):.1f}%")
            print(f"   Total Value Lost: {summary_stats.get('total_value_lost', 0):,.0f} gp")
            
            # Liquidity and viability stats
            print(f"\n💰 ECONOMIC VIABILITY")
            print(f"   Economically Viable Items: {summary_stats.get('economically_viable_items', 0):,}")
            print(f"   High Liquidity Items (≥70): {summary_stats.get('high_liquidity_items', 0):,}")
            print(f"   Medium Liquidity Items (40-69): {summary_stats.get('medium_liquidity_items', 0):,}")
            print(f"   Low Liquidity Items (<40): {summary_stats.get('low_liquidity_items', 0):,}")
            print(f"   Average Liquidity Score: {summary_stats.get('average_liquidity_score', 0):.1f}")
            print(f"   Average Daily Volume: {summary_stats.get('average_daily_volume', 0):.0f}")
            print(f"   Average Slippage: {summary_stats.get('average_slippage_percentage', 0):.1f}%")
            
            # Market quality metrics
            print(f"\n🔍 MARKET QUALITY ANALYSIS")
            print(f"   Enhanced Viable Items: {summary_stats.get('enhanced_viable_items', 0):,}")
            print(f"   Stable Markets: {summary_stats.get('stable_market_items', 0):,}")
            print(f"   Genuine Declines: {summary_stats.get('genuine_decline_items', 0):,}")
            print(f"   Quality Filters Passed: {summary_stats.get('quality_filters_passed', 0):,}")
            print(f"   Average Stability Score: {summary_stats.get('average_stability_score', 0):.1f}")
            print(f"   Average Outlier Ratio: {summary_stats.get('average_outlier_ratio', 0):.1%}")
            
            # Enhanced investment recommendations
            print(f"\n📈 ENHANCED RECOMMENDATIONS (Quality-Filtered)")
            print(f"   Strong Buy: {summary_stats.get('enhanced_strong_buy', 0):,} items")
            print(f"   Buy: {summary_stats.get('enhanced_buy', 0):,} items")
            print(f"   Consider: {summary_stats.get('enhanced_consider', 0):,} items")
            print(f"   Avoid: {summary_stats.get('enhanced_avoid', 0):,} items")
            
            # Execution feasibility
            print(f"\n⚡ EXECUTION FEASIBILITY")
            print(f"   Easy: {summary_stats.get('easy_execution', 0):,} items")
            print(f"   Moderate: {summary_stats.get('moderate_execution', 0):,} items")
            print(f"   Difficult: {summary_stats.get('difficult_execution', 0):,} items")
            print(f"   Very Difficult: {summary_stats.get('very_difficult_execution', 0):,} items")
            print(f"   Impossible: {summary_stats.get('impossible_execution', 0):,} items")
            
            # Data methods
            methods = summary_stats.get('methods_used', {})
            print(f"\n📋 DATA SOURCES")
            print(f"   Historical Data: {methods.get('historical', 0):,} items")
            print(f"   Fallback Method: {methods.get('fallback', 0):,} items")
            
            # Top quality opportunities
            top_opps = summary_stats.get('top_opportunities', [])
            if top_opps:
                print(f"\n🎯 TOP QUALITY OPPORTUNITIES ({len(top_opps)} items)")
                for i, opp in enumerate(top_opps[:5], 1):  # Show top 5
                    print(f"   {i}. {opp['item_name']}")
                    print(f"      Decline: {opp['decline_percentage']:.1f}% | Net Profit: {opp['net_profit_percentage']:.1f}%")
                    print(f"      Liquidity: {opp['liquidity_score']:.0f} | Quality: {opp['quality_score']:.0f}")
                    print(f"      Stable: {'✓' if opp['is_stable_market'] else '✗'} | Genuine: {'✓' if opp['is_genuine_decline'] else '✗'} | {opp['enhanced_recommendation'].upper()}")
                    print()
        
        print("\n")
        
        # Print top declining items table
        self.visualizer.print_console_table(results_df, TOP_DECLINING_ITEMS)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='OSRS Price Decline Tracker')
    parser.add_argument('--force-refresh', action='store_true', 
                       help='Force refresh of all cached data')
    parser.add_argument('--max-items', type=int, 
                       help='Limit analysis to N items (for testing)')
    parser.add_argument('--no-console', action='store_true',
                       help='Skip console output')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving files')
    
    args = parser.parse_args()
    
    tracker = RuneScapePriceTracker()
    
    await tracker.run_analysis(
        force_refresh=args.force_refresh,
        max_items=args.max_items,
        show_console=not args.no_console,
        save_files=not args.no_save
    )


if __name__ == "__main__":
    asyncio.run(main())
