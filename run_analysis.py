#!/usr/bin/env python3
"""
Quick start script for RuneScape Price Tracker
Run this file to start the analysis with default settings
"""

import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.main import RuneScapePriceTracker

async def quick_analysis():
    """Run analysis with default settings"""
    print("🏃‍♂️ Starting OSRS Price Decline Analysis...")
    print("📊 This will fetch current prices and historical data for all tradeable items")
    print("⏱️  This may take several minutes depending on API response times")
    print("📁 Results will be saved to data/exports/ directory")
    print("-" * 60)
    
    tracker = RuneScapePriceTracker()
    
    try:
        await tracker.run_analysis(
            force_refresh=False,  # Use cached data if available
            max_items=None,       # Analyze all items
            show_console=True,    # Show results in console
            save_files=True,      # Save CSV and HTML files
            include_all_items=True  # Include items that might be filtered by quality criteria
        )
        
        print("\n✅ Analysis completed successfully!")
        print("📂 Check the data/exports/ directory for:")
        print("   - CSV file with complete results")
        print("   - Interactive HTML charts")
        print("   - Dashboard with summary statistics")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print("📋 Check price_tracker.log for detailed error information")

if __name__ == "__main__":
    asyncio.run(quick_analysis())
