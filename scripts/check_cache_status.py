#!/usr/bin/env python3
"""
Cache Status Checker for MLB Betting Analytics

This script helps you understand:
1. What's currently cached
2. API usage statistics 
3. Whether running ingestion again will make new API calls
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from backend.data_backend import MLBDataBackendV3

def check_cache_status():
    """Check current cache status and API usage."""
    print("🗄️ MLB Betting Analytics - Cache Status Report")
    print("=" * 60)
    
    # Initialize backend (with dummy API key for cache checking)
    api_key = os.getenv("ODDS_API_KEY", "dummy_key")
    backend = MLBDataBackendV3(api_key=api_key, seasons=[2024])
    
    # Get API usage stats
    usage_stats = backend.get_api_usage_stats()
    
    print("\n📊 API Usage Statistics:")
    print(f"  This Week: {usage_stats['calls_this_week']}/{usage_stats['weekly_limit']}")
    print(f"  Weekly Remaining: {usage_stats['weekly_remaining']}")
    print(f"  Monthly Estimate: {usage_stats['monthly_calls_estimated']}/{usage_stats['monthly_limit']}")
    print(f"  Monthly Remaining: {usage_stats['monthly_remaining_estimated']}")
    print(f"  Current Week: {usage_stats['current_week']}")
    
    # Check cache files
    cache_dir = Path("data/cache")
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.pkl"))
        print(f"\n💾 Cache Files: {len(cache_files)} total")
        
        # Group by date
        files_by_date = {}
        for file in cache_files:
            try:
                # Extract date from filename (format: key_YYYY-MM-DD.pkl)
                date_str = file.stem.split('_')[-1]
                if len(date_str) == 10 and '-' in date_str:  # YYYY-MM-DD format
                    files_by_date.setdefault(date_str, []).append(file.name)
            except:
                continue
        
        # Show cache by date
        today = datetime.now().strftime('%Y-%m-%d')
        for date_str in sorted(files_by_date.keys(), reverse=True):
            file_count = len(files_by_date[date_str])
            age_indicator = "📅 TODAY" if date_str == today else f"📆 {date_str}"
            print(f"  {age_indicator}: {file_count} cached requests")
            
            # Show details for today's cache
            if date_str == today:
                for filename in sorted(files_by_date[date_str])[:5]:  # Show first 5
                    endpoint = filename.replace('.pkl', '').replace(f'_{date_str}', '')
                    # Clean up the endpoint name
                    if 'sports_baseball_mlb_odds' in endpoint:
                        endpoint = "🏀 Moneyline Odds"
                    elif 'sports_baseball_mlb_events' in endpoint and 'pitcher_strikeouts' in endpoint:
                        endpoint = "⚾ Strikeout Props"
                    elif 'sports_baseball_mlb_events' in endpoint and 'batter_' in endpoint:
                        endpoint = "🏏 Batter Props"
                    elif 'sports_baseball_mlb_events' in endpoint:
                        endpoint = "📋 MLB Events"
                    
                    print(f"    - {endpoint}")
                
                if len(files_by_date[date_str]) > 5:
                    print(f"    ... and {len(files_by_date[date_str]) - 5} more")
    
    print("\n🔄 Cache Behavior Analysis:")
    print("  ✅ If you run ingest_odds.py again TODAY:")
    print("     → Will use cached data (NO new API calls)")
    print("     → Data will be identical to previous run")
    print("     → Cache is valid for same day")
    
    print("\n  🔄 If you run ingest_odds.py TOMORROW:")
    print("     → Will make new API calls for fresh data")
    print("     → Will create new cache files for tomorrow")
    print("     → Old cache files remain as backup")
    
    print("\n  ⚠️  If you run ingest_odds.py after hitting weekly limit:")
    print("     → Will try to use stale cache from previous days")
    print("     → Will NOT make new API calls")
    print("     → May return older data if cache exists")
    
    # Check what would happen if we ran ingestion now
    print("\n🧪 Simulation: If you ran ingestion right now...")
    
    # Check if we would use cache for key endpoints
    endpoints_to_check = [
        ("sports/baseball_mlb/odds", {"regions": "us,us2", "markets": "h2h", "oddsFormat": "american", "dateFormat": "iso"}),
        ("sports/baseball_mlb/events", {"dateFormat": "iso"})
    ]
    
    for endpoint, params in endpoints_to_check:
        cache_key = backend._get_cache_key(endpoint, params)
        cache_file = backend._get_cache_file(cache_key)
        
        if backend._is_cache_valid(cache_file):
            print(f"  ✅ {endpoint}: Would use TODAY'S cache (no API call)")
        else:
            # Check for stale cache
            stale_cache = backend._try_stale_cache(cache_key)
            if stale_cache:
                print(f"  🟡 {endpoint}: Would use STALE cache (no API call)")
            else:
                if usage_stats['weekly_remaining'] > 0:
                    print(f"  🔴 {endpoint}: Would make NEW API call")
                else:
                    print(f"  ❌ {endpoint}: NO CACHE + LIMIT EXCEEDED = FAIL")

def main():
    check_cache_status()
    
    print("\n" + "=" * 60)
    print("💡 Tips:")
    print("  • Cache files are named with date stamps")
    print("  • Same day = no new API calls")
    print("  • Weekly limit prevents excessive API usage")
    print("  • Stale cache (up to 7 days old) used when limits hit")
    print("  • Delete cache files to force fresh API calls")
    print("\n🧹 To clear cache: rm -rf data/cache/*.pkl")
    print("🔄 To run ingestion: python src/ingest/ingest_odds.py")

if __name__ == "__main__":
    main() 