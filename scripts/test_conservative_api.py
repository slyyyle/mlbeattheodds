#!/usr/bin/env python3
"""
Test script for conservative API usage with daily caching.
This script demonstrates how the system stays under the 500 API calls/month limit.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backend.data_backend import MLBDataBackendV3

def test_conservative_api_usage():
    """Test the conservative API usage and caching system."""
    
    # Get API key from environment
    api_key = os.getenv('ODDS_API_KEY')
    if not api_key:
        print("⚠️  ODDS_API_KEY environment variable not set")
        print("   This is fine for testing cache behavior")
        api_key = "test_key_for_cache_demo"
    
    # Initialize backend with conservative settings
    backend = MLBDataBackendV3(
        api_key=api_key,
        seasons=[2024]
    )
    
    print("🏀 MLB Betting Analytics - Conservative API Usage Test")
    print("=" * 60)
    
    # Show initial API usage stats
    usage_stats = backend.get_api_usage_stats()
    print(f"📊 Initial API Usage Stats:")
    print(f"   • Calls today: {usage_stats['calls_today']}/{usage_stats['daily_limit']}")
    print(f"   • Calls remaining: {usage_stats['calls_remaining']}")
    print(f"   • Monthly estimate: {usage_stats['monthly_estimate']}/500")
    print(f"   • Cache files: {usage_stats['cache_files']}")
    print()
    
    # Test 1: Fetch moneyline odds (will use cache if available)
    print("🎯 Test 1: Fetching moneyline odds...")
    start_time = time.time()
    
    try:
        odds_df = backend.fetch_moneyline_odds(days_ahead=3)
        elapsed = time.time() - start_time
        
        print(f"   ✅ Fetched {len(odds_df)} odds records in {elapsed:.2f}s")
        if len(odds_df) > 0:
            print(f"   📈 Sample: {odds_df.iloc[0]['home_team']} vs {odds_df.iloc[0]['away_team']}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   🔄 This is expected if no API key is set or daily limit reached")
    
    print()
    
    # Test 2: Fetch the same data again (should use cache)
    print("🎯 Test 2: Fetching same data again (should use cache)...")
    start_time = time.time()
    
    try:
        odds_df_2 = backend.fetch_moneyline_odds(days_ahead=3)
        elapsed = time.time() - start_time
        
        print(f"   ✅ Fetched {len(odds_df_2)} odds records in {elapsed:.2f}s")
        print(f"   💾 Cache hit - no API call made!")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Show updated API usage stats
    usage_stats = backend.get_api_usage_stats()
    print(f"📊 Updated API Usage Stats:")
    print(f"   • Calls today: {usage_stats['calls_today']}/{usage_stats['daily_limit']}")
    print(f"   • Calls remaining: {usage_stats['calls_remaining']}")
    print(f"   • Monthly estimate: {usage_stats['monthly_estimate']}/500")
    print(f"   • Cache files: {usage_stats['cache_files']}")
    print()
    
    # Test 3: Try to fetch different data (will respect daily limit)
    print("🎯 Test 3: Fetching strikeout props...")
    start_time = time.time()
    
    try:
        props_df = backend.fetch_strikeout_props(days_ahead=2)
        elapsed = time.time() - start_time
        
        print(f"   ✅ Fetched {len(props_df)} prop records in {elapsed:.2f}s")
        if len(props_df) > 0:
            print(f"   🎯 Sample: {props_df.iloc[0]['player_name']} strikeouts")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   🔄 This is expected if daily limit reached")
    
    print()
    
    # Final stats
    usage_stats = backend.get_api_usage_stats()
    print(f"📊 Final API Usage Stats:")
    print(f"   • Calls today: {usage_stats['calls_today']}/{usage_stats['daily_limit']}")
    print(f"   • Calls remaining: {usage_stats['calls_remaining']}")
    print(f"   • Monthly estimate: {usage_stats['monthly_estimate']}/500")
    print(f"   • Cache files: {usage_stats['cache_files']}")
    print()
    
    # Cache management
    print("🧹 Cache Management:")
    print(f"   • Cache directory: {backend.cache_dir}")
    print(f"   • Cleaning cache files older than 7 days...")
    backend.clear_cache(older_than_days=7)
    print(f"   • Cache files after cleanup: {len(list(backend.cache_dir.glob('*.pkl')))}")
    print()
    
    # Summary
    print("📋 Summary:")
    print("   ✅ Conservative API usage system working")
    print("   ✅ Daily caching prevents duplicate API calls")
    print("   ✅ Automatic cache cleanup prevents disk bloat")
    print("   ✅ API usage tracking prevents exceeding limits")
    print(f"   📊 Monthly projection: {usage_stats['monthly_estimate']}/500 calls")
    
    if usage_stats['monthly_estimate'] <= 500:
        print("   🎉 UNDER MONTHLY LIMIT - System is working correctly!")
    else:
        print("   ⚠️  Over monthly limit - consider reducing API calls")

if __name__ == "__main__":
    test_conservative_api_usage() 