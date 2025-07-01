#!/usr/bin/env python3
"""
Test script to examine raw API responses from The Odds API.
"""

import os
import json
import requests
import time
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4"

def test_raw_sports_endpoint():
    """Test the sports endpoint to see available sports."""
    print("=" * 60)
    print("TESTING SPORTS ENDPOINT")
    print("=" * 60)
    
    url = f"{BASE_URL}/sports"
    params = {"apiKey": API_KEY}
    
    try:
        response = requests.get(url, params=params)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Number of sports: {len(data)}")
            
            # Find MLB
            mlb_sports = [sport for sport in data if 'baseball' in sport.get('description', '').lower()]
            print(f"MLB/Baseball sports found: {len(mlb_sports)}")
            
            for sport in mlb_sports:
                print(f"  - {sport}")
                
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")


def test_raw_moneyline_odds():
    """Test raw moneyline odds response."""
    print("\n" + "=" * 60)
    print("TESTING RAW MONEYLINE ODDS")
    print("=" * 60)
    
    url = f"{BASE_URL}/sports/baseball_mlb/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us,us2",
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso"
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Number of games: {len(data)}")
            
            if data:
                print("\nFirst game structure:")
                pprint(data[0], width=120, depth=4)
                
                print(f"\nSample bookmaker structure:")
                if data[0].get('bookmakers'):
                    pprint(data[0]['bookmakers'][0], width=120, depth=3)
                    
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")


def test_raw_strikeout_props():
    """Test raw strikeout props response."""
    print("\n" + "=" * 60)
    print("TESTING RAW STRIKEOUT PROPS")
    print("=" * 60)
    
    url = f"{BASE_URL}/sports/baseball_mlb/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us,us2",
        "markets": "pitcher_strikeouts",
        "oddsFormat": "american",
        "dateFormat": "iso"
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Number of games with SO props: {len(data)}")
            
            if data:
                print("\nFirst game with SO props structure:")
                pprint(data[0], width=120, depth=4)
                
                # Look for strikeout markets
                if data[0].get('bookmakers'):
                    for bookmaker in data[0]['bookmakers']:
                        for market in bookmaker.get('markets', []):
                            if market['key'] == 'pitcher_strikeouts':
                                print(f"\nSample strikeout market from {bookmaker['title']}:")
                                pprint(market, width=120, depth=3)
                                break
                        else:
                            continue
                        break
                    
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")


def test_raw_batter_props():
    """Test raw batter props response."""
    print("\n" + "=" * 60)
    print("TESTING RAW BATTER PROPS")
    print("=" * 60)
    
    url = f"{BASE_URL}/sports/baseball_mlb/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us,us2",
        "markets": "batter_hits,batter_total_bases",
        "oddsFormat": "american",
        "dateFormat": "iso"
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Number of games with batter props: {len(data)}")
            
            if data:
                print("\nFirst game with batter props structure:")
                pprint(data[0], width=120, depth=4)
                
                # Look for batter markets
                if data[0].get('bookmakers'):
                    for bookmaker in data[0]['bookmakers']:
                        for market in bookmaker.get('markets', []):
                            if market['key'] in ['batter_hits', 'batter_total_bases']:
                                print(f"\nSample {market['key']} market from {bookmaker['title']}:")
                                pprint(market, width=120, depth=3)
                                break
                        else:
                            continue
                        break
                    
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")


def check_api_usage():
    """Check API usage and limits."""
    print("\n" + "=" * 60)
    print("CHECKING API USAGE")
    print("=" * 60)
    
    url = f"{BASE_URL}/sports"
    params = {"apiKey": API_KEY}
    
    try:
        response = requests.get(url, params=params)
        
        # API usage info is in headers
        usage_headers = {
            'x-requests-remaining': response.headers.get('x-requests-remaining'),
            'x-requests-used': response.headers.get('x-requests-used'),
            'x-requests-last': response.headers.get('x-requests-last'),
        }
        
        print("API Usage Information:")
        for key, value in usage_headers.items():
            if value:
                print(f"  {key}: {value}")
        
        return usage_headers
        
    except Exception as e:
        print(f"Exception checking usage: {e}")
        return {}


def main():
    """Run all raw API tests."""
    print("🔍 RAW API RESPONSE TESTING SUITE", flush=True)
    print("=" * 60, flush=True)
    
    print(f"DEBUG: API_KEY loaded: {API_KEY}", flush=True)
    
    if not API_KEY:
        print("❌ ERROR: ODDS_API_KEY not found in environment", flush=True)
        return
    
    print(f"✅ API Key: {API_KEY[:8]}...", flush=True)
    
    # Check usage first
    print("Starting API usage check...", flush=True)
    check_api_usage()
    
    # Test different endpoints
    print("Testing sports endpoint...", flush=True)
    test_raw_sports_endpoint()
    time.sleep(1)  # Rate limiting
    
    print("Testing moneyline odds...", flush=True)
    test_raw_moneyline_odds()
    time.sleep(1)
    
    print("Testing strikeout props...", flush=True)
    test_raw_strikeout_props()
    time.sleep(1)
    
    print("Testing batter props...", flush=True)
    test_raw_batter_props()
    
    # Final usage check
    print("Final usage check...", flush=True)
    check_api_usage()
    
    print("\n" + "=" * 60, flush=True)
    print("✅ RAW API TESTING COMPLETE", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main() 