#!/usr/bin/env python3
"""
Test script to examine The Odds API response structure and data quality.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from backend.data_backend import MLBDataBackendV3

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_api_connection():
    """Test basic API connection and usage info."""
    print("=" * 60)
    print("TESTING API CONNECTION")
    print("=" * 60)
    
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        print("❌ ERROR: ODDS_API_KEY not found in environment")
        return False
    
    print(f"✅ API Key found: {api_key[:8]}...")
    
    # Initialize backend
    backend = MLBDataBackendV3(api_key=api_key, seasons=[2024])
    
    # Check API usage
    try:
        usage_info = backend.check_api_usage()
        print(f"✅ API Connection successful")
        print(f"📊 Usage info: {usage_info}")
        return True
    except Exception as e:
        print(f"❌ API Connection failed: {e}")
        return False


def test_moneyline_odds():
    """Test moneyline odds data structure."""
    print("\n" + "=" * 60)
    print("TESTING MONEYLINE ODDS")
    print("=" * 60)
    
    api_key = os.getenv("ODDS_API_KEY")
    backend = MLBDataBackendV3(api_key=api_key, seasons=[2024])
    
    try:
        # Fetch moneyline odds
        ml_df = backend.fetch_moneyline_odds(days_ahead=3)
        
        if ml_df.empty:
            print("⚠️  No moneyline odds data returned")
            return
        
        print(f"✅ Retrieved {len(ml_df)} moneyline odds records")
        print(f"📅 Date range: {ml_df['date'].min()} to {ml_df['date'].max()}")
        print(f"🏟️  Unique games: {len(ml_df.groupby(['date', 'home_team', 'away_team']))}")
        print(f"📖 Bookmakers: {ml_df['bookmaker'].unique().tolist()}")
        
        # Show column structure
        print("\n📋 Column Structure:")
        for col in ml_df.columns:
            print(f"  - {col}: {ml_df[col].dtype}")
        
        # Show sample data
        print("\n📊 Sample Records:")
        print(ml_df.head(3).to_string(index=False))
        
        # Show data quality metrics
        print("\n🔍 Data Quality Metrics:")
        print(f"  - Missing home odds: {ml_df['home_odds'].isna().sum()}")
        print(f"  - Missing away odds: {ml_df['away_odds'].isna().sum()}")
        print(f"  - Odds range (home): {ml_df['home_odds'].min():.0f} to {ml_df['home_odds'].max():.0f}")
        print(f"  - Odds range (away): {ml_df['away_odds'].min():.0f} to {ml_df['away_odds'].max():.0f}")
        print(f"  - Avg market total prob: {ml_df['market_total_prob'].mean():.3f}")
        print(f"  - Avg vig: {ml_df['vig'].mean():.3f} ({ml_df['vig'].mean()*100:.1f}%)")
        
        # Show vig by bookmaker
        if len(ml_df['bookmaker'].unique()) > 1:
            print("\n💰 Vig by Bookmaker:")
            vig_by_book = ml_df.groupby('bookmaker')['vig'].mean().sort_values()
            for book, vig in vig_by_book.items():
                print(f"  - {book}: {vig:.3f} ({vig*100:.1f}%)")
        
        return ml_df
        
    except Exception as e:
        print(f"❌ Error fetching moneyline odds: {e}")
        return None


def test_strikeout_props():
    """Test strikeout props data structure."""
    print("\n" + "=" * 60)
    print("TESTING STRIKEOUT PROPS")
    print("=" * 60)
    
    api_key = os.getenv("ODDS_API_KEY")
    backend = MLBDataBackendV3(api_key=api_key, seasons=[2024])
    
    try:
        # Fetch strikeout props
        so_df = backend.fetch_strikeout_props(days_ahead=3)
        
        if so_df.empty:
            print("⚠️  No strikeout props data returned")
            return
        
        print(f"✅ Retrieved {len(so_df)} strikeout prop records")
        print(f"👥 Unique players: {so_df['player_name'].nunique()}")
        print(f"📖 Bookmakers: {so_df['bookmaker'].unique().tolist()}")
        print(f"📊 Line range: {so_df['line'].min():.1f} to {so_df['line'].max():.1f}")
        
        # Show column structure
        print("\n📋 Column Structure:")
        for col in so_df.columns:
            print(f"  - {col}: {so_df[col].dtype}")
        
        # Show sample data
        print("\n📊 Sample Records:")
        print(so_df.head(3).to_string(index=False))
        
        # Show player distribution
        print("\n👥 Top Players by Prop Count:")
        player_counts = so_df['player_name'].value_counts().head(10)
        for player, count in player_counts.items():
            print(f"  - {player}: {count} props")
        
        # Show line distribution
        print("\n📊 Line Distribution:")
        line_dist = so_df['line'].value_counts().sort_index()
        for line, count in line_dist.items():
            print(f"  - {line:.1f} strikeouts: {count} props")
        
        return so_df
        
    except Exception as e:
        print(f"❌ Error fetching strikeout props: {e}")
        return None


def test_hit_tb_props():
    """Test hits/total bases props data structure."""
    print("\n" + "=" * 60)
    print("TESTING HITS/TOTAL BASES PROPS")
    print("=" * 60)
    
    api_key = os.getenv("ODDS_API_KEY")
    backend = MLBDataBackendV3(api_key=api_key, seasons=[2024])
    
    try:
        # Fetch hits/TB props
        hb_df = backend.fetch_hit_tb_props(days_ahead=3)
        
        if hb_df.empty:
            print("⚠️  No hits/total bases props data returned")
            return
        
        print(f"✅ Retrieved {len(hb_df)} hits/total bases prop records")
        print(f"👥 Unique players: {hb_df['player_name'].nunique()}")
        print(f"📊 Market types: {hb_df['market_type'].unique().tolist()}")
        print(f"📖 Bookmakers: {hb_df['bookmaker'].unique().tolist()}")
        print(f"📊 Line range: {hb_df['line'].min():.1f} to {hb_df['line'].max():.1f}")
        
        # Show column structure
        print("\n📋 Column Structure:")
        for col in hb_df.columns:
            print(f"  - {col}: {hb_df[col].dtype}")
        
        # Show sample data
        print("\n📊 Sample Records:")
        print(hb_df.head(3).to_string(index=False))
        
        # Show market type breakdown
        print("\n📊 Market Type Breakdown:")
        market_counts = hb_df['market_type'].value_counts()
        for market, count in market_counts.items():
            print(f"  - {market}: {count} props")
        
        # Show top players
        print("\n👥 Top Players by Prop Count:")
        player_counts = hb_df['player_name'].value_counts().head(10)
        for player, count in player_counts.items():
            print(f"  - {player}: {count} props")
        
        return hb_df
        
    except Exception as e:
        print(f"❌ Error fetching hits/total bases props: {e}")
        return None


def analyze_data_completeness(ml_df, so_df, hb_df):
    """Analyze overall data completeness and quality."""
    print("\n" + "=" * 60)
    print("DATA COMPLETENESS ANALYSIS")
    print("=" * 60)
    
    total_records = 0
    if ml_df is not None:
        total_records += len(ml_df)
    if so_df is not None:
        total_records += len(so_df)
    if hb_df is not None:
        total_records += len(hb_df)
    
    print(f"📊 Total records across all markets: {total_records}")
    
    # Check for overlapping games
    if ml_df is not None and not ml_df.empty:
        ml_games = set(ml_df.apply(lambda x: f"{x['date']}_{x['home_team']}_vs_{x['away_team']}", axis=1))
        print(f"🏟️  Moneyline games: {len(ml_games)}")
        
        if so_df is not None and not so_df.empty:
            so_games = set(so_df.apply(lambda x: f"{x['date']}_{x['home_team']}_vs_{x['away_team']}", axis=1))
            print(f"⚾ Strikeout prop games: {len(so_games)}")
            print(f"🔗 Overlapping games (ML + SO): {len(ml_games & so_games)}")
        
        if hb_df is not None and not hb_df.empty:
            hb_games = set(hb_df.apply(lambda x: f"{x['date']}_{x['home_team']}_vs_{x['away_team']}", axis=1))
            print(f"🏏 Hits/TB prop games: {len(hb_games)}")
            print(f"🔗 Overlapping games (ML + HB): {len(ml_games & hb_games)}")
    
    # Check bookmaker coverage
    all_bookmakers = set()
    if ml_df is not None and not ml_df.empty:
        all_bookmakers.update(ml_df['bookmaker'].unique())
    if so_df is not None and not so_df.empty:
        all_bookmakers.update(so_df['bookmaker'].unique())
    if hb_df is not None and not hb_df.empty:
        all_bookmakers.update(hb_df['bookmaker'].unique())
    
    print(f"📖 Total unique bookmakers: {len(all_bookmakers)}")
    print(f"📖 Bookmakers: {list(all_bookmakers)}")


def main():
    """Run all tests."""
    print("🧪 MLB ODDS API TESTING SUITE")
    print("=" * 60)
    
    # Test API connection
    if not test_api_connection():
        return
    
    # Test each data type
    ml_df = test_moneyline_odds()
    so_df = test_strikeout_props()
    hb_df = test_hit_tb_props()
    
    # Analyze completeness
    analyze_data_completeness(ml_df, so_df, hb_df)
    
    print("\n" + "=" * 60)
    print("✅ TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main() 