#!/usr/bin/env python3
"""
Comprehensive MLB Props Ingestion Script

Fetches all available MLB prop markets from The Odds API including:
- Player props (home runs, hits, strikeouts, RBIs, etc.)
- Game props (NRFI/YRFI, team totals, inning lines, etc.)
- Pitcher props (strikeouts, hits allowed, earned runs, etc.)
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.data_backend import MLBDataBackendV3
from storage.init_db import MLBDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main ingestion function for comprehensive MLB props."""
    logger.info("Starting comprehensive MLB props ingestion...")
    
    # Initialize data backend
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        logger.error("ODDS_API_KEY environment variable not set")
        return
    
    backend = MLBDataBackendV3(
        api_key=api_key,
        seasons=[2024, 2025]
    )
    
    # Initialize database
    db = MLBDatabase()
    
    try:
        # Fetch comprehensive player props
        logger.info("Fetching comprehensive player props...")
        player_props = backend.fetch_comprehensive_mlb_props(days_ahead=7)
        
        if not player_props.empty:
            # Save to parquet
            output_dir = Path("data/raw")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            player_props_file = output_dir / f"comprehensive_props_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            player_props.to_parquet(player_props_file)
            logger.info(f"Saved {len(player_props)} player prop records to {player_props_file}")
            
            # Load to database
            with db.get_connection() as con:
                con.execute("DELETE FROM raw.comprehensive_props WHERE date >= CURRENT_DATE")
                con.execute("INSERT INTO raw.comprehensive_props SELECT * FROM read_parquet(?)", [str(player_props_file)])
                logger.info(f"Loaded {len(player_props)} player prop records to database")
        
        # Fetch game props
        logger.info("Fetching game props...")
        game_props = backend.fetch_game_props(days_ahead=7)
        
        if not game_props.empty:
            game_props_file = output_dir / f"game_props_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            game_props.to_parquet(game_props_file)
            logger.info(f"Saved {len(game_props)} game prop records to {game_props_file}")
            
            # Load to database
            with db.get_connection() as con:
                con.execute("DELETE FROM raw.game_props WHERE date >= CURRENT_DATE")
                con.execute("INSERT INTO raw.game_props SELECT * FROM read_parquet(?)", [str(game_props_file)])
                logger.info(f"Loaded {len(game_props)} game prop records to database")
        
        # Update legacy tables for backward compatibility
        logger.info("Updating legacy prop tables...")
        update_legacy_tables(db)
        
        logger.info("Comprehensive props ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during comprehensive props ingestion: {e}")
        raise

def update_legacy_tables(db: MLBDatabase):
    """Update legacy SO and HB props tables for backward compatibility."""
    with db.get_connection() as con:
        # Update strikeout props table
        con.execute("""
        DELETE FROM raw.so_props WHERE date >= CURRENT_DATE;
        
        INSERT INTO raw.so_props (
            date, game_time, home_team, away_team, bookmaker, 
            player_name, line, over_odds, under_odds, last_update
        )
        SELECT 
            date, game_time, home_team, away_team, bookmaker,
            player_name, line, over_odds, under_odds, last_update
        FROM raw.comprehensive_props
        WHERE market_type = 'pitcher_strikeouts'
        """)
        
        # Update hits/total bases props table
        con.execute("""
        DELETE FROM raw.hb_props WHERE date >= CURRENT_DATE;
        
        INSERT INTO raw.hb_props (
            date, game_time, home_team, away_team, bookmaker,
            market_type, player_name, line, over_odds, under_odds, last_update
        )
        SELECT 
            date, game_time, home_team, away_team, bookmaker,
            market_type, player_name, line, over_odds, under_odds, last_update
        FROM raw.comprehensive_props
        WHERE market_type IN ('batter_hits', 'batter_total_bases')
        """)
        
        logger.info("Legacy tables updated successfully")

if __name__ == "__main__":
    main() 