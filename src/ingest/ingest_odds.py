import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# Add the src directory to the path to import backend
sys.path.append(str(Path(__file__).parent.parent))

from backend.data_backend import MLBDataBackendV3

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to ingest MLB odds data.
    
    This script fetches comprehensive odds data including moneyline,
    strikeout props, and hits/total bases props for upcoming MLB games.
    """
    
    # Configuration
    api_key = os.getenv("ODDS_API_KEY")
    days_ahead = int(os.getenv("ODDS_DAYS_AHEAD", "7"))
    
    if not api_key:
        logger.error("ODDS_API_KEY is required for odds ingestion. Please set it in your .env file.")
        return False
    
    # Initialize backend
    logger.info("Initializing MLB data backend for odds ingestion...")
    backend = MLBDataBackendV3(
        api_key=api_key,
        seasons=[2024]  # Only current season needed for odds
    )
    
    # Create output directories
    raw_data_dir = Path("data/raw")
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Check API usage before making calls
        logger.info("Checking API usage limits...")
        usage_info = backend.check_api_usage()
        logger.info(f"API usage info: {usage_info}")
        
        # Fetch moneyline odds
        logger.info(f"Fetching moneyline odds for next {days_ahead} days...")
        ml_df = backend.fetch_moneyline_odds(days_ahead=days_ahead)
        
        if not ml_df.empty:
            ml_file = raw_data_dir / "ml_odds.parquet"
            logger.info(f"Saving {len(ml_df)} moneyline odds records to {ml_file}")
            ml_df.to_parquet(ml_file, index=False)
            
            # Log moneyline summary
            logger.info("Moneyline odds summary:")
            logger.info(f"  Records: {len(ml_df)}")
            logger.info(f"  Unique games: {len(ml_df.groupby(['date', 'home_team', 'away_team']))}")
            logger.info(f"  Bookmakers: {ml_df['bookmaker'].unique().tolist()}")
            logger.info(f"  Date range: {ml_df['date'].min()} to {ml_df['date'].max()}")
        else:
            logger.warning("No moneyline odds data retrieved")
        
        # Fetch strikeout props
        logger.info(f"Fetching strikeout props for next {days_ahead} days...")
        so_df = backend.fetch_strikeout_props(days_ahead=days_ahead)
        
        if not so_df.empty:
            so_file = raw_data_dir / "so_props.parquet"
            logger.info(f"Saving {len(so_df)} strikeout prop records to {so_file}")
            so_df.to_parquet(so_file, index=False)
            
            # Log strikeout props summary
            logger.info("Strikeout props summary:")
            logger.info(f"  Records: {len(so_df)}")
            logger.info(f"  Unique players: {so_df['player_name'].nunique()}")
            logger.info(f"  Bookmakers: {so_df['bookmaker'].unique().tolist()}")
            logger.info(f"  Line range: {so_df['line'].min():.1f} to {so_df['line'].max():.1f}")
        else:
            logger.warning("No strikeout props data retrieved")
        
        # Fetch hits and total bases props
        logger.info(f"Fetching hits/total bases props for next {days_ahead} days...")
        hb_df = backend.fetch_hit_tb_props(days_ahead=days_ahead)
        
        if not hb_df.empty:
            hb_file = raw_data_dir / "hb_props.parquet"
            logger.info(f"Saving {len(hb_df)} hits/total bases prop records to {hb_file}")
            hb_df.to_parquet(hb_file, index=False)
            
            # Log hits/total bases props summary
            logger.info("Hits/Total bases props summary:")
            logger.info(f"  Records: {len(hb_df)}")
            logger.info(f"  Unique players: {hb_df['player_name'].nunique()}")
            logger.info(f"  Market types: {hb_df['market_type'].unique().tolist()}")
            logger.info(f"  Bookmakers: {hb_df['bookmaker'].unique().tolist()}")
        else:
            logger.warning("No hits/total bases props data retrieved")
        
        # Perform data quality checks
        perform_odds_quality_checks(ml_df, so_df, hb_df)
        
        logger.info("Odds data ingestion completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during odds ingestion: {e}")
        return False


def perform_odds_quality_checks(ml_df, so_df, hb_df):
    """
    Perform data quality checks on odds data.
    """
    logger.info("Performing odds data quality checks...")
    
    issues = []
    
    # Check moneyline odds
    if not ml_df.empty:
        # Check for reasonable odds ranges
        if ml_df['home_odds'].abs().max() > 2000 or ml_df['away_odds'].abs().max() > 2000:
            issues.append("Extreme odds values detected in moneyline data")
        
        # Check implied probabilities (normal range is 1.02-1.2 representing 2-20% vig)
        if (ml_df['market_total_prob'] < 1.02).any() or (ml_df['market_total_prob'] > 1.25).any():
            issues.append("Unusual implied probability totals in moneyline data")
        
        # Check for missing odds
        if ml_df['home_odds'].isna().any() or ml_df['away_odds'].isna().any():
            issues.append("Missing odds values in moneyline data")
    
    # Check strikeout props
    if not so_df.empty:
        # Check for reasonable strikeout line ranges
        if so_df['line'].min() < 0 or so_df['line'].max() > 20:
            issues.append("Unreasonable strikeout line values detected")
        
        # Check for missing player names
        if so_df['player_name'].isna().any() or (so_df['player_name'] == '').any():
            issues.append("Missing player names in strikeout props")
    
    # Check hits/total bases props
    if not hb_df.empty:
        # Check for reasonable line ranges
        if hb_df['line'].min() < 0 or hb_df['line'].max() > 10:
            issues.append("Unreasonable line values in hits/total bases props")
        
        # Check for missing player names
        if hb_df['player_name'].isna().any() or (hb_df['player_name'] == '').any():
            issues.append("Missing player names in hits/total bases props")
    
    # Report issues
    if issues:
        logger.warning("Data quality issues detected:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("All odds data quality checks passed")
    
    return len(issues) == 0


def analyze_odds_market_efficiency(ml_df):
    """
    Analyze market efficiency metrics from moneyline odds.
    """
    if ml_df.empty:
        return
    
    logger.info("Analyzing odds market efficiency...")
    
    # Calculate average vig by bookmaker
    if 'vig' in ml_df.columns:
        avg_vig = ml_df.groupby('bookmaker')['vig'].mean().sort_values()
        logger.info("Average vig by bookmaker:")
        for bookmaker, vig in avg_vig.items():
            logger.info(f"  {bookmaker}: {vig:.3f} ({vig*100:.1f}%)")
    
    # Calculate odds spread (difference between best and worst odds)
    if len(ml_df['bookmaker'].unique()) > 1:
        game_groups = ml_df.groupby(['date', 'home_team', 'away_team'])
        
        spreads = []
        for (date, home, away), group in game_groups:
            if len(group) > 1:
                home_spread = group['home_odds'].max() - group['home_odds'].min()
                away_spread = group['away_odds'].max() - group['away_odds'].min()
                spreads.extend([home_spread, away_spread])
        
        if spreads:
            avg_spread = sum(spreads) / len(spreads)
            logger.info(f"Average odds spread across bookmakers: {avg_spread:.1f}")
    
    # Calculate market consistency
    total_prob_std = ml_df['market_total_prob'].std()
    logger.info(f"Market total probability standard deviation: {total_prob_std:.4f}")


if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    sys.exit(exit_code) 