import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to the path to import backend
sys.path.append(str(Path(__file__).parent.parent))

from backend.data_backend import MLBDataBackendV3

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to ingest MLB games and standings data.
    
    This script fetches comprehensive game results and standings data
    for specified seasons using pybaseball integration, then saves
    the raw data to Parquet files for further processing.
    """
    
    # Configuration
    seasons = [2023, 2024]  # Can be configured via env vars
    api_key = os.getenv("ODDS_API_KEY")
    
    if not api_key:
        logger.warning("ODDS_API_KEY not found in environment. Some features may be limited.")
    
    # Initialize backend
    logger.info("Initializing MLB data backend...")
    backend = MLBDataBackendV3(
        api_key=api_key or "dummy_key",  # Some functionality works without API key
        seasons=seasons
    )
    
    # Create output directories
    raw_data_dir = Path("data/raw")
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Fetch game results and standings
        logger.info("Fetching game results and standings...")
        games_df, standings_df = backend.fetch_game_results()
        
        if games_df.empty:
            logger.error("No games data retrieved. Check your data source.")
            return False
        
        if standings_df.empty:
            logger.error("No standings data retrieved. Check your data source.")
            return False
        
        # Save to Parquet files
        games_file = raw_data_dir / f"games_{'_'.join(map(str, seasons))}.parquet"
        standings_file = raw_data_dir / f"standings_{'_'.join(map(str, seasons))}.parquet"
        
        logger.info(f"Saving {len(games_df)} game records to {games_file}")
        games_df.to_parquet(games_file, index=False)
        
        logger.info(f"Saving {len(standings_df)} standings records to {standings_file}")
        standings_df.to_parquet(standings_file, index=False)
        
        # Display summary statistics
        logger.info("Data ingestion summary:")
        logger.info(f"  Games: {len(games_df)} records")
        logger.info(f"  Seasons: {games_df['season'].unique().tolist()}")
        logger.info(f"  Teams: {games_df['team'].nunique()} unique teams")
        logger.info(f"  Date range: {games_df['date'].min()} to {games_df['date'].max()}")
        
        logger.info(f"  Standings: {len(standings_df)} records")
        logger.info(f"  Standings seasons: {standings_df['season'].unique().tolist()}")
        
        logger.info("Game data ingestion completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        return False


def validate_data_quality(games_df, standings_df):
    """
    Perform basic data quality checks on the ingested data.
    """
    issues = []
    
    # Check for missing dates
    if games_df['date'].isna().any():
        issues.append("Missing dates found in games data")
    
    # Check for missing team names
    if games_df['team'].isna().any():
        issues.append("Missing team names found in games data")
    
    # Check for reasonable win/loss counts
    if 'wins' in standings_df.columns and 'losses' in standings_df.columns:
        total_games = standings_df['wins'] + standings_df['losses']
        if (total_games < 50).any() or (total_games > 200).any():
            issues.append("Unreasonable win/loss totals found")
    
    # Log any issues found
    if issues:
        logger.warning("Data quality issues detected:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Data quality checks passed")
    
    return len(issues) == 0


if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    sys.exit(exit_code) 