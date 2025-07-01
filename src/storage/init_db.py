import os
import logging
from pathlib import Path
import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database(db_path: str = "data/warehouse.duckdb", force_recreate: bool = False):
    """
    Initialize DuckDB warehouse with all raw data tables.
    
    Args:
        db_path: Path to the DuckDB database file
        force_recreate: Whether to drop and recreate existing tables
    """
    
    logger.info(f"Initializing DuckDB warehouse at {db_path}")
    
    # Ensure data directories exist
    Path("data").mkdir(exist_ok=True)
    raw_data_dir = Path("data/raw")
    
    if not raw_data_dir.exists():
        logger.error("Raw data directory not found. Please run ingestion scripts first.")
        return False
    
    try:
        # Connect to DuckDB
        con = duckdb.connect(db_path)
        
        # Install and load necessary extensions
        logger.info("Setting up DuckDB extensions...")
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
        con.execute("INSTALL parquet;")
        con.execute("LOAD parquet;")
        
        # Create schema for organized data
        con.execute("CREATE SCHEMA IF NOT EXISTS raw;")
        con.execute("CREATE SCHEMA IF NOT EXISTS features;")
        con.execute("CREATE SCHEMA IF NOT EXISTS analytics;")
        
        # Initialize games table
        games_files = list(raw_data_dir.glob("games_*.parquet"))
        if games_files:
            logger.info("Creating games table...")
            if force_recreate:
                con.execute("DROP TABLE IF EXISTS raw.games;")
            
            # Create games table from parquet files
            games_pattern = str(raw_data_dir / "games_*.parquet")
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS raw.games AS 
                SELECT * FROM read_parquet('{games_pattern}');
            """)
            
            # Add indexes for performance
            con.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON raw.games(date);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_games_team ON raw.games(team);")
        else:
            logger.warning("No games parquet files found")
        
        # Initialize standings table
        standings_files = list(raw_data_dir.glob("standings_*.parquet"))
        if standings_files:
            logger.info("Creating standings table...")
            if force_recreate:
                con.execute("DROP TABLE IF EXISTS raw.standings;")
            
            standings_pattern = str(raw_data_dir / "standings_*.parquet")
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS raw.standings AS 
                SELECT * FROM read_parquet('{standings_pattern}');
            """)
            
            con.execute("CREATE INDEX IF NOT EXISTS idx_standings_team ON raw.standings(team);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_standings_season ON raw.standings(season);")
        else:
            logger.warning("No standings parquet files found")
        
        # Initialize moneyline odds table
        ml_odds_file = raw_data_dir / "ml_odds.parquet"
        if ml_odds_file.exists():
            logger.info("Creating moneyline odds table...")
            if force_recreate:
                con.execute("DROP TABLE IF EXISTS raw.ml_odds;")
            
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS raw.ml_odds AS 
                SELECT * FROM read_parquet('{ml_odds_file}');
            """)
            
            con.execute("CREATE INDEX IF NOT EXISTS idx_ml_odds_date ON raw.ml_odds(date);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_ml_odds_teams ON raw.ml_odds(home_team, away_team);")
        else:
            logger.warning("No moneyline odds file found")
        
        # Initialize strikeout props table
        so_props_file = raw_data_dir / "so_props.parquet"
        if so_props_file.exists():
            logger.info("Creating strikeout props table...")
            if force_recreate:
                con.execute("DROP TABLE IF EXISTS raw.so_props;")
            
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS raw.so_props AS 
                SELECT * FROM read_parquet('{so_props_file}');
            """)
            
            con.execute("CREATE INDEX IF NOT EXISTS idx_so_props_date ON raw.so_props(date);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_so_props_player ON raw.so_props(player_name);")
        else:
            logger.warning("No strikeout props file found")
        
        # Initialize hits/total bases props table
        hb_props_file = raw_data_dir / "hb_props.parquet"
        if hb_props_file.exists():
            logger.info("Creating hits/total bases props table...")
            if force_recreate:
                con.execute("DROP TABLE IF EXISTS raw.hb_props;")
            
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS raw.hb_props AS 
                SELECT * FROM read_parquet('{hb_props_file}');
            """)
            
            con.execute("CREATE INDEX IF NOT EXISTS idx_hb_props_date ON raw.hb_props(date);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_hb_props_player ON raw.hb_props(player_name);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_hb_props_market ON raw.hb_props(market_type);")
        else:
            logger.warning("No hits/total bases props file found")
        
        # Create useful views for common queries
        create_analytical_views(con)
        
        # Display database statistics
        display_database_stats(con)
        
        con.close()
        logger.info("Database initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False


def create_analytical_views(con):
    """
    Create useful analytical views for common queries.
    """
    logger.info("Creating analytical views...")
    
    # Determine existing raw tables
    try:
        raw_info = con.execute(
            "SELECT table_name FROM duckdb_tables() WHERE schema_name='raw';"
        ).fetchall()
        existing_raw = {row[0] for row in raw_info}
    except Exception:
        existing_raw = set()
        logger.warning("Could not list raw schema tables, proceeding with best effort.")

    # View for latest team standings
    if 'standings' in existing_raw:
        try:
            con.execute("""
                CREATE OR REPLACE VIEW analytics.latest_standings AS
                SELECT *
                FROM raw.standings
                WHERE season = (SELECT MAX(season) FROM raw.standings);
            """)
        except Exception as e:
            logger.warning(f"Skipping latest_standings view: {e}")
    else:
        logger.warning("raw.standings table not found, skipping latest_standings view")

    # View for recent games (last 30 days)
    if 'games' in existing_raw:
        try:
            con.execute("""
                CREATE OR REPLACE VIEW analytics.recent_games AS
                SELECT *
                FROM raw.games
                WHERE date >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY date DESC;
            """)
        except Exception as e:
            logger.warning(f"Skipping recent_games view: {e}")
    else:
        logger.warning("raw.games table not found, skipping recent_games view")

    # View for current odds with best prices
    if 'ml_odds' in existing_raw:
        try:
            con.execute("""
                CREATE OR REPLACE VIEW analytics.best_moneyline_odds AS
                SELECT 
                    date,
                    home_team,
                    away_team,
                    MAX(home_odds) as best_home_odds,
                    MAX(away_odds) as best_away_odds,
                    MIN(vig) as lowest_vig,
                    COUNT(DISTINCT bookmaker) as num_bookmakers
                FROM raw.ml_odds
                GROUP BY date, home_team, away_team
            """)
        except Exception as e:
            logger.warning(f"Skipping best_moneyline_odds view: {e}")
    else:
        logger.warning("raw.ml_odds table not found, skipping best_moneyline_odds view")

    # View for team performance metrics
    if 'games' in existing_raw:
        try:
            con.execute("""
                CREATE OR REPLACE VIEW analytics.team_performance AS
                SELECT 
                    team,
                    season,
                    COUNT(*) as games_played,
                    SUM(win) as wins,
                    COUNT(*) - SUM(win) as losses,
                    AVG(win::FLOAT) as win_pct,
                    AVG(runs_scored) as avg_runs_scored,
                    AVG(runs_allowed) as avg_runs_allowed,
                    AVG(run_differential) as avg_run_diff,
                    SUM(CASE WHEN is_home THEN win ELSE 0 END) as home_wins,
                    SUM(CASE WHEN is_home THEN 1 ELSE 0 END) as home_games,
                    SUM(CASE WHEN NOT is_home THEN win ELSE 0 END) as away_wins,
                    SUM(CASE WHEN NOT is_home THEN 1 ELSE 0 END) as away_games
                FROM raw.games
                GROUP BY team, season
            """)
        except Exception as e:
            logger.warning(f"Skipping team_performance view: {e}")
    else:
        logger.warning("raw.games table not found, skipping team_performance view")

    # Comprehensive props summary
    if 'comprehensive_props' in existing_raw:
        try:
            con.execute("""
                CREATE OR REPLACE VIEW analytics.props_summary AS
                SELECT 
                    market_type,
                    player_name,
                    COUNT(*) as total_props,
                    AVG(line) as avg_line,
                    COUNT(DISTINCT bookmaker) as num_bookmakers,
                    MIN(over_odds) as best_over_odds,
                    MIN(under_odds) as best_under_odds,
                    MAX(last_update) as last_updated
                FROM raw.comprehensive_props
                WHERE player_name IS NOT NULL
                GROUP BY market_type, player_name
            """)
        except Exception as e:
            logger.warning(f"Skipping props_summary view: {e}")
    else:
        logger.warning("raw.comprehensive_props table not found, skipping props_summary view")

    # Game props summary  
    if 'game_props' in existing_raw:
        try:
            con.execute("""
                CREATE OR REPLACE VIEW analytics.game_props_summary AS
                SELECT 
                    market_type,
                    home_team,
                    away_team,
                    date,
                    COUNT(*) as total_lines,
                    AVG(line) as avg_line,
                    AVG(total) as avg_total,
                    AVG(spread) as avg_spread,
                    COUNT(DISTINCT bookmaker) as num_bookmakers
                FROM raw.game_props
                GROUP BY market_type, home_team, away_team, date
            """)
        except Exception as e:
            logger.warning(f"Skipping game_props_summary view: {e}")
    else:
        logger.warning("raw.game_props table not found, skipping game_props_summary view")

    # NRFI/YRFI specific analytics
    if 'game_props' in existing_raw:
        try:
            con.execute("""
                CREATE OR REPLACE VIEW analytics.nrfi_yrfi_odds AS
                SELECT 
                    date,
                    home_team,
                    away_team,
                    bookmaker,
                    CASE WHEN outcome_name = 'Over' THEN odds END as yrfi_odds,
                    CASE WHEN outcome_name = 'Under' THEN odds END as nrfi_odds,
                    total as runs_line
                FROM raw.game_props
                WHERE market_type = 'totals_1st_1_innings'
            """)
        except Exception as e:
            logger.warning(f"Skipping nrfi_yrfi_odds view: {e}")
    else:
        logger.warning("raw.game_props table not found, skipping nrfi_yrfi_odds view")

    # Best prop odds across bookmakers
    if 'comprehensive_props' in existing_raw:
        try:
            con.execute("""
                CREATE OR REPLACE VIEW analytics.best_prop_odds AS
                SELECT 
                    market_type,
                    player_name,
                    line,
                    MAX(over_odds) as best_over_odds,
                    MAX(under_odds) as best_under_odds,
                    COUNT(DISTINCT bookmaker) as num_books_offering
                FROM raw.comprehensive_props
                WHERE player_name IS NOT NULL
                GROUP BY market_type, player_name, line
            """)
        except Exception as e:
            logger.warning(f"Skipping best_prop_odds view: {e}")
    else:
        logger.warning("raw.comprehensive_props table not found, skipping best_prop_odds view")


def display_database_stats(con):
    """
    Display statistics about the initialized database.
    """
    logger.info("Database statistics:")
    
    # List all tables
    tables = con.execute("""
        SELECT schema_name, table_name, estimated_size
        FROM duckdb_tables()
        ORDER BY schema_name, table_name;
    """).fetchall()
    
    for schema, table, size in tables:
        logger.info(f"  {schema}.{table}: {size:,} bytes")
    
    # Count records in each main table
    main_tables = ['raw.games', 'raw.standings', 'raw.ml_odds', 'raw.so_props', 'raw.hb_props']
    
    for table in main_tables:
        try:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info(f"  {table}: {count:,} records")
        except:
            logger.info(f"  {table}: table not found")
    
    # Show views created
    views = con.execute("""
        SELECT schema_name, view_name
        FROM duckdb_views()
        WHERE schema_name = 'analytics'
        ORDER BY view_name;
    """).fetchall()
    
    logger.info(f"Created {len(views)} analytical views:")
    for schema, view in views:
        logger.info(f"  {schema}.{view}")


def test_database_connectivity(db_path: str = "data/warehouse.duckdb"):
    """
    Test database connectivity and run basic queries.
    """
    logger.info("Testing database connectivity...")
    
    try:
        con = duckdb.connect(db_path)
        
        # Test basic queries
        test_queries = [
            "SELECT COUNT(*) as game_count FROM raw.games",
            "SELECT COUNT(DISTINCT team) as team_count FROM raw.games", 
            "SELECT MIN(date) as earliest_date, MAX(date) as latest_date FROM raw.games",
            "SELECT COUNT(*) as odds_count FROM raw.ml_odds"
        ]
        
        for query in test_queries:
            try:
                result = con.execute(query).fetchone()
                logger.info(f"  {query}: {result}")
            except Exception as e:
                logger.warning(f"  Query failed - {query}: {e}")
        
        con.close()
        logger.info("Database connectivity test completed")
        return True
        
    except Exception as e:
        logger.error(f"Database connectivity test failed: {e}")
        return False


def main():
    """
    Main function to initialize the database.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize MLB betting database")
    parser.add_argument("--db-path", default="data/warehouse.duckdb", 
                       help="Path to DuckDB database file")
    parser.add_argument("--force-recreate", action="store_true",
                       help="Force recreation of existing tables")
    parser.add_argument("--test", action="store_true",
                       help="Run connectivity tests after initialization")
    
    args = parser.parse_args()
    
    # Initialize database
    success = init_database(args.db_path, args.force_recreate)
    
    if success and args.test:
        test_database_connectivity(args.db_path)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 