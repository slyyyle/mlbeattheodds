import os
import sys
import logging
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLBFeatureBuilder:
    """
    Comprehensive feature builder for MLB betting models.
    
    Builds features for:
    - Moneyline betting
    - Strikeout props
    - Hits and total bases props
    """
    
    def __init__(self, db_path: str = "data/warehouse.duckdb"):
        self.db_path = db_path
        self.con = None
        
    def __enter__(self):
        self.con = duckdb.connect(self.db_path)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.con:
            self.con.close()
    
    def build_all_features(self, seasons: list = None):
        """
        Build all feature sets for modeling.
        """
        if seasons is None:
            seasons = [2023, 2024]
        
        logger.info(f"Building features for seasons: {seasons}")
        
        # Create features directory
        features_dir = Path("data/features")
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Build each feature set
        self.build_moneyline_features(seasons)
        self.build_strikeout_features(seasons)
        self.build_hits_tb_features(seasons)
        
        logger.info("All feature building completed")
    
    def build_moneyline_features(self, seasons: list):
        """
        Build comprehensive features for moneyline betting models.
        
        Features include:
        - Team performance metrics
        - Head-to-head records
        - Recent form
        - Pitching matchups
        - Home field advantage
        - Market efficiency signals
        """
        logger.info("Building moneyline features...")
        
        # Base SQL query for moneyline features
        sql = f"""
        WITH game_data AS (
            SELECT 
                g.date,
                g.home_team,
                g.away_team,
                g.season,
                CASE WHEN g.team = g.home_team AND g.result = 'W' THEN 1
                     WHEN g.team = g.away_team AND g.result = 'L' THEN 1
                     ELSE 0 END as home_win,
                g.runs_scored,
                g.runs_allowed,
                g.run_differential
            FROM raw.games g
            WHERE g.season IN ({','.join(map(str, seasons))})
                AND g.result IS NOT NULL
                AND g.home_team IS NOT NULL
                AND g.away_team IS NOT NULL
        ),
        
        team_rolling_stats AS (
            SELECT 
                date,
                home_team as team,
                'home' as venue,
                AVG(runs_scored) OVER (
                    PARTITION BY home_team 
                    ORDER BY date 
                    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
                ) as avg_runs_10g,
                AVG(runs_allowed) OVER (
                    PARTITION BY home_team 
                    ORDER BY date 
                    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
                ) as avg_runs_allowed_10g,
                AVG(CASE WHEN home_win = 1 THEN 1.0 ELSE 0.0 END) OVER (
                    PARTITION BY home_team 
                    ORDER BY date 
                    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
                ) as win_pct_10g
            FROM game_data
            
            UNION ALL
            
            SELECT 
                date,
                away_team as team,
                'away' as venue,
                AVG(runs_scored) OVER (
                    PARTITION BY away_team 
                    ORDER BY date 
                    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
                ) as avg_runs_10g,
                AVG(runs_allowed) OVER (
                    PARTITION BY away_team 
                    ORDER BY date 
                    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
                ) as avg_runs_allowed_10g,
                AVG(CASE WHEN home_win = 0 THEN 1.0 ELSE 0.0 END) OVER (
                    PARTITION BY away_team 
                    ORDER BY date 
                    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
                ) as win_pct_10g
            FROM game_data
        ),
        
        standings_features AS (
            SELECT 
                team,
                season,
                wins,
                losses,
                win_pct,
                run_differential,
                games_played,
                ROW_NUMBER() OVER (PARTITION BY season ORDER BY win_pct DESC) as season_rank
            FROM raw.standings
            WHERE season IN ({','.join(map(str, seasons))})
        ),
        
        odds_features AS (
            SELECT 
                date,
                home_team,
                away_team,
                AVG(home_odds) as avg_home_odds,
                AVG(away_odds) as avg_away_odds,
                AVG(home_implied_prob) as avg_home_prob,
                AVG(away_implied_prob) as avg_away_prob,
                AVG(vig) as avg_vig,
                MIN(vig) as min_vig,
                MAX(home_odds) as max_home_odds,
                MIN(home_odds) as min_home_odds,
                MAX(away_odds) as max_away_odds,
                MIN(away_odds) as min_away_odds
            FROM raw.ml_odds
            GROUP BY date, home_team, away_team
        )
        
        SELECT 
            gd.date,
            gd.home_team,
            gd.away_team,
            gd.season,
            gd.home_win,
            
            -- Team performance features
            hs.wins as home_wins,
            hs.losses as home_losses,
            hs.win_pct as home_win_pct,
            hs.run_differential as home_run_diff,
            hs.season_rank as home_rank,
            
            as_.wins as away_wins,
            as_.losses as away_losses,
            as_.win_pct as away_win_pct,
            as_.run_differential as away_run_diff,
            as_.season_rank as away_rank,
            
            -- Win percentage differential
            hs.win_pct - as_.win_pct as win_pct_diff,
            hs.season_rank - as_.season_rank as rank_diff,
            
            -- Recent form (rolling 10 games)
            hrs.avg_runs_10g as home_avg_runs_10g,
            hrs.avg_runs_allowed_10g as home_avg_runs_allowed_10g,
            hrs.win_pct_10g as home_win_pct_10g,
            
            ars.avg_runs_10g as away_avg_runs_10g,
            ars.avg_runs_allowed_10g as away_avg_runs_allowed_10g,
            ars.win_pct_10g as away_win_pct_10g,
            
            -- Form differentials
            hrs.avg_runs_10g - ars.avg_runs_10g as runs_form_diff,
            hrs.win_pct_10g - ars.win_pct_10g as win_form_diff,
            
            -- Odds features
            of.avg_home_odds,
            of.avg_away_odds,
            of.avg_home_prob,
            of.avg_away_prob,
            of.avg_vig,
            of.min_vig,
            of.max_home_odds - of.min_home_odds as home_odds_spread,
            of.max_away_odds - of.min_away_odds as away_odds_spread
            
        FROM game_data gd
        LEFT JOIN standings_features hs ON hs.team = gd.home_team AND hs.season = gd.season
        LEFT JOIN standings_features as_ ON as_.team = gd.away_team AND as_.season = gd.season
        LEFT JOIN team_rolling_stats hrs ON hrs.team = gd.home_team AND hrs.date = gd.date AND hrs.venue = 'home'
        LEFT JOIN team_rolling_stats ars ON ars.team = gd.away_team AND ars.date = gd.date AND ars.venue = 'away'
        LEFT JOIN odds_features of ON of.date = gd.date AND of.home_team = gd.home_team AND of.away_team = gd.away_team
        
        WHERE gd.date >= '{datetime.now().year - 2}-01-01'
        ORDER BY gd.date DESC
        """
        
        try:
            df = self.con.execute(sql).df()
            
            # Add engineered features
            df = self._add_moneyline_engineered_features(df)
            
            # Save features
            output_file = Path("data/features/moneyline.parquet")
            df.to_parquet(output_file, index=False)
            
            logger.info(f"Built {len(df)} moneyline feature records")
            logger.info(f"Features saved to {output_file}")
            
            # Log feature summary
            self._log_feature_summary(df, "moneyline")
            
        except Exception as e:
            logger.error(f"Error building moneyline features: {e}")
    
    def build_strikeout_features(self, seasons: list):
        """
        Build features for strikeout prop betting models.
        
        Features include:
        - Pitcher strikeout rates
        - Batter strikeout rates
        - Matchup history
        - Recent form
        - Park factors
        - Weather conditions
        """
        logger.info("Building strikeout features...")
        
        # This is a simplified version - in practice, you'd need pitcher/batter stats
        sql = f"""
        WITH pitcher_stats AS (
            -- Placeholder for pitcher statistics
            -- In real implementation, integrate with pybaseball pitcher stats
            SELECT 
                'placeholder' as pitcher_name,
                0.0 as k_per_9,
                0.0 as bb_per_9,
                0.0 as era
        ),
        
        batter_matchups AS (
            -- Placeholder for batter vs pitcher matchups
            SELECT 
                'placeholder' as batter_name,
                'placeholder' as pitcher_name,
                0 as career_abs,
                0 as career_strikeouts
        ),
        
        prop_data AS (
            SELECT 
                sp.date,
                sp.home_team,
                sp.away_team,
                sp.player_name,
                sp.line,
                sp.over_odds,
                sp.under_odds,
                -- Add outcome if available (for training)
                NULL as actual_strikeouts
            FROM raw.so_props sp
            WHERE sp.date >= '{datetime.now().year - 1}-01-01'
        )
        
        SELECT 
            pd.*,
            -- Pitcher features would go here
            0.0 as pitcher_k_per_9,
            0.0 as pitcher_recent_form,
            
            -- Opposition features would go here  
            0.0 as opponent_k_rate,
            0.0 as opponent_recent_form,
            
            -- Market features
            (1.0 / (1.0 + ABS(over_odds) / 100.0)) as over_implied_prob,
            (1.0 / (1.0 + ABS(under_odds) / 100.0)) as under_implied_prob
            
        FROM prop_data pd
        ORDER BY pd.date DESC
        """
        
        try:
            df = self.con.execute(sql).df()
            
            # Add engineered features
            df = self._add_strikeout_engineered_features(df)
            
            # Save features
            output_file = Path("data/features/strikeout_props.parquet")
            df.to_parquet(output_file, index=False)
            
            logger.info(f"Built {len(df)} strikeout prop feature records")
            logger.info(f"Features saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error building strikeout features: {e}")
    
    def build_hits_tb_features(self, seasons: list):
        """
        Build features for hits and total bases prop betting models.
        
        Features include:
        - Batter performance metrics
        - Pitcher matchup data
        - Park factors
        - Recent form
        - Weather conditions
        """
        logger.info("Building hits/total bases features...")
        
        # Simplified version - would need actual batter/pitcher stats
        sql = f"""
        WITH batter_stats AS (
            -- Placeholder for batter statistics
            SELECT 
                'placeholder' as batter_name,
                0.0 as avg,
                0.0 as obp,
                0.0 as slg,
                0.0 as tb_per_pa
        ),
        
        prop_data AS (
            SELECT 
                hb.date,
                hb.home_team,
                hb.away_team,
                hb.player_name,
                hb.market_type,
                hb.line,
                hb.over_odds,
                hb.under_odds,
                -- Add outcome if available (for training)
                NULL as actual_hits,
                NULL as actual_tb
            FROM raw.hb_props hb
            WHERE hb.date >= '{datetime.now().year - 1}-01-01'
        )
        
        SELECT 
            pd.*,
            -- Batter features would go here
            0.0 as batter_avg,
            0.0 as batter_recent_form,
            0.0 as batter_vs_pitcher,
            
            -- Pitcher features would go here
            0.0 as pitcher_opponent_avg,
            0.0 as pitcher_recent_form,
            
            -- Park features
            0.0 as park_factor,
            
            -- Market features
            (1.0 / (1.0 + ABS(over_odds) / 100.0)) as over_implied_prob,
            (1.0 / (1.0 + ABS(under_odds) / 100.0)) as under_implied_prob
            
        FROM prop_data pd
        ORDER BY pd.date DESC
        """
        
        try:
            df = self.con.execute(sql).df()
            
            # Add engineered features
            df = self._add_hits_tb_engineered_features(df)
            
            # Save features
            output_file = Path("data/features/hits_tb_props.parquet")
            df.to_parquet(output_file, index=False)
            
            logger.info(f"Built {len(df)} hits/TB prop feature records")
            logger.info(f"Features saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error building hits/TB features: {e}")
    
    def _add_moneyline_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features for moneyline betting.
        """
        # Market efficiency features
        if 'avg_home_prob' in df.columns and 'avg_away_prob' in df.columns:
            df['market_total_prob'] = df['avg_home_prob'] + df['avg_away_prob']
            df['market_edge_home'] = np.where(
                df['home_win_pct'] > df['avg_home_prob'],
                df['home_win_pct'] - df['avg_home_prob'],
                0
            )
            df['market_edge_away'] = np.where(
                df['away_win_pct'] > df['avg_away_prob'],
                df['away_win_pct'] - df['avg_away_prob'],
                0
            )
        
        # Momentum features
        if 'home_win_pct_10g' in df.columns and 'home_win_pct' in df.columns:
            df['home_momentum'] = df['home_win_pct_10g'] - df['home_win_pct']
            df['away_momentum'] = df['away_win_pct_10g'] - df['away_win_pct']
        
        # Strength of schedule proxy
        if 'home_rank' in df.columns and 'away_rank' in df.columns:
            df['matchup_quality'] = 31 - ((df['home_rank'] + df['away_rank']) / 2)
        
        return df
    
    def _add_strikeout_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features for strikeout props.
        """
        # Market efficiency
        if 'over_implied_prob' in df.columns and 'under_implied_prob' in df.columns:
            df['total_implied_prob'] = df['over_implied_prob'] + df['under_implied_prob']
            df['vig'] = df['total_implied_prob'] - 1.0
        
        # Line value assessment
        if 'line' in df.columns:
            df['line_z_score'] = (df['line'] - df['line'].mean()) / df['line'].std()
        
        return df
    
    def _add_hits_tb_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features for hits/total bases props.
        """
        # Market efficiency
        if 'over_implied_prob' in df.columns and 'under_implied_prob' in df.columns:
            df['total_implied_prob'] = df['over_implied_prob'] + df['under_implied_prob']
            df['vig'] = df['total_implied_prob'] - 1.0
        
        # Line difficulty assessment
        if 'line' in df.columns and 'market_type' in df.columns:
            # Different difficulty scales for hits vs total bases
            df['line_difficulty'] = np.where(
                df['market_type'] == 'batter_hits',
                df['line'] / 3.0,  # Hits typically 0-3 range
                df['line'] / 6.0   # Total bases typically 0-6 range
            )
        
        return df
    
    def _log_feature_summary(self, df: pd.DataFrame, feature_type: str):
        """
        Log summary statistics for built features.
        """
        logger.info(f"{feature_type.title()} feature summary:")
        logger.info(f"  Total records: {len(df)}")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  Features: {len(df.columns)} columns")
        
        # Log null percentages for key features
        null_pct = (df.isnull().sum() / len(df) * 100).round(1)
        high_null_features = null_pct[null_pct > 10].sort_values(ascending=False)
        
        if len(high_null_features) > 0:
            logger.warning(f"  Features with >10% null values:")
            for feature, pct in high_null_features.items():
                logger.warning(f"    {feature}: {pct}%")


def main():
    """
    Main function to build all features.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Build MLB betting features")
    parser.add_argument("--db-path", default="data/warehouse.duckdb",
                       help="Path to DuckDB database")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024],
                       help="Seasons to build features for")
    parser.add_argument("--feature-type", choices=["all", "moneyline", "strikeout", "hits_tb"],
                       default="all", help="Type of features to build")
    
    args = parser.parse_args()
    
    try:
        with MLBFeatureBuilder(args.db_path) as builder:
            if args.feature_type == "all":
                builder.build_all_features(args.seasons)
            elif args.feature_type == "moneyline":
                builder.build_moneyline_features(args.seasons)
            elif args.feature_type == "strikeout":
                builder.build_strikeout_features(args.seasons)
            elif args.feature_type == "hits_tb":
                builder.build_hits_tb_features(args.seasons)
        
        logger.info("Feature building completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error in feature building: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 