import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedMLBFeatures:
    """
    Advanced feature engineering for MLB betting analytics.
    
    Features:
    - Sabermetric statistics
    - Player matchup analysis
    - Weather impact modeling
    - Umpire tendencies
    - Park factors
    - Market intelligence
    - Situational analysis
    - Momentum indicators
    """
    
    def __init__(self):
        self.park_factors = self._load_park_factors()
        self.umpire_tendencies = {}
        self.weather_impact_models = {}
        
    def _load_park_factors(self) -> Dict[str, Dict[str, float]]:
        """Load park factors for all MLB stadiums."""
        # Simplified park factors - in production, these would be calculated from historical data
        return {
            'Coors Field': {'runs': 1.15, 'home_runs': 1.25, 'hits': 1.08},
            'Fenway Park': {'runs': 1.05, 'home_runs': 1.10, 'hits': 1.03},
            'Yankee Stadium': {'runs': 1.02, 'home_runs': 1.08, 'hits': 1.01},
            'Petco Park': {'runs': 0.92, 'home_runs': 0.85, 'hits': 0.96},
            'Marlins Park': {'runs': 0.95, 'home_runs': 0.90, 'hits': 0.98},
            # Add more parks as needed
        }
    
    def build_pitcher_features(self, pitcher_data: pd.DataFrame, 
                              opposing_lineup: pd.DataFrame = None) -> pd.DataFrame:
        """
        Build comprehensive pitcher features.
        
        Args:
            pitcher_data: Pitcher statistics and information
            opposing_lineup: Opposing team's batting lineup
            
        Returns:
            DataFrame with advanced pitcher features
        """
        features = pitcher_data.copy()
        
        # Basic rate stats
        if 'IP' in features.columns and features['IP'].sum() > 0:
            features['K_per_9'] = (features['SO'] * 9) / features['IP']
            features['BB_per_9'] = (features['BB'] * 9) / features['IP']
            features['HR_per_9'] = (features['HR'] * 9) / features['IP']
            features['WHIP'] = (features['BB'] + features['H']) / features['IP']
            features['K_BB_ratio'] = features['SO'] / features['BB'].replace(0, np.nan)
        
        # Advanced metrics
        features = self._add_pitcher_advanced_metrics(features)
        
        # Situational splits
        features = self._add_pitcher_situational_features(features)
        
        # Recent form (last 5 starts)
        features = self._add_pitcher_recent_form(features)
        
        # Matchup-specific features
        if opposing_lineup is not None:
            features = self._add_pitcher_matchup_features(features, opposing_lineup)
        
        # Rest and usage patterns
        features = self._add_pitcher_usage_features(features)
        
        return features
    
    def _add_pitcher_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced pitcher metrics."""
        # FIP (Fielding Independent Pitching)
        if all(col in df.columns for col in ['HR', 'BB', 'SO', 'IP']):
            fip_constant = 3.10  # Approximate FIP constant
            df['FIP'] = ((13 * df['HR'] + 3 * df['BB'] - 2 * df['SO']) / df['IP']) + fip_constant
        
        # xFIP (Expected FIP)
        if 'FB' in df.columns and df['FB'].sum() > 0:
            league_hr_fb_rate = 0.105  # Approximate league average
            df['xFIP'] = ((13 * df['FB'] * league_hr_fb_rate + 3 * df['BB'] - 2 * df['SO']) / df['IP']) + 3.10
        
        # SIERA (Skill-Interactive ERA)
        if all(col in df.columns for col in ['SO', 'BB', 'GB', 'FB']):
            # Simplified SIERA calculation
            df['K_rate'] = df['SO'] / (df['SO'] + df['BB'] + df['H'])
            df['BB_rate'] = df['BB'] / (df['SO'] + df['BB'] + df['H'])
            df['GB_rate'] = df['GB'] / (df['GB'] + df['FB'])
            
            df['SIERA'] = (
                6.145 - 16.986 * df['K_rate'] + 11.434 * df['BB_rate'] - 
                1.858 * df['GB_rate'] + 7.653 * df['K_rate']**2
            )
        
        # Velocity trends (if available)
        if 'avg_fastball_velocity' in df.columns:
            df['velocity_trend'] = df['avg_fastball_velocity'].rolling(5).mean().diff()
        
        return df
    
    def _add_pitcher_situational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add situational pitching features."""
        # Home/Away splits
        if 'home_ERA' in df.columns and 'away_ERA' in df.columns:
            df['home_away_ERA_diff'] = df['home_ERA'] - df['away_ERA']
        
        # Day/Night splits
        if 'day_ERA' in df.columns and 'night_ERA' in df.columns:
            df['day_night_ERA_diff'] = df['day_ERA'] - df['night_ERA']
        
        # vs Left/Right handed batters
        if 'vs_LHB_OPS' in df.columns and 'vs_RHB_OPS' in df.columns:
            df['platoon_advantage'] = df['vs_RHB_OPS'] - df['vs_LHB_OPS']
        
        # High leverage situations
        if 'high_leverage_ERA' in df.columns:
            df['clutch_performance'] = df['ERA'] - df['high_leverage_ERA']
        
        return df
    
    def _add_pitcher_recent_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add recent form indicators for pitchers."""
        # Last 5 starts metrics
        recent_cols = ['last_5_ERA', 'last_5_WHIP', 'last_5_K9']
        
        for col in recent_cols:
            if col in df.columns:
                season_col = col.replace('last_5_', '')
                if season_col in df.columns:
                    df[f'{col}_vs_season'] = df[col] - df[season_col]
        
        # Trend indicators
        if 'game_score' in df.columns:
            df['game_score_trend'] = df['game_score'].rolling(3).mean().diff()
        
        # Workload indicators
        if 'pitches_thrown' in df.columns:
            df['recent_workload'] = df['pitches_thrown'].rolling(3).sum()
            df['workload_trend'] = df['pitches_thrown'].rolling(3).mean().diff()
        
        return df
    
    def _add_pitcher_matchup_features(self, pitcher_df: pd.DataFrame, 
                                    opposing_lineup: pd.DataFrame) -> pd.DataFrame:
        """Add pitcher vs opposing lineup features."""
        # Historical performance against opposing team
        if 'vs_team_ERA' in pitcher_df.columns:
            pitcher_df['team_matchup_advantage'] = pitcher_df['ERA'] - pitcher_df['vs_team_ERA']
        
        # Lineup handedness analysis
        if 'handedness' in opposing_lineup.columns:
            lhb_count = (opposing_lineup['handedness'] == 'L').sum()
            rhb_count = (opposing_lineup['handedness'] == 'R').sum()
            
            pitcher_df['opposing_lhb_pct'] = lhb_count / len(opposing_lineup)
            pitcher_df['opposing_rhb_pct'] = rhb_count / len(opposing_lineup)
        
        # Power hitter concentration
        if 'ISO' in opposing_lineup.columns:
            pitcher_df['opposing_power_avg'] = opposing_lineup['ISO'].mean()
            pitcher_df['opposing_power_concentration'] = (opposing_lineup['ISO'] > 0.200).sum() / len(opposing_lineup)
        
        return pitcher_df
    
    def _add_pitcher_usage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pitcher usage and rest features."""
        # Days of rest
        if 'last_start_date' in df.columns:
            df['days_rest'] = (pd.Timestamp.now() - pd.to_datetime(df['last_start_date'])).dt.days
            
            # Rest impact on performance
            df['rest_category'] = pd.cut(df['days_rest'], 
                                       bins=[0, 3, 5, 7, float('inf')], 
                                       labels=['short_rest', 'normal_rest', 'extra_rest', 'long_layoff'])
        
        # Season usage
        if 'innings_pitched_season' in df.columns:
            df['season_workload_pct'] = df['innings_pitched_season'] / 200  # Normalize by 200 IP
        
        return df
    
    def build_batter_features(self, batter_data: pd.DataFrame, 
                            opposing_pitcher: pd.DataFrame = None) -> pd.DataFrame:
        """
        Build comprehensive batter features.
        
        Args:
            batter_data: Batter statistics and information
            opposing_pitcher: Opposing pitcher information
            
        Returns:
            DataFrame with advanced batter features
        """
        features = batter_data.copy()
        
        # Advanced batting metrics
        features = self._add_batter_advanced_metrics(features)
        
        # Situational performance
        features = self._add_batter_situational_features(features)
        
        # Recent form
        features = self._add_batter_recent_form(features)
        
        # Matchup-specific features
        if opposing_pitcher is not None:
            features = self._add_batter_matchup_features(features, opposing_pitcher)
        
        # Lineup context
        features = self._add_batter_lineup_context(features)
        
        return features
    
    def _add_batter_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced batting metrics."""
        # wOBA (Weighted On-Base Average)
        if all(col in df.columns for col in ['BB', 'HBP', '1B', '2B', '3B', 'HR', 'PA']):
            # wOBA weights (approximate 2023 values)
            woba_weights = {'BB': 0.690, 'HBP': 0.722, '1B': 0.888, '2B': 1.271, '3B': 1.616, 'HR': 2.101}
            
            df['wOBA'] = (
                woba_weights['BB'] * df['BB'] + 
                woba_weights['HBP'] * df['HBP'] +
                woba_weights['1B'] * df['1B'] +
                woba_weights['2B'] * df['2B'] +
                woba_weights['3B'] * df['3B'] +
                woba_weights['HR'] * df['HR']
            ) / df['PA']
        
        # wRC+ (Weighted Runs Created Plus)
        if 'wOBA' in df.columns:
            league_woba = 0.320  # Approximate league average
            woba_scale = 1.157   # Approximate scale factor
            df['wRC_plus'] = (((df['wOBA'] - league_woba) / woba_scale) + 1) * 100
        
        # Barrel Rate (if Statcast data available)
        if 'barrels' in df.columns and 'batted_ball_events' in df.columns:
            df['barrel_rate'] = df['barrels'] / df['batted_ball_events']
        
        # Hard Hit Rate
        if 'hard_hit_balls' in df.columns and 'batted_ball_events' in df.columns:
            df['hard_hit_rate'] = df['hard_hit_balls'] / df['batted_ball_events']
        
        # Expected stats (if available)
        if 'xBA' in df.columns and 'AVG' in df.columns:
            df['BA_vs_xBA'] = df['AVG'] - df['xBA']
        
        return df
    
    def _add_batter_situational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add situational batting features."""
        # Clutch performance
        if 'RISP_AVG' in df.columns and 'AVG' in df.columns:
            df['clutch_performance'] = df['RISP_AVG'] - df['AVG']
        
        # Home/Away splits
        if 'home_OPS' in df.columns and 'away_OPS' in df.columns:
            df['home_away_OPS_diff'] = df['home_OPS'] - df['away_OPS']
        
        # vs LHP/RHP splits
        if 'vs_LHP_OPS' in df.columns and 'vs_RHP_OPS' in df.columns:
            df['platoon_split'] = df['vs_RHP_OPS'] - df['vs_LHP_OPS']
        
        # Count performance
        if 'ahead_count_OPS' in df.columns and 'behind_count_OPS' in df.columns:
            df['count_performance'] = df['ahead_count_OPS'] - df['behind_count_OPS']
        
        return df
    
    def _add_batter_recent_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add recent form indicators for batters."""
        # Last 15 games performance
        recent_cols = ['last_15_AVG', 'last_15_OPS', 'last_15_wOBA']
        
        for col in recent_cols:
            if col in df.columns:
                season_col = col.replace('last_15_', '')
                if season_col in df.columns:
                    df[f'{col}_vs_season'] = df[col] - df[season_col]
        
        # Hot/Cold streaks
        if 'current_streak' in df.columns:
            df['streak_momentum'] = np.where(df['current_streak'] > 0, 
                                           np.log1p(df['current_streak']), 
                                           -np.log1p(abs(df['current_streak'])))
        
        # Plate discipline trends
        if 'BB_rate' in df.columns and 'K_rate' in df.columns:
            df['discipline_trend'] = df['BB_rate'].rolling(10).mean().diff() - df['K_rate'].rolling(10).mean().diff()
        
        return df
    
    def _add_batter_matchup_features(self, batter_df: pd.DataFrame, 
                                   opposing_pitcher: pd.DataFrame) -> pd.DataFrame:
        """Add batter vs pitcher matchup features."""
        # Historical performance vs pitcher
        if 'vs_pitcher_AVG' in batter_df.columns:
            batter_df['pitcher_matchup_advantage'] = batter_df['vs_pitcher_AVG'] - batter_df['AVG']
        
        # Handedness matchup
        if 'handedness' in batter_df.columns and 'throws' in opposing_pitcher.columns:
            # Same-handed matchup disadvantage
            batter_df['same_handed_matchup'] = (batter_df['handedness'] == opposing_pitcher['throws'].iloc[0])
        
        # Velocity matchup
        if 'avg_exit_velocity' in batter_df.columns and 'avg_fastball_velocity' in opposing_pitcher.columns:
            batter_df['velocity_matchup'] = batter_df['avg_exit_velocity'] - opposing_pitcher['avg_fastball_velocity'].iloc[0]
        
        return batter_df
    
    def _add_batter_lineup_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lineup context features."""
        # Batting order impact
        if 'batting_order' in df.columns:
            df['leadoff_hitter'] = (df['batting_order'] == 1).astype(int)
            df['cleanup_hitter'] = (df['batting_order'] == 4).astype(int)
            df['bottom_order'] = (df['batting_order'] >= 7).astype(int)
        
        # Protection in lineup (runners on base opportunities)
        if 'RBI_opportunities' in df.columns and 'RBI' in df.columns:
            df['rbi_efficiency'] = df['RBI'] / df['RBI_opportunities'].replace(0, np.nan)
        
        return df
    
    def build_team_features(self, team_data: pd.DataFrame, 
                          opponent_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Build comprehensive team-level features.
        
        Args:
            team_data: Team statistics and information
            opponent_data: Opposing team information
            
        Returns:
            DataFrame with advanced team features
        """
        features = team_data.copy()
        
        # Team performance metrics
        features = self._add_team_performance_features(features)
        
        # Momentum and trends
        features = self._add_team_momentum_features(features)
        
        # Situational team performance
        features = self._add_team_situational_features(features)
        
        # Bullpen analysis
        features = self._add_bullpen_features(features)
        
        # Matchup-specific features
        if opponent_data is not None:
            features = self._add_team_matchup_features(features, opponent_data)
        
        return features
    
    def _add_team_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team performance features."""
        # Pythagorean expectation
        if 'runs_scored' in df.columns and 'runs_allowed' in df.columns:
            df['pythagorean_wins'] = df['runs_scored']**2 / (df['runs_scored']**2 + df['runs_allowed']**2)
            df['pythagorean_diff'] = df['win_pct'] - df['pythagorean_wins']
        
        # Run differential per game
        if 'games_played' in df.columns:
            df['run_diff_per_game'] = (df['runs_scored'] - df['runs_allowed']) / df['games_played']
        
        # Offensive and defensive efficiency
        if 'team_OPS' in df.columns and 'opponent_OPS' in df.columns:
            df['ops_differential'] = df['team_OPS'] - df['opponent_OPS']
        
        return df
    
    def _add_team_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team momentum and trend features."""
        # Recent form (last 10 games)
        if 'last_10_record' in df.columns:
            df['recent_momentum'] = df['last_10_record'].apply(lambda x: int(x.split('-')[0]) / 10 if '-' in str(x) else 0.5)
        
        # Home/Road trends
        if 'home_record' in df.columns and 'road_record' in df.columns:
            home_wins = df['home_record'].apply(lambda x: int(x.split('-')[0]) if '-' in str(x) else 0)
            home_games = df['home_record'].apply(lambda x: sum(map(int, x.split('-'))) if '-' in str(x) else 1)
            road_wins = df['road_record'].apply(lambda x: int(x.split('-')[0]) if '-' in str(x) else 0)
            road_games = df['road_record'].apply(lambda x: sum(map(int, x.split('-'))) if '-' in str(x) else 1)
            
            df['home_win_pct'] = home_wins / home_games
            df['road_win_pct'] = road_wins / road_games
            df['home_road_diff'] = df['home_win_pct'] - df['road_win_pct']
        
        # Streak analysis
        if 'current_streak' in df.columns:
            df['streak_value'] = df['current_streak'].apply(
                lambda x: int(x[1:]) if str(x).startswith('W') else -int(x[1:]) if str(x).startswith('L') else 0
            )
        
        return df
    
    def _add_team_situational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add situational team features."""
        # One-run game performance
        if 'one_run_record' in df.columns:
            df['one_run_win_pct'] = df['one_run_record'].apply(
                lambda x: int(x.split('-')[0]) / sum(map(int, x.split('-'))) if '-' in str(x) else 0.5
            )
        
        # Extra innings performance
        if 'extra_inning_record' in df.columns:
            df['extra_inning_win_pct'] = df['extra_inning_record'].apply(
                lambda x: int(x.split('-')[0]) / sum(map(int, x.split('-'))) if '-' in str(x) else 0.5
            )
        
        # Day/Night performance
        if 'day_record' in df.columns and 'night_record' in df.columns:
            day_wins = df['day_record'].apply(lambda x: int(x.split('-')[0]) if '-' in str(x) else 0)
            day_games = df['day_record'].apply(lambda x: sum(map(int, x.split('-'))) if '-' in str(x) else 1)
            night_wins = df['night_record'].apply(lambda x: int(x.split('-')[0]) if '-' in str(x) else 0)
            night_games = df['night_record'].apply(lambda x: sum(map(int, x.split('-'))) if '-' in str(x) else 1)
            
            df['day_win_pct'] = day_wins / day_games
            df['night_win_pct'] = night_wins / night_games
            df['day_night_diff'] = df['day_win_pct'] - df['night_win_pct']
        
        return df
    
    def _add_bullpen_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add bullpen-specific features."""
        # Bullpen performance metrics
        if 'bullpen_ERA' in df.columns:
            df['bullpen_quality'] = 1 / (1 + df['bullpen_ERA'])  # Inverse relationship
        
        # Save situation performance
        if 'save_pct' in df.columns:
            df['closer_reliability'] = df['save_pct']
        
        # Bullpen usage and fatigue
        if 'bullpen_innings' in df.columns and 'games_played' in df.columns:
            df['bullpen_usage_rate'] = df['bullpen_innings'] / df['games_played']
        
        return df
    
    def _add_team_matchup_features(self, team_df: pd.DataFrame, 
                                 opponent_df: pd.DataFrame) -> pd.DataFrame:
        """Add team vs team matchup features."""
        # Head-to-head record
        if 'h2h_record' in team_df.columns:
            team_df['h2h_advantage'] = team_df['h2h_record'].apply(
                lambda x: (int(x.split('-')[0]) - int(x.split('-')[1])) / sum(map(int, x.split('-'))) 
                if '-' in str(x) else 0
            )
        
        # Style matchups
        if 'team_speed' in team_df.columns and 'opponent_stolen_base_defense' in opponent_df.columns:
            team_df['speed_vs_defense'] = team_df['team_speed'] - opponent_df['opponent_stolen_base_defense'].iloc[0]
        
        # Power vs pitching matchup
        if 'team_power' in team_df.columns and 'opponent_hr_allowed_rate' in opponent_df.columns:
            team_df['power_matchup'] = team_df['team_power'] * opponent_df['opponent_hr_allowed_rate'].iloc[0]
        
        return team_df
    
    def build_weather_features(self, weather_data: pd.DataFrame, 
                             venue: str) -> pd.DataFrame:
        """
        Build weather impact features.
        
        Args:
            weather_data: Weather conditions data
            venue: Stadium name
            
        Returns:
            DataFrame with weather impact features
        """
        features = weather_data.copy()
        
        # Temperature impact on offense
        if 'temperature' in features.columns:
            features['temp_offense_boost'] = np.where(features['temperature'] > 75, 
                                                    (features['temperature'] - 75) * 0.002, 0)
        
        # Wind impact
        if 'wind_speed' in features.columns and 'wind_direction' in features.columns:
            # Simplified wind impact (would need stadium-specific modeling)
            features['wind_factor'] = np.where(
                features['wind_direction'].isin(['out_to_rf', 'out_to_lf', 'out_to_cf']),
                features['wind_speed'] * 0.01,
                -features['wind_speed'] * 0.005
            )
        
        # Humidity impact on ball flight
        if 'humidity' in features.columns:
            features['humidity_factor'] = (100 - features['humidity']) * 0.001
        
        # Venue-specific weather adjustments
        if venue in self.park_factors:
            park_factor = self.park_factors[venue]
            if 'wind_factor' in features.columns:
                features['venue_wind_factor'] = features['wind_factor'] * park_factor.get('runs', 1.0)
        
        return features
    
    def build_market_intelligence_features(self, odds_data: pd.DataFrame, 
                                         betting_volume: pd.DataFrame = None) -> pd.DataFrame:
        """
        Build market intelligence features from betting data.
        
        Args:
            odds_data: Betting odds data from multiple sportsbooks
            betting_volume: Betting volume data (if available)
            
        Returns:
            DataFrame with market intelligence features
        """
        features = odds_data.copy()
        
        # Market consensus
        if 'home_odds' in features.columns:
            features['market_consensus'] = 1 / (1 + np.abs(features['home_odds']) / 100)
        
        # Line movement
        if 'opening_odds' in features.columns and 'current_odds' in features.columns:
            features['line_movement'] = features['current_odds'] - features['opening_odds']
            features['line_movement_pct'] = features['line_movement'] / np.abs(features['opening_odds'])
        
        # Vig analysis
        if 'home_implied_prob' in features.columns and 'away_implied_prob' in features.columns:
            features['total_implied_prob'] = features['home_implied_prob'] + features['away_implied_prob']
            features['vig'] = features['total_implied_prob'] - 1.0
            features['market_efficiency'] = 1 / features['total_implied_prob']
        
        # Betting volume impact (if available)
        if betting_volume is not None and 'volume' in betting_volume.columns:
            features = features.merge(betting_volume, on=['game_id'], how='left')
            features['volume_weighted_odds'] = features['current_odds'] * features['volume']
        
        # Sharp vs public money indicators
        if 'sharp_money_pct' in features.columns:
            features['sharp_public_divergence'] = features['sharp_money_pct'] - 50
        
        return features


def main():
    """Example usage of advanced features."""
    logger.info("Advanced MLB Features - Example Usage")
    
    # Initialize feature builder
    advanced_features = AdvancedMLBFeatures()
    
    # Example pitcher data
    pitcher_data = pd.DataFrame({
        'pitcher_name': ['Gerrit Cole', 'Jacob deGrom'],
        'IP': [200, 180],
        'SO': [250, 220],
        'BB': [50, 40],
        'HR': [25, 20],
        'H': [180, 150],
        'ERA': [3.20, 2.85],
        'last_start_date': ['2024-01-10', '2024-01-12']
    })
    
    # Build pitcher features
    pitcher_features = advanced_features.build_pitcher_features(pitcher_data)
    logger.info(f"Built {len(pitcher_features.columns)} pitcher features")
    
    # Example batter data
    batter_data = pd.DataFrame({
        'batter_name': ['Aaron Judge', 'Mookie Betts'],
        'PA': [600, 650],
        'BB': [80, 70],
        'HBP': [5, 8],
        '1B': [100, 120],
        '2B': [30, 35],
        '3B': [2, 5],
        'HR': [40, 25],
        'AVG': [.280, .295],
        'handedness': ['R', 'R']
    })
    
    # Build batter features
    batter_features = advanced_features.build_batter_features(batter_data)
    logger.info(f"Built {len(batter_features.columns)} batter features")
    
    # Example weather data
    weather_data = pd.DataFrame({
        'temperature': [78, 82],
        'wind_speed': [8, 12],
        'wind_direction': ['out_to_rf', 'in_from_lf'],
        'humidity': [65, 70]
    })
    
    # Build weather features
    weather_features = advanced_features.build_weather_features(weather_data, 'Yankee Stadium')
    logger.info(f"Built {len(weather_features.columns)} weather features")
    
    logger.info("Advanced feature engineering completed!")


if __name__ == "__main__":
    main() 