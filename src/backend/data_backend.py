import os
import time
import pandas as pd
import numpy as np
import requests
import json
import pickle
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
import pybaseball as pb
from tqdm import tqdm
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLBDataBackendV3:
    """
    Comprehensive MLB data backend for fetching games, standings, and odds data.
    Integrates pybaseball for historical data and The Odds API for live betting markets.
    Features aggressive daily caching to minimize API usage (500 calls/month limit).
    """
    
    def __init__(self, api_key: str, seasons: List[int], base_url: str = "https://api.the-odds-api.com/v4"):
        self.api_key = api_key
        self.seasons = seasons
        self.base_url = base_url
        self.sport_key = "baseball_mlb"
        
        # Rate limiting for API calls (conservative for 500/month limit)
        self.api_call_delay = 2.0  # Increased delay between calls
        self.last_api_call = 0
        
        # Daily cache system
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache = {}
        
        # API usage tracking - Weekly based system
        self.api_calls_this_week = 0
        self.weekly_call_limit = 120  # Conservative weekly limit (500/month = ~120/week)
        self.monthly_call_limit = 500
        self.usage_file = self.cache_dir / "api_usage.json"
        
        # Load API usage tracking
        self._load_api_usage()
        
        logger.info(f"Initialized MLBDataBackendV3 for seasons: {seasons}")
        logger.info(f"Weekly API calls used: {self.api_calls_this_week}/{self.weekly_call_limit}")
    
    def _load_api_usage(self):
        """Load API usage tracking from file."""
        if self.usage_file.exists():
            try:
                with open(self.usage_file, 'r') as f:
                    usage_data = json.load(f)
                
                # Get current week (Monday start)
                current_week = datetime.now().strftime('%Y-W%U')
                stored_week = usage_data.get('week', '')
                
                if stored_week == current_week:
                    self.api_calls_this_week = usage_data.get('calls_this_week', 0)
                else:
                    # New week, reset counter but preserve monthly stats
                    self.api_calls_this_week = 0
                    self._save_api_usage()
            except Exception as e:
                logger.warning(f"Could not load API usage: {e}")
                self.api_calls_this_week = 0
    
    def _save_api_usage(self):
        """Save API usage tracking to file."""
        now = datetime.now()
        current_week = now.strftime('%Y-W%U')
        current_month = now.strftime('%Y-%m')
        
        # Load existing data to preserve monthly history
        existing_data = {}
        if self.usage_file.exists():
            try:
                with open(self.usage_file, 'r') as f:
                    existing_data = json.load(f)
            except:
                pass
        
        # Update monthly calls count
        monthly_calls = existing_data.get('monthly_history', {}).get(current_month, 0)
        if existing_data.get('week', '') != current_week:
            # New week, add previous week's calls to monthly total
            monthly_calls += existing_data.get('calls_this_week', 0)
        
        usage_data = {
            'week': current_week,
            'month': current_month,
            'calls_this_week': self.api_calls_this_week,
            'last_updated': now.isoformat(),
            'monthly_history': {
                **existing_data.get('monthly_history', {}),
                current_month: monthly_calls
            },
            'ingestion_history': existing_data.get('ingestion_history', [])
        }
        
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(usage_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save API usage: {e}")
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key for API request."""
        # Create deterministic cache key
        cache_params = {k: v for k, v in params.items() if k != 'apiKey'}
        key_string = f"{endpoint}_{json.dumps(cache_params, sort_keys=True)}"
        return key_string.replace('/', '_').replace(' ', '_')
    
    def _get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path for given key."""
        today = datetime.now().strftime('%Y-%m-%d')
        return self.cache_dir / f"{cache_key}_{today}.pkl"
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file is valid (from today)."""
        if not cache_file.exists():
            return False
        
        # Check if file is from today
        today = datetime.now().strftime('%Y-%m-%d')
        return today in cache_file.name
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load data from cache if valid."""
        # Check memory cache first
        if cache_key in self._memory_cache:
            cached_data, timestamp = self._memory_cache[cache_key]
            if (time.time() - timestamp) < 3600:  # 1 hour memory cache
                logger.info(f"Using memory cache for {cache_key}")
                return cached_data
        
        # Check disk cache
        cache_file = self._get_cache_file(cache_key)
        if self._is_cache_valid(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Store in memory cache
                self._memory_cache[cache_key] = (cached_data, time.time())
                logger.info(f"Using daily cache for {cache_key}")
                return cached_data
            except Exception as e:
                logger.warning(f"Could not load cache {cache_key}: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save data to cache."""
        # Save to memory cache
        self._memory_cache[cache_key] = (data, time.time())
        
        # Save to disk cache
        cache_file = self._get_cache_file(cache_key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Cached data for {cache_key}")
        except Exception as e:
            logger.warning(f"Could not save cache {cache_key}: {e}")
    
    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self.last_api_call
        if elapsed < self.api_call_delay:
            time.sleep(self.api_call_delay - elapsed)
        self.last_api_call = time.time()
    
    def _make_api_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make rate-limited API request with intelligent weekly-based caching."""
        # Check cache first (aggressive caching strategy)
        cache_key = self._get_cache_key(endpoint, params)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Check weekly API limit
        if self.api_calls_this_week >= self.weekly_call_limit:
            logger.warning(f"Weekly API limit reached ({self.weekly_call_limit}). Using stale cache if available.")
            stale_cache = self._try_stale_cache(cache_key)
            if stale_cache:
                logger.info(f"Using stale cache for {cache_key}")
                return stale_cache
            else:
                logger.error(f"No cache available for {cache_key} and weekly limit exceeded.")
                return None
        
        # Make API request
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/{endpoint}"
            params['apiKey'] = self.api_key
            
            logger.info(f"Making API request to {endpoint} (call {self.api_calls_this_week + 1}/{self.weekly_call_limit})")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Update API usage counter
            self.api_calls_this_week += 1
            self._save_api_usage()
            
            # Cache the response
            self._save_to_cache(cache_key, data)
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {endpoint}: {e}")
            # Try to return stale cache if available
            stale_cache = self._try_stale_cache(cache_key)
            if stale_cache:
                logger.info(f"Using stale cache for {cache_key}")
                return stale_cache
            return None
    
    def _try_stale_cache(self, cache_key: str) -> Optional[Dict]:
        """Try to load cache from previous days if current day fails."""
        for days_back in range(1, 8):  # Try up to 7 days back
            date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            stale_file = self.cache_dir / f"{cache_key}_{date}.pkl"
            if stale_file.exists():
                try:
                    with open(stale_file, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    continue
        return None
    
    def clear_cache(self, older_than_days: int = 7):
        """Clear cache files older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                file_date_str = cache_file.stem.split('_')[-1]
                file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
                if file_date < cutoff_date:
                    cache_file.unlink()
                    logger.info(f"Deleted old cache file: {cache_file}")
            except Exception as e:
                logger.warning(f"Could not process cache file {cache_file}: {e}")
    
    def get_api_usage_stats(self) -> Dict:
        """Get comprehensive API usage statistics."""
        # Load full usage data
        usage_data = {}
        if self.usage_file.exists():
            try:
                with open(self.usage_file, 'r') as f:
                    usage_data = json.load(f)
            except:
                pass
        
        current_month = datetime.now().strftime('%Y-%m')
        monthly_calls = usage_data.get('monthly_history', {}).get(current_month, 0)
        monthly_calls += self.api_calls_this_week  # Add current week's calls
        
        return {
            'calls_this_week': self.api_calls_this_week,
            'weekly_limit': self.weekly_call_limit,
            'weekly_remaining': max(0, self.weekly_call_limit - self.api_calls_this_week),
            'monthly_calls_estimated': monthly_calls,
            'monthly_limit': self.monthly_call_limit,
            'monthly_remaining_estimated': max(0, self.monthly_call_limit - monthly_calls),
            'cache_files': len(list(self.cache_dir.glob("*.pkl"))),
            'current_week': datetime.now().strftime('%Y-W%U'),
            'cache_efficiency': self._calculate_cache_efficiency()
        }
    
    def _calculate_cache_efficiency(self) -> float:
        """Calculate cache hit rate based on memory cache usage."""
        if not hasattr(self, '_cache_hits'):
            self._cache_hits = 0
        if not hasattr(self, '_cache_attempts'):
            self._cache_attempts = 0
        
        return (self._cache_hits / max(1, self._cache_attempts)) * 100
    
    def fetch_game_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch comprehensive game results and standings for specified seasons.
        Returns games DataFrame and standings DataFrame.
        """
        logger.info("Fetching game results and standings...")
        
        all_games = []
        all_standings = []
        
        for season in tqdm(self.seasons, desc="Processing seasons"):
            try:
                # Fetch games using pybaseball
                games = pb.schedule_and_record(season, team=None)
                games['season'] = season
                all_games.append(games)
                
                # Fetch standings
                standings = pb.standings(season)
                standings['season'] = season
                all_standings.append(standings)
                
                time.sleep(0.5)  # Be nice to pybaseball servers
                
            except Exception as e:
                logger.error(f"Error fetching data for season {season}: {e}")
                continue
        
        # Combine all seasons
        games_df = pd.concat(all_games, ignore_index=True) if all_games else pd.DataFrame()
        standings_df = pd.concat(all_standings, ignore_index=True) if all_standings else pd.DataFrame()
        
        # Clean and standardize game data
        if not games_df.empty:
            games_df = self._clean_games_data(games_df)
        
        # Clean and standardize standings data  
        if not standings_df.empty:
            standings_df = self._clean_standings_data(standings_df)
        
        logger.info(f"Fetched {len(games_df)} games and {len(standings_df)} standings records")
        return games_df, standings_df
    
    def _clean_games_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize games data."""
        # Standardize column names
        column_mapping = {
            'Date': 'date',
            'Tm': 'team',
            'Opp': 'opponent', 
            'W/L': 'result',
            'R': 'runs_scored',
            'RA': 'runs_allowed',
            'Inn': 'innings'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Add derived features
        if 'runs_scored' in df.columns and 'runs_allowed' in df.columns:
            df['run_differential'] = df['runs_scored'] - df['runs_allowed']
            df['win'] = (df['result'] == 'W').astype(int)
        
        # Add home/away designation
        if '@' in df.columns:
            df['is_home'] = df['@'].isna()
            df['home_team'] = np.where(df['is_home'], df['team'], df['opponent'])
            df['away_team'] = np.where(df['is_home'], df['opponent'], df['team'])
        
        return df
    
    def _clean_standings_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize standings data."""
        # Standardize column names
        column_mapping = {
            'Tm': 'team',
            'W': 'wins',
            'L': 'losses', 
            'W-L%': 'win_pct',
            'GB': 'games_back',
            'RS': 'runs_scored',
            'RA': 'runs_allowed'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Calculate additional metrics
        if 'wins' in df.columns and 'losses' in df.columns:
            df['games_played'] = df['wins'] + df['losses']
            df['win_pct_calc'] = df['wins'] / df['games_played']
        
        if 'runs_scored' in df.columns and 'runs_allowed' in df.columns:
            df['run_differential'] = df['runs_scored'] - df['runs_allowed']
        
        return df
    
    def fetch_moneyline_odds(self, days_ahead: int = 7) -> pd.DataFrame:
        """
        Fetch moneyline odds for upcoming MLB games.
        """
        logger.info(f"Fetching moneyline odds for next {days_ahead} days...")
        
        endpoint = f"sports/{self.sport_key}/odds"
        params = {
            'regions': 'us,us2',
            'markets': 'h2h',  # Head-to-head (moneyline)
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }
        
        data = self._make_api_request(endpoint, params)
        if not data:
            return pd.DataFrame()
        
        # Parse odds data
        odds_records = []
        for game in data:
            game_time = pd.to_datetime(game['commence_time'])
            home_team = game['home_team']
            away_team = game['away_team']
            
            for bookmaker in game.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    if market['key'] == 'h2h':
                        outcomes = {outcome['name']: outcome['price'] 
                                  for outcome in market['outcomes']}
                        
                        odds_records.append({
                            'date': game_time.date(),
                            'game_time': game_time,
                            'home_team': home_team,
                            'away_team': away_team,
                            'bookmaker': bookmaker['title'],
                            'home_odds': outcomes.get(home_team),
                            'away_odds': outcomes.get(away_team),
                            'last_update': pd.to_datetime(bookmaker['last_update'])
                        })
        
        df = pd.DataFrame(odds_records)
        
        if not df.empty:
            # Add implied probabilities
            df['home_implied_prob'] = self._american_to_probability(df['home_odds'])
            df['away_implied_prob'] = self._american_to_probability(df['away_odds'])
            df['market_total_prob'] = df['home_implied_prob'] + df['away_implied_prob']
            df['vig'] = df['market_total_prob'] - 1.0
        
        logger.info(f"Fetched {len(df)} moneyline odds records")
        return df
    
    def fetch_strikeout_props(self, days_ahead: int = 7) -> pd.DataFrame:
        """
        Fetch pitcher strikeout prop bets for upcoming games using Odds API events endpoint.
        Player props require event-specific calls, not the general odds endpoint.
        """
        logger.info(f"Fetching strikeout props for next {days_ahead} days...")
        
        # First get events to find event IDs
        events_endpoint = f"sports/{self.sport_key}/events"
        events_params = {
            'dateFormat': 'iso'
        }
        events_data = self._make_api_request(events_endpoint, events_params) or []
        
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days_ahead)
        props_records = []
        
        # Filter events within our time window
        relevant_events = []
        for event in events_data:
            game_time = pd.to_datetime(event.get('commence_time'))
            if now <= game_time <= cutoff:
                relevant_events.append(event)
        
        logger.info(f"Found {len(relevant_events)} games to check for strikeout props")
        
        # For each event, fetch player props
        for event in relevant_events[:5]:  # Limit to 5 games to conserve API calls
            if self.api_calls_this_week >= self.weekly_call_limit:
                logger.warning("Weekly API limit reached, stopping prop fetch")
                break
                
            event_id = event.get('id')
            if not event_id:
                continue
                
            # Fetch player props for this specific event
            props_endpoint = f"sports/{self.sport_key}/events/{event_id}/odds"
            props_params = {
                'regions': 'us,us2',
                'markets': 'pitcher_strikeouts',  # Correct market name for The Odds API
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            }
            
            event_props = self._make_api_request(props_endpoint, props_params)
            if not event_props:
                continue
            
            game_time = pd.to_datetime(event.get('commence_time'))
            home_team = event.get('home_team')
            away_team = event.get('away_team')
            
            # Process the event props response
            for bookmaker in event_props.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    if market.get('key') != 'pitcher_strikeouts':
                        continue
                    for outcome in market.get('outcomes', []):
                        props_records.append({
                            'date': game_time.date(),
                            'game_time': game_time,
                            'home_team': home_team,
                            'away_team': away_team,
                            'bookmaker': bookmaker.get('title'),
                            'player_name': outcome.get('description', ''),
                            'line': outcome.get('point'),
                            'over_odds': outcome['price'] if outcome['name'] == 'Over' else None,
                            'under_odds': outcome['price'] if outcome['name'] == 'Under' else None,
                            'last_update': pd.to_datetime(market.get('last_update'))
                        })
        
        df = pd.DataFrame(props_records)
        logger.info(f"Fetched {len(df)} strikeout prop records")
        return df
    
    def fetch_hit_tb_props(self, days_ahead: int = 7) -> pd.DataFrame:
        """
        Fetch batter hits and total bases prop bets using Odds API events endpoint.
        Player props require event-specific calls, not the general odds endpoint.
        """
        logger.info(f"Fetching hits/total bases props for next {days_ahead} days...")
        
        # First get events to find event IDs
        events_endpoint = f"sports/{self.sport_key}/events"
        events_params = {
            'dateFormat': 'iso'
        }
        events_data = self._make_api_request(events_endpoint, events_params) or []
        
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days_ahead)
        props_records = []
        
        # Filter events within our time window
        relevant_events = []
        for event in events_data:
            game_time = pd.to_datetime(event.get('commence_time'))
            if now <= game_time <= cutoff:
                relevant_events.append(event)
        
        logger.info(f"Found {len(relevant_events)} games to check for hits/total bases props")
        
        # For each event, fetch player props
        for event in relevant_events[:5]:  # Limit to 5 games to conserve API calls
            if self.api_calls_this_week >= self.weekly_call_limit:
                logger.warning("Weekly API limit reached, stopping prop fetch")
                break
                
            event_id = event.get('id')
            if not event_id:
                continue
                
            # Fetch player props for this specific event
            props_endpoint = f"sports/{self.sport_key}/events/{event_id}/odds"
            props_params = {
                'regions': 'us,us2',
                'markets': 'batter_hits,batter_total_bases',  # Correct market names for The Odds API
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            }
            
            event_props = self._make_api_request(props_endpoint, props_params)
            if not event_props:
                continue
            
            game_time = pd.to_datetime(event.get('commence_time'))
            home_team = event.get('home_team')
            away_team = event.get('away_team')
            
            # Process the event props response
            for bookmaker in event_props.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    if market.get('key') not in ('batter_hits', 'batter_total_bases'):
                        continue
                    market_type = market.get('key')
                    for outcome in market.get('outcomes', []):
                        props_records.append({
                            'date': game_time.date(),
                            'game_time': game_time,
                            'home_team': home_team,
                            'away_team': away_team,
                            'bookmaker': bookmaker.get('title'),
                            'market_type': market_type,
                            'player_name': outcome.get('description', ''),
                            'line': outcome.get('point'),
                            'over_odds': outcome['price'] if outcome['name'] == 'Over' else None,
                            'under_odds': outcome['price'] if outcome['name'] == 'Under' else None,
                            'last_update': pd.to_datetime(market.get('last_update'))
                        })
        
        df = pd.DataFrame(props_records)
        logger.info(f"Fetched {len(df)} hits/total bases prop records")
        return df
    
    def get_pitcher_stats_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch pitcher statistics for a date range using pybaseball.
        """
        logger.info(f"Fetching pitcher stats from {start_date} to {end_date}")
        
        try:
            # Fetch pitching stats
            pitching_stats = pb.pitching_stats(start_date, end_date)
            
            # Add advanced metrics
            if not pitching_stats.empty:
                pitching_stats['K_per_9'] = (pitching_stats['SO'] * 9) / pitching_stats['IP']
                pitching_stats['BB_per_9'] = (pitching_stats['BB'] * 9) / pitching_stats['IP']
                pitching_stats['K_BB_ratio'] = pitching_stats['SO'] / pitching_stats['BB'].replace(0, np.nan)
                pitching_stats['WHIP'] = (pitching_stats['BB'] + pitching_stats['H']) / pitching_stats['IP']
            
            return pitching_stats
            
        except Exception as e:
            logger.error(f"Error fetching pitcher stats: {e}")
            return pd.DataFrame()
    
    def get_batter_stats_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch batter statistics for a date range using pybaseball.
        """
        logger.info(f"Fetching batter stats from {start_date} to {end_date}")
        
        try:
            # Fetch batting stats
            batting_stats = pb.batting_stats(start_date, end_date)
            
            # Add advanced metrics
            if not batting_stats.empty:
                batting_stats['TB_per_PA'] = batting_stats['TB'] / batting_stats['PA']
                batting_stats['H_per_PA'] = batting_stats['H'] / batting_stats['PA']
                batting_stats['ISO'] = batting_stats['SLG'] - batting_stats['AVG']
                batting_stats['wOBA_est'] = (
                    0.690 * batting_stats['uBB'] + 
                    0.722 * batting_stats['HBP'] +
                    0.888 * (batting_stats['H'] - batting_stats['2B'] - batting_stats['3B'] - batting_stats['HR']) +
                    1.271 * batting_stats['2B'] +
                    1.616 * batting_stats['3B'] +
                    2.101 * batting_stats['HR']
                ) / batting_stats['PA']
            
            return batting_stats
            
        except Exception as e:
            logger.error(f"Error fetching batter stats: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def _american_to_probability(odds: pd.Series) -> pd.Series:
        """Convert American odds to implied probability."""
        def convert_odds(odd):
            if pd.isna(odd):
                return np.nan
            if odd > 0:
                return 100 / (odd + 100)
            else:
                return abs(odd) / (abs(odd) + 100)
        
        return odds.apply(convert_odds)
    
    def get_team_recent_performance(self, team: str, games: int = 10) -> Dict:
        """
        Get recent performance metrics for a team.
        """
        # This would typically query recent games from the database
        # For now, return placeholder structure
        return {
            'team': team,
            'last_games': games,
            'wins': 0,
            'losses': 0,
            'runs_per_game': 0.0,
            'runs_allowed_per_game': 0.0,
            'win_pct': 0.0
        }
    
    def check_api_usage(self) -> Dict:
        """
        Check remaining API calls for The Odds API.
        """
        endpoint = "sports"
        params = {}
        
        response = self._make_api_request(endpoint, params)
        
        # The Odds API returns usage info in headers
        # This is a simplified version
        return {
            'remaining_calls': 'Check headers in actual implementation',
            'reset_time': 'Check headers in actual implementation'
        }
    
    def fetch_comprehensive_mlb_props(self, days_ahead: int = 7) -> pd.DataFrame:
        """
        Fetch comprehensive MLB prop bets including all available markets.
        Note: Player props require event-specific API calls, not the general odds endpoint.
        """
        logger.info(f"Fetching comprehensive MLB props for next {days_ahead} days...")
        
        # First get events to find event IDs
        events_endpoint = f"sports/{self.sport_key}/events"
        events_params = {
            'dateFormat': 'iso'
        }
        events_data = self._make_api_request(events_endpoint, events_params) or []
        
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days_ahead)
        
        # Filter events within our time window
        relevant_events = []
        for event in events_data:
            game_time = pd.to_datetime(event.get('commence_time'))
            if now <= game_time <= cutoff:
                relevant_events.append(event)
        
        logger.info(f"Found {len(relevant_events)} games to check for player props")
        
        # Available MLB player prop markets from The Odds API
        player_prop_markets = [
            'batter_home_runs',
            'batter_hits', 
            'batter_total_bases',
            'batter_rbis',
            'batter_runs_scored',
            'batter_hits_runs_rbis',
            'batter_singles',
            'batter_doubles', 
            'batter_triples',
            'batter_walks',
            'batter_strikeouts',
            'batter_stolen_bases',
            'pitcher_strikeouts',
            'pitcher_hits_allowed',
            'pitcher_walks',
            'pitcher_earned_runs',
            'pitcher_outs',
            'pitcher_record_a_win'
        ]
        
        all_props = []
        
        # For each event, fetch player props (limit to conserve API calls)
        for event in relevant_events[:3]:  # Limit to 3 games to conserve API calls
            if self.api_calls_this_week >= self.weekly_call_limit:
                logger.warning("Weekly API limit reached, stopping comprehensive prop fetch")
                break
                
            event_id = event.get('id')
            if not event_id:
                continue
            
            game_time = pd.to_datetime(event.get('commence_time'))
            home_team = event.get('home_team')
            away_team = event.get('away_team')
            
            # Fetch all available player props for this event
            # Note: We'll try a few key markets to see what's available
            key_markets = ['pitcher_strikeouts', 'batter_hits', 'batter_total_bases', 'batter_home_runs']
            markets_str = ','.join(key_markets)
            
            props_endpoint = f"sports/{self.sport_key}/events/{event_id}/odds"
            props_params = {
                'regions': 'us,us2',
                'markets': markets_str,
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            }
            
            event_props = self._make_api_request(props_endpoint, props_params)
            if not event_props:
                continue
            
            # Process the event props response
            for bookmaker in event_props.get('bookmakers', []):
                for market_data in bookmaker.get('markets', []):
                    market_type = market_data.get('key')
                    if market_type not in key_markets:
                        continue
                        
                    for outcome in market_data.get('outcomes', []):
                        all_props.append({
                            'date': game_time.date(),
                            'game_time': game_time,
                            'home_team': home_team,
                            'away_team': away_team,
                            'bookmaker': bookmaker.get('title'),
                            'market_type': market_type,
                            'player_name': outcome.get('description', ''),
                            'line': outcome.get('point'),
                            'over_odds': outcome['price'] if outcome['name'] == 'Over' else None,
                            'under_odds': outcome['price'] if outcome['name'] == 'Under' else None,
                            'yes_odds': outcome['price'] if outcome['name'] == 'Yes' else None,
                            'no_odds': outcome['price'] if outcome['name'] == 'No' else None,
                            'last_update': pd.to_datetime(market_data.get('last_update'))
                        })
        
        df = pd.DataFrame(all_props)
        logger.info(f"Fetched {len(df)} comprehensive MLB prop records")
        return df

    def fetch_game_props(self, days_ahead: int = 7) -> pd.DataFrame:
        """
        Fetch game-level props like NRFI/YRFI, team totals, alternate totals, etc.
        """
        logger.info(f"Fetching game props for next {days_ahead} days...")
        
        game_markets = [
            'totals',  # Over/under total runs
            'team_totals',  # Team-specific run totals
            'alternate_totals',  # Alternate total lines
            'alternate_team_totals',  # Alternate team total lines
            'h2h_1st_1_innings',  # First inning moneyline
            'h2h_1st_3_innings',  # First 3 innings moneyline
            'h2h_1st_5_innings',  # First 5 innings moneyline
            'h2h_1st_7_innings',  # First 7 innings moneyline
            'totals_1st_1_innings',  # First inning totals (NRFI/YRFI)
            'totals_1st_3_innings',  # First 3 innings totals
            'totals_1st_5_innings',  # First 5 innings totals
            'totals_1st_7_innings',  # First 7 innings totals
            'spreads_1st_1_innings',  # First inning run line
            'spreads_1st_5_innings',  # First 5 innings run line
        ]
        
        all_game_props = []
        
        for market in game_markets:
            logger.info(f"Fetching {market} game props...")
            endpoint = f"sports/{self.sport_key}/odds"
            params = {
                'regions': 'us,us2', 
                'markets': market,
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            }
            
            data = self._make_api_request(endpoint, params) or []
            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(days=days_ahead)
            
            for game in data:
                game_time = pd.to_datetime(game.get('commence_time'))
                if game_time < now or game_time > cutoff:
                    continue
                    
                home_team = game.get('home_team')
                away_team = game.get('away_team')
                
                for bookmaker in game.get('bookmakers', []):
                    for market_data in bookmaker.get('markets', []):
                        if market_data.get('key') != market:
                            continue
                            
                        for outcome in market_data.get('outcomes', []):
                            all_game_props.append({
                                'date': game_time.date(),
                                'game_time': game_time,
                                'home_team': home_team,
                                'away_team': away_team,
                                'bookmaker': bookmaker.get('title'),
                                'market_type': market,
                                'outcome_name': outcome.get('name'),
                                'line': outcome.get('point'),
                                'spread': outcome.get('point') if 'spreads' in market else None,
                                'total': outcome.get('point') if 'totals' in market else None,
                                'odds': outcome.get('price'),
                                'last_update': pd.to_datetime(bookmaker.get('last_update'))
                            })
            
            time.sleep(1.5)  # Rate limiting
        
        df = pd.DataFrame(all_game_props)
        logger.info(f"Fetched {len(df)} game prop records")
        return df 