import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
import duckdb

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dashboard-specific models
class DashboardMetrics(BaseModel):
    total_recommendations: int
    high_confidence_bets: int
    average_confidence: float
    expected_roi: float
    active_games: int
    system_uptime: int

class SystemStatus(BaseModel):
    status: str
    uptime: int

class ApiUsageStats(BaseModel):
    calls_today: int
    daily_limit: int
    calls_remaining: int
    monthly_estimate: int
    cache_files: int

# Pydantic models for API responses
class BacktestMetrics(BaseModel):
    strategy: str
    model: str
    roi: float
    win_rate: float
    total_bets: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    model_auc: Optional[float] = None

class OddsComparison(BaseModel):
    date: str
    home_team: str
    away_team: str
    bookmaker: str
    home_odds: float
    away_odds: float
    home_implied_prob: float
    away_implied_prob: float
    vig: float

class BettingRecommendation(BaseModel):
    date: str
    game_id: str
    bet_type: str
    recommendation: str  # "home", "away", "over", "under", "pass"
    confidence: float
    expected_value: float
    recommended_stake: float
    reasoning: str

class TeamPerformance(BaseModel):
    team: str
    season: int
    wins: int
    losses: int
    win_pct: float
    avg_runs_scored: float
    avg_runs_allowed: float
    recent_form: Optional[float] = None

class PropBet(BaseModel):
    date: str
    player_name: str
    prop_type: str  # "strikeouts", "hits", "total_bases"
    line: float
    over_odds: float
    under_odds: float
    bookmaker: str
    recommendation: Optional[str] = None
    edge: Optional[float] = None

# FastAPI app
app = FastAPI(
    title="MLB Betting Analytics API",
    description="Comprehensive API for MLB betting analysis and recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MLBAnalyticsAPI:
    """
    Core analytics API class that handles data access and computations.
    """
    
    def __init__(self):
        self.db_path = "data/warehouse.duckdb"
        self.results_dir = Path("data/results")
        self.features_dir = Path("data/features")
        
        # Cache for frequently accessed data
        self._cache = {}
        self._cache_expiry = {}
        self.cache_ttl = 300  # 5 minutes
    
    def _get_db_connection(self):
        """Get DuckDB connection."""
        return duckdb.connect(self.db_path)
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid."""
        if key not in self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry[key]
    
    def _set_cache(self, key: str, value: Any):
        """Set cache entry with TTL."""
        self._cache[key] = value
        self._cache_expiry[key] = datetime.now() + timedelta(seconds=self.cache_ttl)
    
    def get_dashboard_metrics(self) -> DashboardMetrics:
        """Get dashboard metrics."""
        cache_key = "dashboard_metrics"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            # Get recommendations
            recommendations = self.get_betting_recommendations()
            high_confidence = [r for r in recommendations if r.confidence > 0.7]
            avg_confidence = np.mean([r.confidence for r in recommendations]) if recommendations else 0.0
            expected_roi = np.mean([r.expected_value for r in recommendations]) if recommendations else 0.0
            
            # Get active games count
            odds = self.get_current_odds(days_ahead=1)
            active_games = len(set(f"{o.home_team}_vs_{o.away_team}" for o in odds))
            
            metrics = DashboardMetrics(
                total_recommendations=len(recommendations),
                high_confidence_bets=len(high_confidence),
                average_confidence=float(avg_confidence),
                expected_roi=float(expected_roi),
                active_games=active_games,
                system_uptime=3600  # Placeholder: 1 hour
            )
            
            self._set_cache(cache_key, metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting dashboard metrics: {e}")
            return DashboardMetrics(
                total_recommendations=0,
                high_confidence_bets=0,
                average_confidence=0.0,
                expected_roi=0.0,
                active_games=0,
                system_uptime=0
            )
    
    def get_system_status(self) -> SystemStatus:
        """Get system status."""
        try:
            # Test database connection
            with self._get_db_connection() as con:
                con.execute("SELECT 1").fetchone()
            
            return SystemStatus(
                status="healthy",
                uptime=3600  # Placeholder: 1 hour
            )
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return SystemStatus(
                status="unhealthy",
                uptime=0
            )
    
    def get_api_usage_stats(self) -> ApiUsageStats:
        """Get API usage statistics."""
        # This would integrate with the backend's API usage tracking
        # For now, return placeholder data
        return ApiUsageStats(
            calls_today=5,
            daily_limit=16,
            calls_remaining=11,
            monthly_estimate=155,
            cache_files=12
        )
    
    def get_backtest_metrics(self, strategy: Optional[str] = None) -> List[BacktestMetrics]:
        """Get backtest performance metrics."""
        cache_key = f"backtest_metrics_{strategy}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            # Load backtest results
            summary_file = self.results_dir / "backtest_summary.parquet"
            
            if not summary_file.exists():
                logger.warning("No backtest summary file found")
                return []
            
            df = pd.read_parquet(summary_file)
            
            if strategy:
                df = df[df['bet_type'] == strategy]
            
            metrics = []
            for _, row in df.iterrows():
                metrics.append(BacktestMetrics(
                    strategy=row['bet_type'],
                    model=row['model'],
                    roi=float(row.get('roi', 0)),
                    win_rate=float(row.get('win_rate', 0)),
                    total_bets=int(row.get('total_bets', 0)),
                    total_return=float(row.get('total_return', 0)),
                    sharpe_ratio=float(row.get('sharpe_ratio', 0)),
                    max_drawdown=float(row.get('max_drawdown', 0)),
                    model_auc=float(row.get('model_auc', 0)) if 'model_auc' in row else None
                ))
            
            self._set_cache(cache_key, metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting backtest metrics: {e}")
            return []
    
    def get_current_odds(self, days_ahead: int = 7) -> List[OddsComparison]:
        """Get current odds for upcoming games."""
        cache_key = f"current_odds_{days_ahead}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            with self._get_db_connection() as con:
                query = """
                SELECT 
                    date,
                    home_team,
                    away_team,
                    bookmaker,
                    home_odds,
                    away_odds,
                    home_implied_prob,
                    away_implied_prob,
                    vig
                FROM raw.ml_odds
                WHERE date >= CURRENT_DATE
                    AND date <= CURRENT_DATE + INTERVAL '{} days'
                ORDER BY date, home_team, bookmaker
                """.format(days_ahead)
                
                df = con.execute(query).df()
                
                odds = []
                for _, row in df.iterrows():
                    odds.append(OddsComparison(
                        date=row['date'].strftime('%Y-%m-%d'),
                        home_team=row['home_team'],
                        away_team=row['away_team'],
                        bookmaker=row['bookmaker'],
                        home_odds=float(row['home_odds']),
                        away_odds=float(row['away_odds']),
                        home_implied_prob=float(row['home_implied_prob']),
                        away_implied_prob=float(row['away_implied_prob']),
                        vig=float(row['vig'])
                    ))
                
                self._set_cache(cache_key, odds)
                return odds
                
        except Exception as e:
            logger.error(f"Error getting current odds: {e}")
            return []
    
    def get_team_performance(self, season: Optional[int] = None) -> List[TeamPerformance]:
        """Get team performance metrics."""
        if season is None:
            season = datetime.now().year
        
        cache_key = f"team_performance_{season}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            with self._get_db_connection() as con:
                query = """
                SELECT *
                FROM analytics.team_performance
                WHERE season = {}
                ORDER BY win_pct DESC
                """.format(season)
                
                df = con.execute(query).df()
                
                teams = []
                for _, row in df.iterrows():
                    teams.append(TeamPerformance(
                        team=row['team'],
                        season=int(row['season']),
                        wins=int(row['wins']),
                        losses=int(row['losses']),
                        win_pct=float(row['win_pct']),
                        avg_runs_scored=float(row['avg_runs_scored']),
                        avg_runs_allowed=float(row['avg_runs_allowed']),
                        recent_form=None  # Would need to calculate separately
                    ))
                
                self._set_cache(cache_key, teams)
                return teams
                
        except Exception as e:
            logger.error(f"Error getting team performance: {e}")
            return []
    
    def get_comprehensive_props(self, prop_type: str = "all", days_ahead: int = 3) -> List[PropBet]:
        """Get comprehensive prop betting opportunities including all MLB markets."""
        cache_key = f"comprehensive_props_{prop_type}_{days_ahead}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            props = []
            
            # Player props
            if prop_type in ["all", "player_props"]:
                props.extend(self._get_comprehensive_player_props(days_ahead))
            
            # Game props (NRFI/YRFI, team totals, etc.)
            if prop_type in ["all", "game_props"]:
                props.extend(self._get_comprehensive_game_props(days_ahead))
            
            # Legacy support for existing prop types
            if prop_type in ["strikeouts"]:
                props.extend(self._get_strikeout_props(days_ahead))
            elif prop_type in ["hits", "total_bases"]:
                props.extend(self._get_hits_tb_props(days_ahead))
            
            self._set_cache(cache_key, props)
            return props
            
        except Exception as e:
            logger.error(f"Error getting comprehensive props: {e}")
            return []
    
    def _get_comprehensive_player_props(self, days_ahead: int) -> List[PropBet]:
        """Get all available player prop bets."""
        try:
            with self._get_db_connection() as con:
                query = """
                SELECT 
                    date,
                    player_name,
                    market_type,
                    line,
                    over_odds,
                    under_odds,
                    yes_odds,
                    no_odds,
                    bookmaker
                FROM raw.comprehensive_props
                WHERE date >= CURRENT_DATE
                    AND date <= CURRENT_DATE + INTERVAL '{} days'
                    AND player_name IS NOT NULL
                    AND player_name != ''
                ORDER BY date, player_name, market_type
                """.format(days_ahead)
                
                df = con.execute(query).df()
                
                props = []
                for _, row in df.iterrows():
                    # Map market types to user-friendly names
                    prop_type_mapping = {
                        'batter_home_runs': 'home_runs',
                        'batter_hits': 'hits',
                        'batter_total_bases': 'total_bases',
                        'batter_rbis': 'rbis',
                        'batter_runs_scored': 'runs_scored',
                        'batter_walks': 'walks',
                        'batter_stolen_bases': 'stolen_bases',
                        'pitcher_strikeouts': 'strikeouts',
                        'pitcher_hits_allowed': 'hits_allowed',
                        'pitcher_walks': 'walks_allowed',
                        'pitcher_earned_runs': 'earned_runs'
                    }
                    
                    prop_type = prop_type_mapping.get(row['market_type'], row['market_type'])
                    
                    props.append(PropBet(
                        date=row['date'].strftime('%Y-%m-%d'),
                        player_name=row['player_name'],
                        prop_type=prop_type,
                        line=float(row['line']) if row['line'] is not None else None,
                        over_odds=float(row['over_odds']) if row['over_odds'] is not None else None,
                        under_odds=float(row['under_odds']) if row['under_odds'] is not None else None,
                        bookmaker=row['bookmaker']
                    ))
                
                return props
                
        except Exception as e:
            logger.error(f"Error getting comprehensive player props: {e}")
            return []
    
    def _get_comprehensive_game_props(self, days_ahead: int) -> List[PropBet]:
        """Get game-level props like NRFI/YRFI, team totals, etc."""
        try:
            with self._get_db_connection() as con:
                query = """
                SELECT 
                    date,
                    home_team,
                    away_team,
                    market_type,
                    outcome_name,
                    line,
                    total,
                    spread,
                    odds,
                    bookmaker
                FROM raw.game_props
                WHERE date >= CURRENT_DATE
                    AND date <= CURRENT_DATE + INTERVAL '{} days'
                ORDER BY date, market_type
                """.format(days_ahead)
                
                df = con.execute(query).df()
                
                props = []
                for _, row in df.iterrows():
                    # Create a display name for game props
                    if 'totals_1st_1_innings' in row['market_type']:
                        prop_type = 'nrfi_yrfi'
                        player_name = f"{row['home_team']} vs {row['away_team']} - First Inning"
                    elif 'team_totals' in row['market_type']:
                        prop_type = 'team_totals'
                        player_name = f"{row['outcome_name']} Team Total"
                    elif 'h2h_1st' in row['market_type']:
                        prop_type = 'first_innings_ml'
                        player_name = f"{row['home_team']} vs {row['away_team']} - {row['market_type'].replace('h2h_1st_', '').replace('_innings', ' Innings')}"
                    else:
                        prop_type = row['market_type']
                        player_name = f"{row['home_team']} vs {row['away_team']}"
                    
                    props.append(PropBet(
                        date=row['date'].strftime('%Y-%m-%d'),
                        player_name=player_name,
                        prop_type=prop_type,
                        line=float(row['line']) if row['line'] is not None else float(row['total']) if row['total'] is not None else float(row['spread']) if row['spread'] is not None else None,
                        over_odds=float(row['odds']) if row['outcome_name'] in ['Over', 'Yes'] else None,
                        under_odds=float(row['odds']) if row['outcome_name'] in ['Under', 'No'] else None,
                        bookmaker=row['bookmaker']
                    ))
                
                return props
                
        except Exception as e:
            logger.error(f"Error getting comprehensive game props: {e}")
            return []
    
    def get_betting_recommendations(self, 
                                  bet_type: str = "all", 
                                  min_edge: float = 0.02,
                                  days_ahead: int = 3) -> List[BettingRecommendation]:
        """Get AI-powered betting recommendations."""
        cache_key = f"recommendations_{bet_type}_{min_edge}_{days_ahead}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        # This would integrate with trained models to generate actual recommendations
        # For now, return placeholder recommendations
        recommendations = []
        
        try:
            # Get today's games for moneyline recommendations
            if bet_type in ["all", "moneyline"]:
                odds = self.get_current_odds(days_ahead)
                
                for odd in odds[:5]:  # Limit to first 5 games
                    # Simple value betting logic (placeholder)
                    if odd.vig < 0.05:  # Low vig games
                        expected_value = np.random.uniform(-0.02, 0.05)  # Placeholder
                        
                        if expected_value > min_edge:
                            recommendations.append(BettingRecommendation(
                                date=odd.date,
                                game_id=f"{odd.home_team}_vs_{odd.away_team}",
                                bet_type="moneyline",
                                recommendation="home" if odd.home_implied_prob < 0.5 else "away",
                                confidence=min(expected_value * 10, 1.0),
                                expected_value=expected_value,
                                recommended_stake=min(expected_value * 2, 0.05),  # Max 5% of bankroll
                                reasoning=f"Model identifies value due to low vig ({odd.vig:.3f}) and favorable odds"
                            ))
            
            self._set_cache(cache_key, recommendations)
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting betting recommendations: {e}")
            return []

# Initialize API instance
analytics_api = MLBAnalyticsAPI()

# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MLB Betting Analytics API",
        "version": "1.0.0",
        "description": "Comprehensive API for MLB betting analysis",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        with analytics_api._get_db_connection() as con:
            con.execute("SELECT 1").fetchone()
        
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Dashboard Routes
@app.get("/dashboard/metrics", response_model=DashboardMetrics)
async def get_dashboard_metrics():
    """Get dashboard metrics."""
    try:
        metrics = analytics_api.get_dashboard_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error in dashboard metrics endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching dashboard metrics: {str(e)}")

@app.get("/dashboard/status", response_model=SystemStatus)
async def get_dashboard_status():
    """Get system status for dashboard."""
    try:
        status = analytics_api.get_system_status()
        return status
    except Exception as e:
        logger.error(f"Error in dashboard status endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching system status: {str(e)}")

@app.get("/dashboard/api-usage", response_model=ApiUsageStats)
async def get_api_usage():
    """Get API usage statistics."""
    try:
        usage = analytics_api.get_api_usage_stats()
        return usage
    except Exception as e:
        logger.error(f"Error in API usage endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching API usage: {str(e)}")

@app.get("/metrics/{strategy}", response_model=List[BacktestMetrics])
async def get_metrics(strategy: str):
    """Get backtest metrics for a specific strategy."""
    if strategy not in ["moneyline", "strikeout_props", "hits_tb_props", "all"]:
        raise HTTPException(status_code=400, detail="Invalid strategy")
    
    metrics = analytics_api.get_backtest_metrics(strategy if strategy != "all" else None)
    return metrics

# Original odds endpoint
@app.get("/odds", response_model=List[OddsComparison])
async def get_odds(days_ahead: int = Query(7, ge=1, le=14)):
    """Get current odds for upcoming games."""
    odds = analytics_api.get_current_odds(days_ahead)
    return odds

@app.get("/teams", response_model=List[TeamPerformance])
async def get_teams(season: Optional[int] = Query(None)):
    """Get team performance metrics."""
    teams = analytics_api.get_team_performance(season)
    return teams

@app.get("/props", response_model=List[PropBet])
async def get_props(
    prop_type: str = Query("all", regex="^(all|strikeouts|hits|total_bases)$"),
    days_ahead: int = Query(3, ge=1, le=7)
):
    """Get prop betting opportunities."""
    props = analytics_api.get_comprehensive_props(prop_type, days_ahead)
    return props

@app.get("/recommendations", response_model=List[BettingRecommendation])
async def get_recommendations(
    bet_type: str = Query("all", regex="^(all|moneyline|props)$"),
    min_edge: float = Query(0.02, ge=0.01, le=0.1),
    days_ahead: int = Query(3, ge=1, le=7)
):
    """Get AI-powered betting recommendations."""
    recommendations = analytics_api.get_betting_recommendations(bet_type, min_edge, days_ahead)
    return recommendations

@app.get("/analytics/best-odds")
async def get_best_odds():
    """Get best available odds across all bookmakers."""
    try:
        with analytics_api._get_db_connection() as con:
            query = """
            SELECT *
            FROM analytics.best_moneyline_odds
            ORDER BY date, home_team
            """
            df = con.execute(query).df()
            return df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching best odds: {str(e)}")

@app.get("/analytics/team-trends/{team}")
async def get_team_trends(team: str, games: int = Query(20, ge=5, le=50)):
    """Get recent performance trends for a specific team."""
    try:
        with analytics_api._get_db_connection() as con:
            query = """
            SELECT 
                date,
                opponent,
                result,
                runs_scored,
                runs_allowed,
                run_differential
            FROM raw.games
            WHERE team = '{}'
            ORDER BY date DESC
            LIMIT {}
            """.format(team, games)
            
            df = con.execute(query).df()
            return df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching team trends: {str(e)}")

@app.post("/analytics/refresh-cache")
async def refresh_cache(background_tasks: BackgroundTasks):
    """Refresh the API cache."""
    def clear_cache():
        analytics_api._cache.clear()
        analytics_api._cache_expiry.clear()
        logger.info("API cache cleared")
    
    background_tasks.add_task(clear_cache)
    return {"message": "Cache refresh initiated"}

@app.get("/db-browser")
async def redirect_to_db_browser():
    """Redirect to database browser for development."""
    return {
        "message": "Database browser available",
        "url": "http://localhost:5001",
        "note": "Run 'python scripts/start_db_browser.py' to start the database browser"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 