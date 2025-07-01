// Team and Game Types
export interface Team {
  id: string;
  name: string;
  abbreviation: string;
  wins: number;
  losses: number;
  win_percentage: number;
  runs_scored: number;
  runs_allowed: number;
  run_differential: number;
}

export interface Game {
  id: string;
  date: string;
  home_team: string;
  away_team: string;
  home_score?: number;
  away_score?: number;
  status: 'scheduled' | 'live' | 'completed';
  inning?: number;
}

// Odds Types
export interface MoneylineOdds {
  id: string;
  game_id: string;
  home_odds: number;
  away_odds: number;
  bookmaker: string;
  last_updated: string;
}

export interface PropBet {
  id: string;
  game_id: string;
  player_name: string;
  player_team: string;
  bet_type: 'strikeouts' | 'hits' | 'total_bases';
  line: number;
  over_odds: number;
  under_odds: number;
  bookmaker: string;
  last_updated: string;
}

// Backtest Types
export interface BacktestResult {
  strategy_name: string;
  total_bets: number;
  winning_bets: number;
  losing_bets: number;
  win_rate: number;
  total_return: number;
  roi: number;
  sharpe_ratio: number;
  max_drawdown: number;
  profit_factor: number;
  avg_odds: number;
}

export interface BacktestMetrics {
  daily_returns: Array<{ date: string; return: number; cumulative: number }>;
  monthly_summary: Array<{ month: string; bets: number; roi: number }>;
  strategy_comparison: BacktestResult[];
}

// Recommendation Types
export interface BettingRecommendation {
  id: string;
  type: 'moneyline' | 'strikeout_prop' | 'hits_prop' | 'total_bases_prop';
  game_id: string;
  recommendation: string;
  confidence: number;
  expected_value: number;
  reasoning: string;
  best_odds: number;
  best_bookmaker: string;
  player_name?: string;
  line?: number;
  created_at: string;
}

// Dashboard Types
export interface DashboardMetrics {
  total_recommendations: number;
  high_confidence_bets: number;
  average_confidence: number;
  expected_roi: number;
  active_games: number;
  system_uptime: number;
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
  timestamp: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  limit: number;
  has_next: boolean;
}

// Filter Types
export interface DateRange {
  start_date: string;
  end_date: string;
}

export interface OddsFilter {
  bookmakers?: string[];
  min_odds?: number;
  max_odds?: number;
  date_range?: DateRange;
}

export interface PropFilter {
  bet_types?: string[];
  players?: string[];
  teams?: string[];
  min_line?: number;
  max_line?: number;
}

// Chart Data Types
export interface ChartDataPoint {
  x: string | number;
  y: number;
  label?: string;
}

export interface PerformanceChart {
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
    backgroundColor?: string;
    borderColor?: string;
    fill?: boolean;
  }>;
}

// Odds Comparison Type
export interface OddsComparison {
  date: string;
  home_team: string;
  away_team: string;
  bookmaker: string;
  home_odds: number;
  away_odds: number;
  home_implied_prob: number;
  away_implied_prob: number;
  vig: number;
}

// Moneyline Odds Type (for legacy usage)
export interface MoneylineOdds {
  id: string;
  game_id: string;
  home_odds: number;
  away_odds: number;
  bookmaker: string;
  last_updated: string;
} 