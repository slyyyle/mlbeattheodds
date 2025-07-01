import axios from 'axios';
import type { AxiosResponse } from 'axios';
import type {
  ApiResponse,
  Team,
  Game,
  MoneylineOdds,
  PropBet,
  BacktestMetrics,
  BacktestResult,
  BettingRecommendation,
  DashboardMetrics,
  OddsComparison,
  PropFilter,
  DateRange,
} from '../types';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: '/api',
  timeout: 30000, // Increased timeout for cached responses
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling and retry logic
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const { config, response } = error;
    
    // Log detailed error information
    console.error('API Error:', {
      url: config?.url,
      status: response?.status,
      message: response?.data?.message || error.message,
      timestamp: new Date().toISOString()
    });
    
    // Don't retry if it's a client error (4xx) or if we've already retried
    if (response?.status >= 400 && response?.status < 500) {
      return Promise.reject(error);
    }
    
    // For server errors or network issues, the backend will use cache
    // So we don't need aggressive retries on the frontend
    return Promise.reject(error);
  }
);

// Helper function to handle API responses
const handleResponse = <T>(response: AxiosResponse<ApiResponse<T>>): T => {
  if (response.data.success) {
    return response.data.data;
  }
  throw new Error(response.data.message || 'API request failed');
};

// Helper function for direct responses (no wrapper)
const handleDirectResponse = <T>(response: AxiosResponse<T>): T => {
  return response.data;
};

// Dashboard API
export const dashboardApi = {
  getMetrics: async (): Promise<DashboardMetrics> => {
    const response = await api.get<DashboardMetrics>('/dashboard/metrics');
    return handleDirectResponse(response);
  },

  getSystemStatus: async (): Promise<{ status: string; uptime: number }> => {
    const response = await api.get<{ status: string; uptime: number }>('/dashboard/status');
    return handleDirectResponse(response);
  },

  getApiUsage: async (): Promise<{
    calls_today: number;
    daily_limit: number;
    calls_remaining: number;
    monthly_estimate: number;
    cache_files: number;
  }> => {
    const response = await api.get<{
      calls_today: number;
      daily_limit: number;
      calls_remaining: number;
      monthly_estimate: number;
      cache_files: number;
    }>('/dashboard/api-usage');
    return handleDirectResponse(response);
  },
};

// Teams API
export const teamsApi = {
  getAll: async (): Promise<Team[]> => {
    const response = await api.get<ApiResponse<Team[]>>('/teams');
    return handleResponse(response);
  },

  getById: async (id: string): Promise<Team> => {
    const response = await api.get<ApiResponse<Team>>(`/teams/${id}`);
    return handleResponse(response);
  },

  getStandings: async (): Promise<Team[]> => {
    const response = await api.get<ApiResponse<Team[]>>('/teams/standings');
    return handleResponse(response);
  },
};

// Games API
export const gamesApi = {
  getToday: async (): Promise<Game[]> => {
    const response = await api.get<ApiResponse<Game[]>>('/games/today');
    return handleResponse(response);
  },

  getByDate: async (date: string): Promise<Game[]> => {
    const response = await api.get<ApiResponse<Game[]>>(`/games/date/${date}`);
    return handleResponse(response);
  },

  getById: async (id: string): Promise<Game> => {
    const response = await api.get<ApiResponse<Game>>(`/games/${id}`);
    return handleResponse(response);
  },
};

// Odds API (Conservative - these use external API calls)
export const oddsApi = {
  getMoneyline: async (daysAhead = 7): Promise<OddsComparison[]> => {
    const response = await api.get<OddsComparison[]>('/odds', {
      params: { days_ahead: daysAhead },
    });
    return handleDirectResponse(response);
  },

  getMoneylineByGame: async (gameId: string): Promise<MoneylineOdds[]> => {
    const response = await api.get<ApiResponse<MoneylineOdds[]>>(`/odds/moneyline/game/${gameId}`);
    return handleResponse(response);
  },

  getBestOdds: async (gameId: string): Promise<{ home: MoneylineOdds; away: MoneylineOdds }> => {
    const response = await api.get<ApiResponse<{ home: MoneylineOdds; away: MoneylineOdds }>>(`/odds/best/${gameId}`);
    return handleResponse(response);
  },
};

// Props API (Conservative - these use external API calls)
export const propsApi = {
  getAll: async (filters?: PropFilter): Promise<PropBet[]> => {
    const response = await api.get<ApiResponse<PropBet[]>>('/props', {
      params: filters,
    });
    return handleResponse(response);
  },

  getByGame: async (gameId: string): Promise<PropBet[]> => {
    const response = await api.get<ApiResponse<PropBet[]>>(`/props/game/${gameId}`);
    return handleResponse(response);
  },

  getByPlayer: async (playerName: string): Promise<PropBet[]> => {
    const response = await api.get<ApiResponse<PropBet[]>>(`/props/player/${encodeURIComponent(playerName)}`);
    return handleResponse(response);
  },

  getStrikeouts: async (): Promise<PropBet[]> => {
    const response = await api.get<ApiResponse<PropBet[]>>('/props/strikeouts');
    return handleResponse(response);
  },

  getHits: async (): Promise<PropBet[]> => {
    const response = await api.get<ApiResponse<PropBet[]>>('/props/hits');
    return handleResponse(response);
  },

  getTotalBases: async (): Promise<PropBet[]> => {
    const response = await api.get<ApiResponse<PropBet[]>>('/props/total-bases');
    return handleResponse(response);
  },
};

// Backtests API (Uses historical data - no external API calls)
export const backtestsApi = {
  getMetrics: async (dateRange?: DateRange): Promise<BacktestMetrics> => {
    const response = await api.get<ApiResponse<BacktestMetrics>>('/backtests/metrics', {
      params: dateRange,
    });
    return handleResponse(response);
  },

  getResults: async (strategy?: string): Promise<BacktestResult[]> => {
    const response = await api.get<ApiResponse<BacktestResult[]>>('/backtests/results', {
      params: strategy ? { strategy } : {},
    });
    return handleResponse(response);
  },

  runBacktest: async (strategy: string, params: Record<string, unknown>): Promise<BacktestResult> => {
    const response = await api.post<ApiResponse<BacktestResult>>('/backtests/run', {
      strategy,
      params,
    });
    return handleResponse(response);
  },
};

// Recommendations API (Uses cached data and models - minimal external API calls)
export const recommendationsApi = {
  getAll: async (limit = 50): Promise<BettingRecommendation[]> => {
    const response = await api.get<ApiResponse<BettingRecommendation[]>>('/recommendations', {
      params: { limit },
    });
    return handleResponse(response);
  },

  getHighConfidence: async (minConfidence = 0.7): Promise<BettingRecommendation[]> => {
    const response = await api.get<ApiResponse<BettingRecommendation[]>>('/recommendations/high-confidence', {
      params: { min_confidence: minConfidence },
    });
    return handleResponse(response);
  },

  getByType: async (type: string): Promise<BettingRecommendation[]> => {
    const response = await api.get<ApiResponse<BettingRecommendation[]>>(`/recommendations/type/${type}`);
    return handleResponse(response);
  },

  getByGame: async (gameId: string): Promise<BettingRecommendation[]> => {
    const response = await api.get<ApiResponse<BettingRecommendation[]>>(`/recommendations/game/${gameId}`);
    return handleResponse(response);
  },
};

// Export the configured axios instance for custom requests
export { api as default }; 