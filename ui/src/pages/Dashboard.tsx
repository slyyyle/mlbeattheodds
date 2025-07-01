import React from 'react';
import {
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Alert,
} from '@mui/material';
import {
  TrendingUp,
  Assessment,
  MonetizationOn,
  Sports,
  CheckCircle,
  Warning,
  Api,
  Storage,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';

// Import API functions
import { dashboardApi } from '../services/api';

const Dashboard: React.FC = () => {
  const { data: metrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['dashboard-metrics'],
    queryFn: dashboardApi.getMetrics,
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: 1,
  });

  const { data: systemStatus, isLoading: statusLoading } = useQuery({
    queryKey: ['system-status'],
    queryFn: dashboardApi.getSystemStatus,
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: 1,
  });

  const { data: apiUsage } = useQuery({
    queryKey: ['api-usage'],
    queryFn: dashboardApi.getApiUsage,
    staleTime: 2 * 60 * 1000, // 2 minutes
    retry: 1,
  });

  const MetricCard: React.FC<{
    title: string;
    value: string | number;
    icon: React.ReactNode;
    change?: string;
    color?: string;
  }> = ({ title, value, icon, change, color = 'primary' }) => (
    <Card sx={{ height: '100%', minWidth: 200 }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Box sx={{ color: `${color}.main`, mr: 1 }}>{icon}</Box>
          <Typography variant="h6" color="text.secondary">
            {title}
          </Typography>
        </Box>
        <Typography variant="h4" sx={{ mb: 1 }}>
          {value}
        </Typography>
        {change && (
          <Chip
            label={change}
            size="small"
            color={change.startsWith('+') ? 'success' : 'error'}
            variant="outlined"
          />
        )}
      </CardContent>
    </Card>
  );

  const getApiUsageColor = () => {
    if (!apiUsage) return 'info';
    const usage = apiUsage.calls_today / apiUsage.daily_limit;
    if (usage < 0.5) return 'success';
    if (usage < 0.8) return 'warning';
    return 'error';
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      {statusLoading ? (
        <LinearProgress sx={{ mb: 3 }} />
      ) : (
        <Alert
          severity={systemStatus?.status === 'healthy' ? 'success' : 'warning'}
          icon={systemStatus?.status === 'healthy' ? <CheckCircle /> : <Warning />}
          sx={{ mb: 3 }}
        >
          System Status: {systemStatus?.status || 'Unknown'} | 
          Uptime: {systemStatus?.uptime ? `${Math.round(systemStatus.uptime / 3600)}h` : 'N/A'}
        </Alert>
      )}

      {/* API Usage Alert */}
      {apiUsage && (
        <Alert
          severity={getApiUsageColor()}
          sx={{ mb: 3 }}
          icon={<Api />}
        >
          <Typography variant="subtitle2">
            Daily API Usage: {apiUsage.calls_today}/{apiUsage.daily_limit} calls
          </Typography>
          <Typography variant="body2">
            Remaining: {apiUsage.calls_remaining} | 
            Monthly Estimate: {apiUsage.monthly_estimate}/500 | 
            Cache Files: {apiUsage.cache_files}
          </Typography>
          <LinearProgress
            variant="determinate"
            value={(apiUsage.calls_today / apiUsage.daily_limit) * 100}
            sx={{ mt: 1 }}
            color={getApiUsageColor()}
          />
        </Alert>
      )}

      {/* Key Metrics Row */}
      <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap' }}>
        <MetricCard
          title="Total Recommendations"
          value={metrics?.total_recommendations || '-'}
          icon={<Assessment />}
          change={metricsLoading ? undefined : '+12%'}
        />
        <MetricCard
          title="High Confidence Bets"
          value={metrics?.high_confidence_bets || '-'}
          icon={<TrendingUp />}
          change={metricsLoading ? undefined : '+8%'}
          color="success"
        />
        <MetricCard
          title="Expected ROI"
          value={metrics?.expected_roi ? `${(metrics.expected_roi * 100).toFixed(1)}%` : '-'}
          icon={<MonetizationOn />}
          change={metricsLoading ? undefined : '+3.2%'}
          color="warning"
        />
        <MetricCard
          title="Active Games"
          value={metrics?.active_games || '-'}
          icon={<Sports />}
          color="info"
        />
      </Box>

      {/* API Usage & Cache Status Row */}
      <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap' }}>
        <MetricCard
          title="API Calls Today"
          value={apiUsage ? `${apiUsage.calls_today}/${apiUsage.daily_limit}` : '-'}
          icon={<Api />}
          color={getApiUsageColor()}
        />
        <MetricCard
          title="Cache Files"
          value={apiUsage?.cache_files || '-'}
          icon={<Storage />}
          color="info"
        />
      </Box>

      {/* Recent Activity */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Recent Activity
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Box>
            <Typography variant="subtitle2" color="primary">
              Cache Hit - No API Call
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Loaded odds data from daily cache
            </Typography>
            <Typography variant="caption" color="text.secondary">
              1 minute ago
            </Typography>
          </Box>
          <Box>
            <Typography variant="subtitle2" color="success.main">
              New Recommendation Generated
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Yankees ML vs Red Sox - 87% confidence
            </Typography>
            <Typography variant="caption" color="text.secondary">
              2 minutes ago
            </Typography>
          </Box>
          <Box>
            <Typography variant="subtitle2" color="warning.main">
              Odds Update (Cached)
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Using cached odds data - API limit preserved
            </Typography>
            <Typography variant="caption" color="text.secondary">
              5 minutes ago
            </Typography>
          </Box>
          <Box>
            <Typography variant="subtitle2" color="info.main">
              Cache Cleanup
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Removed 3 cache files older than 7 days
            </Typography>
            <Typography variant="caption" color="text.secondary">
              15 minutes ago
            </Typography>
          </Box>
        </Box>
      </Paper>
    </Box>
  );
};

export default Dashboard; 