import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for MLB betting analytics.
    
    Features:
    - Model performance tracking
    - Data quality monitoring
    - System health checks
    - Alert generation
    - Performance degradation detection
    - Real-time metrics dashboard
    """
    
    def __init__(self, monitoring_dir: str = "monitoring"):
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance history
        self.model_performance_history = []
        self.data_quality_history = []
        self.system_health_history = []
        self.alerts = []
        
        # Thresholds for alerting
        self.performance_thresholds = {
            'auc_min': 0.55,
            'accuracy_min': 0.52,
            'roi_min': -0.05,
            'sharpe_min': 0.1
        }
        
        self.data_quality_thresholds = {
            'missing_data_max': 0.1,
            'outlier_ratio_max': 0.05,
            'drift_score_max': 0.15
        }
    
    def track_model_performance(self, model_name: str, predictions: Dict[str, Any], 
                               actual_outcomes: pd.Series, 
                               betting_results: Optional[Dict] = None) -> Dict[str, float]:
        """Track and log model performance metrics."""
        
        # Calculate prediction metrics
        y_true = actual_outcomes
        y_pred_proba = predictions.get('probability', [])
        y_pred = predictions.get('prediction', [])
        
        if len(y_pred_proba) == 0 or len(y_pred) == 0:
            logger.warning(f"No predictions available for {model_name}")
            return {}
        
        # Core ML metrics
        from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, precision_score, recall_score
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'n_predictions': len(y_true),
            'auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5,
            'accuracy': accuracy_score(y_true, y_pred),
            'log_loss': log_loss(y_true, y_pred_proba),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0)
        }
        
        # Add betting-specific metrics if available
        if betting_results:
            metrics.update({
                'roi': betting_results.get('roi', 0),
                'win_rate': betting_results.get('win_rate', 0),
                'total_bets': betting_results.get('total_bets', 0),
                'profit_loss': betting_results.get('total_pnl', 0),
                'sharpe_ratio': betting_results.get('sharpe_ratio', 0),
                'max_drawdown': betting_results.get('max_drawdown', 0)
            })
        
        # Store in history
        self.model_performance_history.append(metrics)
        
        # Check for performance degradation
        self._check_performance_alerts(metrics)
        
        # Save to file
        self._save_performance_metrics(metrics)
        
        logger.info(f"Tracked performance for {model_name}: AUC={metrics['auc']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        
        return metrics
    
    def monitor_data_quality(self, data: pd.DataFrame, data_source: str) -> Dict[str, Any]:
        """Monitor data quality and detect anomalies."""
        
        quality_metrics = {
            'timestamp': datetime.now().isoformat(),
            'data_source': data_source,
            'n_records': len(data),
            'n_features': len(data.columns)
        }
        
        # Missing data analysis
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        quality_metrics['missing_data_ratio'] = missing_ratio
        
        # Feature-level missing data
        feature_missing = (data.isnull().sum() / len(data)).to_dict()
        quality_metrics['feature_missing_ratios'] = feature_missing
        
        # Outlier detection (for numeric columns)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_ratios = {}
        
        for col in numeric_cols:
            if data[col].notna().sum() > 0:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                outlier_ratios[col] = outliers / len(data)
        
        quality_metrics['outlier_ratios'] = outlier_ratios
        quality_metrics['avg_outlier_ratio'] = np.mean(list(outlier_ratios.values())) if outlier_ratios else 0
        
        # Data freshness (if date column exists)
        date_cols = data.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            latest_date = data[date_cols[0]].max()
            days_old = (datetime.now() - latest_date).days if pd.notna(latest_date) else float('inf')
            quality_metrics['data_age_days'] = days_old
        
        # Data distribution checks
        quality_metrics['data_distribution'] = self._analyze_data_distribution(data)
        
        # Store in history
        self.data_quality_history.append(quality_metrics)
        
        # Check for data quality alerts
        self._check_data_quality_alerts(quality_metrics)
        
        # Save to file
        self._save_data_quality_metrics(quality_metrics)
        
        logger.info(f"Data quality check for {data_source}: {len(data)} records, {missing_ratio:.2%} missing")
        
        return quality_metrics
    
    def check_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check."""
        
        health_metrics = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy'
        }
        
        # Database connectivity
        health_metrics['database'] = self._check_database_health()
        
        # API connectivity
        health_metrics['api'] = self._check_api_health()
        
        # Model availability
        health_metrics['models'] = self._check_model_health()
        
        # Data pipeline status
        health_metrics['data_pipeline'] = self._check_pipeline_health()
        
        # Resource usage
        health_metrics['resources'] = self._check_resource_usage()
        
        # Overall health status
        failed_components = [
            comp for comp, status in health_metrics.items() 
            if isinstance(status, dict) and status.get('status') == 'failed'
        ]
        
        if failed_components:
            health_metrics['status'] = 'degraded' if len(failed_components) < 3 else 'failed'
            health_metrics['failed_components'] = failed_components
        
        # Store in history
        self.system_health_history.append(health_metrics)
        
        # Generate alerts for failed components
        if failed_components:
            self._generate_system_alert(health_metrics)
        
        # Save to file
        self._save_system_health_metrics(health_metrics)
        
        logger.info(f"System health check: {health_metrics['status'].upper()}")
        
        return health_metrics
    
    def _analyze_data_distribution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data distribution for anomalies."""
        distribution_info = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
            if data[col].notna().sum() > 0:
                distribution_info[col] = {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'skewness': float(data[col].skew()),
                    'kurtosis': float(data[col].kurtosis()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max())
                }
        
        return distribution_info
    
    def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            import duckdb
            
            # Test connection
            con = duckdb.connect("data/warehouse.duckdb")
            
            # Test query
            result = con.execute("SELECT COUNT(*) FROM information_schema.tables").fetchone()
            
            con.close()
            
            return {
                'status': 'healthy',
                'tables_count': result[0] if result else 0,
                'response_time_ms': 50  # Placeholder
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _check_api_health(self) -> Dict[str, Any]:
        """Check external API connectivity."""
        try:
            import requests
            
            # Test The Odds API
            response = requests.get(
                "https://api.the-odds-api.com/v4/sports",
                params={'apiKey': os.getenv('ODDS_API_KEY', 'test')},
                timeout=10
            )
            
            return {
                'status': 'healthy' if response.status_code == 200 else 'degraded',
                'response_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _check_model_health(self) -> Dict[str, Any]:
        """Check model availability and performance."""
        models_dir = Path("models")
        
        if not models_dir.exists():
            return {
                'status': 'failed',
                'error': 'Models directory not found'
            }
        
        model_files = list(models_dir.glob("*.joblib"))
        
        return {
            'status': 'healthy' if len(model_files) > 0 else 'failed',
            'available_models': len(model_files),
            'model_files': [f.name for f in model_files[:5]]  # Limit to 5
        }
    
    def _check_pipeline_health(self) -> Dict[str, Any]:
        """Check data pipeline status."""
        # Check if recent data exists
        data_dir = Path("data/raw")
        
        if not data_dir.exists():
            return {
                'status': 'failed',
                'error': 'Data directory not found'
            }
        
        # Check for recent data files
        recent_files = []
        cutoff_time = datetime.now() - timedelta(days=1)
        
        for file_path in data_dir.glob("*.parquet"):
            if file_path.stat().st_mtime > cutoff_time.timestamp():
                recent_files.append(file_path.name)
        
        return {
            'status': 'healthy' if len(recent_files) > 0 else 'degraded',
            'recent_files': len(recent_files),
            'file_names': recent_files[:5]
        }
    
    def _check_resource_usage(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'status': 'healthy'
            }
            
        except ImportError:
            return {
                'status': 'unknown',
                'error': 'psutil not available'
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _check_performance_alerts(self, metrics: Dict[str, float]):
        """Check for performance degradation and generate alerts."""
        alerts = []
        
        # Check AUC threshold
        if metrics.get('auc', 1.0) < self.performance_thresholds['auc_min']:
            alerts.append({
                'type': 'performance_degradation',
                'severity': 'high',
                'message': f"Model AUC ({metrics['auc']:.4f}) below threshold ({self.performance_thresholds['auc_min']})",
                'model_name': metrics['model_name'],
                'metric': 'auc',
                'value': metrics['auc'],
                'threshold': self.performance_thresholds['auc_min']
            })
        
        # Check accuracy threshold
        if metrics.get('accuracy', 1.0) < self.performance_thresholds['accuracy_min']:
            alerts.append({
                'type': 'performance_degradation',
                'severity': 'medium',
                'message': f"Model accuracy ({metrics['accuracy']:.4f}) below threshold ({self.performance_thresholds['accuracy_min']})",
                'model_name': metrics['model_name'],
                'metric': 'accuracy',
                'value': metrics['accuracy'],
                'threshold': self.performance_thresholds['accuracy_min']
            })
        
        # Check ROI threshold (if available)
        if 'roi' in metrics and metrics['roi'] < self.performance_thresholds['roi_min']:
            alerts.append({
                'type': 'betting_performance',
                'severity': 'high',
                'message': f"ROI ({metrics['roi']:.2%}) below threshold ({self.performance_thresholds['roi_min']:.2%})",
                'model_name': metrics['model_name'],
                'metric': 'roi',
                'value': metrics['roi'],
                'threshold': self.performance_thresholds['roi_min']
            })
        
        # Add alerts to the global alerts list
        for alert in alerts:
            alert['timestamp'] = datetime.now().isoformat()
            self.alerts.append(alert)
            logger.warning(f"ALERT: {alert['message']}")
    
    def _check_data_quality_alerts(self, metrics: Dict[str, Any]):
        """Check for data quality issues and generate alerts."""
        alerts = []
        
        # Check missing data threshold
        if metrics.get('missing_data_ratio', 0) > self.data_quality_thresholds['missing_data_max']:
            alerts.append({
                'type': 'data_quality',
                'severity': 'medium',
                'message': f"High missing data ratio ({metrics['missing_data_ratio']:.2%}) in {metrics['data_source']}",
                'data_source': metrics['data_source'],
                'metric': 'missing_data_ratio',
                'value': metrics['missing_data_ratio'],
                'threshold': self.data_quality_thresholds['missing_data_max']
            })
        
        # Check outlier ratio
        if metrics.get('avg_outlier_ratio', 0) > self.data_quality_thresholds['outlier_ratio_max']:
            alerts.append({
                'type': 'data_quality',
                'severity': 'low',
                'message': f"High outlier ratio ({metrics['avg_outlier_ratio']:.2%}) in {metrics['data_source']}",
                'data_source': metrics['data_source'],
                'metric': 'avg_outlier_ratio',
                'value': metrics['avg_outlier_ratio'],
                'threshold': self.data_quality_thresholds['outlier_ratio_max']
            })
        
        # Check data freshness
        if metrics.get('data_age_days', 0) > 2:
            alerts.append({
                'type': 'data_freshness',
                'severity': 'medium',
                'message': f"Data in {metrics['data_source']} is {metrics['data_age_days']} days old",
                'data_source': metrics['data_source'],
                'metric': 'data_age_days',
                'value': metrics['data_age_days'],
                'threshold': 2
            })
        
        # Add alerts to the global alerts list
        for alert in alerts:
            alert['timestamp'] = datetime.now().isoformat()
            self.alerts.append(alert)
            logger.warning(f"DATA QUALITY ALERT: {alert['message']}")
    
    def _generate_system_alert(self, health_metrics: Dict[str, Any]):
        """Generate system health alerts."""
        alert = {
            'type': 'system_health',
            'severity': 'high' if health_metrics['status'] == 'failed' else 'medium',
            'message': f"System health degraded: {', '.join(health_metrics.get('failed_components', []))}",
            'timestamp': datetime.now().isoformat(),
            'failed_components': health_metrics.get('failed_components', []),
            'overall_status': health_metrics['status']
        }
        
        self.alerts.append(alert)
        logger.error(f"SYSTEM ALERT: {alert['message']}")
    
    def get_active_alerts(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get active alerts from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        active_alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]
        
        # Sort by severity and timestamp
        severity_order = {'high': 3, 'medium': 2, 'low': 1}
        active_alerts.sort(
            key=lambda x: (severity_order.get(x['severity'], 0), x['timestamp']),
            reverse=True
        )
        
        return active_alerts
    
    def get_performance_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Get performance summary for the last N days."""
        cutoff_time = datetime.now() - timedelta(days=days_back)
        
        recent_performance = [
            perf for perf in self.model_performance_history
            if datetime.fromisoformat(perf['timestamp']) > cutoff_time
        ]
        
        if not recent_performance:
            return {'error': 'No recent performance data available'}
        
        # Aggregate metrics
        summary = {
            'period_days': days_back,
            'total_predictions': sum(p.get('n_predictions', 0) for p in recent_performance),
            'avg_auc': np.mean([p.get('auc', 0) for p in recent_performance]),
            'avg_accuracy': np.mean([p.get('accuracy', 0) for p in recent_performance]),
            'models_tracked': len(set(p.get('model_name', 'unknown') for p in recent_performance))
        }
        
        # Betting metrics (if available)
        betting_metrics = [p for p in recent_performance if 'roi' in p]
        if betting_metrics:
            summary.update({
                'avg_roi': np.mean([p['roi'] for p in betting_metrics]),
                'avg_win_rate': np.mean([p.get('win_rate', 0) for p in betting_metrics]),
                'total_bets': sum(p.get('total_bets', 0) for p in betting_metrics),
                'total_pnl': sum(p.get('profit_loss', 0) for p in betting_metrics)
            })
        
        return summary
    
    def _save_performance_metrics(self, metrics: Dict[str, Any]):
        """Save performance metrics to file."""
        file_path = self.monitoring_dir / "performance_metrics.jsonl"
        
        with open(file_path, 'a') as f:
            import json
            f.write(json.dumps(metrics) + '\n')
    
    def _save_data_quality_metrics(self, metrics: Dict[str, Any]):
        """Save data quality metrics to file."""
        file_path = self.monitoring_dir / "data_quality_metrics.jsonl"
        
        with open(file_path, 'a') as f:
            import json
            f.write(json.dumps(metrics) + '\n')
    
    def _save_system_health_metrics(self, metrics: Dict[str, Any]):
        """Save system health metrics to file."""
        file_path = self.monitoring_dir / "system_health_metrics.jsonl"
        
        with open(file_path, 'a') as f:
            import json
            f.write(json.dumps(metrics) + '\n')
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_health': self.check_system_health(),
            'performance_summary': self.get_performance_summary(),
            'active_alerts': self.get_active_alerts(),
            'alert_counts': {
                'total': len(self.alerts),
                'last_24h': len(self.get_active_alerts(24)),
                'by_severity': {}
            }
        }
        
        # Count alerts by severity
        for alert in self.get_active_alerts(24):
            severity = alert.get('severity', 'unknown')
            report['alert_counts']['by_severity'][severity] = report['alert_counts']['by_severity'].get(severity, 0) + 1
        
        return report


def main():
    """Example usage of the performance monitor."""
    logger.info("Performance Monitor - Example Usage")
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    # Example model performance tracking
    fake_predictions = {
        'probability': np.random.random(100),
        'prediction': np.random.binomial(1, 0.5, 100)
    }
    fake_outcomes = pd.Series(np.random.binomial(1, 0.5, 100))
    fake_betting_results = {
        'roi': 0.08,
        'win_rate': 0.55,
        'total_bets': 100,
        'total_pnl': 800,
        'sharpe_ratio': 1.2,
        'max_drawdown': 0.05
    }
    
    performance_metrics = monitor.track_model_performance(
        "ensemble_v1", 
        fake_predictions, 
        fake_outcomes, 
        fake_betting_results
    )
    
    # Example data quality monitoring
    fake_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000),
        'date': pd.date_range('2024-01-01', periods=1000, freq='H')
    })
    
    # Add some missing values
    fake_data.loc[np.random.choice(1000, 50, replace=False), 'feature1'] = np.nan
    
    quality_metrics = monitor.monitor_data_quality(fake_data, "test_data_source")
    
    # System health check
    health_metrics = monitor.check_system_health()
    
    # Generate monitoring report
    report = monitor.generate_monitoring_report()
    
    logger.info("Monitoring Report Generated:")
    logger.info(f"  System Status: {report['system_health']['status']}")
    logger.info(f"  Active Alerts: {report['alert_counts']['last_24h']}")
    logger.info(f"  Performance Summary: {len(report['performance_summary'])} metrics")


if __name__ == "__main__":
    main() 