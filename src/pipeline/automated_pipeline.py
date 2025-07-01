import os
import sys
import logging
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from backend.data_backend import MLBDataBackendV3
from features.build_features import MLBFeatureBuilder
from ml.models import MLBBettingModel
from ml.risk_management import BankrollManager
from monitoring.performance_monitor import PerformanceMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutomatedMLBPipeline:
    """
    Automated pipeline for continuous MLB betting analytics.
    
    Features:
    - Scheduled data ingestion
    - Automated feature engineering
    - Model retraining triggers
    - Performance monitoring
    - Alert generation
    - Risk management integration
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._load_default_config()
        
        # Initialize components
        self.data_backend = MLBDataBackendV3(
            api_key=os.getenv('ODDS_API_KEY'),
            seasons=[2023, 2024]
        )
        
        self.feature_builder = MLBFeatureBuilder()
        self.model = MLBBettingModel(model_type="ensemble")
        self.bankroll_manager = BankrollManager(
            initial_bankroll=self.config.get('initial_bankroll', 10000)
        )
        self.monitor = PerformanceMonitor()
        
        # Pipeline state
        self.last_data_update = None
        self.last_model_training = None
        self.pipeline_status = "initialized"
        
    def _load_default_config(self) -> Dict:
        """Load default pipeline configuration."""
        return {
            'data_update_interval_hours': 6,
            'model_retrain_interval_days': 7,
            'performance_check_interval_hours': 1,
            'min_model_performance_threshold': 0.55,
            'max_daily_risk': 0.10,
            'initial_bankroll': 10000,
            'auto_betting_enabled': False,
            'notification_enabled': True
        }
    
    def start_pipeline(self):
        """Start the automated pipeline with scheduled tasks."""
        logger.info("Starting automated MLB betting pipeline...")
        
        # Schedule data ingestion
        schedule.every(self.config['data_update_interval_hours']).hours.do(
            self._run_data_ingestion
        )
        
        # Schedule feature engineering
        schedule.every(self.config['data_update_interval_hours']).hours.do(
            self._run_feature_engineering
        ).tag('features')
        
        # Schedule model training
        schedule.every(self.config['model_retrain_interval_days']).days.do(
            self._run_model_training
        ).tag('training')
        
        # Schedule performance monitoring
        schedule.every(self.config['performance_check_interval_hours']).hours.do(
            self._run_performance_monitoring
        ).tag('monitoring')
        
        # Schedule daily risk assessment
        schedule.every().day.at("09:00").do(
            self._run_daily_risk_assessment
        ).tag('risk')
        
        # Schedule betting recommendations (if enabled)
        if self.config.get('auto_betting_enabled', False):
            schedule.every().day.at("10:00").do(
                self._generate_daily_recommendations
            ).tag('betting')
        
        # Initial pipeline run
        self._run_initial_setup()
        
        # Main pipeline loop
        self.pipeline_status = "running"
        logger.info("Pipeline started successfully. Running scheduled tasks...")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user")
            self.pipeline_status = "stopped"
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.pipeline_status = "error"
            raise
    
    def _run_initial_setup(self):
        """Run initial setup tasks."""
        logger.info("Running initial pipeline setup...")
        
        try:
            # Initial data ingestion
            self._run_data_ingestion()
            
            # Initial feature engineering
            self._run_feature_engineering()
            
            # Load or train initial models
            if not self.model.load_latest_models():
                logger.info("No existing models found, training initial models...")
                self._run_model_training()
            
            # Initial performance monitoring
            self._run_performance_monitoring()
            
            logger.info("Initial setup completed successfully")
            
        except Exception as e:
            logger.error(f"Initial setup failed: {e}")
            raise
    
    def _run_data_ingestion(self):
        """Run data ingestion tasks."""
        logger.info("Running data ingestion...")
        
        try:
            # Fetch games data
            games_data = self.data_backend.fetch_games_data()
            if not games_data.empty:
                games_data.to_parquet("data/raw/games_latest.parquet")
                logger.info(f"Ingested {len(games_data)} games records")
            
            # Fetch standings data
            standings_data = self.data_backend.fetch_standings_data()
            if not standings_data.empty:
                standings_data.to_parquet("data/raw/standings_latest.parquet")
                logger.info(f"Ingested {len(standings_data)} standings records")
            
            # Fetch odds data
            odds_data = self.data_backend.fetch_moneyline_odds()
            if not odds_data.empty:
                odds_data.to_parquet("data/raw/odds_latest.parquet")
                logger.info(f"Ingested {len(odds_data)} odds records")
            
            # Fetch prop bets
            props_data = self.data_backend.fetch_comprehensive_mlb_props()
            if not props_data.empty:
                props_data.to_parquet("data/raw/props_latest.parquet")
                logger.info(f"Ingested {len(props_data)} prop bet records")
            
            self.last_data_update = datetime.now()
            
            # Data quality monitoring
            if not games_data.empty:
                self.monitor.monitor_data_quality(games_data, "games_data")
            if not odds_data.empty:
                self.monitor.monitor_data_quality(odds_data, "odds_data")
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            self._handle_pipeline_error("data_ingestion", e)
    
    def _run_feature_engineering(self):
        """Run feature engineering tasks."""
        logger.info("Running feature engineering...")
        
        try:
            with self.feature_builder as fb:
                # Build all feature sets
                fb.build_all_features([2023, 2024])
                
            logger.info("Feature engineering completed")
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            self._handle_pipeline_error("feature_engineering", e)
    
    def _run_model_training(self):
        """Run model training tasks."""
        logger.info("Running model training...")
        
        try:
            # Load latest features
            features_dir = Path("data/features")
            
            if (features_dir / "moneyline.parquet").exists():
                # Load moneyline features
                df = pd.read_parquet(features_dir / "moneyline.parquet")
                
                # Prepare training data
                feature_cols = [col for col in df.columns if col not in ['date', 'home_team', 'away_team', 'home_win']]
                X = df[feature_cols].fillna(0)
                y = df['home_win']
                
                # Train model
                results = self.model.train_model(X, y, hyperparameter_tuning=False)
                
                self.last_model_training = datetime.now()
                logger.info(f"Model training completed. Results: {results}")
                
                # Track model performance
                predictions = self.model.predict(X[:1000])  # Sample for performance tracking
                self.monitor.track_model_performance(
                    "ensemble_v1", 
                    predictions, 
                    y[:1000]
                )
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            self._handle_pipeline_error("model_training", e)
    
    def _run_performance_monitoring(self):
        """Run performance monitoring tasks."""
        logger.info("Running performance monitoring...")
        
        try:
            # System health check
            health_metrics = self.monitor.check_system_health()
            
            # Generate monitoring report
            report = self.monitor.generate_monitoring_report()
            
            # Check for critical alerts
            critical_alerts = [
                alert for alert in self.monitor.get_active_alerts(24)
                if alert.get('severity') == 'high'
            ]
            
            if critical_alerts:
                logger.warning(f"Found {len(critical_alerts)} critical alerts")
                self._handle_critical_alerts(critical_alerts)
            
            logger.info("Performance monitoring completed")
            
        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")
            self._handle_pipeline_error("performance_monitoring", e)
    
    def _run_daily_risk_assessment(self):
        """Run daily risk assessment."""
        logger.info("Running daily risk assessment...")
        
        try:
            # Get risk assessment
            risk_assessment = self.bankroll_manager.get_risk_assessment()
            
            logger.info(f"Risk Level: {risk_assessment['risk_level']}")
            logger.info(f"Current Drawdown: {risk_assessment['current_drawdown']:.2%}")
            logger.info(f"Bankroll Health: {risk_assessment['bankroll_health']:.2f}")
            
            # Generate alerts for high risk situations
            if risk_assessment['risk_level'] == 'HIGH':
                alert = {
                    'type': 'risk_management',
                    'severity': 'high',
                    'message': f"High risk exposure detected: {risk_assessment['risk_recommendation']}",
                    'timestamp': datetime.now().isoformat()
                }
                self.monitor.alerts.append(alert)
                logger.warning(f"RISK ALERT: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            self._handle_pipeline_error("risk_assessment", e)
    
    def _generate_daily_recommendations(self):
        """Generate daily betting recommendations."""
        logger.info("Generating daily betting recommendations...")
        
        try:
            # Get today's odds
            odds_data = self.data_backend.fetch_moneyline_odds(days_ahead=1)
            
            if odds_data.empty:
                logger.info("No odds data available for recommendations")
                return
            
            # Load latest features for prediction
            features_dir = Path("data/features")
            if not (features_dir / "moneyline.parquet").exists():
                logger.warning("No feature data available for recommendations")
                return
            
            # Generate recommendations (simplified)
            recommendations = []
            
            for _, game in odds_data.head(10).iterrows():  # Limit to 10 games
                # Create dummy features for demonstration
                game_features = pd.DataFrame({
                    'feature_1': [0.5],
                    'feature_2': [0.3],
                    'feature_3': [0.7]
                })
                
                # Get model predictions
                predictions = self.model.predict(game_features)
                
                # Calculate expected value
                home_odds = game.get('home_odds', -110)
                expected_values = self.model.calculate_expected_value(predictions, np.array([home_odds]))
                
                if expected_values[0] > 0.02:  # Minimum edge threshold
                    # Get bet sizing recommendation
                    bet_info = {
                        'game_id': f"{game.get('home_team', 'Unknown')}_vs_{game.get('away_team', 'Unknown')}",
                        'bet_type': 'moneyline',
                        'win_probability': predictions['probability'][0],
                        'odds': home_odds,
                        'confidence': predictions.get('confidence', [0.5])[0]
                    }
                    
                    sizing = self.bankroll_manager.calculate_bet_size(
                        bet_info['win_probability'],
                        bet_info['odds'],
                        bet_info['confidence']
                    )
                    
                    recommendation = {
                        'game_id': bet_info['game_id'],
                        'bet_type': bet_info['bet_type'],
                        'recommendation': 'home',
                        'confidence': bet_info['confidence'],
                        'expected_value': expected_values[0],
                        'recommended_stake': sizing['final_fraction'],
                        'bet_amount': sizing['bet_amount'],
                        'reasoning': f"Model identifies {expected_values[0]:.1%} edge with {bet_info['confidence']:.1%} confidence"
                    }
                    
                    recommendations.append(recommendation)
            
            # Save recommendations
            if recommendations:
                recommendations_df = pd.DataFrame(recommendations)
                recommendations_df.to_parquet(f"data/results/recommendations_{datetime.now().strftime('%Y%m%d')}.parquet")
                logger.info(f"Generated {len(recommendations)} betting recommendations")
            else:
                logger.info("No profitable betting opportunities found")
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            self._handle_pipeline_error("recommendations", e)
    
    def _handle_pipeline_error(self, component: str, error: Exception):
        """Handle pipeline errors and generate alerts."""
        alert = {
            'type': 'pipeline_error',
            'severity': 'high',
            'message': f"Pipeline component '{component}' failed: {str(error)}",
            'component': component,
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        }
        
        self.monitor.alerts.append(alert)
        logger.error(f"PIPELINE ERROR: {alert['message']}")
        
        # Optionally send notifications
        if self.config.get('notification_enabled', True):
            self._send_notification(alert)
    
    def _handle_critical_alerts(self, alerts: List[Dict]):
        """Handle critical alerts that require immediate attention."""
        for alert in alerts:
            logger.critical(f"CRITICAL ALERT: {alert['message']}")
            
            # Take automated actions based on alert type
            if alert.get('type') == 'performance_degradation':
                self._handle_performance_degradation(alert)
            elif alert.get('type') == 'system_health':
                self._handle_system_health_issue(alert)
            elif alert.get('type') == 'risk_management':
                self._handle_risk_management_alert(alert)
    
    def _handle_performance_degradation(self, alert: Dict):
        """Handle model performance degradation."""
        logger.info("Handling performance degradation...")
        
        # Trigger model retraining
        if alert.get('metric') == 'auc' and alert.get('value', 0) < 0.52:
            logger.info("Triggering emergency model retraining due to severe performance degradation")
            self._run_model_training()
    
    def _handle_system_health_issue(self, alert: Dict):
        """Handle system health issues."""
        logger.info("Handling system health issue...")
        
        failed_components = alert.get('failed_components', [])
        
        # Try to restart failed components
        if 'database' in failed_components:
            logger.info("Attempting to reinitialize database connection...")
            # Database restart logic would go here
        
        if 'api' in failed_components:
            logger.info("API connectivity issues detected, reducing API call frequency...")
            # API throttling logic would go here
    
    def _handle_risk_management_alert(self, alert: Dict):
        """Handle risk management alerts."""
        logger.info("Handling risk management alert...")
        
        # Reduce bet sizing or pause betting
        if "high risk exposure" in alert.get('message', '').lower():
            self.config['max_daily_risk'] *= 0.5  # Reduce risk limit by half
            logger.info(f"Reduced daily risk limit to {self.config['max_daily_risk']:.1%}")
    
    def _send_notification(self, alert: Dict):
        """Send notification for critical alerts."""
        # Placeholder for notification system
        # Could integrate with email, Slack, SMS, etc.
        logger.info(f"NOTIFICATION: {alert['message']}")
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status and metrics."""
        return {
            'status': self.pipeline_status,
            'last_data_update': self.last_data_update.isoformat() if self.last_data_update else None,
            'last_model_training': self.last_model_training.isoformat() if self.last_model_training else None,
            'active_alerts': len(self.monitor.get_active_alerts(24)),
            'system_health': self.monitor.check_system_health()['status'],
            'bankroll_status': {
                'current_bankroll': self.bankroll_manager.current_bankroll,
                'roi': (self.bankroll_manager.current_bankroll - self.bankroll_manager.initial_bankroll) / self.bankroll_manager.initial_bankroll,
                'max_drawdown': self.bankroll_manager.max_drawdown
            }
        }
    
    def stop_pipeline(self):
        """Stop the automated pipeline."""
        logger.info("Stopping automated pipeline...")
        schedule.clear()
        self.pipeline_status = "stopped"
        logger.info("Pipeline stopped successfully")


def main():
    """Main function to run the automated pipeline."""
    logger.info("Starting MLB Betting Analytics Automated Pipeline")
    
    # Load configuration
    config = {
        'data_update_interval_hours': 6,
        'model_retrain_interval_days': 7,
        'performance_check_interval_hours': 1,
        'initial_bankroll': 10000,
        'auto_betting_enabled': False,  # Set to True to enable automated betting
        'notification_enabled': True
    }
    
    # Initialize and start pipeline
    pipeline = AutomatedMLBPipeline(config)
    
    try:
        pipeline.start_pipeline()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        pipeline.stop_pipeline()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        pipeline.stop_pipeline()
        raise


if __name__ == "__main__":
    main() 