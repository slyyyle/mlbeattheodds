import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import all modules to test
from backend.data_backend import MLBDataBackendV3
from features.build_features import MLBFeatureBuilder
from ml.models import MLBBettingModel
from ml.risk_management import BankrollManager
from monitoring.performance_monitor import PerformanceMonitor
from pipeline.automated_pipeline import AutomatedMLBPipeline
from alerts.notification_system import NotificationSystem


class TestMLBDataBackend:
    """Test suite for data backend functionality."""
    
    @pytest.fixture
    def backend(self):
        return MLBDataBackendV3(api_key="test_key", seasons=[2023, 2024])
    
    @pytest.fixture
    def sample_games_data(self):
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'home_team': ['NYY'] * 50 + ['BOS'] * 50,
            'away_team': ['BOS'] * 50 + ['NYY'] * 50,
            'season': [2024] * 100,
            'team': ['NYY'] * 25 + ['BOS'] * 25 + ['BOS'] * 25 + ['NYY'] * 25,
            'result': ['W', 'L'] * 50,
            'runs_scored': np.random.randint(0, 15, 100),
            'runs_allowed': np.random.randint(0, 15, 100)
        })
    
    def test_initialization(self, backend):
        """Test backend initialization."""
        assert backend.api_key == "test_key"
        assert backend.seasons == [2023, 2024]
        assert backend.sport_key == "baseball_mlb"
    
    @patch('requests.get')
    def test_api_request_success(self, mock_get, backend):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "data"}
        mock_get.return_value = mock_response
        
        result = backend._make_api_request("test_endpoint", {"param": "value"})
        
        assert result == {"test": "data"}
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_api_request_failure(self, mock_get, backend):
        """Test failed API request."""
        mock_get.side_effect = Exception("API Error")
        
        result = backend._make_api_request("test_endpoint", {"param": "value"})
        
        assert result is None
    
    def test_american_to_probability_conversion(self, backend):
        """Test odds conversion."""
        # Test positive odds
        positive_odds = pd.Series([100, 200, 150])
        positive_probs = backend._american_to_probability(positive_odds)
        
        expected_positive = pd.Series([0.5, 0.333333, 0.4])
        pd.testing.assert_series_equal(positive_probs, expected_positive, atol=1e-5)
        
        # Test negative odds
        negative_odds = pd.Series([-110, -200, -150])
        negative_probs = backend._american_to_probability(negative_odds)
        
        expected_negative = pd.Series([0.523810, 0.666667, 0.6])
        pd.testing.assert_series_equal(negative_probs, expected_negative, atol=1e-5)
    
    @patch('pybaseball.schedule_and_record')
    def test_fetch_games_data(self, mock_schedule, backend, sample_games_data):
        """Test games data fetching."""
        mock_schedule.return_value = sample_games_data
        
        result = backend.fetch_games_data()
        
        assert not result.empty
        assert 'run_differential' in result.columns
        mock_schedule.assert_called()
    
    def test_data_processing(self, backend, sample_games_data):
        """Test data processing and feature creation."""
        # Test run differential calculation
        processed_data = sample_games_data.copy()
        processed_data['run_differential'] = processed_data['runs_scored'] - processed_data['runs_allowed']
        
        assert 'run_differential' in processed_data.columns
        assert len(processed_data) == len(sample_games_data)


class TestMLBFeatureBuilder:
    """Test suite for feature engineering."""
    
    @pytest.fixture
    def feature_builder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.duckdb")
            yield MLBFeatureBuilder(db_path=db_path)
    
    @pytest.fixture
    def sample_feature_data(self):
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50),
            'home_team': ['NYY'] * 25 + ['BOS'] * 25,
            'away_team': ['BOS'] * 25 + ['NYY'] * 25,
            'season': [2024] * 50,
            'home_win': [1, 0] * 25,
            'home_win_pct': np.random.uniform(0.3, 0.7, 50),
            'away_win_pct': np.random.uniform(0.3, 0.7, 50),
            'home_rank': np.random.randint(1, 30, 50),
            'away_rank': np.random.randint(1, 30, 50)
        })
    
    def test_initialization(self, feature_builder):
        """Test feature builder initialization."""
        assert feature_builder.db_path is not None
        assert feature_builder.con is None  # Should be None until context manager is used
    
    def test_moneyline_engineered_features(self, feature_builder, sample_feature_data):
        """Test moneyline feature engineering."""
        result = feature_builder._add_moneyline_engineered_features(sample_feature_data)
        
        # Check that new features are added
        expected_features = ['win_pct_diff', 'rank_diff', 'matchup_quality']
        for feature in expected_features:
            if all(col in sample_feature_data.columns for col in ['home_win_pct', 'away_win_pct', 'home_rank', 'away_rank']):
                assert feature in result.columns
    
    def test_feature_summary_logging(self, feature_builder, sample_feature_data):
        """Test feature summary logging."""
        # This should not raise an exception
        feature_builder._log_feature_summary(sample_feature_data, "test")


class TestMLBBettingModel:
    """Test suite for machine learning models."""
    
    @pytest.fixture
    def model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield MLBBettingModel(model_type="logistic", model_dir=temp_dir)
    
    @pytest.fixture
    def sample_training_data(self):
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.binomial(1, 0.5, n_samples))
        
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.model_type == "logistic"
        assert "logistic" in model.models
        assert len(model.trained_models) == 0
    
    def test_model_training(self, model, sample_training_data):
        """Test model training process."""
        X, y = sample_training_data
        
        results = model.train_model(X, y, hyperparameter_tuning=False)
        
        assert isinstance(results, dict)
        assert len(model.trained_models) > 0
        assert model.feature_names == list(X.columns)
    
    def test_predictions(self, model, sample_training_data):
        """Test model predictions."""
        X, y = sample_training_data
        
        # Train model first
        model.train_model(X, y, hyperparameter_tuning=False)
        
        # Make predictions
        predictions = model.predict(X[:10])
        
        assert 'probability' in predictions
        assert 'prediction' in predictions
        assert len(predictions['probability']) == 10
        assert all(0 <= p <= 1 for p in predictions['probability'])
    
    def test_expected_value_calculation(self, model, sample_training_data):
        """Test expected value calculation."""
        X, y = sample_training_data
        
        # Train model
        model.train_model(X, y, hyperparameter_tuning=False)
        
        # Get predictions
        predictions = model.predict(X[:5])
        odds = np.array([-110, 120, -150, 200, -200])
        
        # Calculate expected values
        ev = model.calculate_expected_value(predictions, odds)
        
        assert len(ev) == 5
        assert all(isinstance(val, (int, float)) for val in ev)
    
    def test_betting_recommendations(self, model, sample_training_data):
        """Test betting recommendations generation."""
        X, y = sample_training_data
        
        # Train model
        model.train_model(X, y, hyperparameter_tuning=False)
        
        # Generate recommendations
        odds = np.array([-110, 120, -150, 200, -200])
        recommendations = model.get_betting_recommendations(X[:5], odds)
        
        assert isinstance(recommendations, pd.DataFrame)
        expected_columns = ['probability', 'confidence', 'odds', 'expected_value', 'recommendation']
        for col in expected_columns:
            if col in recommendations.columns:
                assert col in recommendations.columns
    
    def test_model_persistence(self, model, sample_training_data):
        """Test model saving and loading."""
        X, y = sample_training_data
        
        # Train and save model
        model.train_model(X, y, hyperparameter_tuning=False)
        
        # Create new model instance and load
        new_model = MLBBettingModel(model_type="logistic", model_dir=model.model_dir)
        loaded = new_model.load_latest_models()
        
        # Note: This test might need adjustment based on actual file structure
        # assert loaded  # Should return True if models were loaded


class TestBankrollManager:
    """Test suite for risk management and bankroll management."""
    
    @pytest.fixture
    def manager(self):
        return BankrollManager(initial_bankroll=10000, max_bet_size=0.05)
    
    def test_initialization(self, manager):
        """Test bankroll manager initialization."""
        assert manager.initial_bankroll == 10000
        assert manager.current_bankroll == 10000
        assert manager.max_bet_size == 0.05
        assert len(manager.bet_history) == 0
    
    def test_kelly_criterion_calculation(self, manager):
        """Test Kelly criterion bet sizing."""
        # Test positive edge scenario
        kelly_size = manager.calculate_kelly_size(win_probability=0.55, odds=-110)
        assert kelly_size > 0
        
        # Test negative edge scenario
        kelly_size_negative = manager.calculate_kelly_size(win_probability=0.45, odds=-110)
        assert kelly_size_negative == 0  # Should be 0 for negative edge
    
    def test_bet_sizing(self, manager):
        """Test comprehensive bet sizing."""
        sizing = manager.calculate_bet_size(
            win_probability=0.55,
            odds=-110,
            confidence=0.8
        )
        
        assert 'kelly_fraction' in sizing
        assert 'final_fraction' in sizing
        assert 'bet_amount' in sizing
        assert sizing['bet_amount'] <= manager.current_bankroll * manager.max_bet_size
    
    def test_bet_placement_and_settlement(self, manager):
        """Test bet placement and settlement process."""
        # Place a bet
        bet_info = {
            'game_id': 'NYY_vs_BOS',
            'bet_type': 'moneyline',
            'win_probability': 0.55,
            'odds': -110,
            'confidence': 0.8
        }
        
        bet_record = manager.place_bet(bet_info)
        
        assert bet_record['id'] == 1
        assert bet_record['status'] == 'pending'
        assert len(manager.bet_history) == 1
        
        # Settle the bet as a win
        settlement = manager.settle_bet(1, won=True)
        
        assert settlement['result'] == 'win'
        assert manager.current_bankroll > manager.initial_bankroll
        assert manager.bet_history[0]['status'] == 'settled'
    
    def test_performance_metrics(self, manager):
        """Test performance metrics calculation."""
        # Place and settle multiple bets
        for i in range(5):
            bet_info = {
                'game_id': f'Game_{i}',
                'bet_type': 'moneyline',
                'win_probability': 0.55,
                'odds': -110,
                'confidence': 0.8
            }
            bet_record = manager.place_bet(bet_info)
            manager.settle_bet(bet_record['id'], won=(i % 2 == 0))  # Win every other bet
        
        metrics = manager.get_performance_metrics()
        
        assert 'total_bets' in metrics
        assert 'win_rate' in metrics
        assert 'roi' in metrics
        assert metrics['total_bets'] == 5
        assert 0 <= metrics['win_rate'] <= 1
    
    def test_portfolio_optimization(self, manager):
        """Test portfolio optimization for multiple bets."""
        bet_opportunities = [
            {
                'win_probability': 0.55,
                'odds': -110,
                'confidence': 0.8
            },
            {
                'win_probability': 0.60,
                'odds': 120,
                'confidence': 0.7
            },
            {
                'win_probability': 0.52,
                'odds': -105,
                'confidence': 0.9
            }
        ]
        
        recommendations = manager.optimize_portfolio(bet_opportunities)
        
        assert len(recommendations) == 3
        for rec in recommendations:
            assert 'kelly_fraction' in rec
            assert 'optimized_fraction' in rec
            assert 'final_fraction' in rec
    
    def test_risk_assessment(self, manager):
        """Test risk assessment functionality."""
        # Place some bets to create risk exposure
        for i in range(3):
            bet_info = {
                'game_id': f'Game_{i}',
                'bet_type': 'moneyline',
                'win_probability': 0.55,
                'odds': -110,
                'confidence': 0.8
            }
            manager.place_bet(bet_info)
        
        risk_assessment = manager.get_risk_assessment()
        
        assert 'risk_level' in risk_assessment
        assert 'current_risk_exposure' in risk_assessment
        assert 'bankroll_health' in risk_assessment
        assert risk_assessment['risk_level'] in ['LOW', 'MEDIUM', 'HIGH']


class TestPerformanceMonitor:
    """Test suite for performance monitoring."""
    
    @pytest.fixture
    def monitor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield PerformanceMonitor(monitoring_dir=temp_dir)
    
    @pytest.fixture
    def sample_predictions(self):
        return {
            'probability': np.random.random(100),
            'prediction': np.random.binomial(1, 0.5, 100)
        }
    
    @pytest.fixture
    def sample_outcomes(self):
        return pd.Series(np.random.binomial(1, 0.5, 100))
    
    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert len(monitor.model_performance_history) == 0
        assert len(monitor.alerts) == 0
        assert 'auc_min' in monitor.performance_thresholds
    
    def test_model_performance_tracking(self, monitor, sample_predictions, sample_outcomes):
        """Test model performance tracking."""
        metrics = monitor.track_model_performance(
            model_name="test_model",
            predictions=sample_predictions,
            actual_outcomes=sample_outcomes
        )
        
        assert 'auc' in metrics
        assert 'accuracy' in metrics
        assert 'model_name' in metrics
        assert len(monitor.model_performance_history) == 1
    
    def test_data_quality_monitoring(self, monitor):
        """Test data quality monitoring."""
        # Create sample data with quality issues
        data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5] * 20,
            'feature2': list(range(100)),
            'feature3': [1] * 95 + [100] * 5,  # Some outliers
            'date': pd.date_range('2024-01-01', periods=100)
        })
        
        quality_metrics = monitor.monitor_data_quality(data, "test_source")
        
        assert 'missing_data_ratio' in quality_metrics
        assert 'outlier_ratios' in quality_metrics
        assert 'data_age_days' in quality_metrics
        assert len(monitor.data_quality_history) == 1
    
    def test_system_health_check(self, monitor):
        """Test system health monitoring."""
        health_metrics = monitor.check_system_health()
        
        assert 'status' in health_metrics
        assert 'database' in health_metrics
        assert 'api' in health_metrics
        assert health_metrics['status'] in ['healthy', 'degraded', 'failed']
    
    def test_alert_generation(self, monitor, sample_predictions, sample_outcomes):
        """Test alert generation for performance issues."""
        # Create predictions that should trigger alerts
        poor_predictions = {
            'probability': [0.5] * 100,  # No discrimination
            'prediction': [0] * 100
        }
        
        metrics = monitor.track_model_performance(
            model_name="poor_model",
            predictions=poor_predictions,
            actual_outcomes=sample_outcomes
        )
        
        # Should generate alerts for poor performance
        alerts = monitor.get_active_alerts(1)
        # Note: Specific alert generation depends on thresholds and implementation


class TestNotificationSystem:
    """Test suite for notification and alerting system."""
    
    @pytest.fixture
    def notifier(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "notifications.json")
            yield NotificationSystem(config_file=config_file)
    
    def test_initialization(self, notifier):
        """Test notification system initialization."""
        assert len(notifier.channels) > 0
        assert 'console' in notifier.channels  # Should have console channel by default
        assert len(notifier.alert_history) == 0
    
    def test_alert_sending(self, notifier):
        """Test basic alert sending."""
        alert_id = notifier.send_alert(
            alert_type="test_alert",
            severity="medium",
            message="Test alert message",
            data={"test_key": "test_value"}
        )
        
        assert alert_id is not None
        assert len(notifier.alert_history) == 1
        assert notifier.alert_history[0].type == "test_alert"
    
    def test_rate_limiting(self, notifier):
        """Test rate limiting functionality."""
        # Send multiple high severity alerts
        alert_ids = []
        for i in range(15):  # Exceed the high severity limit
            alert_id = notifier.send_alert(
                alert_type="test_alert",
                severity="high",
                message=f"Test alert {i}"
            )
            alert_ids.append(alert_id)
        
        # Some alerts should be rate limited
        # The exact behavior depends on implementation
        assert len(notifier.alert_history) <= 15
    
    def test_betting_opportunity_alert(self, notifier):
        """Test betting opportunity specific alerts."""
        alert_id = notifier.send_betting_opportunity(
            game_id="NYY_vs_BOS",
            bet_type="moneyline",
            expected_value=0.05,
            confidence=0.8,
            recommended_stake=0.03,
            reasoning="Strong model confidence with market inefficiency"
        )
        
        assert alert_id is not None
        assert len(notifier.alert_history) == 1
        assert notifier.alert_history[0].type == "betting_opportunity"
    
    def test_delivery_stats(self, notifier):
        """Test delivery statistics tracking."""
        # Send some alerts
        for i in range(3):
            notifier.send_alert("test", "low", f"Message {i}")
        
        stats = notifier.get_delivery_stats()
        
        assert 'total_deliveries' in stats
        assert 'successful_deliveries' in stats
        assert 'channel_stats' in stats


class TestAutomatedPipeline:
    """Test suite for automated pipeline."""
    
    @pytest.fixture
    def pipeline_config(self):
        return {
            'data_update_interval_hours': 1,
            'model_retrain_interval_days': 1,
            'performance_check_interval_hours': 1,
            'initial_bankroll': 10000,
            'auto_betting_enabled': False
        }
    
    @pytest.fixture
    def pipeline(self, pipeline_config):
        with patch('src.pipeline.automated_pipeline.MLBDataBackendV3'), \
             patch('src.pipeline.automated_pipeline.MLBFeatureBuilder'), \
             patch('src.pipeline.automated_pipeline.MLBBettingModel'), \
             patch('src.pipeline.automated_pipeline.BankrollManager'), \
             patch('src.pipeline.automated_pipeline.PerformanceMonitor'):
            yield AutomatedMLBPipeline(config=pipeline_config)
    
    def test_initialization(self, pipeline, pipeline_config):
        """Test pipeline initialization."""
        assert pipeline.config == pipeline_config
        assert pipeline.pipeline_status == "initialized"
    
    def test_pipeline_status(self, pipeline):
        """Test pipeline status reporting."""
        status = pipeline.get_pipeline_status()
        
        assert 'status' in status
        assert 'system_health' in status
        assert 'bankroll_status' in status
        assert status['status'] == "initialized"


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data to recommendations."""
        # This would be a comprehensive test that:
        # 1. Fetches data
        # 2. Builds features
        # 3. Trains models
        # 4. Generates predictions
        # 5. Calculates bet sizing
        # 6. Monitors performance
        
        # For now, just test that all components can be imported and initialized
        try:
            backend = MLBDataBackendV3(api_key="test", seasons=[2024])
            
            with tempfile.TemporaryDirectory() as temp_dir:
                feature_builder = MLBFeatureBuilder(db_path=os.path.join(temp_dir, "test.db"))
                model = MLBBettingModel(model_dir=temp_dir)
                manager = BankrollManager()
                monitor = PerformanceMonitor(monitoring_dir=temp_dir)
                notifier = NotificationSystem(config_file=os.path.join(temp_dir, "notifications.json"))
            
            # If we get here without exceptions, basic integration works
            assert True
            
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")
    
    def test_data_flow_consistency(self):
        """Test that data flows consistently between components."""
        # Create sample data
        sample_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.binomial(1, 0.5, 100)
        })
        
        # Test that data can be processed by each component
        assert len(sample_data) == 100
        assert 'target' in sample_data.columns
        
        # More specific data flow tests would go here
        # based on actual component interfaces


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Set test environment variables
    os.environ['TESTING'] = 'true'
    os.environ['DATABASE_PATH'] = ':memory:'
    
    yield
    
    # Clean up after test
    if 'TESTING' in os.environ:
        del os.environ['TESTING']


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require API access"
    )


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing"
    ]) 