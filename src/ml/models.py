import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, roc_auc_score, log_loss, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
from scipy import stats

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLBBettingModel:
    """
    Advanced machine learning model for MLB betting predictions.
    
    Features:
    - Multiple algorithms (Logistic, RF, XGBoost, Ensemble)
    - Time series cross-validation
    - Feature importance analysis
    - Model persistence and versioning
    - Prediction confidence intervals
    - Model drift detection
    """
    
    def __init__(self, model_type: str = "ensemble", model_dir: str = "models"):
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.models = {
            'logistic': Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(f_classif, k=20)),
                ('classifier', LogisticRegression(
                    random_state=42, 
                    max_iter=2000,
                    class_weight='balanced'
                ))
            ]),
            'random_forest': Pipeline([
                ('scaler', RobustScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                ))
            ]),
            'xgboost': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                ))
            ]),
            'gradient_boosting': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42
                ))
            ])
        }
        
        # Model metadata
        self.trained_models = {}
        self.feature_names = []
        self.model_metrics = {}
        self.training_history = []
        
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   validation_split: float = 0.2,
                   hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train the betting model with comprehensive validation.
        """
        logger.info(f"Training {self.model_type} model with {len(X)} samples")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        results = {}
        
        if self.model_type == "ensemble":
            # Train all models and ensemble them
            results = self._train_ensemble(X, y, tscv, hyperparameter_tuning)
        else:
            # Train single model
            results = self._train_single_model(X, y, tscv, self.model_type, hyperparameter_tuning)
        
        # Save training metadata
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'n_samples': len(X),
            'n_features': len(X.columns),
            'results': results
        }
        self.training_history.append(training_record)
        
        # Save models and metadata
        self._save_models()
        
        return results
    
    def _train_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                       tscv: TimeSeriesSplit, hyperparameter_tuning: bool) -> Dict[str, Any]:
        """Train ensemble of multiple models."""
        logger.info("Training ensemble of models...")
        
        ensemble_results = {}
        model_predictions = {}
        
        # Train each base model
        for model_name in ['logistic', 'random_forest', 'xgboost', 'gradient_boosting']:
            logger.info(f"Training {model_name}...")
            
            try:
                model_result = self._train_single_model(X, y, tscv, model_name, hyperparameter_tuning)
                ensemble_results[model_name] = model_result
                
                # Get cross-validation predictions for ensemble training
                model_predictions[model_name] = self._get_cv_predictions(X, y, tscv, model_name)
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Train meta-learner for ensemble
        if len(model_predictions) >= 2:
            ensemble_results['ensemble'] = self._train_meta_learner(
                model_predictions, y, tscv
            )
        
        return ensemble_results
    
    def _train_single_model(self, X: pd.DataFrame, y: pd.Series, 
                           tscv: TimeSeriesSplit, model_name: str,
                           hyperparameter_tuning: bool) -> Dict[str, Any]:
        """Train a single model with comprehensive evaluation."""
        
        model = self.models[model_name]
        
        # Hyperparameter tuning if requested
        if hyperparameter_tuning:
            model = self._tune_hyperparameters(model, X, y, tscv, model_name)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc')
        cv_log_loss = cross_val_score(model, X, y, cv=tscv, scoring='neg_log_loss')
        
        # Fit final model
        model.fit(X, y)
        
        # Feature importance (if available)
        feature_importance = self._get_feature_importance(model, X.columns)
        
        # Store trained model
        self.trained_models[model_name] = model
        
        results = {
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'cv_log_loss_mean': -cv_log_loss.mean(),
            'cv_log_loss_std': cv_log_loss.std(),
            'feature_importance': feature_importance
        }
        
        logger.info(f"{model_name} - AUC: {results['cv_auc_mean']:.4f} (+/- {results['cv_auc_std']:.4f})")
        
        return results
    
    def _tune_hyperparameters(self, model: Pipeline, X: pd.DataFrame, y: pd.Series,
                             tscv: TimeSeriesSplit, model_name: str) -> Pipeline:
        """Perform hyperparameter tuning for the model."""
        logger.info(f"Tuning hyperparameters for {model_name}...")
        
        param_grids = {
            'logistic': {
                'classifier__C': [0.01, 0.1, 1.0, 10.0],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga'],
                'selector__k': [10, 15, 20, 25]
            },
            'random_forest': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [5, 10, 15, None],
                'classifier__min_samples_split': [10, 20, 50],
                'classifier__min_samples_leaf': [5, 10, 20]
            },
            'xgboost': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [3, 6, 9],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__subsample': [0.8, 0.9, 1.0]
            },
            'gradient_boosting': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [3, 6, 9],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__subsample': [0.8, 0.9, 1.0]
            }
        }
        
        if model_name in param_grids:
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name],
                cv=tscv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X, y)
            logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            return grid_search.best_estimator_
        
        return model
    
    def _get_cv_predictions(self, X: pd.DataFrame, y: pd.Series, 
                           tscv: TimeSeriesSplit, model_name: str) -> np.ndarray:
        """Get cross-validation predictions for ensemble training."""
        model = self.models[model_name]
        predictions = np.zeros(len(y))
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            
            model.fit(X_train, y_train)
            predictions[val_idx] = model.predict_proba(X_val)[:, 1]
        
        return predictions
    
    def _train_meta_learner(self, model_predictions: Dict[str, np.ndarray], 
                           y: pd.Series, tscv: TimeSeriesSplit) -> Dict[str, Any]:
        """Train meta-learner for ensemble predictions."""
        logger.info("Training ensemble meta-learner...")
        
        # Create ensemble features
        ensemble_X = pd.DataFrame(model_predictions)
        
        # Simple averaging ensemble
        ensemble_pred = ensemble_X.mean(axis=1)
        ensemble_auc = roc_auc_score(y, ensemble_pred)
        
        # Weighted ensemble using cross-validation
        weights = self._optimize_ensemble_weights(ensemble_X, y, tscv)
        weighted_pred = np.average(ensemble_X.values, axis=1, weights=weights)
        weighted_auc = roc_auc_score(y, weighted_pred)
        
        # Store ensemble configuration
        self.ensemble_weights = weights
        
        return {
            'simple_ensemble_auc': ensemble_auc,
            'weighted_ensemble_auc': weighted_auc,
            'ensemble_weights': weights.tolist()
        }
    
    def _optimize_ensemble_weights(self, ensemble_X: pd.DataFrame, 
                                  y: pd.Series, tscv: TimeSeriesSplit) -> np.ndarray:
        """Optimize ensemble weights using cross-validation."""
        from scipy.optimize import minimize
        
        def objective(weights):
            weights = weights / weights.sum()  # Normalize
            pred = np.average(ensemble_X.values, axis=1, weights=weights)
            return -roc_auc_score(y, pred)
        
        # Initial equal weights
        n_models = ensemble_X.shape[1]
        initial_weights = np.ones(n_models) / n_models
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = minimize(
            objective, 
            initial_weights, 
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        
        return result.x
    
    def _get_feature_importance(self, model: Pipeline, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from trained model."""
        try:
            if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                # Tree-based models
                importance = model.named_steps['classifier'].feature_importances_
                
                # Handle feature selection
                if 'selector' in model.named_steps:
                    selected_features = model.named_steps['selector'].get_support()
                    full_importance = np.zeros(len(feature_names))
                    full_importance[selected_features] = importance
                    importance = full_importance
                
                return dict(zip(feature_names, importance))
                
            elif hasattr(model.named_steps['classifier'], 'coef_'):
                # Linear models
                coef = model.named_steps['classifier'].coef_[0]
                
                # Handle feature selection
                if 'selector' in model.named_steps:
                    selected_features = model.named_steps['selector'].get_support()
                    full_coef = np.zeros(len(feature_names))
                    full_coef[selected_features] = coef
                    coef = full_coef
                
                return dict(zip(feature_names, np.abs(coef)))
                
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        
        return {}
    
    def predict(self, X: pd.DataFrame, return_confidence: bool = True) -> Dict[str, Any]:
        """
        Make predictions with confidence intervals.
        """
        if not self.trained_models:
            raise ValueError("No trained models available. Please train first.")
        
        predictions = {}
        
        if self.model_type == "ensemble" and len(self.trained_models) > 1:
            # Ensemble predictions
            model_preds = {}
            
            for model_name, model in self.trained_models.items():
                if model_name != 'ensemble':
                    pred_proba = model.predict_proba(X)[:, 1]
                    model_preds[model_name] = pred_proba
            
            if model_preds:
                # Simple ensemble average
                ensemble_pred = np.mean(list(model_preds.values()), axis=0)
                
                # Weighted ensemble if weights available
                if hasattr(self, 'ensemble_weights'):
                    model_names = list(model_preds.keys())
                    pred_array = np.array([model_preds[name] for name in model_names])
                    weighted_pred = np.average(pred_array, axis=0, weights=self.ensemble_weights)
                    ensemble_pred = weighted_pred
                
                predictions['probability'] = ensemble_pred
                predictions['prediction'] = (ensemble_pred > 0.5).astype(int)
                
                if return_confidence:
                    # Calculate prediction confidence based on model agreement
                    pred_std = np.std(list(model_preds.values()), axis=0)
                    confidence = 1 - (pred_std / 0.5)  # Normalize by max possible std
                    predictions['confidence'] = np.clip(confidence, 0, 1)
                    
                    # Individual model predictions for analysis
                    predictions['individual_models'] = model_preds
        
        else:
            # Single model prediction
            model_name = self.model_type if self.model_type in self.trained_models else list(self.trained_models.keys())[0]
            model = self.trained_models[model_name]
            
            pred_proba = model.predict_proba(X)[:, 1]
            predictions['probability'] = pred_proba
            predictions['prediction'] = (pred_proba > 0.5).astype(int)
            
            if return_confidence:
                # Use prediction probability as confidence (distance from 0.5)
                confidence = np.abs(pred_proba - 0.5) * 2
                predictions['confidence'] = confidence
        
        return predictions
    
    def calculate_expected_value(self, predictions: Dict[str, Any], 
                               odds: np.ndarray) -> np.ndarray:
        """
        Calculate expected value for betting decisions.
        
        EV = (probability * (odds - 1)) - (1 - probability)
        """
        prob = predictions['probability']
        
        # Convert American odds to decimal odds
        decimal_odds = np.where(odds > 0, (odds / 100) + 1, (100 / np.abs(odds)) + 1)
        
        # Calculate expected value
        ev = (prob * (decimal_odds - 1)) - (1 - prob)
        
        return ev
    
    def get_betting_recommendations(self, X: pd.DataFrame, odds: np.ndarray,
                                  min_edge: float = 0.02,
                                  min_confidence: float = 0.6) -> pd.DataFrame:
        """
        Generate betting recommendations based on model predictions.
        """
        predictions = self.predict(X, return_confidence=True)
        expected_values = self.calculate_expected_value(predictions, odds)
        
        # Create recommendations dataframe
        recommendations = pd.DataFrame({
            'probability': predictions['probability'],
            'confidence': predictions.get('confidence', 0.5),
            'odds': odds,
            'expected_value': expected_values,
            'recommendation': 'hold'
        })
        
        # Apply betting criteria
        bet_mask = (
            (expected_values > min_edge) & 
            (predictions.get('confidence', 0.5) > min_confidence)
        )
        
        recommendations.loc[bet_mask, 'recommendation'] = 'bet'
        
        # Calculate Kelly criterion bet sizing
        recommendations['kelly_fraction'] = self._calculate_kelly_sizing(
            recommendations['probability'], 
            recommendations['odds']
        )
        
        # Apply maximum bet size constraint
        recommendations['recommended_stake'] = np.minimum(
            recommendations['kelly_fraction'], 
            0.05  # Max 5% of bankroll
        )
        
        return recommendations[recommendations['recommendation'] == 'bet'].copy()
    
    def _calculate_kelly_sizing(self, probabilities: np.ndarray, odds: np.ndarray) -> np.ndarray:
        """Calculate Kelly criterion bet sizing."""
        # Convert to decimal odds
        decimal_odds = np.where(odds > 0, (odds / 100) + 1, (100 / np.abs(odds)) + 1)
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds-1, p = probability, q = 1-p
        b = decimal_odds - 1
        p = probabilities
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Only positive Kelly fractions (negative means don't bet)
        return np.maximum(kelly_fraction, 0)
    
    def detect_model_drift(self, X_new: pd.DataFrame, threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect if model performance has drifted from training data.
        """
        if not self.trained_models or not self.training_history:
            return {'drift_detected': False, 'reason': 'No baseline available'}
        
        drift_metrics = {}
        
        # Feature distribution drift
        if hasattr(self, 'training_features_stats'):
            feature_drift = self._detect_feature_drift(X_new)
            drift_metrics['feature_drift'] = feature_drift
        
        # Prediction distribution drift
        predictions = self.predict(X_new, return_confidence=False)
        pred_drift = self._detect_prediction_drift(predictions['probability'])
        drift_metrics['prediction_drift'] = pred_drift
        
        # Overall drift assessment
        drift_detected = (
            drift_metrics.get('feature_drift', {}).get('max_drift', 0) > threshold or
            drift_metrics.get('prediction_drift', {}).get('drift_score', 0) > threshold
        )
        
        return {
            'drift_detected': drift_detected,
            'drift_metrics': drift_metrics,
            'threshold': threshold
        }
    
    def _detect_feature_drift(self, X_new: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift in feature distributions."""
        if not hasattr(self, 'training_features_stats'):
            return {'error': 'No training feature statistics available'}
        
        drift_scores = {}
        
        for feature in X_new.columns:
            if feature in self.training_features_stats:
                # Kolmogorov-Smirnov test for distribution drift
                training_stats = self.training_features_stats[feature]
                
                # Compare means and stds
                new_mean = X_new[feature].mean()
                new_std = X_new[feature].std()
                
                mean_drift = abs(new_mean - training_stats['mean']) / training_stats['std']
                std_drift = abs(new_std - training_stats['std']) / training_stats['std']
                
                drift_scores[feature] = {
                    'mean_drift': mean_drift,
                    'std_drift': std_drift,
                    'combined_drift': (mean_drift + std_drift) / 2
                }
        
        max_drift = max([scores['combined_drift'] for scores in drift_scores.values()])
        
        return {
            'feature_scores': drift_scores,
            'max_drift': max_drift
        }
    
    def _detect_prediction_drift(self, new_predictions: np.ndarray) -> Dict[str, Any]:
        """Detect drift in prediction distributions."""
        if not hasattr(self, 'training_prediction_stats'):
            return {'error': 'No training prediction statistics available'}
        
        # Compare prediction distributions
        training_mean = self.training_prediction_stats['mean']
        training_std = self.training_prediction_stats['std']
        
        new_mean = new_predictions.mean()
        new_std = new_predictions.std()
        
        mean_drift = abs(new_mean - training_mean) / training_std
        std_drift = abs(new_std - training_std) / training_std
        
        return {
            'mean_drift': mean_drift,
            'std_drift': std_drift,
            'drift_score': (mean_drift + std_drift) / 2
        }
    
    def _save_models(self):
        """Save trained models and metadata to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        for model_name, model in self.trained_models.items():
            model_file = self.model_dir / f"{model_name}_{timestamp}.joblib"
            joblib.dump(model, model_file)
            logger.info(f"Saved {model_name} model to {model_file}")
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'model_metrics': self.model_metrics,
            'timestamp': timestamp
        }
        
        if hasattr(self, 'ensemble_weights'):
            metadata['ensemble_weights'] = self.ensemble_weights.tolist()
        
        metadata_file = self.model_dir / f"metadata_{timestamp}.joblib"
        joblib.dump(metadata, metadata_file)
        
        # Create symlink to latest model
        latest_dir = self.model_dir / "latest"
        latest_dir.mkdir(exist_ok=True)
        
        for model_name in self.trained_models.keys():
            model_file = self.model_dir / f"{model_name}_{timestamp}.joblib"
            latest_file = latest_dir / f"{model_name}.joblib"
            
            if latest_file.exists():
                latest_file.unlink()
            latest_file.symlink_to(model_file.relative_to(latest_dir))
    
    def load_latest_models(self) -> bool:
        """Load the latest trained models."""
        latest_dir = self.model_dir / "latest"
        
        if not latest_dir.exists():
            logger.warning("No latest models directory found")
            return False
        
        try:
            # Load models
            for model_file in latest_dir.glob("*.joblib"):
                if model_file.stem != "metadata":
                    model_name = model_file.stem
                    model = joblib.load(model_file)
                    self.trained_models[model_name] = model
                    logger.info(f"Loaded {model_name} model")
            
            # Load metadata
            metadata_file = latest_dir / "metadata.joblib"
            if metadata_file.exists():
                metadata = joblib.load(metadata_file)
                self.feature_names = metadata.get('feature_names', [])
                self.training_history = metadata.get('training_history', [])
                self.model_metrics = metadata.get('model_metrics', {})
                
                if 'ensemble_weights' in metadata:
                    self.ensemble_weights = np.array(metadata['ensemble_weights'])
            
            return len(self.trained_models) > 0
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False


class MLBModelManager:
    """
    Manager class for handling multiple MLB betting models.
    
    Features:
    - Model versioning and deployment
    - A/B testing of models
    - Performance monitoring
    - Automated retraining
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.active_models = {}
        self.performance_history = []
        
    def register_model(self, model_name: str, model: MLBBettingModel):
        """Register a new model."""
        self.models[model_name] = model
        logger.info(f"Registered model: {model_name}")
    
    def deploy_model(self, model_name: str, strategy: str = "moneyline"):
        """Deploy a model for a specific betting strategy."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        self.active_models[strategy] = model_name
        logger.info(f"Deployed {model_name} for {strategy} strategy")
    
    def get_predictions(self, strategy: str, X: pd.DataFrame, odds: np.ndarray) -> Dict[str, Any]:
        """Get predictions from the active model for a strategy."""
        if strategy not in self.active_models:
            raise ValueError(f"No active model for strategy: {strategy}")
        
        model_name = self.active_models[strategy]
        model = self.models[model_name]
        
        predictions = model.predict(X)
        recommendations = model.get_betting_recommendations(X, odds)
        
        return {
            'model_name': model_name,
            'predictions': predictions,
            'recommendations': recommendations
        }
    
    def monitor_performance(self, strategy: str, actual_outcomes: pd.Series,
                          predictions: Dict[str, Any]) -> Dict[str, float]:
        """Monitor model performance and log metrics."""
        model_name = self.active_models.get(strategy)
        if not model_name:
            return {}
        
        # Calculate performance metrics
        y_true = actual_outcomes
        y_pred_proba = predictions['predictions']['probability']
        y_pred = predictions['predictions']['prediction']
        
        metrics = {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba),
            'accuracy': (y_true == y_pred).mean(),
            'precision': ((y_pred == 1) & (y_true == 1)).sum() / (y_pred == 1).sum() if (y_pred == 1).sum() > 0 else 0,
            'recall': ((y_pred == 1) & (y_true == 1)).sum() / (y_true == 1).sum() if (y_true == 1).sum() > 0 else 0
        }
        
        # Log performance
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'strategy': strategy,
            'metrics': metrics,
            'n_samples': len(y_true)
        }
        
        self.performance_history.append(performance_record)
        
        logger.info(f"Performance for {model_name} ({strategy}): AUC={metrics['auc']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        
        return metrics
    
    def should_retrain(self, strategy: str, performance_threshold: float = 0.05) -> bool:
        """Determine if a model should be retrained based on performance degradation."""
        if not self.performance_history:
            return False
        
        # Get recent performance for the strategy
        strategy_history = [
            record for record in self.performance_history 
            if record['strategy'] == strategy
        ]
        
        if len(strategy_history) < 10:  # Need enough history
            return False
        
        # Compare recent performance to historical average
        recent_performance = np.mean([
            record['metrics']['auc'] for record in strategy_history[-5:]
        ])
        
        historical_performance = np.mean([
            record['metrics']['auc'] for record in strategy_history[:-5]
        ])
        
        performance_drop = historical_performance - recent_performance
        
        return performance_drop > performance_threshold


def main():
    """
    Example usage of the MLB betting models.
    """
    logger.info("MLB Betting Models - Example Usage")
    
    # This would typically load real feature data
    # For demo purposes, create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.binomial(1, 0.5, n_samples))
    odds = np.random.choice([-110, -120, -105, 100, 110, 120], n_samples)
    
    # Create and train model
    model = MLBBettingModel(model_type="ensemble")
    
    logger.info("Training model...")
    results = model.train_model(X, y, hyperparameter_tuning=False)  # Skip tuning for demo
    
    logger.info("Training Results:")
    for model_name, metrics in results.items():
        if isinstance(metrics, dict) and 'cv_auc_mean' in metrics:
            logger.info(f"  {model_name}: AUC = {metrics['cv_auc_mean']:.4f}")
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(X[:10])
    logger.info(f"Sample predictions: {predictions['probability'][:5]}")
    
    # Get betting recommendations
    recommendations = model.get_betting_recommendations(X[:10], odds[:10])
    logger.info(f"Generated {len(recommendations)} betting recommendations")
    
    # Model manager example
    manager = MLBModelManager()
    manager.register_model("ensemble_v1", model)
    manager.deploy_model("ensemble_v1", "moneyline")
    
    logger.info("Model training and deployment completed!")


if __name__ == "__main__":
    main() 