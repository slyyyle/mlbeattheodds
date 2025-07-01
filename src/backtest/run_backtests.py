import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, log_loss

# Sports betting imports  
try:
    from sportsbet import ClassifierBettor, backtest
    from sportsbet.evaluation import expected_value
    SPORTSBET_AVAILABLE = True
except ImportError:
    logger.warning("sports-betting package not available. Some functionality will be limited.")
    SPORTSBET_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLBBacktester:
    """
    Comprehensive backtesting framework for MLB betting strategies.
    
    Supports:
    - Moneyline betting strategies
    - Strikeout prop strategies  
    - Hits/Total bases prop strategies
    """
    
    def __init__(self, features_dir: str = "data/features", results_dir: str = "data/results"):
        self.features_dir = Path(features_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        }
        
        # Backtesting parameters
        self.min_edge = 0.02  # Minimum edge required to place bet
        self.max_bet_size = 0.05  # Maximum bet size as fraction of bankroll
        self.initial_bankroll = 10000  # Starting bankroll
        
    def run_all_backtests(self):
        """
        Run backtests for all available betting strategies.
        """
        logger.info("Starting comprehensive backtesting...")
        
        results_summary = {}
        
        # Backtest moneyline strategies
        if (self.features_dir / "moneyline.parquet").exists():
            ml_results = self.backtest_moneyline()
            results_summary['moneyline'] = ml_results
        
        # Backtest strikeout prop strategies
        if (self.features_dir / "strikeout_props.parquet").exists():
            so_results = self.backtest_strikeout_props()
            results_summary['strikeout_props'] = so_results
        
        # Backtest hits/TB prop strategies
        if (self.features_dir / "hits_tb_props.parquet").exists():
            hb_results = self.backtest_hits_tb_props()
            results_summary['hits_tb_props'] = hb_results
        
        # Save overall summary
        self._save_summary_report(results_summary)
        
        logger.info("All backtesting completed!")
        return results_summary
    
    def backtest_moneyline(self):
        """
        Backtest moneyline betting strategies.
        """
        logger.info("Backtesting moneyline strategies...")
        
        # Load features
        df = pd.read_parquet(self.features_dir / "moneyline.parquet")
        
        if df.empty:
            logger.warning("No moneyline data available for backtesting")
            return None
        
        # Prepare data for modeling
        X, y, odds_data = self._prepare_moneyline_data(df)
        
        if X.empty:
            logger.warning("No valid moneyline training data")
            return None
        
        results = {}
        
        # Test each model
        for model_name, model in self.models.items():
            logger.info(f"Testing {model_name} for moneyline...")
            
            try:
                # Fit model and run backtest
                model_results = self._run_single_backtest(
                    X, y, odds_data, model, f"moneyline_{model_name}"
                )
                results[model_name] = model_results
                
            except Exception as e:
                logger.error(f"Error in {model_name} moneyline backtest: {e}")
                continue
        
        # Save results
        self._save_backtest_results(results, "moneyline")
        
        return results
    
    def backtest_strikeout_props(self):
        """
        Backtest strikeout prop betting strategies.
        """
        logger.info("Backtesting strikeout prop strategies...")
        
        # Load features
        df = pd.read_parquet(self.features_dir / "strikeout_props.parquet")
        
        if df.empty:
            logger.warning("No strikeout props data available for backtesting")
            return None
        
        # Since we don't have actual outcomes, create synthetic data for demonstration
        df = self._add_synthetic_outcomes(df, 'strikeout')
        
        # Prepare data
        X, y, odds_data = self._prepare_props_data(df, 'strikeout')
        
        if X.empty:
            logger.warning("No valid strikeout props training data")
            return None
        
        results = {}
        
        # Test models
        for model_name, model in self.models.items():
            logger.info(f"Testing {model_name} for strikeout props...")
            
            try:
                model_results = self._run_single_backtest(
                    X, y, odds_data, model, f"strikeout_{model_name}"
                )
                results[model_name] = model_results
                
            except Exception as e:
                logger.error(f"Error in {model_name} strikeout backtest: {e}")
                continue
        
        # Save results
        self._save_backtest_results(results, "strikeout_props")
        
        return results
    
    def backtest_hits_tb_props(self):
        """
        Backtest hits and total bases prop betting strategies.
        """
        logger.info("Backtesting hits/total bases prop strategies...")
        
        # Load features
        df = pd.read_parquet(self.features_dir / "hits_tb_props.parquet")
        
        if df.empty:
            logger.warning("No hits/TB props data available for backtesting")
            return None
        
        # Add synthetic outcomes for demonstration
        df = self._add_synthetic_outcomes(df, 'hits_tb')
        
        # Prepare data
        X, y, odds_data = self._prepare_props_data(df, 'hits_tb')
        
        if X.empty:
            logger.warning("No valid hits/TB props training data")
            return None
        
        results = {}
        
        # Test models
        for model_name, model in self.models.items():
            logger.info(f"Testing {model_name} for hits/TB props...")
            
            try:
                model_results = self._run_single_backtest(
                    X, y, odds_data, model, f"hits_tb_{model_name}"
                )
                results[model_name] = model_results
                
            except Exception as e:
                logger.error(f"Error in {model_name} hits/TB backtest: {e}")
                continue
        
        # Save results
        self._save_backtest_results(results, "hits_tb_props")
        
        return results
    
    def _prepare_moneyline_data(self, df: pd.DataFrame):
        """
        Prepare moneyline data for modeling.
        """
        # Select features (excluding target and identifiers)
        feature_cols = [col for col in df.columns if col not in [
            'date', 'home_team', 'away_team', 'season', 'home_win'
        ]]
        
        # Handle missing values
        X = df[feature_cols].fillna(0)
        y = df['home_win']
        
        # Create odds data for backtesting
        odds_data = df[['avg_home_odds', 'avg_away_odds']].fillna(100)  # Default odds
        
        # Filter out rows with missing target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        odds_data = odds_data[valid_mask]
        
        return X, y, odds_data
    
    def _prepare_props_data(self, df: pd.DataFrame, prop_type: str):
        """
        Prepare props data for modeling.
        """
        # Select numeric features
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and identifier columns
        exclude_cols = ['date', 'line', 'over_odds', 'under_odds', 'actual_outcome']
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
        
        X = df[feature_cols].fillna(0)
        y = df.get('actual_outcome', pd.Series([0] * len(df)))  # Use synthetic if no actual
        
        # Create odds data
        odds_data = df[['over_odds', 'under_odds']].fillna(100)
        
        # Filter valid data
        valid_mask = ~y.isna() & ~X.isna().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        odds_data = odds_data[valid_mask]
        
        return X, y, odds_data
    
    def _run_single_backtest(self, X, y, odds_data, model, strategy_name):
        """
        Run backtest for a single model/strategy combination.
        """
        if not SPORTSBET_AVAILABLE:
            return self._run_simple_backtest(X, y, odds_data, model, strategy_name)
        
        try:
            # Create bettor with the model
            bettor = ClassifierBettor(model)
            
            # Convert odds to the format expected by sports-betting
            odds_array = odds_data.values
            
            # Run backtest with time series split
            results = backtest(
                bettor=bettor,
                X=X,
                y=y,
                odds=odds_array,
                cv=TimeSeriesSplit(n_splits=5),
                return_details=True
            )
            
            # Calculate performance metrics
            total_return = results['profit'].sum() if 'profit' in results.columns else 0
            total_bets = len(results) if hasattr(results, '__len__') else 0
            win_rate = (results['profit'] > 0).mean() if 'profit' in results.columns else 0
            roi = total_return / self.initial_bankroll if total_bets > 0 else 0
            
            performance = {
                'strategy': strategy_name,
                'total_bets': total_bets,
                'total_return': total_return,
                'roi': roi,
                'win_rate': win_rate,
                'sharpe_ratio': self._calculate_sharpe_ratio(results.get('profit', [])),
                'max_drawdown': self._calculate_max_drawdown(results.get('profit', [])),
                'details': results
            }
            
            logger.info(f"{strategy_name} - ROI: {roi:.3f}, Win Rate: {win_rate:.3f}, Total Bets: {total_bets}")
            
            return performance
            
        except Exception as e:
            logger.error(f"Error in sports-betting backtest for {strategy_name}: {e}")
            return self._run_simple_backtest(X, y, odds_data, model, strategy_name)
    
    def _run_simple_backtest(self, X, y, odds_data, model, strategy_name):
        """
        Run a simplified backtest without sports-betting package.
        """
        logger.info(f"Running simplified backtest for {strategy_name}")
        
        # Simple train/test split (last 20% for testing)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        odds_train, odds_test = odds_data.iloc[:split_idx], odds_data.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit model
        model.fit(X_train_scaled, y_train)
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Simple betting strategy: bet when model probability differs significantly from market
        if odds_test.shape[1] >= 2:  # Moneyline
            market_prob = 1 / (1 + np.abs(odds_test.iloc[:, 0]) / 100)
            edge = y_pred_proba - market_prob
            
            # Place bets where edge > threshold
            bet_mask = np.abs(edge) > self.min_edge
            bets_placed = bet_mask.sum()
            
            if bets_placed > 0:
                # Calculate returns (simplified)
                bet_outcomes = y_test[bet_mask]
                bet_odds = odds_test.iloc[bet_mask, 0]
                
                # Calculate profit/loss
                profits = []
                for outcome, odd in zip(bet_outcomes, bet_odds):
                    if outcome == 1:  # Win
                        profit = np.abs(odd) / 100 if odd > 0 else 100 / np.abs(odd)
                    else:  # Loss
                        profit = -1
                    profits.append(profit)
                
                total_return = sum(profits)
                win_rate = (np.array(profits) > 0).mean()
                roi = total_return / bets_placed if bets_placed > 0 else 0
            else:
                total_return = 0
                win_rate = 0
                roi = 0
                profits = []
        else:
            bets_placed = 0
            total_return = 0
            win_rate = 0
            roi = 0
            profits = []
        
        performance = {
            'strategy': strategy_name,
            'total_bets': bets_placed,
            'total_return': total_return,
            'roi': roi,
            'win_rate': win_rate,
            'sharpe_ratio': self._calculate_sharpe_ratio(profits),
            'max_drawdown': self._calculate_max_drawdown(profits),
            'model_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        logger.info(f"{strategy_name} - ROI: {roi:.3f}, Win Rate: {win_rate:.3f}, Total Bets: {bets_placed}")
        
        return performance
    
    def _add_synthetic_outcomes(self, df: pd.DataFrame, prop_type: str):
        """
        Add synthetic outcomes for demonstration purposes.
        """
        np.random.seed(42)
        
        if prop_type == 'strikeout':
            # Simulate strikeout outcomes based on line
            df['actual_outcome'] = np.random.binomial(
                1, 0.52, size=len(df)  # Slightly over 50% to simulate over bias
            )
        elif prop_type == 'hits_tb':
            # Simulate hits/TB outcomes
            df['actual_outcome'] = np.random.binomial(
                1, 0.48, size=len(df)  # Slightly under 50% for under bias
            )
        
        return df
    
    def _calculate_sharpe_ratio(self, returns):
        """
        Calculate Sharpe ratio from returns.
        """
        if len(returns) == 0:
            return 0
        
        returns = np.array(returns)
        if returns.std() == 0:
            return 0
        
        return returns.mean() / returns.std() * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, returns):
        """
        Calculate maximum drawdown from returns.
        """
        if len(returns) == 0:
            return 0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        
        return drawdown.min()
    
    def _save_backtest_results(self, results, bet_type):
        """
        Save backtest results to files.
        """
        # Save summary
        summary_data = []
        for model_name, model_results in results.items():
            if model_results:
                summary_data.append({
                    'bet_type': bet_type,
                    'model': model_name,
                    'roi': model_results.get('roi', 0),
                    'win_rate': model_results.get('win_rate', 0),
                    'total_bets': model_results.get('total_bets', 0),
                    'total_return': model_results.get('total_return', 0),
                    'sharpe_ratio': model_results.get('sharpe_ratio', 0),
                    'max_drawdown': model_results.get('max_drawdown', 0),
                    'model_auc': model_results.get('model_auc', 0)
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = self.results_dir / f"{bet_type}_backtest.parquet"
            summary_df.to_parquet(summary_file, index=False)
            logger.info(f"Saved {bet_type} backtest results to {summary_file}")
    
    def _save_summary_report(self, results_summary):
        """
        Save overall summary report.
        """
        all_results = []
        
        for bet_type, bet_results in results_summary.items():
            if bet_results:
                for model_name, model_results in bet_results.items():
                    if model_results:
                        all_results.append({
                            'bet_type': bet_type,
                            'model': model_name,
                            **{k: v for k, v in model_results.items() if k != 'details'}
                        })
        
        if all_results:
            summary_df = pd.DataFrame(all_results)
            summary_file = self.results_dir / "backtest_summary.parquet"
            summary_df.to_parquet(summary_file, index=False)
            
            # Also save as CSV for easy viewing
            summary_df.to_csv(self.results_dir / "backtest_summary.csv", index=False)
            
            logger.info(f"Saved overall backtest summary to {summary_file}")
            
            # Log best performers
            logger.info("Best performing strategies:")
            best_roi = summary_df.loc[summary_df['roi'].idxmax()]
            logger.info(f"  Best ROI: {best_roi['bet_type']} - {best_roi['model']} ({best_roi['roi']:.3f})")
            
            if 'sharpe_ratio' in summary_df.columns:
                best_sharpe = summary_df.loc[summary_df['sharpe_ratio'].idxmax()]
                logger.info(f"  Best Sharpe: {best_sharpe['bet_type']} - {best_sharpe['model']} ({best_sharpe['sharpe_ratio']:.3f})")


def main():
    """
    Main function to run backtests.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MLB betting backtests")
    parser.add_argument("--features-dir", default="data/features",
                       help="Directory containing feature files")
    parser.add_argument("--results-dir", default="data/results",
                       help="Directory to save results")
    parser.add_argument("--strategy", choices=["all", "moneyline", "strikeout", "hits_tb"],
                       default="all", help="Strategy to backtest")
    
    args = parser.parse_args()
    
    try:
        backtester = MLBBacktester(args.features_dir, args.results_dir)
        
        if args.strategy == "all":
            results = backtester.run_all_backtests()
        elif args.strategy == "moneyline":
            results = backtester.backtest_moneyline()
        elif args.strategy == "strikeout":
            results = backtester.backtest_strikeout_props()
        elif args.strategy == "hits_tb":
            results = backtester.backtest_hits_tb_props()
        
        logger.info("Backtesting completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 