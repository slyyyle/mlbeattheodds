import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BankrollManager:
    """
    Advanced bankroll management system for MLB betting.
    
    Features:
    - Kelly Criterion sizing
    - Fractional Kelly for risk reduction
    - Dynamic sizing based on confidence
    - Portfolio optimization across multiple bets
    - Drawdown protection
    - Risk-adjusted returns
    """
    
    def __init__(self, initial_bankroll: float = 10000, 
                 max_bet_size: float = 0.05,
                 kelly_fraction: float = 0.25,
                 max_daily_risk: float = 0.10):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.max_bet_size = max_bet_size  # Maximum bet as fraction of bankroll
        self.kelly_fraction = kelly_fraction  # Fractional Kelly (0.25 = quarter Kelly)
        self.max_daily_risk = max_daily_risk  # Maximum daily risk exposure
        
        # Track betting history
        self.bet_history = []
        self.daily_pnl = []
        self.bankroll_history = [initial_bankroll]
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_bankroll = initial_bankroll
        
    def calculate_kelly_size(self, win_probability: float, odds: float) -> float:
        """
        Calculate Kelly Criterion bet size.
        
        Kelly formula: f = (bp - q) / b
        where:
        - b = decimal odds - 1
        - p = win probability
        - q = 1 - p (lose probability)
        """
        # Convert American odds to decimal odds
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1
        
        b = decimal_odds - 1
        p = win_probability
        q = 1 - p
        
        # Kelly fraction
        kelly_f = (b * p - q) / b
        
        # Apply fractional Kelly for risk reduction
        kelly_f *= self.kelly_fraction
        
        # Ensure non-negative
        return max(0, kelly_f)
    
    def calculate_bet_size(self, win_probability: float, odds: float, 
                          confidence: float = 1.0) -> Dict[str, float]:
        """
        Calculate optimal bet size considering multiple factors.
        
        Args:
            win_probability: Model predicted win probability
            odds: American odds
            confidence: Model confidence (0-1)
            
        Returns:
            Dictionary with bet sizing information
        """
        # Base Kelly size
        kelly_size = self.calculate_kelly_size(win_probability, odds)
        
        # Adjust for confidence
        confidence_adjusted_size = kelly_size * confidence
        
        # Apply maximum bet size constraint
        max_constrained_size = min(confidence_adjusted_size, self.max_bet_size)
        
        # Check current risk exposure
        current_risk_exposure = self._calculate_current_risk_exposure()
        available_risk = max(0, self.max_daily_risk - current_risk_exposure)
        
        # Final bet size considering risk limits
        final_size = min(max_constrained_size, available_risk)
        
        # Calculate bet amount in dollars
        bet_amount = final_size * self.current_bankroll
        
        return {
            'kelly_fraction': kelly_size,
            'confidence_adjusted_fraction': confidence_adjusted_size,
            'max_constrained_fraction': max_constrained_size,
            'final_fraction': final_size,
            'bet_amount': bet_amount,
            'current_bankroll': self.current_bankroll,
            'risk_exposure': current_risk_exposure,
            'available_risk': available_risk
        }
    
    def _calculate_current_risk_exposure(self) -> float:
        """Calculate current risk exposure from pending bets."""
        today = datetime.now().date()
        
        # Sum up risk from today's bets
        daily_risk = 0.0
        for bet in self.bet_history:
            if bet['date'].date() == today and bet['status'] == 'pending':
                daily_risk += bet['risk_amount'] / self.current_bankroll
        
        return daily_risk
    
    def place_bet(self, bet_info: Dict) -> Dict[str, any]:
        """
        Place a bet and update bankroll tracking.
        
        Args:
            bet_info: Dictionary containing bet details
            
        Returns:
            Updated bet information with sizing
        """
        # Calculate bet size
        sizing = self.calculate_bet_size(
            bet_info['win_probability'],
            bet_info['odds'],
            bet_info.get('confidence', 1.0)
        )
        
        # Create bet record
        bet_record = {
            'id': len(self.bet_history) + 1,
            'date': datetime.now(),
            'game_id': bet_info.get('game_id', 'unknown'),
            'bet_type': bet_info.get('bet_type', 'unknown'),
            'win_probability': bet_info['win_probability'],
            'odds': bet_info['odds'],
            'confidence': bet_info.get('confidence', 1.0),
            'bet_amount': sizing['bet_amount'],
            'bet_fraction': sizing['final_fraction'],
            'risk_amount': sizing['bet_amount'],
            'potential_profit': self._calculate_potential_profit(sizing['bet_amount'], bet_info['odds']),
            'status': 'pending',
            'result': None,
            'pnl': 0.0
        }
        
        # Add to history
        self.bet_history.append(bet_record)
        
        logger.info(f"Placed bet: ${sizing['bet_amount']:.2f} ({sizing['final_fraction']:.1%} of bankroll)")
        
        return bet_record
    
    def settle_bet(self, bet_id: int, won: bool) -> Dict[str, float]:
        """
        Settle a bet and update bankroll.
        
        Args:
            bet_id: ID of the bet to settle
            won: Whether the bet won
            
        Returns:
            Settlement information
        """
        # Find the bet
        bet = None
        for b in self.bet_history:
            if b['id'] == bet_id:
                bet = b
                break
        
        if not bet:
            raise ValueError(f"Bet with ID {bet_id} not found")
        
        if bet['status'] != 'pending':
            raise ValueError(f"Bet {bet_id} is already settled")
        
        # Calculate P&L
        if won:
            pnl = bet['potential_profit']
        else:
            pnl = -bet['bet_amount']
        
        # Update bet record
        bet['status'] = 'settled'
        bet['result'] = 'win' if won else 'loss'
        bet['pnl'] = pnl
        
        # Update bankroll
        self.current_bankroll += pnl
        self.bankroll_history.append(self.current_bankroll)
        
        # Update drawdown metrics
        self._update_drawdown_metrics()
        
        # Track daily P&L
        self.daily_pnl.append({
            'date': datetime.now().date(),
            'pnl': pnl,
            'bankroll': self.current_bankroll
        })
        
        logger.info(f"Settled bet {bet_id}: {'WIN' if won else 'LOSS'} - P&L: ${pnl:.2f}")
        logger.info(f"Current bankroll: ${self.current_bankroll:.2f}")
        
        return {
            'bet_id': bet_id,
            'result': 'win' if won else 'loss',
            'pnl': pnl,
            'new_bankroll': self.current_bankroll,
            'roi': (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll
        }
    
    def _calculate_potential_profit(self, bet_amount: float, odds: float) -> float:
        """Calculate potential profit from a bet."""
        if odds > 0:
            return bet_amount * (odds / 100)
        else:
            return bet_amount * (100 / abs(odds))
    
    def _update_drawdown_metrics(self):
        """Update drawdown tracking metrics."""
        if self.current_bankroll > self.peak_bankroll:
            self.peak_bankroll = self.current_bankroll
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if len(self.bet_history) == 0:
            return {}
        
        settled_bets = [bet for bet in self.bet_history if bet['status'] == 'settled']
        
        if len(settled_bets) == 0:
            return {}
        
        # Basic metrics
        total_bets = len(settled_bets)
        winning_bets = len([bet for bet in settled_bets if bet['result'] == 'win'])
        win_rate = winning_bets / total_bets
        
        # P&L metrics
        total_pnl = sum(bet['pnl'] for bet in settled_bets)
        total_wagered = sum(bet['bet_amount'] for bet in settled_bets)
        roi = (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll
        
        # Risk metrics
        returns = np.array([bet['pnl'] / bet['bet_amount'] for bet in settled_bets])
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        
        # Streak analysis
        win_streak, loss_streak = self._calculate_streaks(settled_bets)
        
        return {
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_wagered': total_wagered,
            'roi': roi,
            'current_bankroll': self.current_bankroll,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'longest_win_streak': win_streak,
            'longest_loss_streak': loss_streak,
            'avg_bet_size': total_wagered / total_bets,
            'avg_win': np.mean([bet['pnl'] for bet in settled_bets if bet['result'] == 'win']) if winning_bets > 0 else 0,
            'avg_loss': np.mean([bet['pnl'] for bet in settled_bets if bet['result'] == 'loss']) if (total_bets - winning_bets) > 0 else 0
        }
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for the betting strategy."""
        if len(returns) < 2:
            return 0.0
        
        # Annualize returns (assuming daily betting)
        annual_return = np.mean(returns) * 365
        annual_volatility = np.std(returns) * np.sqrt(365)
        
        if annual_volatility == 0:
            return 0.0
        
        return (annual_return - risk_free_rate) / annual_volatility
    
    def _calculate_streaks(self, settled_bets: List[Dict]) -> Tuple[int, int]:
        """Calculate longest winning and losing streaks."""
        if not settled_bets:
            return 0, 0
        
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for bet in settled_bets:
            if bet['result'] == 'win':
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        return max_win_streak, max_loss_streak
    
    def optimize_portfolio(self, bet_opportunities: List[Dict]) -> List[Dict]:
        """
        Optimize bet sizing across multiple simultaneous opportunities.
        
        Uses portfolio optimization to maximize expected return while
        controlling for correlation and risk.
        """
        if len(bet_opportunities) <= 1:
            # Single bet - use standard sizing
            if bet_opportunities:
                return [self.calculate_bet_size(
                    bet_opportunities[0]['win_probability'],
                    bet_opportunities[0]['odds'],
                    bet_opportunities[0].get('confidence', 1.0)
                )]
            return []
        
        # Extract bet characteristics
        probabilities = np.array([bet['win_probability'] for bet in bet_opportunities])
        odds = np.array([bet['odds'] for bet in bet_opportunities])
        confidences = np.array([bet.get('confidence', 1.0) for bet in bet_opportunities])
        
        # Calculate expected returns and Kelly fractions
        expected_returns = []
        kelly_fractions = []
        
        for i, bet in enumerate(bet_opportunities):
            kelly_f = self.calculate_kelly_size(probabilities[i], odds[i])
            kelly_fractions.append(kelly_f)
            
            # Expected return calculation
            decimal_odds = (odds[i] / 100) + 1 if odds[i] > 0 else (100 / abs(odds[i])) + 1
            expected_return = probabilities[i] * (decimal_odds - 1) - (1 - probabilities[i])
            expected_returns.append(expected_return)
        
        expected_returns = np.array(expected_returns)
        kelly_fractions = np.array(kelly_fractions)
        
        # Portfolio optimization
        optimized_fractions = self._optimize_portfolio_weights(
            expected_returns, kelly_fractions, confidences
        )
        
        # Create sizing recommendations
        recommendations = []
        for i, bet in enumerate(bet_opportunities):
            sizing = {
                'kelly_fraction': kelly_fractions[i],
                'optimized_fraction': optimized_fractions[i],
                'final_fraction': min(optimized_fractions[i], self.max_bet_size),
                'bet_amount': min(optimized_fractions[i], self.max_bet_size) * self.current_bankroll,
                'expected_return': expected_returns[i],
                'confidence': confidences[i]
            }
            recommendations.append(sizing)
        
        return recommendations
    
    def _optimize_portfolio_weights(self, expected_returns: np.ndarray, 
                                   kelly_fractions: np.ndarray,
                                   confidences: np.ndarray) -> np.ndarray:
        """Optimize portfolio weights using mean-variance optimization."""
        n_bets = len(expected_returns)
        
        # Objective function: maximize expected return adjusted for confidence
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns * confidences)
            return -portfolio_return  # Minimize negative return
        
        # Constraints
        constraints = [
            {'type': 'ineq', 'fun': lambda w: self.max_daily_risk - np.sum(w)},  # Total risk constraint
            {'type': 'ineq', 'fun': lambda w: w}  # Non-negative weights
        ]
        
        # Bounds: each bet between 0 and max_bet_size
        bounds = [(0, self.max_bet_size) for _ in range(n_bets)]
        
        # Initial guess: scaled Kelly fractions
        initial_weights = kelly_fractions * self.kelly_fraction
        initial_weights = np.minimum(initial_weights, self.max_bet_size)
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return result.x
        else:
            # Fallback to scaled Kelly fractions
            logger.warning("Portfolio optimization failed, using scaled Kelly fractions")
            return np.minimum(kelly_fractions * self.kelly_fraction, self.max_bet_size)
    
    def get_risk_assessment(self) -> Dict[str, any]:
        """Get current risk assessment and recommendations."""
        current_risk = self._calculate_current_risk_exposure()
        
        # Risk level assessment
        if current_risk > 0.08:
            risk_level = "HIGH"
            recommendation = "Reduce position sizes or avoid new bets"
        elif current_risk > 0.05:
            risk_level = "MEDIUM"
            recommendation = "Monitor closely, consider smaller positions"
        else:
            risk_level = "LOW"
            recommendation = "Normal betting operations"
        
        # Drawdown assessment
        if self.current_drawdown > 0.15:
            drawdown_level = "SEVERE"
            drawdown_recommendation = "Consider stopping betting until recovery"
        elif self.current_drawdown > 0.10:
            drawdown_level = "HIGH"
            drawdown_recommendation = "Reduce bet sizes significantly"
        elif self.current_drawdown > 0.05:
            drawdown_level = "MODERATE"
            drawdown_recommendation = "Reduce bet sizes moderately"
        else:
            drawdown_level = "LOW"
            drawdown_recommendation = "Normal operations"
        
        return {
            'current_risk_exposure': current_risk,
            'risk_level': risk_level,
            'risk_recommendation': recommendation,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'drawdown_level': drawdown_level,
            'drawdown_recommendation': drawdown_recommendation,
            'available_risk_capacity': max(0, self.max_daily_risk - current_risk),
            'bankroll_health': self.current_bankroll / self.initial_bankroll
        }


def main():
    """Example usage of the bankroll manager."""
    logger.info("Bankroll Manager - Example Usage")
    
    # Initialize bankroll manager
    manager = BankrollManager(initial_bankroll=10000)
    
    # Example bet opportunities
    bet_opportunities = [
        {
            'game_id': 'NYY_vs_BOS',
            'bet_type': 'moneyline',
            'win_probability': 0.55,
            'odds': -110,
            'confidence': 0.8
        },
        {
            'game_id': 'LAD_vs_SF',
            'bet_type': 'moneyline',
            'win_probability': 0.60,
            'odds': 120,
            'confidence': 0.7
        }
    ]
    
    # Place bets
    for bet_info in bet_opportunities:
        bet_record = manager.place_bet(bet_info)
        logger.info(f"Placed bet: {bet_record['game_id']} - ${bet_record['bet_amount']:.2f}")
    
    # Simulate settling bets
    manager.settle_bet(1, won=True)
    manager.settle_bet(2, won=False)
    
    # Get performance metrics
    metrics = manager.get_performance_metrics()
    logger.info("Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Risk assessment
    risk_assessment = manager.get_risk_assessment()
    logger.info(f"Risk Level: {risk_assessment['risk_level']}")
    logger.info(f"Recommendation: {risk_assessment['risk_recommendation']}")


if __name__ == "__main__":
    main() 