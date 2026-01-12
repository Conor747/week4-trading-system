"""
Week 3: Performance Metrics Module
Calculates key performance indicators for trading strategies
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for trading strategies
    """
    
    @staticmethod
    def calculate_total_return(initial_capital: float, final_capital: float) -> float:
        """
        Calculate total return percentage
        
        Args:
            initial_capital: Starting capital
            final_capital: Ending capital
            
        Returns:
            Total return as percentage
        """
        return (final_capital - initial_capital) / initial_capital * 100
    
    @staticmethod
    def calculate_win_rate(trades: List[Dict]) -> float:
        """
        Calculate percentage of winning trades
        
        Args:
            trades: List of trade dictionaries with 'pnl' field
            
        Returns:
            Win rate as percentage (0-100)
        """
        if not trades:
            return 0.0
        
        # Filter sell trades (which have PnL)
        sell_trades = [t for t in trades if t.get('action') == 'SELL']
        if not sell_trades:
            return 0.0
        
        winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        return len(winning_trades) / len(sell_trades) * 100
    
    @staticmethod
    def calculate_profit_factor(trades: List[Dict]) -> float:
        """
        Calculate profit factor (gross profit / gross loss)
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Profit factor (higher is better)
        """
        sell_trades = [t for t in trades if t.get('action') == 'SELL']
        if not sell_trades:
            return 0.0
        
        gross_profit = sum(t.get('pnl', 0) for t in sell_trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in sell_trades if t.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> tuple[float, int, int]:
        """
        Calculate maximum drawdown
        
        Args:
            equity_curve: List of portfolio values over time
            
        Returns:
            (max_drawdown_pct, peak_idx, trough_idx)
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0, 0, 0
        
        peak = equity_curve[0]
        peak_idx = 0
        max_dd = 0.0
        max_dd_peak_idx = 0
        max_dd_trough_idx = 0
        
        for i, value in enumerate(equity_curve):
            if value > peak:
                peak = value
                peak_idx = i
            
            drawdown = (peak - value) / peak * 100
            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_peak_idx = peak_idx
                max_dd_trough_idx = i
        
        return max_dd, max_dd_peak_idx, max_dd_trough_idx
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: List[float],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 365
    ) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted return)
        
        Args:
            returns: List of period returns (daily/hourly)
            risk_free_rate: Annual risk-free rate (default 2%)
            periods_per_year: Number of periods per year (365 for daily, 8760 for hourly)
            
        Returns:
            Sharpe ratio
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        
        # Calculate excess returns
        risk_free_per_period = risk_free_rate / periods_per_year
        excess_returns = returns_array - risk_free_per_period
        
        # Calculate Sharpe ratio
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: List[float],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 365
    ) -> float:
        """
        Calculate Sortino ratio (downside risk-adjusted return)
        
        Args:
            returns: List of period returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year
            
        Returns:
            Sortino ratio
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        risk_free_per_period = risk_free_rate / periods_per_year
        excess_returns = returns_array - risk_free_per_period
        
        # Only consider negative returns for downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        downside_std = np.std(downside_returns)
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)
        
        return sortino
    
    @staticmethod
    def calculate_calmar_ratio(total_return_pct: float, max_drawdown_pct: float) -> float:
        """
        Calculate Calmar ratio (return / max drawdown)
        
        Args:
            total_return_pct: Total return percentage
            max_drawdown_pct: Maximum drawdown percentage
            
        Returns:
            Calmar ratio
        """
        if max_drawdown_pct == 0:
            return 0.0
        return total_return_pct / max_drawdown_pct
    
    @staticmethod
    def calculate_average_trade_metrics(trades: List[Dict]) -> Dict:
        """
        Calculate average trade statistics
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dict with average trade metrics
        """
        sell_trades = [t for t in trades if t.get('action') == 'SELL']
        if not sell_trades:
            return {
                'avg_trade_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_trade_duration': 0.0
            }
        
        winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('pnl', 0) < 0]
        
        return {
            'avg_trade_pnl': np.mean([t.get('pnl', 0) for t in sell_trades]),
            'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0.0,
            'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0.0,
            'largest_win': max([t['pnl'] for t in winning_trades]) if winning_trades else 0.0,
            'largest_loss': min([t['pnl'] for t in losing_trades]) if losing_trades else 0.0,
        }
    
    @staticmethod
    def calculate_consecutive_wins_losses(trades: List[Dict]) -> Dict:
        """
        Calculate max consecutive wins and losses
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dict with consecutive stats
        """
        sell_trades = [t for t in trades if t.get('action') == 'SELL']
        if not sell_trades:
            return {
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'current_streak': 0
            }
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in sell_trades:
            if trade.get('pnl', 0) > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        # Current streak
        if sell_trades[-1].get('pnl', 0) > 0:
            current_streak = current_wins
        else:
            current_streak = -current_losses
        
        return {
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses,
            'current_streak': current_streak
        }
    
    @staticmethod
    def calculate_comprehensive_metrics(
        initial_capital: float,
        final_capital: float,
        equity_curve: List[float],
        trades: List[Dict],
        returns: List[float] = None
    ) -> Dict:
        """
        Calculate all performance metrics
        
        Args:
            initial_capital: Starting capital
            final_capital: Ending capital
            equity_curve: List of portfolio values
            trades: List of trade dictionaries
            returns: List of returns (optional, will calculate if not provided)
            
        Returns:
            Dict with all metrics
        """
        # Calculate returns if not provided
        if returns is None and len(equity_curve) > 1:
            returns = [
                (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                for i in range(1, len(equity_curve))
            ]
        
        # Basic metrics
        total_return = PerformanceMetrics.calculate_total_return(initial_capital, final_capital)
        win_rate = PerformanceMetrics.calculate_win_rate(trades)
        profit_factor = PerformanceMetrics.calculate_profit_factor(trades)
        
        # Risk metrics
        max_dd, peak_idx, trough_idx = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns) if returns else 0.0
        sortino = PerformanceMetrics.calculate_sortino_ratio(returns) if returns else 0.0
        calmar = PerformanceMetrics.calculate_calmar_ratio(total_return, max_dd)
        
        # Trade metrics
        trade_metrics = PerformanceMetrics.calculate_average_trade_metrics(trades)
        consecutive_metrics = PerformanceMetrics.calculate_consecutive_wins_losses(trades)
        
        # Count trades
        sell_trades = [t for t in trades if t.get('action') == 'SELL']
        winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('pnl', 0) < 0]
        
        return {
            # Capital metrics
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'total_pnl': final_capital - initial_capital,
            
            # Trade metrics
            'total_trades': len(sell_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            
            # Profit metrics
            'profit_factor': profit_factor,
            'avg_trade_pnl': trade_metrics['avg_trade_pnl'],
            'avg_win': trade_metrics['avg_win'],
            'avg_loss': trade_metrics['avg_loss'],
            'largest_win': trade_metrics['largest_win'],
            'largest_loss': trade_metrics['largest_loss'],
            
            # Risk metrics
            'max_drawdown_pct': max_dd,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            
            # Consecutive trades
            'max_consecutive_wins': consecutive_metrics['max_consecutive_wins'],
            'max_consecutive_losses': consecutive_metrics['max_consecutive_losses'],
            'current_streak': consecutive_metrics['current_streak'],
            
            # Total fees
            'total_fees': sum(t.get('fee', 0) for t in trades)
        }
    
    @staticmethod
    def print_metrics_report(metrics: Dict):
        """
        Print formatted metrics report
        
        Args:
            metrics: Dict from calculate_comprehensive_metrics()
        """
        print("\n" + "="*70)
        print("PERFORMANCE METRICS REPORT")
        print("="*70)
        
        print("\n📊 CAPITAL METRICS:")
        print(f"Initial Capital:     ${metrics['initial_capital']:,.2f}")
        print(f"Final Capital:       ${metrics['final_capital']:,.2f}")
        print(f"Total Return:        {metrics['total_return_pct']:+.2f}%")
        print(f"Total PnL:           ${metrics['total_pnl']:+,.2f}")
        
        print("\n📈 TRADE METRICS:")
        print(f"Total Trades:        {metrics['total_trades']}")
        print(f"Winning Trades:      {metrics['winning_trades']}")
        print(f"Losing Trades:       {metrics['losing_trades']}")
        print(f"Win Rate:            {metrics['win_rate']:.2f}%")
        
        print("\n💰 PROFIT METRICS:")
        print(f"Profit Factor:       {metrics['profit_factor']:.2f}")
        print(f"Avg Trade PnL:       ${metrics['avg_trade_pnl']:+,.2f}")
        print(f"Avg Win:             ${metrics['avg_win']:,.2f}")
        print(f"Avg Loss:            ${metrics['avg_loss']:,.2f}")
        print(f"Largest Win:         ${metrics['largest_win']:,.2f}")
        print(f"Largest Loss:        ${metrics['largest_loss']:,.2f}")
        
        print("\n⚠️  RISK METRICS:")
        print(f"Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:       {metrics['sortino_ratio']:.2f}")
        print(f"Calmar Ratio:        {metrics['calmar_ratio']:.2f}")
        
        print("\n🔄 CONSECUTIVE TRADES:")
        print(f"Max Consecutive Wins:   {metrics['max_consecutive_wins']}")
        print(f"Max Consecutive Losses: {metrics['max_consecutive_losses']}")
        print(f"Current Streak:         {metrics['current_streak']:+d}")
        
        print("\n💸 COSTS:")
        print(f"Total Fees:          ${metrics['total_fees']:,.2f}")
        
        print("="*70)


# Example usage
if __name__ == "__main__":
    # Sample data
    equity_curve = [10000, 10200, 10150, 10400, 10300, 10600, 10500, 10800]
    
    trades = [
        {'action': 'SELL', 'pnl': 200, 'fee': 10},
        {'action': 'SELL', 'pnl': -50, 'fee': 10},
        {'action': 'SELL', 'pnl': 250, 'fee': 10},
        {'action': 'SELL', 'pnl': -100, 'fee': 10},
        {'action': 'SELL', 'pnl': 300, 'fee': 10},
    ]
    
    # Calculate metrics
    metrics = PerformanceMetrics.calculate_comprehensive_metrics(
        initial_capital=10000,
        final_capital=10800,
        equity_curve=equity_curve,
        trades=trades
    )
    
    # Print report
    PerformanceMetrics.print_metrics_report(metrics)
    
    print("\n✅ Metrics module loaded successfully!")
