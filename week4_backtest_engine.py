"""
Week 4 Improved Backtest Engine
Integrates all improvements based on Week 3 analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Import Week 3 modules
sys.path.append('/mnt/user-data/outputs')
try:
    from portfolio import Portfolio
    from risk_manager import RiskManager
    from metrics import PerformanceMetrics
except ImportError:
    print("⚠️ Week 3 modules not found in /mnt/user-data/outputs")
    print("   Please ensure portfolio.py, risk_manager.py, metrics.py are available")
    Portfolio = None
    RiskManager = None
    PerformanceMetrics = None

# Import Week 4 improvements
from market_regime import MarketRegimeDetector
from improved_llm_strategy import ImprovedLLMStrategy


class Week4BacktestEngine:
    """
    Improved backtesting engine with:
    1. Market regime detection
    2. Meta-strategy rules
    3. Better LLM integration
    4. Performance tracking
    """
    
    def __init__(self, 
                 initial_capital=10000,
                 fee_rate=0.001,
                 slippage_pct=0.001,
                 risk_per_trade=0.02,
                 stop_loss_pct=0.02,
                 api_key=None,
                 trading_mode='BALANCED'):  # NEW: Control trade frequency
        
        # Core components
        if Portfolio and RiskManager and PerformanceMetrics:
            self.portfolio = Portfolio(initial_capital, fee_rate)
            self.risk_manager = RiskManager(
                risk_per_trade_pct=risk_per_trade,
                stop_loss_pct=stop_loss_pct
            )
            self.metrics = PerformanceMetrics()
        else:
            raise ImportError("Week 3 modules required. Run Week 3 first!")
        
        # Week 4 improvements
        self.regime_detector = MarketRegimeDetector()
        
        # Import adjustable strategy
        try:
            from improved_llm_strategy_adjustable import ImprovedLLMStrategy as AdjustableStrategy
            self.llm_strategy = AdjustableStrategy(api_key=api_key, trading_mode=trading_mode)
            print(f"✅ Using adjustable strategy in {trading_mode} mode")
        except ImportError:
            # Fallback to original
            self.llm_strategy = ImprovedLLMStrategy(api_key=api_key)
            print("⚠️ Using original strategy (adjustable version not found)")
        
        # Configuration
        self.slippage_pct = slippage_pct
        self.initial_capital = initial_capital
        self.trading_mode = trading_mode
        
        # Tracking
        self.trades = []
        self.regime_history = []
        self.signal_history = []
        
    def run(self, data, symbol='BTC/USDT', strategy_type='improved_llm'):
        """
        Run backtest with Week 4 improvements
        
        Args:
            data: DataFrame with OHLCV and indicators
            symbol: Trading symbol
            strategy_type: 'improved_llm', 'rsi', or 'buy_hold'
            
        Returns:
            dict: Complete results
        """
        print(f"\n{'='*70}")
        print(f"WEEK 4 BACKTEST - {strategy_type.upper()}")
        print(f"{'='*70}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Data Points: {len(data)}")
        print(f"Date Range: {data.index[0]} to {data.index[-1]}")
        print(f"{'='*70}\n")
        
        # Need at least 100 bars for regime detection
        if len(data) < 100:
            raise ValueError("Need at least 100 data points for regime detection")
        
        # Track progress
        total_signals = 0
        signals_by_type = {'LONG': 0, 'SHORT': 0, 'HOLD': 0}
        regime_blocks = {'LONG': 0, 'SHORT': 0}
        meta_blocks = 0
        
        # Iterate through data
        for i in range(100, len(data)):  # Start after 100 bars
            current_data = data.iloc[:i+1].copy()  # Make copy to avoid issues
            current_bar = data.iloc[i]
            timestamp = data.index[i]
            
            # 1. Detect market regime
            try:
                regime = self.regime_detector.detect_regime(current_data)
            except Exception as e:
                print(f"⚠️ Regime detection failed at {timestamp}: {e}")
                continue
                
            self.regime_history.append({
                'timestamp': timestamp,
                'regime': regime['regime'],
                'confidence': regime['confidence']
            })
            
            # 2. Generate signal
            try:
                signal = self.llm_strategy.generate_signal_with_context(
                    df=current_data,
                    regime_info=regime,
                    index=-1
                )
            except Exception as e:
                print(f"⚠️ Signal generation failed at {timestamp}: {e}")
                signal = {'signal': 'HOLD', 'confidence': 0, 'reasoning': f'Error: {e}', 'meta_override': True}
            
            # Track signal
            total_signals += 1
            signals_by_type[signal['signal']] += 1
            
            if signal.get('meta_override'):
                meta_blocks += 1
            
            self.signal_history.append({
                'timestamp': timestamp,
                'signal': signal['signal'],
                'confidence': signal['confidence'],
                'reasoning': signal['reasoning']
            })
            
            # 3. Execute trade if signal is LONG
            if signal['signal'] == 'LONG':
                self._execute_long_entry(
                    data=current_data,
                    timestamp=timestamp,
                    signal=signal,
                    regime=regime
                )
            
            # 4. Check exits on existing positions
            self._check_exits(current_data, timestamp)
            
            # 5. Record equity (Week 3 uses 'current_prices' parameter)
            self.portfolio.record_equity(
                timestamp=timestamp,
                current_prices={symbol: current_bar['close']}
            )
            
            # Progress update every 100 bars
            if i % 100 == 0:
                print(f"Progress: {i}/{len(data)} ({i/len(data)*100:.1f}%) | "
                      f"Trades: {len(self.trades)} | "
                      f"Regime: {regime['regime']}")
        
        # Print signal summary
        print(f"\n{'='*70}")
        print("SIGNAL SUMMARY")
        print(f"{'='*70}")
        print(f"Total Signals: {total_signals}")
        print(f"  LONG:  {signals_by_type['LONG']} ({signals_by_type['LONG']/total_signals*100:.1f}%)")
        print(f"  SHORT: {signals_by_type['SHORT']} ({signals_by_type['SHORT']/total_signals*100:.1f}%)")
        print(f"  HOLD:  {signals_by_type['HOLD']} ({signals_by_type['HOLD']/total_signals*100:.1f}%)")
        print(f"\nMeta-Strategy Blocks: {meta_blocks} ({meta_blocks/total_signals*100:.1f}%)")
        print(f"{'='*70}\n")
        
        # Calculate results
        return self._calculate_results(data, symbol)
    
    def _execute_long_entry(self, data, timestamp, signal, regime):
        """Execute long entry with Week 4 improvements"""
        current_price = data['close'].iloc[-1]
        
        # Check if we can buy
        if not self.portfolio.can_buy('BTC/USDT', current_price, 0.001):
            return
        
        # Calculate position size with regime adjustment
        portfolio_value = self.portfolio.get_total_value({'BTC/USDT': current_price})
        
        base_size = self.risk_manager.calculate_position_size_fixed_pct(
            portfolio_value=portfolio_value,
            price=current_price
        )
        
        # Adjust for regime (Week 4 improvement!)
        regime_adj = signal.get('confidence', 100) / 100.0
        quantity = base_size * regime_adj
        
        # Calculate stops
        stop_price = current_price * (1 - self.risk_manager.stop_loss_pct)
        take_profit_price = current_price * (1 + self.risk_manager.stop_loss_pct * 2)
        
        # Execute with slippage
        entry_price = current_price * (1 + self.slippage_pct)
        
        # Buy
        trade_record = self.portfolio.buy(
            symbol='BTC/USDT',
            price=entry_price,
            quantity=quantity,
            timestamp=timestamp
        )
        
        if trade_record:
            # Add metadata
            trade_record.update({
                'stop_price': stop_price,
                'take_profit_price': take_profit_price,
                'entry_regime': regime['regime'],
                'entry_confidence': signal['confidence'],
                'entry_reasoning': signal['reasoning'],
                'closed': False
            })
            self.trades.append(trade_record)
    
    def _check_exits(self, data, timestamp):
        """Check stop-loss and take-profit on open positions"""
        current_price = data['close'].iloc[-1]
        
        for trade in self.trades:
            if trade['action'] == 'BUY' and not trade.get('closed', False):
                # Check stop-loss
                if current_price <= trade['stop_price']:
                    self._close_position(trade, current_price, timestamp, 'STOP_LOSS')
                # Check take-profit
                elif current_price >= trade['take_profit_price']:
                    self._close_position(trade, current_price, timestamp, 'TAKE_PROFIT')
    
    def _close_position(self, trade, price, timestamp, reason):
        """Close position and update LLM strategy"""
        # Apply slippage
        exit_price = price * (1 - self.slippage_pct)
        
        # Sell
        sell_record = self.portfolio.sell(
            symbol='BTC/USDT',
            price=exit_price,
            quantity=trade['quantity'],
            timestamp=timestamp
        )
        
        if sell_record:
            # Mark trade as closed
            trade['closed'] = True
            trade['exit_timestamp'] = timestamp
            trade['exit_reason'] = reason
            trade['pnl'] = sell_record['pnl']
            
            # Update LLM strategy (meta-learning!)
            self.llm_strategy.update_trade_result(timestamp, sell_record['pnl'])
            
            # Record as separate trade
            self.trades.append(sell_record)
    
    def _calculate_results(self, data, symbol):
        """Calculate comprehensive results"""
        # Get trades DataFrame
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # Get equity curve from portfolio (returns DataFrame)
        equity_df = self.portfolio.get_equity_dataframe()
        
        # Convert to list of values for metrics calculation
        if not equity_df.empty:
            equity_curve = equity_df['total_value'].tolist()
        else:
            equity_curve = [self.initial_capital]
        
        # Calculate metrics
        if not trades_df.empty and len(trades_df) > 0:
            final_capital = self.portfolio.get_total_value({symbol: data['close'].iloc[-1]})
            
            metrics = self.metrics.calculate_comprehensive_metrics(
                initial_capital=self.initial_capital,
                final_capital=final_capital,
                equity_curve=equity_curve,  # List[float]
                trades=self.trades  # List[Dict]
            )
        else:
            metrics = {
                'total_return_pct': -100,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown_pct': 100,
                'sharpe_ratio': -999,
                'sortino_ratio': -999,
                'calmar_ratio': -999,
                'final_capital': 0,
                'total_fees': 0
            }
        
        return {
            'metrics': metrics,
            'trades': trades_df,
            'equity_curve': equity_df,  # DataFrame for plotting
            'regime_history': pd.DataFrame(self.regime_history),
            'signal_history': pd.DataFrame(self.signal_history)
        }
    
    def print_results(self, results):
        """Print formatted results"""
        metrics = results['metrics']
        
        print("\n" + "="*70)
        print("PERFORMANCE METRICS REPORT")
        print("="*70)
        
        print(f"\n💰 CAPITAL METRICS:")
        print(f"Initial Capital:  ${self.initial_capital:,.2f}")
        print(f"Final Capital:    ${metrics.get('final_capital', 0):,.2f}")
        print(f"Total Return:     {metrics.get('total_return_pct', 0):+.2f}%")
        print(f"Total PnL:        ${metrics.get('final_capital', 0) - self.initial_capital:+,.2f}")
        
        print(f"\n📈 TRADE METRICS:")
        print(f"Total Trades:     {metrics.get('total_trades', 0)}")
        print(f"Winning Trades:   {metrics.get('winning_trades', 0)}")
        print(f"Losing Trades:    {metrics.get('losing_trades', 0)}")
        print(f"Win Rate:         {metrics.get('win_rate', 0):.2f}%")
        
        print(f"\n💵 PROFIT METRICS:")
        print(f"Profit Factor:    {metrics.get('profit_factor', 0):.2f}")
        print(f"Avg Trade PnL:    ${metrics.get('avg_trade_pnl', 0):,.2f}")
        
        print(f"\n⚠️  RISK METRICS:")
        print(f"Max Drawdown:     {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Sortino Ratio:    {metrics.get('sortino_ratio', 0):.2f}")
        
        print(f"\n💸 COSTS:")
        print(f"Total Fees:       ${metrics.get('total_fees', 0):,.2f}")
        
        print("\n" + "="*70)


# Module test
if __name__ == '__main__':
    print("=" * 70)
    print("WEEK 4 IMPROVED BACKTEST ENGINE")
    print("=" * 70)
    print("\n✅ Module loaded successfully!")
    print("\nKey Improvements:")
    print("  1. Market regime detection (prevents Week 3 mistakes)")
    print("  2. Meta-strategy rules (cooldown after losses)")
    print("  3. Better LLM integration (rich context)")
    print("  4. Performance tracking (learn from mistakes)")
    print("  5. Regime-adjusted position sizing")
    print("\nUsage:")
    print("  engine = Week4BacktestEngine(initial_capital=10000)")
    print("  results = engine.run(data)")
    print("  engine.print_results(results)")
    print("\n" + "=" * 70)
