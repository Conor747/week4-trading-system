"""
Week 3: Portfolio Management Module
Tracks portfolio state, positions, and equity
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Portfolio:
    """
    Manages portfolio state including cash, positions, and equity tracking
    """
    
    def __init__(self, initial_capital: float = 10000.0, fee_rate: float = 0.001):
        """
        Initialize portfolio
        
        Args:
            initial_capital: Starting capital in USD
            fee_rate: Trading fee as decimal (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.fee_rate = fee_rate
        
        # Position tracking
        self.positions = {}  # {symbol: quantity}
        self.position_entries = {}  # {symbol: entry_info}
        
        # History tracking
        self.equity_curve = []
        self.trades = []
        self.daily_returns = []
        
        logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")
    
    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Calculate current value of position"""
        quantity = self.positions.get(symbol, 0)
        return quantity * current_price
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value (cash + positions)
        
        Args:
            current_prices: Dict of {symbol: price}
            
        Returns:
            Total portfolio value
        """
        holdings_value = sum(
            self.get_position_value(symbol, current_prices.get(symbol, 0))
            for symbol in self.positions
        )
        return self.cash + holdings_value
    
    def can_buy(self, symbol: str, price: float, quantity: float) -> bool:
        """
        Check if we have enough cash to buy
        
        Args:
            symbol: Trading symbol
            price: Current price
            quantity: Quantity to buy
            
        Returns:
            True if sufficient cash available
        """
        total_cost = price * quantity * (1 + self.fee_rate)
        return self.cash >= total_cost
    
    def buy(self, symbol: str, price: float, quantity: float, timestamp: datetime) -> Optional[Dict]:
        """
        Execute buy order
        
        Args:
            symbol: Trading symbol
            price: Execution price
            quantity: Quantity to buy
            timestamp: Trade timestamp
            
        Returns:
            Trade record or None if failed
        """
        # Calculate costs
        fee = price * quantity * self.fee_rate
        total_cost = price * quantity + fee
        
        # Check if we can afford it
        if not self.can_buy(symbol, price, quantity):
            logger.warning(f"Insufficient funds to buy {quantity} {symbol} at ${price}")
            return None
        
        # Execute trade
        self.cash -= total_cost
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        
        # Record entry info
        self.position_entries[symbol] = {
            'entry_price': price,
            'entry_time': timestamp,
            'quantity': quantity
        }
        
        # Create trade record
        trade = {
            'timestamp': timestamp,
            'action': 'BUY',
            'symbol': symbol,
            'price': price,
            'quantity': quantity,
            'fee': fee,
            'total_cost': total_cost,
            'cash_after': self.cash,
            'pnl': 0
        }
        
        self.trades.append(trade)
        logger.info(f"BUY {quantity:.6f} {symbol} @ ${price:,.2f} | Fee: ${fee:.2f}")
        
        return trade
    
    def can_sell(self, symbol: str, quantity: float = None) -> bool:
        """
        Check if we have position to sell
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to sell (None = sell all)
            
        Returns:
            True if we have the position
        """
        current_position = self.positions.get(symbol, 0)
        if quantity is None:
            return current_position > 0
        return current_position >= quantity
    
    def sell(self, symbol: str, price: float, quantity: float = None, timestamp: datetime = None) -> Optional[Dict]:
        """
        Execute sell order
        
        Args:
            symbol: Trading symbol
            price: Execution price
            quantity: Quantity to sell (None = sell all)
            timestamp: Trade timestamp
            
        Returns:
            Trade record or None if failed
        """
        # Default to selling all
        if quantity is None:
            quantity = self.positions.get(symbol, 0)
        
        # Check if we have position
        if not self.can_sell(symbol, quantity):
            logger.warning(f"No position to sell for {symbol}")
            return None
        
        # Calculate proceeds
        fee = price * quantity * self.fee_rate
        revenue = price * quantity - fee
        
        # Calculate PnL if we have entry info
        pnl = 0
        if symbol in self.position_entries:
            entry_info = self.position_entries[symbol]
            entry_price = entry_info['entry_price']
            pnl = (price - entry_price) * quantity - fee - (entry_price * quantity * self.fee_rate)
        
        # Execute trade
        self.cash += revenue
        self.positions[symbol] = self.positions.get(symbol, 0) - quantity
        
        # Clean up if position closed
        if self.positions[symbol] <= 0.0001:  # Account for floating point
            self.positions[symbol] = 0
            if symbol in self.position_entries:
                del self.position_entries[symbol]
        
        # Create trade record
        trade = {
            'timestamp': timestamp or datetime.now(),
            'action': 'SELL',
            'symbol': symbol,
            'price': price,
            'quantity': quantity,
            'fee': fee,
            'revenue': revenue,
            'cash_after': self.cash,
            'pnl': pnl
        }
        
        self.trades.append(trade)
        logger.info(f"SELL {quantity:.6f} {symbol} @ ${price:,.2f} | PnL: ${pnl:,.2f}")
        
        return trade
    
    def record_equity(self, timestamp: datetime, current_prices: Dict[str, float]):
        """
        Record current portfolio value
        
        Args:
            timestamp: Current timestamp
            current_prices: Dict of current prices
        """
        total_value = self.get_total_value(current_prices)
        
        equity_record = {
            'timestamp': timestamp,
            'cash': self.cash,
            'holdings_value': total_value - self.cash,
            'total_value': total_value,
            'return_pct': (total_value - self.initial_capital) / self.initial_capital * 100
        }
        
        self.equity_curve.append(equity_record)
        
        # Calculate daily return if we have previous value
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2]['total_value']
            daily_return = (total_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
    
    def get_equity_dataframe(self) -> pd.DataFrame:
        """Convert equity curve to DataFrame"""
        return pd.DataFrame(self.equity_curve)
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Convert trade history to DataFrame"""
        return pd.DataFrame(self.trades)
    
    def get_summary(self) -> Dict:
        """
        Get portfolio summary statistics
        
        Returns:
            Dict with summary metrics
        """
        if not self.equity_curve:
            return {}
        
        final_value = self.equity_curve[-1]['total_value']
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate winning/losing trades
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        
        winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('pnl', 0) < 0]
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return * 100,
            'total_trades': len(sell_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(sell_trades) * 100 if sell_trades else 0,
            'total_pnl': sum(t.get('pnl', 0) for t in sell_trades),
            'total_fees': sum(t.get('fee', 0) for t in self.trades),
            'avg_win': sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0,
        }
    
    def print_summary(self):
        """Print portfolio summary"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("PORTFOLIO SUMMARY")
        print("="*60)
        print(f"Initial Capital:  ${summary.get('initial_capital', 0):,.2f}")
        print(f"Final Value:      ${summary.get('final_value', 0):,.2f}")
        print(f"Total Return:     {summary.get('total_return', 0):,.2f}%")
        print(f"\nTotal Trades:     {summary.get('total_trades', 0)}")
        print(f"Winning Trades:   {summary.get('winning_trades', 0)}")
        print(f"Losing Trades:    {summary.get('losing_trades', 0)}")
        print(f"Win Rate:         {summary.get('win_rate', 0):.2f}%")
        print(f"\nTotal PnL:        ${summary.get('total_pnl', 0):,.2f}")
        print(f"Total Fees:       ${summary.get('total_fees', 0):,.2f}")
        print(f"Avg Win:          ${summary.get('avg_win', 0):,.2f}")
        print(f"Avg Loss:         ${summary.get('avg_loss', 0):,.2f}")
        print("="*60)


# Example usage
if __name__ == "__main__":
    # Create portfolio
    portfolio = Portfolio(initial_capital=10000)
    
    # Simulate some trades
    portfolio.buy('BTC/USDT', price=42000, quantity=0.2, timestamp=datetime.now())
    portfolio.record_equity(datetime.now(), {'BTC/USDT': 42000})
    
    portfolio.sell('BTC/USDT', price=43000, quantity=0.2, timestamp=datetime.now())
    portfolio.record_equity(datetime.now(), {'BTC/USDT': 43000})
    
    # Print summary
    portfolio.print_summary()
    
    print("\n✅ Portfolio module loaded successfully!")
