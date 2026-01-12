"""
Week 3: Risk Management Module
Handles position sizing, stop-loss, and risk limits
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manages trading risk through position sizing and stop-loss logic
    """
    
    def __init__(
        self,
        max_position_pct: float = 0.95,
        risk_per_trade_pct: float = 0.02,
        stop_loss_pct: float = 0.02,
        max_positions: int = 1
    ):
        """
        Initialize risk manager
        
        Args:
            max_position_pct: Maximum % of capital per position (0.95 = 95%)
            risk_per_trade_pct: Maximum risk per trade (0.02 = 2%)
            stop_loss_pct: Stop-loss percentage (0.02 = 2%)
            max_positions: Maximum number of concurrent positions
        """
        self.max_position_pct = max_position_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_positions = max_positions
        
        logger.info(f"RiskManager initialized: max_pos={max_position_pct*100}%, "
                   f"risk_per_trade={risk_per_trade_pct*100}%, "
                   f"stop_loss={stop_loss_pct*100}%")
    
    def calculate_position_size_fixed_pct(
        self,
        portfolio_value: float,
        price: float
    ) -> float:
        """
        Calculate position size as fixed percentage of portfolio
        
        Args:
            portfolio_value: Current portfolio value
            price: Current price per unit
            
        Returns:
            Quantity to buy
        """
        position_value = portfolio_value * self.max_position_pct
        quantity = position_value / price
        return quantity
    
    def calculate_position_size_risk_based(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Calculate position size based on risk per trade
        
        Args:
            portfolio_value: Current portfolio value
            entry_price: Planned entry price
            stop_loss_price: Stop-loss price
            
        Returns:
            Quantity to buy
        """
        # Amount we're willing to lose on this trade
        risk_amount = portfolio_value * self.risk_per_trade_pct
        
        # Price distance to stop-loss
        price_risk = abs(entry_price - stop_loss_price)
        
        # Position size that risks exactly risk_amount
        if price_risk > 0:
            quantity = risk_amount / price_risk
        else:
            # Fallback to fixed percentage
            quantity = self.calculate_position_size_fixed_pct(portfolio_value, entry_price)
        
        return quantity
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: str = 'long'
    ) -> float:
        """
        Calculate stop-loss price
        
        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            
        Returns:
            Stop-loss price
        """
        if direction == 'long':
            return entry_price * (1 - self.stop_loss_pct)
        else:  # short
            return entry_price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        risk_reward_ratio: float = 2.0,
        direction: str = 'long'
    ) -> float:
        """
        Calculate take-profit price based on risk-reward ratio
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop-loss price
            risk_reward_ratio: Desired risk:reward ratio (2.0 = 2:1)
            direction: 'long' or 'short'
            
        Returns:
            Take-profit price
        """
        risk_distance = abs(entry_price - stop_loss_price)
        reward_distance = risk_distance * risk_reward_ratio
        
        if direction == 'long':
            return entry_price + reward_distance
        else:  # short
            return entry_price - reward_distance
    
    def should_stop_out(
        self,
        entry_price: float,
        current_price: float,
        stop_loss_price: float,
        direction: str = 'long'
    ) -> bool:
        """
        Check if stop-loss should be triggered
        
        Args:
            entry_price: Entry price
            current_price: Current market price
            stop_loss_price: Stop-loss price
            direction: 'long' or 'short'
            
        Returns:
            True if stop-loss hit
        """
        if direction == 'long':
            return current_price <= stop_loss_price
        else:  # short
            return current_price >= stop_loss_price
    
    def should_take_profit(
        self,
        entry_price: float,
        current_price: float,
        take_profit_price: float,
        direction: str = 'long'
    ) -> bool:
        """
        Check if take-profit should be triggered
        
        Args:
            entry_price: Entry price
            current_price: Current market price
            take_profit_price: Take-profit price
            direction: 'long' or 'short'
            
        Returns:
            True if take-profit hit
        """
        if direction == 'long':
            return current_price >= take_profit_price
        else:  # short
            return current_price <= take_profit_price
    
    def check_position_limits(self, current_positions: int) -> bool:
        """
        Check if we can open a new position
        
        Args:
            current_positions: Number of open positions
            
        Returns:
            True if allowed to open new position
        """
        return current_positions < self.max_positions
    
    def calculate_atr_stop_loss(
        self,
        entry_price: float,
        atr: float,
        multiplier: float = 2.0,
        direction: str = 'long'
    ) -> float:
        """
        Calculate ATR-based stop-loss
        
        Args:
            entry_price: Entry price
            atr: Average True Range value
            multiplier: ATR multiplier (2.0 = 2 x ATR)
            direction: 'long' or 'short'
            
        Returns:
            Stop-loss price
        """
        stop_distance = atr * multiplier
        
        if direction == 'long':
            return entry_price - stop_distance
        else:  # short
            return entry_price + stop_distance
    
    def trailing_stop_loss(
        self,
        entry_price: float,
        highest_price: float,
        trailing_pct: float = 0.03,
        direction: str = 'long'
    ) -> float:
        """
        Calculate trailing stop-loss
        
        Args:
            entry_price: Entry price
            highest_price: Highest price since entry
            trailing_pct: Trailing stop percentage (0.03 = 3%)
            direction: 'long' or 'short'
            
        Returns:
            Trailing stop price
        """
        if direction == 'long':
            # Stop trails below highest price
            return highest_price * (1 - trailing_pct)
        else:  # short
            # Stop trails above lowest price
            return highest_price * (1 + trailing_pct)
    
    def validate_trade(
        self,
        portfolio_value: float,
        position_value: float,
        num_positions: int
    ) -> tuple[bool, str]:
        """
        Validate if trade meets risk criteria
        
        Args:
            portfolio_value: Current portfolio value
            position_value: Proposed position value
            num_positions: Current number of positions
            
        Returns:
            (is_valid, reason) tuple
        """
        # Check position limit
        if not self.check_position_limits(num_positions):
            return False, f"Max positions reached ({self.max_positions})"
        
        # Check position size
        position_pct = position_value / portfolio_value
        if position_pct > self.max_position_pct:
            return False, f"Position too large ({position_pct*100:.1f}% > {self.max_position_pct*100:.1f}%)"
        
        return True, "Trade approved"
    
    def get_position_stats(
        self,
        entry_price: float,
        current_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float
    ) -> dict:
        """
        Get current position statistics
        
        Args:
            entry_price: Entry price
            current_price: Current price
            quantity: Position quantity
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            
        Returns:
            Dict with position stats
        """
        unrealized_pnl = (current_price - entry_price) * quantity
        unrealized_pnl_pct = (current_price - entry_price) / entry_price * 100
        
        risk_amount = abs(entry_price - stop_loss) * quantity
        reward_amount = abs(take_profit - entry_price) * quantity
        risk_reward = reward_amount / risk_amount if risk_amount > 0 else 0
        
        distance_to_stop_pct = abs(current_price - stop_loss) / current_price * 100
        distance_to_target_pct = abs(take_profit - current_price) / current_price * 100
        
        return {
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'risk_amount': risk_amount,
            'reward_amount': reward_amount,
            'risk_reward_ratio': risk_reward,
            'distance_to_stop_pct': distance_to_stop_pct,
            'distance_to_target_pct': distance_to_target_pct
        }


# Example usage
if __name__ == "__main__":
    # Create risk manager
    rm = RiskManager(
        max_position_pct=0.95,
        risk_per_trade_pct=0.02,
        stop_loss_pct=0.02
    )
    
    # Example calculations
    portfolio_value = 10000
    entry_price = 42000
    
    # Position sizing
    quantity = rm.calculate_position_size_fixed_pct(portfolio_value, entry_price)
    print(f"Position size (fixed %): {quantity:.6f} BTC")
    
    # Stop-loss and take-profit
    stop_loss = rm.calculate_stop_loss(entry_price, direction='long')
    take_profit = rm.calculate_take_profit(entry_price, stop_loss, risk_reward_ratio=2.0)
    
    print(f"Entry: ${entry_price:,.2f}")
    print(f"Stop-Loss: ${stop_loss:,.2f} ({rm.stop_loss_pct*100}% below)")
    print(f"Take-Profit: ${take_profit:,.2f} (2:1 R:R)")
    
    # Check position stats
    current_price = 43000
    stats = rm.get_position_stats(entry_price, current_price, quantity, stop_loss, take_profit)
    print(f"\nUnrealized PnL: ${stats['unrealized_pnl']:,.2f} ({stats['unrealized_pnl_pct']:.2f}%)")
    print(f"Risk:Reward: {stats['risk_reward_ratio']:.2f}:1")
    
    print("\n✅ Risk Manager module loaded successfully!")
