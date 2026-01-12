"""
Market Regime Detection Module
Week 4 Improvement: Detect market conditions to avoid Week 3 mistakes

Key Insight from Week 3:
- Both RSI and LLM failed during strong downtrend (Oct 10-14)
- Mean reversion strategies don't work in trending markets
- Need to detect regime BEFORE trading
"""

import pandas as pd
import numpy as np


class MarketRegimeDetector:
    """
    Detects market regime to determine appropriate trading strategy
    
    Regimes:
    - STRONG_TREND_UP: Bull market, avoid shorts
    - STRONG_TREND_DOWN: Bear market, avoid mean reversion longs
    - TREND_UP: Mild uptrend, careful with shorts
    - TREND_DOWN: Mild downtrend, careful with longs
    - RANGE: Sideways, mean reversion works well
    - VOLATILE: High volatility, reduce position sizes
    - TRANSITIONAL: Unclear, be cautious
    """
    
    def __init__(self, ma_short=20, ma_medium=50, ma_long=200):
        self.ma_short = ma_short
        self.ma_medium = ma_medium
        self.ma_long = ma_long
        
    def detect_regime(self, df):
        """
        Detect current market regime
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            dict: {
                'regime': str,
                'confidence': float (0-100),
                'trend_direction': str,
                'trend_strength': float,
                'volatility_state': str,
                'signals': dict
            }
        """
        # Safety check
        if len(df) < 50:
            return {
                'regime': 'TRANSITIONAL',
                'confidence': 0,
                'recommendation': 'WAIT',
                'reason': 'Insufficient data for regime detection',
                'details': {}
            }
        
        # Calculate required indicators if not present
        df = self._ensure_indicators(df)
        
        # Get latest values
        current = df.iloc[-1]
        
        # 1. Trend Analysis
        trend_info = self._analyze_trend(df)
        
        # 2. Volatility Analysis
        volatility_info = self._analyze_volatility(df)
        
        # 3. Momentum Analysis
        momentum_info = self._analyze_momentum(df)
        
        # 4. Range Detection
        range_info = self._detect_range(df)
        
        # 5. Combine to determine regime
        regime = self._determine_regime(
            trend_info,
            volatility_info,
            momentum_info,
            range_info
        )
        
        return regime
    
    def _ensure_indicators(self, df):
        """Calculate required indicators"""
        df = df.copy()
        
        # Moving averages
        if f'MA_{self.ma_short}' not in df.columns:
            df[f'MA_{self.ma_short}'] = df['close'].rolling(self.ma_short).mean()
        if f'MA_{self.ma_medium}' not in df.columns:
            df[f'MA_{self.ma_medium}'] = df['close'].rolling(self.ma_medium).mean()
        if f'MA_{self.ma_long}' not in df.columns:
            df[f'MA_{self.ma_long}'] = df['close'].rolling(self.ma_long).mean()
            
        # RSI
        if 'RSI' not in df.columns:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR
        if 'ATR' not in df.columns:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(14).mean()
        
        return df
    
    def _analyze_trend(self, df):
        """Analyze trend direction and strength"""
        current = df.iloc[-1]
        recent = df.tail(50)
        
        ma_20 = current[f'MA_{self.ma_short}']
        ma_50 = current[f'MA_{self.ma_medium}']
        ma_200 = current[f'MA_{self.ma_long}'] if f'MA_{self.ma_long}' in current else ma_50
        price = current['close']
        
        # Trend direction
        if ma_20 > ma_50 and price > ma_20:
            direction = 'UP'
            if ma_50 > ma_200:
                strength = 'STRONG'
            else:
                strength = 'MODERATE'
        elif ma_20 < ma_50 and price < ma_20:
            direction = 'DOWN'
            if ma_50 < ma_200:
                strength = 'STRONG'
            else:
                strength = 'MODERATE'
        else:
            direction = 'NEUTRAL'
            strength = 'WEAK'
        
        # Calculate trend consistency
        ma_20_slope = (recent[f'MA_{self.ma_short}'].iloc[-1] - 
                       recent[f'MA_{self.ma_short}'].iloc[0]) / len(recent)
        
        trend_consistency = self._calculate_consistency(recent['close'])
        
        return {
            'direction': direction,
            'strength': strength,
            'consistency': trend_consistency,
            'ma_alignment': self._check_ma_alignment(current),
            'slope': ma_20_slope
        }
    
    def _analyze_volatility(self, df):
        """Analyze volatility state"""
        current = df.iloc[-1]
        recent = df.tail(100)
        
        atr = current['ATR']
        atr_mean = recent['ATR'].mean()
        atr_std = recent['ATR'].std()
        
        # Calculate percentile
        atr_percentile = (recent['ATR'] <= atr).sum() / len(recent) * 100
        
        # Classify volatility
        if atr > atr_mean + 2 * atr_std:
            state = 'EXTREMELY_HIGH'
        elif atr > atr_mean + atr_std:
            state = 'HIGH'
        elif atr < atr_mean - atr_std:
            state = 'LOW'
        else:
            state = 'NORMAL'
        
        return {
            'state': state,
            'atr': atr,
            'atr_mean': atr_mean,
            'percentile': atr_percentile,
            'relative': atr / atr_mean if atr_mean > 0 else 1.0
        }
    
    def _analyze_momentum(self, df):
        """Analyze momentum"""
        current = df.iloc[-1]
        recent = df.tail(20)
        
        rsi = current['RSI']
        
        # RSI state
        if rsi > 70:
            rsi_state = 'OVERBOUGHT'
        elif rsi > 60:
            rsi_state = 'STRONG'
        elif rsi > 40:
            rsi_state = 'NEUTRAL'
        elif rsi > 30:
            rsi_state = 'WEAK'
        else:
            rsi_state = 'OVERSOLD'
        
        # Price momentum
        price_change_pct = ((current['close'] - recent['close'].iloc[0]) / 
                            recent['close'].iloc[0] * 100)
        
        return {
            'rsi': rsi,
            'rsi_state': rsi_state,
            'price_change_20': price_change_pct,
            'momentum_direction': 'UP' if price_change_pct > 0 else 'DOWN'
        }
    
    def _detect_range(self, df):
        """Detect if market is ranging"""
        recent = df.tail(50)
        
        # Calculate range metrics
        high = recent['high'].max()
        low = recent['low'].min()
        range_pct = (high - low) / low * 100
        
        # Check if price is bouncing between levels
        upper_touches = (recent['high'] > high * 0.98).sum()
        lower_touches = (recent['low'] < low * 1.02).sum()
        
        # Ranging if: tight range + multiple touches
        is_ranging = (range_pct < 10 and 
                     upper_touches >= 2 and 
                     lower_touches >= 2)
        
        return {
            'is_ranging': is_ranging,
            'range_pct': range_pct,
            'support': low,
            'resistance': high,
            'upper_touches': upper_touches,
            'lower_touches': lower_touches
        }
    
    def _determine_regime(self, trend, volatility, momentum, range_info):
        """Combine all factors to determine regime"""
        
        # High volatility overrides everything
        if volatility['state'] in ['EXTREMELY_HIGH', 'HIGH']:
            if volatility['percentile'] > 90:
                return {
                    'regime': 'VOLATILE',
                    'confidence': 90,
                    'recommendation': 'REDUCE_SIZE',
                    'reason': f"Extreme volatility (ATR {volatility['percentile']:.0f}th percentile)",
                    'details': {
                        'trend': trend,
                        'volatility': volatility,
                        'momentum': momentum,
                        'range': range_info
                    }
                }
        
        # Ranging market
        if range_info['is_ranging']:
            return {
                'regime': 'RANGE',
                'confidence': 80,
                'recommendation': 'MEAN_REVERSION',
                'reason': f"Market ranging {range_info['range_pct']:.1f}% with clear support/resistance",
                'details': {
                    'trend': trend,
                    'volatility': volatility,
                    'momentum': momentum,
                    'range': range_info
                }
            }
        
        # Strong trending markets
        if trend['strength'] == 'STRONG':
            if trend['direction'] == 'UP':
                return {
                    'regime': 'STRONG_TREND_UP',
                    'confidence': 85,
                    'recommendation': 'TREND_FOLLOWING',
                    'reason': f"Strong uptrend (MA alignment: {trend['ma_alignment']})",
                    'avoid': 'SHORT positions and mean reversion shorts',
                    'details': {
                        'trend': trend,
                        'volatility': volatility,
                        'momentum': momentum,
                        'range': range_info
                    }
                }
            else:  # DOWN
                return {
                    'regime': 'STRONG_TREND_DOWN',
                    'confidence': 85,
                    'recommendation': 'AVOID_LONGS',
                    'reason': f"Strong downtrend (MA alignment: {trend['ma_alignment']})",
                    'avoid': 'LONG positions and mean reversion longs (WEEK 3 MISTAKE!)',
                    'details': {
                        'trend': trend,
                        'volatility': volatility,
                        'momentum': momentum,
                        'range': range_info
                    }
                }
        
        # Moderate trends
        if trend['direction'] == 'UP':
            return {
                'regime': 'TREND_UP',
                'confidence': 70,
                'recommendation': 'CAREFUL_LONGS',
                'reason': f"Mild uptrend, be selective",
                'details': {
                    'trend': trend,
                    'volatility': volatility,
                    'momentum': momentum,
                    'range': range_info
                }
            }
        elif trend['direction'] == 'DOWN':
            return {
                'regime': 'TREND_DOWN',
                'confidence': 70,
                'recommendation': 'CAREFUL_SHORTS',
                'reason': f"Mild downtrend, be selective with longs",
                'details': {
                    'trend': trend,
                    'volatility': volatility,
                    'momentum': momentum,
                    'range': range_info
                }
            }
        
        # Unclear/transitional
        return {
            'regime': 'TRANSITIONAL',
            'confidence': 50,
            'recommendation': 'WAIT',
            'reason': "Market regime unclear, wait for clearer setup",
            'details': {
                'trend': trend,
                'volatility': volatility,
                'momentum': momentum,
                'range': range_info
            }
        }
    
    def _check_ma_alignment(self, current):
        """Check if MAs are properly aligned"""
        ma_20 = current[f'MA_{self.ma_short}']
        ma_50 = current[f'MA_{self.ma_medium}']
        price = current['close']
        
        if price > ma_20 > ma_50:
            return 'BULLISH'
        elif price < ma_20 < ma_50:
            return 'BEARISH'
        else:
            return 'MIXED'
    
    def _calculate_consistency(self, series):
        """Calculate how consistent the trend is"""
        # Count directional changes
        changes = (series.diff().fillna(0) > 0).astype(int).diff().fillna(0)
        direction_changes = abs(changes).sum()
        
        # Fewer changes = more consistent
        consistency = max(0, 100 - (direction_changes / len(series) * 100))
        return consistency
    
    def should_allow_trade(self, regime, signal_direction):
        """
        Check if trade is allowed in current regime
        
        CRITICAL: This prevents Week 3 mistakes!
        
        Args:
            regime: dict from detect_regime()
            signal_direction: 'LONG' or 'SHORT'
            
        Returns:
            dict: {
                'allowed': bool,
                'reason': str,
                'confidence_adjustment': float (0-1)
            }
        """
        regime_type = regime['regime']
        
        # RULE 1: No mean reversion longs in strong downtrends
        if regime_type == 'STRONG_TREND_DOWN' and signal_direction == 'LONG':
            return {
                'allowed': False,
                'reason': 'BLOCKED: Mean reversion LONG in strong downtrend (Week 3 mistake!)',
                'confidence_adjustment': 0.0
            }
        
        # RULE 2: No mean reversion shorts in strong uptrends
        if regime_type == 'STRONG_TREND_UP' and signal_direction == 'SHORT':
            return {
                'allowed': False,
                'reason': 'BLOCKED: Mean reversion SHORT in strong uptrend',
                'confidence_adjustment': 0.0
            }
        
        # RULE 3: Reduce confidence in volatile markets
        if regime_type == 'VOLATILE':
            return {
                'allowed': True,
                'reason': 'ALLOWED but high volatility - reduce size',
                'confidence_adjustment': 0.5  # 50% size
            }
        
        # RULE 4: Wait in transitional markets
        if regime_type == 'TRANSITIONAL':
            return {
                'allowed': False,
                'reason': 'BLOCKED: Market regime unclear, wait for better setup',
                'confidence_adjustment': 0.0
            }
        
        # RULE 5: Prefer trend-aligned trades
        if regime_type in ['TREND_UP', 'TREND_DOWN']:
            if regime_type == 'TREND_UP' and signal_direction == 'SHORT':
                return {
                    'allowed': True,
                    'reason': 'ALLOWED but counter-trend - reduce confidence',
                    'confidence_adjustment': 0.7  # 70% size
                }
            elif regime_type == 'TREND_DOWN' and signal_direction == 'LONG':
                return {
                    'allowed': True,
                    'reason': 'ALLOWED but counter-trend - reduce confidence',
                    'confidence_adjustment': 0.7  # 70% size
                }
        
        # All other cases: allow with full confidence
        return {
            'allowed': True,
            'reason': f'ALLOWED: {regime_type} supports {signal_direction}',
            'confidence_adjustment': 1.0
        }


# Module test
if __name__ == '__main__':
    print("=" * 70)
    print("MARKET REGIME DETECTOR - Week 4 Improvement")
    print("=" * 70)
    print("\n✅ Module loaded successfully!")
    print("\nKey Features:")
    print("  1. Detects 7 market regimes")
    print("  2. Prevents Week 3 mistakes (mean reversion in trends)")
    print("  3. Adjusts position sizing based on regime")
    print("  4. Provides clear recommendations")
    print("\nUsage:")
    print("  detector = MarketRegimeDetector()")
    print("  regime = detector.detect_regime(df)")
    print("  allowed = detector.should_allow_trade(regime, 'LONG')")
    print("\n" + "=" * 70)
