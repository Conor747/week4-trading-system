"""
Improved LLM Trading Strategy - Week 4 (Adjustable Version)
Now with TRADING_MODE to control aggressiveness
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Try to import LLM module from Week 2
try:
    from llm_signal import LLMSignalGenerator
except ImportError:
    print("⚠️ llm_signal.py not found. Using mock for testing.")
    LLMSignalGenerator = None


class ImprovedLLMStrategy:
    """
    Enhanced LLM trading strategy with adjustable aggressiveness
    
    TRADING_MODE options:
    - 'CONSERVATIVE': Original (very few trades, high quality)
    - 'BALANCED': Medium activity (recommended)
    - 'AGGRESSIVE': More trades, lower threshold
    """
    
    def __init__(self, api_key=None, model="deepseek-chat", trading_mode='BALANCED'):
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.model = model
        self.trading_mode = trading_mode  # NEW: Control aggressiveness
        
        # Initialize LLM generator if available
        if LLMSignalGenerator and self.api_key:
            self.llm = LLMSignalGenerator(
                api_key=self.api_key,
                model=self.model,
                provider="deepseek"
            )
        else:
            self.llm = None
            print("⚠️ LLM not initialized - using rule-based fallback")
        
        # Trade history for meta-learning
        self.trade_history = []
        self.consecutive_losses = 0
        self.total_trades = 0
        
        # Set parameters based on trading mode
        self._configure_mode()
        
        print(f"✅ Trading Mode: {self.trading_mode}")
        print(f"   Max Consecutive Losses: {self.max_consecutive_losses}")
        print(f"   Min Confidence: {self.min_confidence}")
        print(f"   Extreme Volatility Threshold: {self.volatility_threshold}%")
    
    def _configure_mode(self):
        """Configure parameters based on trading mode"""
        if self.trading_mode == 'CONSERVATIVE':
            # Original settings - very selective
            self.max_consecutive_losses = 2
            self.min_confidence = 70
            self.volatility_threshold = 95
            self.allow_transitional = False
            self.rsi_oversold = 30
            self.rsi_overbought = 70
            
        elif self.trading_mode == 'AGGRESSIVE':
            # More trades - lower standards
            self.max_consecutive_losses = 3
            self.min_confidence = 50
            self.volatility_threshold = 98
            self.allow_transitional = True
            self.rsi_oversold = 35
            self.rsi_overbought = 65
            
        else:  # BALANCED (recommended)
            # Moderate - good balance
            self.max_consecutive_losses = 2
            self.min_confidence = 60
            self.volatility_threshold = 97
            self.allow_transitional = False
            self.rsi_oversold = 32
            self.rsi_overbought = 68
    
    def generate_signal_with_context(self, df, regime_info, index=-1):
        """
        Generate trading signal with full context
        
        Returns:
            dict: {
                'signal': 'LONG'/'SHORT'/'HOLD',
                'confidence': 0-100,
                'reasoning': str,
                'meta_override': bool
            }
        """
        # STEP 1: Meta-strategy checks (fast, free)
        meta_check = self._meta_strategy_check(regime_info)
        if not meta_check['allowed']:
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'reasoning': meta_check['reason'],
                'meta_override': True
            }
        
        # STEP 2: Build rich context
        context = self._build_rich_context(df, regime_info, index)
        
        # STEP 3: Generate signal
        if self.llm:
            signal = self._llm_signal_with_context(context)
        else:
            signal = self._rule_based_fallback(context)
        
        # STEP 4: Apply regime filter
        regime_check = self._regime_filter(signal, regime_info)
        if not regime_check['allowed']:
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'reasoning': regime_check['reason'],
                'meta_override': True
            }
        
        # STEP 5: Check minimum confidence threshold
        if signal['confidence'] < self.min_confidence:
            return {
                'signal': 'HOLD',
                'confidence': signal['confidence'],
                'reasoning': f"Confidence {signal['confidence']:.0f}% below threshold {self.min_confidence}%",
                'meta_override': True
            }
        
        # STEP 6: Adjust confidence based on regime
        signal['confidence'] *= regime_check.get('confidence_adjustment', 1.0)
        
        return signal
    
    def _meta_strategy_check(self, regime_info):
        """
        High-level checks before even calling LLM
        
        Adjusted based on trading_mode
        """
        # Rule 1: Cooldown after losses (adjusted by mode)
        if self.consecutive_losses >= self.max_consecutive_losses:
            return {
                'allowed': False,
                'reason': f'COOLDOWN: Lost {self.consecutive_losses} in a row'
            }
        
        # Rule 2: Don't trade in transitional markets (unless aggressive)
        if regime_info['regime'] == 'TRANSITIONAL' and not self.allow_transitional:
            return {
                'allowed': False,
                'reason': 'WAIT: Market regime unclear'
            }
        
        # Rule 3: Avoid extreme volatility (threshold adjusted by mode)
        if regime_info['regime'] == 'VOLATILE':
            try:
                percentile = regime_info['details']['volatility']['percentile']
                if percentile > self.volatility_threshold:
                    return {
                        'allowed': False,
                        'reason': f'WAIT: Extreme volatility ({percentile:.0f}th percentile)'
                    }
            except (KeyError, TypeError):
                pass
        
        return {'allowed': True}
    
    def _build_rich_context(self, df, regime_info, index=-1):
        """Build comprehensive context for LLM"""
        # Handle negative index properly
        if index < 0:
            index = len(df) + index
        
        current = df.iloc[index]
        recent_100 = df.iloc[max(0, index-100):index+1]
        recent_20 = df.iloc[max(0, index-20):index+1]
        
        # Safety checks
        if len(recent_100) == 0 or len(recent_20) == 0:
            raise ValueError(f"Insufficient data at index {index}")
        
        context = {
            # Current state
            'timestamp': current.name if hasattr(current, 'name') else 'N/A',
            'price': current['close'],
            'rsi': current['RSI'],
            'ma_20': current['MA_20'],
            'ma_50': current['MA_50'],
            'atr': current['ATR'],
            
            # Regime information
            'regime': regime_info['regime'],
            'regime_confidence': regime_info['confidence'],
            'regime_reason': regime_info['reason'],
            'regime_recommendation': regime_info['recommendation'],
            
            # Historical context
            'price_change_20': ((current['close'] - recent_20['close'].iloc[0]) / 
                               recent_20['close'].iloc[0] * 100),
            'price_change_100': ((current['close'] - recent_100['close'].iloc[0]) / 
                                recent_100['close'].iloc[0] * 100),
            
            # Technical summary
            'trend_direction': 'UP' if current['MA_20'] > current['MA_50'] else 'DOWN',
            'price_vs_ma20': ((current['close'] - current['MA_20']) / 
                             current['MA_20'] * 100),
            'ma20_vs_ma50': ((current['MA_20'] - current['MA_50']) / 
                            current['MA_50'] * 100),
            
            # Performance history
            'recent_trades': self._summarize_recent_trades(),
            'consecutive_losses': self.consecutive_losses,
            'total_trades': self.total_trades,
            'win_rate': self._calculate_win_rate(),
            
            # Key levels
            'support': recent_100['low'].min(),
            'resistance': recent_100['high'].max(),
            
            # Volatility context
            'volatility_state': regime_info['details'].get('volatility', {}).get('state', 'N/A') if 'details' in regime_info else 'N/A',
            'volatility_percentile': regime_info['details'].get('volatility', {}).get('percentile', 0) if 'details' in regime_info else 0,
            
            # Trading mode context
            'trading_mode': self.trading_mode,
            'min_confidence': self.min_confidence
        }
        
        return context
    
    def _llm_signal_with_context(self, context):
        """Call LLM with improved prompt"""
        prompt = self._build_improved_prompt(context)
        
        try:
            response = self.llm.generate_signal_raw(prompt, temperature=0.7)
            signal = self._parse_llm_response(response)
            return signal
            
        except Exception as e:
            print(f"⚠️ LLM call failed: {e}")
            return self._rule_based_fallback(context)
    
    def _build_improved_prompt(self, context):
        """Build prompt adjusted for trading mode"""
        mode_guidance = {
            'CONSERVATIVE': "Only take the VERY BEST setups. When in doubt, HOLD.",
            'BALANCED': "Take good setups with reasonable confidence. Balance quality and quantity.",
            'AGGRESSIVE': "Be more active. Take decent setups even with moderate confidence."
        }
        
        prompt = f"""
# TRADING SIGNAL ANALYSIS

## Trading Mode: {context['trading_mode']}
**Guidance:** {mode_guidance.get(context['trading_mode'], '')}
**Minimum Confidence Required:** {context['min_confidence']}%

## Current Market State
Price: ${context['price']:,.2f}
RSI: {context['rsi']:.1f}
MA(20): ${context['ma_20']:,.2f}
MA(50): ${context['ma_50']:,.2f}

## Market Regime
Regime: {context['regime']}
Recommendation: {context['regime_recommendation']}
Reason: {context['regime_reason']}

## Recent Performance
{context['recent_trades']}
Win Rate: {context['win_rate']:.1f}%

---

## YOUR TASK:
Given the {context['trading_mode']} trading mode, should we enter a trade?

Respond in JSON format:
{{
  "signal": "LONG" | "SHORT" | "HOLD",
  "confidence": 0-100,
  "reasoning": "Brief explanation"
}}

RESPOND ONLY WITH VALID JSON.
"""
        return prompt
    
    def _parse_llm_response(self, response):
        """Parse LLM JSON response"""
        try:
            if isinstance(response, str):
                response = response.strip()
                if response.startswith('```'):
                    response = response.split('\n', 1)[1]
                if response.endswith('```'):
                    response = response.rsplit('\n', 1)[0]
                response = response.strip()
                
                data = json.loads(response)
            else:
                data = response
            
            return {
                'signal': data.get('signal', 'HOLD').upper(),
                'confidence': float(data.get('confidence', 50)),
                'reasoning': data.get('reasoning', 'No reasoning provided'),
                'risk_factors': data.get('risk_factors', ''),
                'learning_note': data.get('learning_note', '')
            }
            
        except Exception as e:
            print(f"⚠️ Failed to parse LLM response: {e}")
            response_lower = str(response).lower()
            if 'long' in response_lower and 'short' not in response_lower:
                signal = 'LONG'
            elif 'short' in response_lower:
                signal = 'SHORT'
            else:
                signal = 'HOLD'
            
            return {
                'signal': signal,
                'confidence': 50,
                'reasoning': 'Parsed from text response',
                'risk_factors': '',
                'learning_note': ''
            }
    
    def _rule_based_fallback(self, context):
        """Rule-based strategy with adjustable thresholds"""
        # Check regime
        if context['regime'] == 'STRONG_TREND_DOWN':
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'reasoning': 'Strong downtrend - avoid longs',
                'risk_factors': '',
                'learning_note': ''
            }
        
        # RSI-based signals with adjusted thresholds
        rsi = context['rsi']
        
        if rsi < self.rsi_oversold and context['trend_direction'] == 'UP':
            confidence = 70 + (self.rsi_oversold - rsi) * 2  # Higher if more oversold
            return {
                'signal': 'LONG',
                'confidence': min(confidence, 90),
                'reasoning': f'Oversold (RSI={rsi:.1f}) in uptrend',
                'risk_factors': '',
                'learning_note': ''
            }
        elif rsi > self.rsi_overbought:
            confidence = 60 + (rsi - self.rsi_overbought) * 2
            return {
                'signal': 'SHORT',
                'confidence': min(confidence, 90),
                'reasoning': f'Overbought (RSI={rsi:.1f})',
                'risk_factors': '',
                'learning_note': ''
            }
        else:
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'reasoning': f'No clear setup (RSI={rsi:.1f})',
                'risk_factors': '',
                'learning_note': ''
            }
    
    def _regime_filter(self, signal, regime_info):
        """Filter signal based on market regime"""
        regime = regime_info['regime']
        signal_direction = signal['signal']
        
        # Import to use regime detector
        from market_regime import MarketRegimeDetector
        detector = MarketRegimeDetector()
        check = detector.should_allow_trade(regime_info, signal_direction)
        
        return check
    
    def _summarize_recent_trades(self, n=5):
        """Summarize recent trade performance"""
        if not self.trade_history:
            return "No previous trades"
        
        recent = self.trade_history[-n:]
        summary = []
        
        for t in recent:
            result = "WIN" if t['pnl'] > 0 else "LOSS"
            summary.append(f"{t['timestamp']}: {result} ${t['pnl']:+.2f}")
        
        return "\n".join(summary) if summary else "No recent trades"
    
    def _calculate_win_rate(self):
        """Calculate win rate from history"""
        if not self.trade_history:
            return 0.0
        
        wins = sum(1 for t in self.trade_history if t['pnl'] > 0)
        return (wins / len(self.trade_history)) * 100
    
    def update_trade_result(self, timestamp, pnl):
        """Update trade history"""
        self.trade_history.append({
            'timestamp': timestamp,
            'pnl': pnl
        })
        
        self.total_trades += 1
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        if self.total_trades % 10 == 0:
            self._analyze_performance_patterns()
    
    def _analyze_performance_patterns(self):
        """Analyze performance patterns"""
        recent_20 = self.trade_history[-20:] if len(self.trade_history) >= 20 else self.trade_history
        
        if not recent_20:
            return
        
        wins = sum(1 for t in recent_20 if t['pnl'] > 0)
        win_rate = (wins / len(recent_20)) * 100
        avg_win = sum(t['pnl'] for t in recent_20 if t['pnl'] > 0) / max(wins, 1)
        avg_loss = sum(t['pnl'] for t in recent_20 if t['pnl'] < 0) / max(len(recent_20) - wins, 1)
        
        print(f"\n📊 Performance Analysis (Last {len(recent_20)} trades):")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Avg Win: ${avg_win:.2f}")
        print(f"   Avg Loss: ${avg_loss:.2f}")
        print(f"   Consecutive Losses: {self.consecutive_losses}")


if __name__ == '__main__':
    print("=" * 70)
    print("IMPROVED LLM STRATEGY - Adjustable Version")
    print("=" * 70)
    
    print("\nAvailable Trading Modes:")
    print("  1. CONSERVATIVE - Very selective (original)")
    print("  2. BALANCED - Moderate activity (recommended)")
    print("  3. AGGRESSIVE - More trades, lower standards")
    
    for mode in ['CONSERVATIVE', 'BALANCED', 'AGGRESSIVE']:
        print(f"\n{mode} Mode:")
        strategy = ImprovedLLMStrategy(trading_mode=mode)
