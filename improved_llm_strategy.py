"""
Improved LLM Trading Strategy - Week 4
Based on Week 3 failure analysis

Key Improvements:
1. Rich context (not just single data point)
2. Recent trade history (learn from mistakes)
3. Market regime awareness
4. Meta-strategy rules (cooldown after losses)
5. Better prompts with specific instructions
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
    Enhanced LLM trading strategy with meta-learning
    """
    
    def __init__(self, api_key=None, model="deepseek-chat"):
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.model = model
        
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
        
    def generate_signal_with_context(self, df, regime_info, index=-1):
        """
        Generate trading signal with full context
        
        Args:
            df: DataFrame with OHLCV + indicators
            regime_info: Dict from MarketRegimeDetector
            index: Which row to analyze
            
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
        
        # STEP 5: Adjust confidence based on regime
        signal['confidence'] *= regime_check.get('confidence_adjustment', 1.0)
        
        return signal
    
    def _meta_strategy_check(self, regime_info):
        """
        High-level checks before even calling LLM
        
        PREVENTS WEEK 3 MISTAKES:
        - No trading after 2 consecutive losses
        - No trading in unclear markets
        - No trading in extreme volatility
        """
        # Rule 1: Cooldown after losses
        if self.consecutive_losses >= 2:
            return {
                'allowed': False,
                'reason': f'COOLDOWN: Lost {self.consecutive_losses} in a row (Week 4 improvement)'
            }
        
        # Rule 2: Don't trade in transitional markets
        if regime_info['regime'] == 'TRANSITIONAL':
            return {
                'allowed': False,
                'reason': 'WAIT: Market regime unclear'
            }
        
        # Rule 3: Avoid extreme volatility
        if regime_info['regime'] == 'VOLATILE':
            if regime_info['details']['volatility']['percentile'] > 95:
                return {
                    'allowed': False,
                    'reason': 'WAIT: Extreme volatility (95th+ percentile)'
                }
        
        return {'allowed': True}
    
    def _build_rich_context(self, df, regime_info, index=-1):
        """
        Build comprehensive context for LLM
        
        This is CRITICAL - Week 3 failed because LLM only saw single data point!
        """
        # Handle negative index properly
        if index < 0:
            index = len(df) + index  # Convert -1 to actual index
        
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
            'volatility_state': regime_info['details']['volatility']['state'],
            'volatility_percentile': regime_info['details']['volatility']['percentile']
        }
        
        return context
    
    def _llm_signal_with_context(self, context):
        """
        Call LLM with improved prompt
        """
        prompt = self._build_improved_prompt(context)
        
        try:
            # Call LLM
            response = self.llm.generate_signal_raw(prompt, temperature=0.7)
            
            # Parse response
            signal = self._parse_llm_response(response)
            
            return signal
            
        except Exception as e:
            print(f"⚠️ LLM call failed: {e}")
            return self._rule_based_fallback(context)
    
    def _build_improved_prompt(self, context):
        """
        Build much better prompt than Week 3
        
        Week 3 prompt was: "Price is X, RSI is Y, what signal?"
        Week 4 prompt is: Full context + recent mistakes + regime awareness
        """
        prompt = f"""
# TRADING SIGNAL ANALYSIS - AVOID WEEK 3 MISTAKES!

## Current Market State
Price: ${context['price']:,.2f}
RSI: {context['rsi']:.1f}
MA(20): ${context['ma_20']:,.2f}
MA(50): ${context['ma_50']:,.2f}
ATR: ${context['atr']:.2f}

## Price Position
vs MA(20): {context['price_vs_ma20']:+.2f}%
MA(20) vs MA(50): {context['ma20_vs_ma50']:+.2f}%
Trend Direction: {context['trend_direction']}

## Market Regime Analysis ⚠️ CRITICAL
Regime: {context['regime']}
Confidence: {context['regime_confidence']}%
Recommendation: {context['regime_recommendation']}
Reason: {context['regime_reason']}

## Recent Price Action
20-period change: {context['price_change_20']:+.2f}%
100-period change: {context['price_change_100']:+.2f}%
Support: ${context['support']:,.2f}
Resistance: ${context['resistance']:,.2f}

## Volatility Context
State: {context['volatility_state']}
Percentile: {context['volatility_percentile']:.0f}%

## Your Recent Performance
{context['recent_trades']}
Current Streak: {context['consecutive_losses']} losses in a row
Win Rate (all time): {context['win_rate']:.1f}%

---

## ⚠️ CRITICAL LESSONS FROM WEEK 3 FAILURES:

1. **NEVER mean reversion in strong trends!**
   - If regime is STRONG_TREND_DOWN, DO NOT go LONG
   - Week 3 lost $2,066 making this mistake

2. **Learn from recent losses!**
   - If you just lost money, ask: "Why did that fail?"
   - Don't immediately re-enter similar setup

3. **Respect the regime!**
   - Regime: {context['regime']} recommends: {context['regime_recommendation']}
   - Follow this recommendation!

4. **When in doubt, HOLD!**
   - Better to miss a trade than lose money
   - Quality > Quantity

---

## YOUR TASK:
Given this context, should we enter a trade?

Think step-by-step:
1. What is the market regime? Does my strategy fit this regime?
2. Have I recently lost money on similar setups?
3. Is the risk/reward favorable?
4. What could go wrong?

Respond in this EXACT JSON format:
{{
  "signal": "LONG" | "SHORT" | "HOLD",
  "confidence": 0-100,
  "reasoning": "Brief explanation of your logic",
  "risk_factors": "What could make this trade fail?",
  "learning_note": "Any insight from recent trades?"
}}

RESPOND ONLY WITH VALID JSON, NO EXTRA TEXT.
"""
        return prompt
    
    def _parse_llm_response(self, response):
        """Parse LLM JSON response"""
        try:
            # Try to parse as JSON
            if isinstance(response, str):
                # Remove markdown code blocks if present
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
            print(f"Response was: {response}")
            
            # Fallback: extract signal from text
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
        """
        Rule-based strategy if LLM not available
        
        Improved from Week 3:
        - Uses regime information
        - Considers recent performance
        - More conservative
        """
        # Check regime
        if context['regime'] == 'STRONG_TREND_DOWN':
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'reasoning': 'Strong downtrend - avoid longs (Week 4 rule)',
                'risk_factors': 'Mean reversion fails in trends',
                'learning_note': 'Learned from Week 3 failures'
            }
        
        # RSI-based signals with regime filter
        rsi = context['rsi']
        
        if rsi < 30 and context['trend_direction'] == 'UP':
            return {
                'signal': 'LONG',
                'confidence': 70,
                'reasoning': f'Oversold (RSI={rsi:.1f}) in uptrend',
                'risk_factors': 'Could stay oversold if trend reverses',
                'learning_note': ''
            }
        elif rsi > 70:
            return {
                'signal': 'SHORT',
                'confidence': 60,
                'reasoning': f'Overbought (RSI={rsi:.1f})',
                'risk_factors': 'Could stay overbought in strong uptrend',
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
        """
        Filter signal based on market regime
        
        THIS IS THE KEY WEEK 4 IMPROVEMENT!
        """
        regime = regime_info['regime']
        signal_direction = signal['signal']
        
        # Check if trade allowed in this regime
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
        """
        Update trade history - CRITICAL for meta-learning!
        
        This is what Week 3 was missing!
        """
        self.trade_history.append({
            'timestamp': timestamp,
            'pnl': pnl
        })
        
        self.total_trades += 1
        
        # Update consecutive loss counter
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Analyze patterns every 10 trades
        if self.total_trades % 10 == 0:
            self._analyze_performance_patterns()
    
    def _analyze_performance_patterns(self):
        """
        Analyze what's working and what's not
        
        Future improvement: Could ask LLM to analyze this!
        """
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


# Test function
def test_improved_strategy():
    """Test the improved strategy"""
    print("=" * 70)
    print("IMPROVED LLM STRATEGY - Week 4")
    print("=" * 70)
    
    strategy = ImprovedLLMStrategy()
    
    print("\n✅ Strategy initialized")
    print("\nKey Improvements:")
    print("  1. Rich context (not just price/RSI)")
    print("  2. Regime awareness (prevents Week 3 mistakes)")
    print("  3. Meta-strategy rules (cooldown after losses)")
    print("  4. Trade history tracking (learn from mistakes)")
    print("  5. Better prompts (specific instructions)")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_improved_strategy()
