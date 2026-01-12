"""
Week 2: LLM Signal Generator Module
Provides classes and utilities for generating trading signals using LLMs
"""

import json
import re
import time
from typing import Optional, Dict, List, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """
    Structured trading signal with all relevant information
    """
    action: str  # 'long', 'short', 'hold', 'close'
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Explanation from LLM
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Validate signal after initialization"""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        
        # Validate action
        valid_actions = ['long', 'short', 'hold', 'close']
        if self.action.lower() not in valid_actions:
            raise ValueError(f"Invalid action: {self.action}. Must be one of {valid_actions}")
        
        # Validate confidence
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1: {self.confidence}")
    
    def to_dict(self) -> Dict:
        """Convert signal to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert signal to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class PromptTemplate:
    """
    Manages prompt templates for LLM signal generation
    """
    
    @staticmethod
    def basic_template(market_data: Dict) -> str:
        """
        Basic prompt template - zero-shot approach
        
        Args:
            market_data: Dictionary containing market information
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a professional quantitative trader. Analyze the following market data and provide a trading signal.

MARKET DATA:
Symbol: {market_data.get('symbol', 'N/A')}
Current Price: ${market_data.get('price', 0):,.2f}
24h Change: {market_data.get('change_24h', 0):+.2f}%

TECHNICAL INDICATORS:
RSI(14): {market_data.get('rsi', 0):.2f}
MA(20): ${market_data.get('ma_20', 0):,.2f}
MA(50): ${market_data.get('ma_50', 0):,.2f}

TASK:
Provide a trading recommendation in the following JSON format:
{{
    "action": "long|short|hold",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation (max 150 words)",
    "entry_price": <suggested entry price>,
    "stop_loss": <suggested stop loss>,
    "take_profit": <suggested take profit>
}}

Respond ONLY with valid JSON, no additional text."""
        
        return prompt
    
    @staticmethod
    def few_shot_template(market_data: Dict) -> str:
        """
        Few-shot prompt template with examples
        
        Args:
            market_data: Dictionary containing market information
            
        Returns:
            Formatted prompt string with examples
        """
        prompt = f"""You are a professional quantitative trader. I will show you examples of how to analyze market data, then you will analyze new data.

EXAMPLE 1:
Market Data: BTC/USDT, Price: $40,000, RSI: 75, MA(20): $38,500
Analysis: RSI shows overbought conditions (>70) but price is well above MA(20), indicating strong uptrend. However, overbought RSI suggests potential pullback.
Signal: {{"action": "hold", "confidence": 0.6, "reasoning": "Overbought RSI in uptrend - wait for pullback or RSI cooldown before entering"}}

EXAMPLE 2:
Market Data: BTC/USDT, Price: $38,000, RSI: 28, MA(20): $39,000
Analysis: RSI shows oversold conditions (<30) but price is below MA(20), indicating downtrend. Oversold in downtrend can persist longer.
Signal: {{"action": "short", "confidence": 0.7, "reasoning": "Oversold in downtrend - momentum likely to continue down. Enter short with tight stops"}}

NOW ANALYZE THIS:
Symbol: {market_data.get('symbol', 'N/A')}
Current Price: ${market_data.get('price', 0):,.2f}
24h Change: {market_data.get('change_24h', 0):+.2f}%
RSI(14): {market_data.get('rsi', 0):.2f}
MA(20): ${market_data.get('ma_20', 0):,.2f}
MA(50): ${market_data.get('ma_50', 0):,.2f}
Volume vs Avg: {market_data.get('volume_vs_avg', '+0')}%

Provide your analysis in the same JSON format. Respond ONLY with valid JSON."""
        
        return prompt
    
    @staticmethod
    def chain_of_thought_template(market_data: Dict) -> str:
        """
        Chain-of-thought prompt for detailed reasoning
        
        Args:
            market_data: Dictionary containing market information
            
        Returns:
            Formatted prompt string encouraging step-by-step thinking
        """
        prompt = f"""You are a professional quantitative trader. Analyze this market data using a step-by-step approach.

MARKET DATA:
Symbol: {market_data.get('symbol', 'N/A')}
Current Price: ${market_data.get('price', 0):,.2f}
24h Change: {market_data.get('change_24h', 0):+.2f}%
RSI(14): {market_data.get('rsi', 0):.2f}
MA(20): ${market_data.get('ma_20', 0):,.2f}
MA(50): ${market_data.get('ma_50', 0):,.2f}
Volume: {market_data.get('volume_vs_avg', '+0')}% vs average

ANALYSIS STEPS:
1. TREND ANALYSIS: Is the price above or below key moving averages? What does this tell us?
2. MOMENTUM: What does the RSI indicate about current momentum?
3. VOLUME: Is volume confirming the price action?
4. SYNTHESIS: Combining all factors, what's the overall picture?
5. DECISION: Based on the analysis, what action should we take?

Think through each step, then provide your final decision in JSON format:
{{
    "action": "long|short|hold",
    "confidence": 0.0-1.0,
    "reasoning": "Your step-by-step analysis (max 200 words)",
    "entry_price": <price>,
    "stop_loss": <price>,
    "take_profit": <price>
}}

Respond with your analysis followed by the JSON."""
        
        return prompt
    
    @staticmethod
    def risk_aware_template(market_data: Dict) -> str:
        """
        Risk-focused prompt template
        
        Args:
            market_data: Dictionary containing market information
            
        Returns:
            Formatted prompt emphasizing risk management
        """
        prompt = f"""You are a risk-conscious quantitative trader. Your primary goal is capital preservation.

MARKET DATA:
Symbol: {market_data.get('symbol', 'N/A')}
Price: ${market_data.get('price', 0):,.2f}
RSI(14): {market_data.get('rsi', 0):.2f}
MA(20): ${market_data.get('ma_20', 0):,.2f}
MA(50): ${market_data.get('ma_50', 0):,.2f}

RISK MANAGEMENT RULES:
1. Never risk more than 2% of capital per trade
2. Always set stop-loss orders
3. Aim for minimum 2:1 risk-reward ratio
4. Avoid trading in unclear market conditions

TASK:
Analyze the data with risk as your top priority. Consider:
- What could go wrong with this trade?
- Where should we place stop-loss to limit downside?
- Is the risk-reward ratio favorable?
- Is there sufficient edge to justify the risk?

Provide your risk-aware analysis in JSON format:
{{
    "action": "long|short|hold",
    "confidence": 0.0-1.0,
    "reasoning": "Risk-focused explanation",
    "entry_price": <price>,
    "stop_loss": <price>,
    "take_profit": <price>,
    "risk_reward_ratio": <ratio>
}}

Respond ONLY with valid JSON."""
        
        return prompt


class LLMSignalGenerator:
    """
    Main class for generating trading signals using LLM APIs
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4", provider: str = "openai", base_url: str = None):
        """
        Initialize the signal generator
        
        Args:
            api_key: API key for LLM provider
            model: Model name (e.g., 'gpt-4', 'claude-3-opus', 'deepseek-chat')
            provider: API provider ('openai', 'anthropic', 'deepseek')
            base_url: Custom base URL for API (used for DeepSeek)
        """
        self.api_key = api_key
        self.model = model
        self.provider = provider.lower()
        self.prompt_template = PromptTemplate()
        
        # Initialize API client based on provider
        if self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
                
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        elif self.provider == "deepseek":
            try:
                import openai
                # DeepSeek uses OpenAI-compatible API
                self.client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url or "https://api.deepseek.com"
                )
                logger.info(f"Using DeepSeek API endpoint: {base_url or 'https://api.deepseek.com'}")
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        
        else:
            raise ValueError(f"Unsupported provider: {provider}. Choose from: openai, anthropic, deepseek")
        
        logger.info(f"Initialized LLMSignalGenerator with {provider}/{model}")
    
    def generate_signal(
        self,
        market_data: Dict,
        template_type: str = "basic",
        max_retries: int = 3,
        temperature: float = 0.7
    ) -> Optional[TradingSignal]:
        """
        Generate a trading signal from market data
        
        Args:
            market_data: Dictionary containing market information
            template_type: Type of prompt template ('basic', 'few_shot', 'cot', 'risk_aware')
            max_retries: Maximum number of API call retries
            temperature: LLM temperature parameter (0-1)
            
        Returns:
            TradingSignal object or None if generation fails
        """
        # Select prompt template
        if template_type == "few_shot":
            prompt = self.prompt_template.few_shot_template(market_data)
        elif template_type == "cot":
            prompt = self.prompt_template.chain_of_thought_template(market_data)
        elif template_type == "risk_aware":
            prompt = self.prompt_template.risk_aware_template(market_data)
        else:
            prompt = self.prompt_template.basic_template(market_data)
        
        logger.info(f"Generating signal using {template_type} template")
        
        # Call LLM API with retry logic
        for attempt in range(max_retries):
            try:
                response_text = self._call_api(prompt, temperature)
                
                if response_text:
                    signal = self._parse_response(response_text)
                    if signal:
                        logger.info(f"Successfully generated signal: {signal.action}")
                        return signal
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        logger.error("Failed to generate signal after all retries")
        return None
    
    def _call_api(self, prompt: str, temperature: float) -> Optional[str]:
        """
        Call the LLM API
        
        Args:
            prompt: Formatted prompt string
            temperature: LLM temperature parameter
            
        Returns:
            Response text or None if call fails
        """
        if self.provider == "openai" or self.provider == "deepseek":
            # Both OpenAI and DeepSeek use the same API format
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional quantitative trader providing structured trading signals."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=500
            )
            return response.choices[0].message.content
        
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        
        return None
    
    def _parse_response(self, response_text: str) -> Optional[TradingSignal]:
        """
        Parse LLM response into TradingSignal object
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            TradingSignal object or None if parsing fails
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    logger.error("No JSON found in response")
                    return None
            
            # Parse JSON
            signal_dict = json.loads(json_str)
            
            # Create TradingSignal object
            signal = TradingSignal(
                action=signal_dict.get('action', 'hold'),
                confidence=signal_dict.get('confidence', 0.5),
                reasoning=signal_dict.get('reasoning', 'No reasoning provided'),
                entry_price=signal_dict.get('entry_price'),
                stop_loss=signal_dict.get('stop_loss'),
                take_profit=signal_dict.get('take_profit'),
                risk_reward_ratio=signal_dict.get('risk_reward_ratio')
            )
            
            return signal
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response text: {response_text}")
            return None
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None
    
    def batch_generate(
        self,
        market_data_list: List[Dict],
        template_type: str = "basic",
        delay: float = 1.0
    ) -> List[Optional[TradingSignal]]:
        """
        Generate signals for multiple market snapshots
        
        Args:
            market_data_list: List of market data dictionaries
            template_type: Prompt template to use
            delay: Delay between API calls (seconds)
            
        Returns:
            List of TradingSignal objects
        """
        signals = []
        
        for i, market_data in enumerate(market_data_list):
            logger.info(f"Processing snapshot {i+1}/{len(market_data_list)}")
            signal = self.generate_signal(market_data, template_type)
            signals.append(signal)
            
            # Add delay to avoid rate limiting
            if i < len(market_data_list) - 1:
                time.sleep(delay)
        
        return signals


def format_market_data(df, index: int = -1) -> Dict:
    """
    Format pandas DataFrame row into market data dictionary
    
    Args:
        df: DataFrame with OHLCV data and indicators
        index: Row index to extract data from (default: -1 for latest)
        
    Returns:
        Dictionary with formatted market data
    """
    row = df.iloc[index]
    
    market_data = {
        'symbol': 'BTC/USDT',  # Can be parameterized
        'timestamp': str(df.index[index]),
        'price': float(row['close']),
        'change_24h': float((row['close'] - df.iloc[index-24]['close']) / df.iloc[index-24]['close'] * 100) if index >= 24 else 0,
        'rsi': float(row.get('RSI', 50)),
        'ma_20': float(row.get('MA_20', row['close'])),
        'ma_50': float(row.get('MA_50', row['close'])) if 'MA_50' in row else float(row['close']),
        'volume': float(row.get('volume', 0)),
        'volume_vs_avg': '+0'  # Can be calculated if needed
    }
    
    return market_data


# Example usage
if __name__ == "__main__":
    # Example market data
    sample_data = {
        'symbol': 'BTC/USDT',
        'price': 42350.00,
        'change_24h': 3.2,
        'rsi': 68.5,
        'ma_20': 41200.00,
        'ma_50': 40500.00,
        'volume_vs_avg': '+15'
    }
    
    # Initialize generator 
    # generator = LLMSignalGenerator(api_key="your-api-key-here", provider="openai")
    
    # Generate signal
    # signal = generator.generate_signal(sample_data, template_type="basic")
    
    # if signal:
    #     print(signal.to_json())
    
    print("LLM Signal Generator module loaded successfully!")
    print("Configure your API key to start generating signals.")
