# Week 4: Advanced AI Trading System with Regime Detection

**A sophisticated cryptocurrency trading system featuring multi-layer decision architecture, intelligent regime detection, and meta-strategy risk management.**

---

##  Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Results Summary](#results-summary)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

---

##  Overview

This project represents the culmination of a four-week quantitative trading system development, focusing on intelligent market regime detection and adaptive risk management. The system achieved a **69% reduction in risk** (maximum drawdown) and **80% reduction in losses** compared to previous iterations.

### Project Evolution

- **Week 1**: Basic RSI momentum strategy
- **Week 2**: LLM-enhanced signal generation with DeepSeek API
- **Week 3**: Risk management layer and comprehensive backtesting framework
- **Week 4**: Regime detection, meta-strategy rules, and multi-mode architecture **(Current)**

### Key Achievement

Built a trading system that **knows when NOT to trade**, blocking 89.4% of signals during unfavorable market conditions while preserving capital.

---

##  Key Features

###  Five-Layer Decision Architecture

1. **Regime Detection**: Classifies market into 7 distinct regimes (STRONG_TREND_UP/DOWN, TREND_UP/DOWN, RANGE, VOLATILE, TRANSITIONAL)
2. **Meta-Strategy Rules**: Enforces cooldowns after losses, blocks trades during extreme volatility
3. **LLM Signal Generation**: DeepSeek AI analyzes market with full regime context
4. **Regime Filter**: Validates signals against current market conditions
5. **Risk Management**: Dynamic position sizing and stop-loss mechanisms

###  Three Trading Modes

- **CONSERVATIVE**: 70% confidence threshold, strict RSI levels (30/70), minimal risk
- **BALANCED**: 60% confidence threshold, moderate RSI levels (32/68), balanced approach
- **AGGRESSIVE**: 50% confidence threshold, relaxed RSI levels (35/65), maximum participation

###  Interactive Dashboard

- Real-time market monitoring with live regime detection
- Comprehensive backtest result visualization
- Multi-strategy performance comparison
- Regime analysis and signal distribution charts

###  AI Provider Flexibility

- Supports multiple LLM providers (DeepSeek, ChatGPT, Claude)
- Cost-effective implementation with DeepSeek (95% performance at 5% cost)
- Easy API provider switching

---

##  Project Structure

```
Week4_Project/
├── Core Trading System
│   ├── market_regime.py                    # 7-regime detection system
│   ├── improved_llm_strategy.py            # Original LLM strategy
│   ├── improved_llm_strategy_adjustable.py # Multi-mode strategy (NEW)
│   ├── week4_backtest_engine.py            # Complete backtest engine
│   ├── portfolio.py                        # Portfolio management (Week 3)
│   ├── risk_manager.py                     # Risk controls (Week 3)
│   └── metrics.py                          # Performance metrics (Week 3)
│
├── Jupyter Notebooks
│   ├── 04_week4_testing.ipynb              # Main analysis notebook
│   
│
├── Dashboard
│   └── dashboard.py                        # Streamlit interactive dashboard
│
├── Results & Data
│   ├── week4_results/                      # Backtest outputs
│   │   ├── equity_curve.csv
│   │   ├── trades.csv
│   │   ├── regime_history.csv
│   │   ├── signal_history.csv
│   │   ├── metrics.json
│   │   └── week3_vs_week4_comparison.csv
│   ├── week4_all_modes_comparison.csv      # Multi-mode comparison
│   └── week4_mode_comparison.png           # Visualization
│
├── Documentation
│   ├── README.md                           # This file
│   ├── Week4_Analysis_Report.docx          # Comprehensive report
│   ├── presentation_script_20min.md        # Presentation guide
│   ├── dashboard_demo_script.md            # Dashboard demo guide
│   └── AI_Provider_Impact_Analysis.md      # AI provider comparison
│
├── Configuration
    ├── .env                                # API keys (not in git)
                     

```

---

##  Installation

### Prerequisites

- Python 3.8 or higher
- Conda (recommended) or pip
- Jupyter Notebook
- API Key from DeepSeek (or other LLM provider)


### Configure API Keys

Create a `.env` file in the project root:

```bash
# .env file
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Optional: Add other providers
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_claude_key_here
```

**Get API Keys:**
- DeepSeek: https://platform.deepseek.com/
- OpenAI: https://platform.openai.com/
- Anthropic: https://console.anthropic.com/

---

##  Quick Start

### Option 1: Run Complete Analysis 
```bash
# Activate environment
conda activate quant

# Launch Jupyter
jupyter notebook

# Open: 04_week4_testing.ipynb
# Run all cells (Cell -> Run All)
```

**What this does:**
1. Loads historical market data
2. Tests regime detection system
3. Runs Week 4 backtest (CONSERVATIVE mode)
4. Compares all three trading modes
5. Generates comparison charts
6. Saves results to `week4_results/`

**Expected runtime:** ~30-45 minutes with live API calls

### Option 2: Launch Dashboard

```bash
# Activate environment
conda activate quant

# Run dashboard
streamlit run dashboard.py
```

**Dashboard URL:** http://localhost:8501

**Features:**
- View backtest results
- Compare strategies
- Monitor live market
- Analyze regime detection

### Option 3: Run Quick Backtest (Python Script)

```python
from week4_backtest_engine import Week4BacktestEngine
import pandas as pd
import os

# Load data
df = pd.read_csv('data/btc_usdt_1h.csv', parse_dates=['timestamp'])

# Initialize engine
engine = Week4BacktestEngine(
    initial_capital=10000,
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    trading_mode='BALANCED'  # or CONSERVATIVE, AGGRESSIVE
)

# Run backtest
results = engine.run(df, symbol='BTC/USDT')

# Print results
engine.print_results(results)
```

---

##  System Architecture

### Five-Layer Decision Flow

```
┌─────────────────────────────────────────┐
│  1. REGIME DETECTION                    │
│     Analyzes: Trend, Volatility,        │
│              Momentum, Structure         │
│     Output: STRONG_TREND_DOWN (92% conf)│
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  2. META-STRATEGY RULES                 │
│     Checks: Consecutive losses,         │
│             Extreme volatility,          │
│             Transitional regimes         │
│     Decision: ALLOW or BLOCK            │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  3. LLM SIGNAL GENERATION               │
│     Input: Price history + Regime info  │
│     Provider: DeepSeek (or GPT/Claude)  │
│     Output: LONG/SHORT/HOLD + Confidence│
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  4. REGIME FILTER                       │
│     Validates: Signal vs Current Regime │
│     Example: Block LONG in downtrend    │
│     Decision: PASS or REJECT            │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  5. RISK MANAGEMENT                     │
│     Calculates: Position size, Stops    │
│     Adjusts: Based on regime confidence │
│     Executes: Trade with risk controls  │
└──────────────┬──────────────────────────┘
               ↓
          TRADE EXECUTION
```

### Regime Classification System

| Regime | Characteristics | Trading Approach |
|--------|----------------|------------------|
| **STRONG_TREND_UP** | Strong upward momentum, high confidence | Aggressive long entries |
| **TREND_UP** | Moderate bullish trend | Selective long entries |
| **RANGE** | Sideways movement, mean-reverting | Mean reversion trades |
| **VOLATILE** | High uncertainty, chaotic | Reduced size or avoid |
| **TRANSITIONAL** | Unclear direction | Wait for clarity |
| **TREND_DOWN** | Moderate bearish trend | Cautious, reduced activity |
| **STRONG_TREND_DOWN** | Strong downward momentum | Avoid longs, preserve capital |

### Trading Mode Configurations

| Parameter | CONSERVATIVE | BALANCED | AGGRESSIVE |
|-----------|--------------|----------|------------|
| Min Confidence | 70% | 60% | 50% |
| RSI Oversold | < 30 | < 32 | < 35 |
| RSI Overbought | > 70 | > 68 | > 65 |
| Transitional Trade | ❌ Blocked | ❌ Blocked | ✅ Allowed |
| Max Consecutive Losses | 2 | 2 | 3 |
| Volatility Threshold | 95th % | 97th % | 98th % |

---

##  Results Summary

### Week 3 vs Week 4 Performance

| Strategy | Return (%) | Trades | Win Rate (%) | Max DD (%) |
|----------|------------|--------|--------------|------------|
| **RSI** | +1.09 | 12 | 50.00 | 7.45 |
| **LLM (Week 3)** | -4.75 | 16 | 31.25 | 13.05 |
| **Buy & Hold** | -7.62 | 1 | 0.00 | 20.01 |
| **Week4 CONSERVATIVE** | **-0.92** | **3** | **33.33** | **4.18** ✅ |
| **Week4 BALANCED** | -0.98 | 3 | 33.33 | 4.41 |
| **Week4 AGGRESSIVE** | -5.44 | 3 | 0.00 | 7.51 |

### Key Improvements (vs Week 3)

- ✅ **+3.83%** Return Improvement (80% loss reduction)
- ✅ **-13** Fewer Trades (quality over quantity)
- ✅ **+2.08%** Win Rate Improvement
- ✅ **-8.87%** Drawdown Reduction (69% risk reduction)

### Signal Filtering Effectiveness

- **Total Signals Generated:** 701
- **HOLD Signals (Blocked):** 627 (89.4%)
- **Trading Signals:** 74 (10.6%)
- **Actual Trades Executed:** 3 (0.4%)

**Interpretation:** Meta-strategy and regime detection successfully prevented 99.6% of potential trades during unfavorable conditions.

---

##  Usage Guide

### Running Backtests

#### Basic Backtest

```python
from week4_backtest_engine import Week4BacktestEngine
import pandas as pd

# Load data
df = pd.read_csv('data/btc_usdt_1h.csv', parse_dates=['timestamp'], index_col='timestamp')

# Create engine
engine = Week4BacktestEngine(
    initial_capital=10000,
    fee_rate=0.001,
    trading_mode='CONSERVATIVE'
)

# Run
results = engine.run(df, symbol='BTC/USDT')

# View results
engine.print_results(results)
```

#### Multi-Mode Comparison

```python
modes = ['CONSERVATIVE', 'BALANCED', 'AGGRESSIVE']
all_results = {}

for mode in modes:
    engine = Week4BacktestEngine(
        initial_capital=10000,
        trading_mode=mode
    )
    all_results[mode] = engine.run(df, symbol='BTC/USDT')

# Compare
import pandas as pd
comparison = pd.DataFrame({
    mode: results['metrics'] 
    for mode, results in all_results.items()
}).T
print(comparison)
```

### Using the Dashboard

#### Launch Dashboard

```bash
streamlit run dashboard.py
```

#### Features

**1. Backtest Results Mode:**
- View equity curves
- Compare all strategies
- Analyze regime distribution
- Review individual trades

**2. Live Market Mode:**
- Real-time price monitoring
- Current regime detection
- Live indicator values
- Trading signal status

#### Dashboard Navigation

```
Sidebar:
├── View Mode
│   ├── Live Market Data
│   └── Backtest Results
├── Symbol Selection (BTC/USDT, ETH/USDT)
├── Timeframe (1h, 4h, 1d)
└── Refresh Controls

Main Tabs:
├── Performance (Equity curves, metrics)
├── Week 3 vs Week 4 (Comparison)
├── Regime Analysis (Distribution, signals)
└── Trades (Individual executions)
```

### Regime Detection

```python
from market_regime import MarketRegimeDetector

# Initialize
detector = MarketRegimeDetector()

# Detect regime
regime_info = detector.detect_regime(df)

print(f"Regime: {regime_info['regime']}")
print(f"Confidence: {regime_info['confidence']}%")
print(f"Reason: {regime_info['reason']}")
print(f"Recommendation: {regime_info['recommendation']}")

# Check if trading allowed
allowed = detector.should_allow_trade(regime_info, signal='LONG')
print(f"Trade allowed: {allowed['allowed']}")
```

### LLM Strategy

```python
from improved_llm_strategy_adjustable import ImprovedLLMStrategy

# Initialize with trading mode
strategy = ImprovedLLMStrategy(
    api_key='your_api_key',
    trading_mode='BALANCED'
)

# Generate signal
signal = strategy.generate_signal_with_context(
    df=df,
    regime_info=regime_info,
    index=-1
)

print(f"Signal: {signal['signal']}")
print(f"Confidence: {signal['confidence']}%")
print(f"Reasoning: {signal['reasoning']}")
```

---

##  Configuration

### Environment Variables

```bash
# Required
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx

# Optional (for testing different providers)
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
```

### Trading Mode Selection

```python
# In notebook or script
trading_mode = 'CONSERVATIVE'  # Options: CONSERVATIVE, BALANCED, AGGRESSIVE

engine = Week4BacktestEngine(
    trading_mode=trading_mode
)
```

### Risk Parameters

```python
engine = Week4BacktestEngine(
    initial_capital=10000,          # Starting capital
    fee_rate=0.001,                 # 0.1% per trade
    slippage_pct=0.001,             # 0.1% slippage
    risk_per_trade=0.02,            # 2% risk per trade
    stop_loss_pct=0.02,             # 2% stop loss
    trading_mode='BALANCED'
)
```

### Regime Detection Sensitivity

Edit `market_regime.py` to adjust:

```python
class MarketRegimeDetector:
    def __init__(self):
        self.trend_threshold = 0.02      # 2% for trend detection
        self.strong_trend_threshold = 0.05  # 5% for strong trends
        self.volatility_window = 20      # Lookback for volatility
        # ... more parameters
```

---

##  Performance Metrics

### Metrics Calculated

- **Return Metrics**: Total return (%), total PnL ($)
- **Trade Metrics**: Total trades, winning trades, losing trades, win rate (%)
- **Risk Metrics**: Maximum drawdown (%), Sharpe ratio, Sortino ratio
- **Efficiency Metrics**: Profit factor, average win/loss, total fees

### Accessing Metrics

```python
results = engine.run(df, symbol='BTC/USDT')

# All metrics in dictionary
metrics = results['metrics']

print(f"Total Return: {metrics['total_return_pct']:.2f}%")
print(f"Win Rate: {metrics['win_rate']:.2f}%")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

---


---

##  Documentation

### Available Documents

- ** Week4_Analysis_Report.docx** - Comprehensive 15-page analysis report

### Academic References

This project builds on concepts from:
- **Technical Analysis**: RSI, Moving Averages, ATR
- **Regime Detection**: Market state classification, volatility analysis
- **Risk Management**: Position sizing, drawdown control, meta-strategy
- **Machine Learning**: LLM-enhanced signal generation
- **Behavioral Finance**: Cooldown mechanisms, loss aversion

---

##  Contributing

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement

- [ ] Implement short selling capability
- [ ] Add multi-asset support (stocks, forex, commodities)
- [ ] Extend to longer backtesting periods (6-12 months)
- [ ] A/B test multiple AI providers systematically
- [ ] Add paper trading mode for live validation
- [ ] Implement walk-forward optimization
- [ ] Add more technical indicators
- [ ] Create mobile-responsive dashboard
- [ ] Add email/SMS alerts for signals

---

##  License


---

## Acknowledgments

### Technologies Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Plotly**: Interactive charts
- **Streamlit**: Dashboard framework
- **CCXT**: Cryptocurrency exchange connectivity
- **DeepSeek API**: LLM-powered signal generation
- **Jupyter**: Interactive development environment

### Inspiration

- Academic research on regime detection
- Quantitative finance best practices
- Behavioral trading psychology

---


---

##  Project Objectives Achieved

✅ **Primary Goal**: Reduce risk while maintaining reasonable returns  
✅ **Architecture Goal**: Build multi-layer intelligent decision system  
✅ **Technical Goal**: Implement regime detection and meta-strategy  
✅ **Learning Goal**: Understand when NOT to trade is as important as when to trade  
✅ **Practical Goal**: Create usable, deployable trading system with dashboard  

---

##  Quick Reference

### Key Commands

```bash
# Environment
conda activate quant

# Analysis
jupyter notebook 04_week4_testing.ipynb

# Dashboard
streamlit run dashboard.py

# Quick test
python -c "from week4_backtest_engine import Week4BacktestEngine; print('✅ Ready')"
```

### Key Files

```bash
market_regime.py                    # Core regime detection
improved_llm_strategy_adjustable.py # Trading modes
week4_backtest_engine.py           # Main engine
dashboard.py                        # Visualization
04_week4_testing.ipynb             # Analysis notebook
```

### Key Results

```bash
week4_results/                      # All outputs
week4_all_modes_comparison.csv     # Performance comparison
week4_mode_comparison.png          # Visual comparison
Week4_Analysis_Report.docx         # Full report
```

---

**Last Updated:** January 2026  
**Version:** 1.0.0  
**Status:**  Complete & Production Ready

---

*"The best trade is sometimes the one you don't take."*

---
