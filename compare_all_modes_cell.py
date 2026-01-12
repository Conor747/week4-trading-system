# Week 4: Compare All Trading Modes
# Add this as a NEW cell in your notebook (don't replace existing!)

"""
This cell compares CONSERVATIVE, BALANCED, and AGGRESSIVE trading modes
while keeping your original Week 3 results for comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
from week4_backtest_engine import Week4BacktestEngine
import os

print("="*70)
print("WEEK 4: TRADING MODE COMPARISON")
print("="*70)

# Your Week 3 results (from previous backtest)
week3_results = {
    'RSI': {'return': 1.09, 'trades': 12, 'win_rate': 50.0, 'max_dd': 7.45},
    'LLM': {'return': -4.75, 'trades': 16, 'win_rate': 31.25, 'max_dd': 13.05},
    'B&H': {'return': -7.62, 'trades': 1, 'win_rate': 0.0, 'max_dd': 20.01}
}

# Run Week 4 in all 3 modes
modes = ['CONSERVATIVE', 'BALANCED', 'AGGRESSIVE']
week4_results = {}

for mode in modes:
    print(f"\n{'='*70}")
    print(f"Testing Week 4 - {mode} Mode")
    print(f"{'='*70}")
    
    # Create engine with specific mode
    engine = Week4BacktestEngine(
        initial_capital=10000,
        fee_rate=0.001,
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        trading_mode=mode
    )
    
    # Run backtest
    results = engine.run(df, symbol='BTC/USDT')
    
    # Store results
    week4_results[f'Week4_{mode}'] = {
        'return': results['metrics']['total_return_pct'],
        'trades': results['metrics']['total_trades'],
        'win_rate': results['metrics']['win_rate'],
        'max_dd': results['metrics']['max_drawdown_pct']
    }
    
    # Print summary
    print(f"\n📊 {mode} Mode Results:")
    print(f"   Total Trades: {results['metrics']['total_trades']}")
    print(f"   Win Rate: {results['metrics']['win_rate']:.2f}%")
    print(f"   Return: {results['metrics']['total_return_pct']:+.2f}%")
    print(f"   Max Drawdown: {results['metrics']['max_drawdown_pct']:.2f}%")

# Combine all results
all_results = {**week3_results, **week4_results}

# Create comparison DataFrame
comparison_df = pd.DataFrame(all_results).T
comparison_df.columns = ['Return (%)', 'Trades', 'Win Rate (%)', 'Max DD (%)']

print("\n" + "="*70)
print("COMPLETE COMPARISON - All Strategies & Modes")
print("="*70)
print(comparison_df)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

strategies = comparison_df.index.tolist()
colors = ['steelblue', 'coral', 'lightgreen', 
          'purple', 'orange', 'pink']  # 6 strategies

# Plot 1: Return
axes[0, 0].bar(strategies, comparison_df['Return (%)'], color=colors)
axes[0, 0].set_title('Total Return (%)')
axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Total Trades
axes[0, 1].bar(strategies, comparison_df['Trades'], color=colors)
axes[0, 1].set_title('Total Trades')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Win Rate
axes[1, 0].bar(strategies, comparison_df['Win Rate (%)'], color=colors)
axes[1, 0].set_title('Win Rate (%)')
axes[1, 0].axhline(y=50, color='gray', linestyle='--', alpha=0.3, label='50% (Random)')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Max Drawdown
axes[1, 1].bar(strategies, comparison_df['Max DD (%)'], color='red', alpha=0.6)
axes[1, 1].set_title('Max Drawdown (%) - Lower is Better')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('week4_mode_comparison.png', dpi=300, bbox_inches='tight')
print("\n📊 Comparison chart saved: week4_mode_comparison.png")
plt.show()

# Analysis
print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

# Find best in each category
best_return = comparison_df['Return (%)'].idxmax()
most_trades = comparison_df['Trades'].idxmax()
best_win_rate = comparison_df['Win Rate (%)'].idxmax()
lowest_dd = comparison_df['Max DD (%)'].idxmin()

print(f"\n🏆 Best Return: {best_return} ({comparison_df.loc[best_return, 'Return (%)']:+.2f}%)")
print(f"📊 Most Trades: {most_trades} ({int(comparison_df.loc[most_trades, 'Trades'])} trades)")
print(f"🎯 Best Win Rate: {best_win_rate} ({comparison_df.loc[best_win_rate, 'Win Rate (%)']:.2f}%)")
print(f"🛡️ Lowest Drawdown: {lowest_dd} ({comparison_df.loc[lowest_dd, 'Max DD (%)']:.2f}%)")

# Week 4 improvements across modes
print(f"\n📈 Week 4 Trading Activity:")
for mode in modes:
    strategy_name = f'Week4_{mode}'
    trades = int(comparison_df.loc[strategy_name, 'Trades'])
    print(f"   {mode:15s}: {trades:2d} trades")

print(f"\n💡 Week 3 LLM had: {int(comparison_df.loc['LLM', 'Trades'])} trades")

# Calculate improvements vs Week 3 LLM
week3_llm = comparison_df.loc['LLM']
print(f"\n🎯 Week 4 Improvements (vs Week 3 LLM):")
for mode in modes:
    strategy_name = f'Week4_{mode}'
    week4 = comparison_df.loc[strategy_name]
    
    return_diff = week4['Return (%)'] - week3_llm['Return (%)']
    trades_diff = week4['Trades'] - week3_llm['Trades']
    wr_diff = week4['Win Rate (%)'] - week3_llm['Win Rate (%)']
    dd_diff = week4['Max DD (%)'] - week3_llm['Max DD (%)']
    
    print(f"\n{mode} Mode:")
    print(f"   Return Change: {return_diff:+.2f}%")
    print(f"   Trade Difference: {int(trades_diff):+d}")
    print(f"   Win Rate Change: {wr_diff:+.2f}%")
    print(f"   Drawdown Change: {dd_diff:+.2f}%")

# Save detailed comparison
comparison_df.to_csv('week4_all_modes_comparison.csv')
print(f"\n💾 Full comparison saved: week4_all_modes_comparison.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
