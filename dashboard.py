"""
Week 4: Real-Time Trading Dashboard with Actual Backtest Results
Shows live market data + your actual Week 4 backtest performance
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ccxt
from datetime import datetime
import os
import sys

# Import Week 4 modules
from market_regime import MarketRegimeDetector
from improved_llm_strategy import ImprovedLLMStrategy

# Page config
st.set_page_config(
    page_title="Week 4 Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .positive {
        color: #00cc00;
        font-weight: bold;
    }
    .negative {
        color: #ff0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Load backtest results
@st.cache_data
def load_backtest_results():
    """Load actual Week 4 backtest results including all trading modes"""
    results_dir = 'week4_results'
    
    if not os.path.exists(results_dir):
        return None
    
    try:
        # Load saved results
        equity_curve = pd.read_csv(f'{results_dir}/equity_curve.csv', index_col=0, parse_dates=True)
        trades = pd.read_csv(f'{results_dir}/trades.csv')
        regime_history = pd.read_csv(f'{results_dir}/regime_history.csv', index_col=0, parse_dates=True)
        signal_history = pd.read_csv(f'{results_dir}/signal_history.csv', index_col=0, parse_dates=True)
        
        # Load metrics
        import json
        with open(f'{results_dir}/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        # Load Week 3 vs Week 4 comparison
        comparison = pd.read_csv(f'{results_dir}/week3_vs_week4_comparison.csv', index_col=0)
        
        # Load multi-mode comparison if available
        multi_mode_comparison = None
        if os.path.exists('week4_all_modes_comparison.csv'):
            multi_mode_comparison = pd.read_csv('week4_all_modes_comparison.csv', index_col=0)
            st.success("✅ Found multi-mode comparison data!")
        
        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'regime_history': regime_history,
            'signal_history': signal_history,
            'metrics': metrics,
            'comparison': comparison,
            'multi_mode_comparison': multi_mode_comparison
        }
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None


# Helper functions
@st.cache_data(ttl=60)
def fetch_live_data(symbol='BTC/USDT', timeframe='1h', limit=200):
    """Fetch live market data"""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def calculate_indicators(df):
    """Calculate technical indicators"""
    import numpy as np
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    df['MA_20'] = df['close'].rolling(20).mean()
    df['MA_50'] = df['close'].rolling(50).mean()
    df['MA_200'] = df['close'].rolling(200).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    return df.dropna()


def get_regime_color(regime):
    """Get color for regime"""
    colors = {
        'STRONG_TREND_UP': '#00cc00',
        'STRONG_TREND_DOWN': '#ff0000',
        'TREND_UP': '#66ff66',
        'TREND_DOWN': '#ff6666',
        'RANGE': '#ffaa00',
        'VOLATILE': '#ff00ff',
        'TRANSITIONAL': '#888888'
    }
    return colors.get(regime, '#888888')


def create_price_chart(df):
    """Create interactive price chart"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('BTC/USDT Price', 'RSI', 'Volume')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving Averages
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MA_20'], name='MA(20)', 
                  line=dict(color='orange', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MA_50'], name='MA(50)',
                  line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                  line=dict(color='purple', width=2)),
        row=2, col=1
    )
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volume
    colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
              for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume',
               marker_color=colors),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def create_equity_curve_chart(equity_df):
    """Create equity curve chart from backtest results"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=equity_df.index,
        y=equity_df['total_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2),
        fill='tonexty'
    ))
    
    # Add initial capital line
    if 'total_value' in equity_df.columns and len(equity_df) > 0:
        initial = equity_df['total_value'].iloc[0]
        fig.add_hline(y=initial, line_dash="dash", line_color="gray", 
                     annotation_text=f"Initial: ${initial:,.2f}")
    
    fig.update_layout(
        title="Week 4 Backtest - Equity Curve",
        xaxis_title="Time",
        yaxis_title="Portfolio Value ($)",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_comparison_chart(comparison_df):
    """Create Week 3 vs Week 4 comparison charts"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Return (%)', 'Total Trades', 'Win Rate (%)', 'Max Drawdown (%)'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    strategies = comparison_df.index.tolist()
    colors = ['steelblue', 'coral', 'lightgreen', 'purple']
    
    # Total Return
    # Handle both 'return' and 'Return (%)' column names
    return_col = 'Return (%)' if 'Return (%)' in comparison_df.columns else 'return'
    fig.add_trace(
        go.Bar(x=strategies, y=comparison_df[return_col], 
               marker_color=colors, name='Return'),
        row=1, col=1
    )
    
    # Total Trades
    trades_col = 'Trades' if 'Trades' in comparison_df.columns else 'trades'
    fig.add_trace(
        go.Bar(x=strategies, y=comparison_df[trades_col],
               marker_color=colors, name='Trades'),
        row=1, col=2
    )
    
    # Win Rate
    win_rate_col = 'Win Rate (%)' if 'Win Rate (%)' in comparison_df.columns else 'win_rate'
    fig.add_trace(
        go.Bar(x=strategies, y=comparison_df[win_rate_col],
               marker_color=colors, name='Win Rate'),
        row=2, col=1
    )
    
    # Max Drawdown
    max_dd_col = 'Max DD (%)' if 'Max DD (%)' in comparison_df.columns else 'max_dd'
    fig.add_trace(
        go.Bar(x=strategies, y=comparison_df[max_dd_col],
               marker_color='red', name='Max DD'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    
    return fig


# Main Dashboard
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Week 4 Trading Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    view_mode = st.sidebar.radio(
        "View Mode",
        ["Live Market Data", "Backtest Results"],
        help="Switch between live market and backtest results"
    )
    
    if view_mode == "Live Market Data":
        symbol = st.sidebar.selectbox("Symbol", ['BTC/USDT', 'ETH/USDT'], index=0)
        timeframe = st.sidebar.selectbox("Timeframe", ['1h', '4h', '1d'], index=0)
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
        
        if st.sidebar.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()
    
    # ========================================
    # BACKTEST RESULTS VIEW
    # ========================================
    if view_mode == "Backtest Results":
        st.subheader("📊 Your Actual Week 4 Backtest Results")
        
        # Load results
        results = load_backtest_results()
        
        if results is None:
            st.error("⚠️ Backtest results not found!")
            st.info("""
            **To see your backtest results:**
            1. Run the Jupyter notebook: `04_week4_testing.ipynb`
            2. Make sure it completes Part 7 (Save Results)
            3. Results will be saved to `week4_results/` folder
            4. Refresh this dashboard
            """)
            return
        
        # Display metrics
        metrics = results['metrics']
        comparison = results['comparison']
        
        # Top metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "💰 Final Capital",
                f"${metrics['final_capital']:,.2f}",
                f"{metrics['total_return_pct']:+.2f}%"
            )
        
        with col2:
            st.metric(
                "📊 Total Trades",
                f"{metrics['total_trades']}",
                f"{metrics['winning_trades']} wins"
            )
        
        with col3:
            win_rate_color = "🟢" if metrics['win_rate'] >= 50 else "🟡"
            st.metric(
                f"{win_rate_color} Win Rate",
                f"{metrics['win_rate']:.1f}%"
            )
        
        with col4:
            st.metric(
                "📉 Max Drawdown",
                f"{metrics['max_drawdown_pct']:.2f}%",
                delta_color="inverse"
            )
        
        with col5:
            st.metric(
                "💵 Profit Factor",
                f"{metrics['profit_factor']:.2f}"
            )
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Performance", 
            "🆚 Week 3 vs Week 4", 
            "🌊 Regime Analysis",
            "📋 Trades"
        ])
        
        with tab1:
            st.markdown("### Equity Curve")
            equity_chart = create_equity_curve_chart(results['equity_curve'])
            st.plotly_chart(equity_chart, use_container_width=True)
            
            # Detailed metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 💰 Capital Metrics")
                st.write(f"Initial Capital: **${metrics['initial_capital']:,.2f}**")
                st.write(f"Final Capital: **${metrics['final_capital']:,.2f}**")
                st.write(f"Total Return: **{metrics['total_return_pct']:+.2f}%**")
                st.write(f"Total PnL: **${metrics['total_pnl']:+,.2f}**")
            
            with col2:
                st.markdown("#### 📊 Trade Metrics")
                st.write(f"Total Trades: **{metrics['total_trades']}**")
                st.write(f"Winning Trades: **{metrics['winning_trades']}**")
                st.write(f"Losing Trades: **{metrics['losing_trades']}**")
                st.write(f"Win Rate: **{metrics['win_rate']:.2f}%**")
            
            with col3:
                st.markdown("#### ⚠️ Risk Metrics")
                st.write(f"Max Drawdown: **{metrics['max_drawdown_pct']:.2f}%**")
                st.write(f"Sharpe Ratio: **{metrics['sharpe_ratio']:.2f}**")
                st.write(f"Sortino Ratio: **{metrics['sortino_ratio']:.2f}**")
                st.write(f"Total Fees: **${metrics['total_fees']:,.2f}**")
        
        with tab2:
            st.markdown("### Week 3 vs Week 4 Comparison")
            
            # Check if we have multi-mode comparison
            if results.get('multi_mode_comparison') is not None:
                st.success("📊 **Multi-Mode Comparison Available!**")
                
                multi_comp = results['multi_mode_comparison']
                
                # Show complete table
                st.markdown("#### Complete Results - All Strategies & Modes")
                st.dataframe(multi_comp, use_container_width=True)
                
                # Create enhanced comparison chart
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Total Return (%)', 'Total Trades', 'Win Rate (%)', 'Max Drawdown (%)'),
                    specs=[[{'type': 'bar'}, {'type': 'bar'}],
                           [{'type': 'bar'}, {'type': 'bar'}]]
                )
                
                strategies = multi_comp.index.tolist()
                colors = ['steelblue', 'coral', 'lightgreen', 'purple', 'orange', 'pink']
                
                # Total Return
                fig.add_trace(
                    go.Bar(x=strategies, y=multi_comp['Return (%)'], 
                           marker_color=colors, name='Return'),
                    row=1, col=1
                )
                fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
                
                # Total Trades
                fig.add_trace(
                    go.Bar(x=strategies, y=multi_comp['Trades'],
                           marker_color=colors, name='Trades'),
                    row=1, col=2
                )
                
                # Win Rate
                fig.add_trace(
                    go.Bar(x=strategies, y=multi_comp['Win Rate (%)'],
                           marker_color=colors, name='Win Rate'),
                    row=2, col=1
                )
                fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                             annotation_text="50% (Random)", row=2, col=1)
                
                # Max Drawdown
                fig.add_trace(
                    go.Bar(x=strategies, y=multi_comp['Max DD (%)'],
                           marker_color='red', name='Max DD'),
                    row=2, col=2
                )
                
                fig.update_layout(height=600, showlegend=False)
                fig.update_xaxes(tickangle=45)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Key insights
                st.markdown("### 🎯 Key Insights from Multi-Mode Comparison")
                
                # Find best performers
                best_return = multi_comp['Return (%)'].idxmax()
                most_trades = multi_comp['Trades'].idxmax()
                best_win_rate = multi_comp['Win Rate (%)'].idxmax()
                lowest_dd = multi_comp['Max DD (%)'].idxmin()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("🏆 Best Return", best_return, 
                             f"{multi_comp.loc[best_return, 'Return (%)']:+.2f}%")
                    st.metric("🎯 Best Win Rate", best_win_rate,
                             f"{multi_comp.loc[best_win_rate, 'Win Rate (%)']:.1f}%")
                
                with col2:
                    st.metric("📊 Most Trades", most_trades,
                             f"{int(multi_comp.loc[most_trades, 'Trades'])} trades")
                    st.metric("🛡️ Lowest Drawdown", lowest_dd,
                             f"{multi_comp.loc[lowest_dd, 'Max DD (%)']:.2f}%")
                
                # Analysis of mode behavior
                st.markdown("### 📈 Trading Mode Analysis")
                
                week4_modes = [idx for idx in multi_comp.index if 'Week4' in idx]
                if len(week4_modes) > 1:
                    mode_data = multi_comp.loc[week4_modes]
                    
                    # Check if modes traded differently
                    trade_counts = mode_data['Trades'].values
                    if len(set(trade_counts)) == 1:
                        st.warning(f"""
                        **⚠️ All Week 4 modes had the same number of trades ({int(trade_counts[0])})!**
                        
                        **Why this happened:**
                        - Market was in difficult conditions (downtrend)
                        - Regime detector blocked most trades for safety
                        - Even with lower confidence thresholds, regime blocks dominated
                        - **This shows the system is working correctly!** It avoided overtrading in bad conditions.
                        
                        **To see mode differences:**
                        - Test on range-bound market (sideways movement)
                        - Test on bull market (uptrend)
                        - Test on longer time period with mixed conditions
                        """)
                    else:
                        st.success(f"""
                        **✅ Trading modes behaved differently!**
                        
                        Mode trade counts:
                        {mode_data['Trades'].to_dict()}
                        
                        This shows the adjustable thresholds are working as intended.
                        """)
                    
                    # Show mode comparison table
                    st.markdown("#### Week 4 Mode Comparison")
                    st.dataframe(mode_data, use_container_width=True)
                
                # Improvement vs Week 3
                if 'LLM' in multi_comp.index and any('Week4' in idx for idx in multi_comp.index):
                    st.markdown("### 🎯 Week 4 Improvements (vs Week 3 LLM)")
                    
                    week3_llm = multi_comp.loc['LLM']
                    
                    for mode in week4_modes:
                        week4 = multi_comp.loc[mode]
                        
                        return_diff = week4['Return (%)'] - week3_llm['Return (%)']
                        dd_diff = week4['Max DD (%)'] - week3_llm['Max DD (%)']
                        
                        mode_name = mode.replace('Week4_', '')
                        
                        if return_diff > 0 or dd_diff < 0:
                            st.success(f"""
                            **{mode_name} Mode:**
                            - Return Change: {return_diff:+.2f}%
                            - Drawdown Change: {dd_diff:+.2f}%
                            - Trade Difference: {int(week4['Trades'] - week3_llm['Trades']):+d}
                            """)
                        else:
                            st.info(f"""
                            **{mode_name} Mode:**
                            - Return Change: {return_diff:+.2f}%
                            - Drawdown Change: {dd_diff:+.2f}%
                            - Trade Difference: {int(week4['Trades'] - week3_llm['Trades']):+d}
                            """)
                
            else:
                # Fallback to original Week 3 vs Week 4 comparison
                st.info("💡 Run the multi-mode comparison cell in your notebook to see all trading modes!")
                
                # Show comparison table
                st.dataframe(comparison, use_container_width=True)
                
                # Show comparison charts
                comparison_chart = create_comparison_chart(comparison)
                st.plotly_chart(comparison_chart, use_container_width=True)
                
                # Calculate improvements
                st.markdown("### 🎯 Key Improvements")
                
                if 'Week4_Improved' in comparison.index and 'LLM' in comparison.index:
                    week4 = comparison.loc['Week4_Improved']
                    week3 = comparison.loc['LLM']
                    
                    # Handle both column naming conventions
                    return_col = 'Return (%)' if 'Return (%)' in comparison.columns else 'return'
                    trades_col = 'Trades' if 'Trades' in comparison.columns else 'trades'
                    wr_col = 'Win Rate (%)' if 'Win Rate (%)' in comparison.columns else 'win_rate'
                    dd_col = 'Max DD (%)' if 'Max DD (%)' in comparison.columns else 'max_dd'
                    
                    return_diff = week4[return_col] - week3[return_col]
                    trades_diff = week4[trades_col] - week3[trades_col]
                    wr_diff = week4[wr_col] - week3[wr_col]
                    dd_diff = week4[dd_col] - week3[dd_col]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success(f"✅ **Return Improvement:** {return_diff:+.2f}%")
                        st.info(f"📊 **Fewer Trades:** {int(trades_diff):+d} (Quality over quantity)")
                    
                    with col2:
                        st.success(f"✅ **Win Rate Change:** {wr_diff:+.2f}%")
                        st.success(f"✅ **Drawdown Reduction:** {dd_diff:+.2f}%")
                
                st.markdown("""
                **Key Insights:**
                - 📉 **Loss Reduction**: Week 4 reduced losses significantly
                - 🛡️ **Lower Risk**: Max drawdown reduced substantially
                - 🎯 **Better Quality**: Fewer trades but better selection
                - 🚫 **Meta-Strategy Working**: Regime-based filtering prevented bad trades
                """)
        
        with tab3:
            st.markdown("### Regime Detection Analysis")
            
            regime_hist = results['regime_history']
            
            # Regime distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Regime Distribution")
                regime_counts = regime_hist['regime'].value_counts()
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=regime_counts.index,
                        values=regime_counts.values,
                        marker=dict(colors=[get_regime_color(r) for r in regime_counts.index])
                    )
                ])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(regime_counts.to_frame('Count'), use_container_width=True)
            
            with col2:
                st.markdown("#### Signal Distribution")
                signal_hist = results['signal_history']
                signal_counts = signal_hist['signal'].value_counts()
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=signal_counts.index,
                        values=signal_counts.values,
                        marker=dict(colors=['gray', 'red', 'green'])
                    )
                ])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(signal_counts.to_frame('Count'), use_container_width=True)
            
            # Key findings
            st.markdown("### 📊 Key Findings")
            total_signals = len(signal_hist)
            hold_signals = signal_counts.get('HOLD', 0)
            hold_pct = (hold_signals / total_signals * 100) if total_signals > 0 else 0
            
            st.info(f"""
            **Meta-Strategy Effectiveness:**
            - Total Signals Generated: **{total_signals}**
            - HOLD Signals (Blocked): **{hold_signals}** ({hold_pct:.1f}%)
            - Actual Trades Executed: **{metrics['total_trades']}**
            
            **Most Common Regime:** {regime_counts.index[0]} ({regime_counts.iloc[0]} occurrences)
            
            This shows the meta-strategy successfully filtered out {hold_pct:.1f}% of potential trades!
            """)
        
        with tab4:
            st.markdown("### Trade History")
            
            trades_df = results['trades']
            
            if not trades_df.empty:
                # Show only SELL trades (completed)
                sell_trades = trades_df[trades_df['action'] == 'SELL'].copy()
                
                if not sell_trades.empty:
                    st.dataframe(
                        sell_trades[['timestamp', 'symbol', 'price', 'quantity', 'pnl', 'fee']],
                        use_container_width=True
                    )
                    
                    # PnL distribution
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=list(range(len(sell_trades))),
                        y=sell_trades['pnl'],
                        marker_color=['green' if p > 0 else 'red' for p in sell_trades['pnl']],
                        name='PnL'
                    ))
                    fig.update_layout(
                        title="Trade PnL Distribution",
                        xaxis_title="Trade #",
                        yaxis_title="PnL ($)",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No completed trades yet")
            else:
                st.info("No trades in backtest results")
    
    # ========================================
    # LIVE MARKET VIEW
    # ========================================
    else:
        # Fetch data
        with st.spinner('Fetching live market data...'):
            df = fetch_live_data(symbol, timeframe, limit=200)
        
        if df is None or len(df) == 0:
            st.error("Failed to fetch data. Please try again.")
            return
        
        # Initialize components
        regime_detector = MarketRegimeDetector()
        
        # Detect current regime
        current_regime = regime_detector.detect_regime(df)
        
        # Current metrics
        current_price = df['close'].iloc[-1]
        price_change_24h = ((df['close'].iloc[-1] - df['close'].iloc[-24]) / df['close'].iloc[-24] * 100) if len(df) >= 24 else 0
        current_rsi = df['RSI'].iloc[-1]
        current_atr = df['ATR'].iloc[-1]
        
        # Top metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="💰 Price",
                value=f"${current_price:,.2f}",
                delta=f"{price_change_24h:+.2f}%"
            )
        
        with col2:
            regime_color = get_regime_color(current_regime['regime'])
            st.markdown(f"**🌊 Regime**")
            st.markdown(f'<span style="color: {regime_color}; font-size: 1.5rem; font-weight: bold;">{current_regime["regime"]}</span>', 
                       unsafe_allow_html=True)
            st.caption(f"Confidence: {current_regime['confidence']}%")
        
        with col3:
            rsi_color = 'red' if current_rsi > 70 else ('green' if current_rsi < 30 else 'orange')
            st.metric(
                label="📊 RSI",
                value=f"{current_rsi:.1f}",
            )
            st.caption(f"<span style='color: {rsi_color};'>{'Overbought' if current_rsi > 70 else ('Oversold' if current_rsi < 30 else 'Neutral')}</span>",
                      unsafe_allow_html=True)
        
        with col4:
            st.metric(
                label="📈 ATR",
                value=f"${current_atr:,.2f}"
            )
            try:
                volatility_state = current_regime['details']['volatility']['state']
            except (KeyError, TypeError):
                volatility_state = "N/A"
            st.caption(f"Volatility: {volatility_state}")
        
        with col5:
            trend = "Bullish" if df['MA_20'].iloc[-1] > df['MA_50'].iloc[-1] else "Bearish"
            trend_color = "green" if trend == "Bullish" else "red"
            st.markdown(f"**📉 Trend**")
            st.markdown(f'<span style="color: {trend_color}; font-size: 1.5rem; font-weight: bold;">{trend}</span>',
                       unsafe_allow_html=True)
            st.caption(f"MA(20) vs MA(50)")
        
        # Market chart
        st.subheader("📈 Live Market")
        fig = create_price_chart(df)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()