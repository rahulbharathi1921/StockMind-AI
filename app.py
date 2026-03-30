import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# Import our modules
from data_engine import YahooFinanceEngine
from feature_engine import MarketIntelligenceEngine
from model_engine import MultiModelPredictionEngine
from visualization import StockVisualizationEngine

# Page configuration
st.set_page_config(
    page_title="AI Stock Intelligence Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stock-header {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #1E88E5;
    }
    .company-name {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .company-details {
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .positive {
        color: #26a69a;
        font-weight: bold;
    }
    .negative {
        color: #ef5350;
        font-weight: bold;
    }
    .prediction-bullish {
        background-color: rgba(38, 166, 154, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #26a69a;
    }
    .prediction-bearish {
        background-color: rgba(239, 83, 80, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ef5350;
    }
    .prediction-neutral {
        background-color: rgba(120, 144, 156, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #78909c;
    }
    .logo-container {
        text-align: center;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


class StockIntelligenceDashboard:
    """Main dashboard application"""

    def __init__(self):
        # Initialize engines
        self.data_engine = YahooFinanceEngine()
        self.feature_engine = MarketIntelligenceEngine()
        self.model_engine = MultiModelPredictionEngine()
        self.viz_engine = StockVisualizationEngine()

        # Session state initialization
        if 'symbol' not in st.session_state:
            st.session_state.symbol = 'AAPL'
        if 'interval' not in st.session_state:
            st.session_state.interval = '1d'
        if 'period' not in st.session_state:
            st.session_state.period = '3mo'
        if 'data' not in st.session_state:
            st.session_state.data = pd.DataFrame()
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
        if 'stock_info' not in st.session_state:
            st.session_state.stock_info = {}

        # Popular symbols with names
        self.popular_symbols_with_names = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc. (Google)',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'JPM': 'JPMorgan Chase & Co.',
            'JNJ': 'Johnson & Johnson',
            'V': 'Visa Inc.',
            'WMT': 'Walmart Inc.',
            'PG': 'Procter & Gamble Co.',
            'DIS': 'The Walt Disney Company',
            'NFLX': 'Netflix Inc.',
            'AMD': 'Advanced Micro Devices Inc.',
            'INTC': 'Intel Corporation',
            'CSCO': 'Cisco Systems Inc.',
            'ORCL': 'Oracle Corporation',
            'IBM': 'International Business Machines',
            'T': 'AT&T Inc.',
            'SPY': 'SPDR S&P 500 ETF',
            'QQQ': 'Invesco QQQ Trust',
            'DIA': 'SPDR Dow Jones Industrial Average ETF',
            'IWM': 'iShares Russell 2000 ETF',
            'VTI': 'Vanguard Total Stock Market ETF'
        }

        # Interval options
        self.interval_options = {
            '1m': '1 Minute',
            '5m': '5 Minutes',
            '1h': '1 Hour',
            '1d': '1 Day',
            '1wk': '1 Week',
            '1mo': '1 Month'
        }

        # Period options
        self.period_options = {
            '1d': '1 Day',
            '5d': '5 Days',
            '1mo': '1 Month',
            '3mo': '3 Months',
            '6mo': '6 Months',
            '1y': '1 Year',
            '2y': '2 Years'
        }

    def run(self):
        """Run the dashboard application"""

        # Sidebar
        self._render_sidebar()

        # Main content
        st.markdown('<h1 class="main-header">📈 AI Stock Intelligence Dashboard</h1>',
                    unsafe_allow_html=True)

        # Check if we have data
        if st.session_state.data.empty:
            st.warning("Please select a stock symbol and click 'Fetch & Analyze' to begin.")
            self._render_welcome_screen()
            return

        # Render dashboard
        self._render_dashboard()

    def _render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.markdown("### 🔍 Stock Selection")

            # Symbol selection with names
            symbol_options = list(self.popular_symbols_with_names.keys())
            symbol_display = [f"{sym} - {name}" for sym, name in self.popular_symbols_with_names.items()]

            # Find current index
            current_display = f"{st.session_state.symbol} - {self.popular_symbols_with_names.get(st.session_state.symbol, 'Custom')}"
            current_index = 0
            if current_display in symbol_display:
                current_index = symbol_display.index(current_display)

            selected_option = st.selectbox(
                "Select Stock",
                options=symbol_display,
                index=current_index,
                key="symbol_select"
            )

            # Extract symbol from selection
            selected_symbol = selected_option.split(" - ")[0]

            # Custom symbol input (overrides dropdown)
            custom_symbol = st.text_input(
                "Or enter custom symbol",
                value="",
                placeholder="e.g., BTC-USD, ETH-USD, NFLX"
            )

            # Use custom symbol if provided
            if custom_symbol:
                selected_symbol = custom_symbol.upper()
                selected_option = f"{selected_symbol} - Custom Symbol"

            # Interval selection
            selected_interval = st.selectbox(
                "Time Interval",
                options=list(self.interval_options.keys()),
                format_func=lambda x: self.interval_options[x],
                index=list(self.interval_options.keys()).index(st.session_state.interval)
                if st.session_state.interval in self.interval_options else 3,
                key="interval_select"
            )

            # Period selection
            selected_period = st.selectbox(
                "Time Period",
                options=list(self.period_options.keys()),
                format_func=lambda x: self.period_options[x],
                index=list(self.period_options.keys()).index(st.session_state.period)
                if st.session_state.period in self.period_options else 3,
                key="period_select"
            )

            # Fetch button
            if st.button("🚀 Fetch & Analyze", type="primary", use_container_width=True):
                with st.spinner(f"Fetching data for {selected_symbol}..."):
                    try:
                        # Fetch stock info first
                        stock_info = self.data_engine.get_stock_info(selected_symbol)
                        st.session_state.stock_info = stock_info

                        # Fetch data
                        data = self.data_engine.get_stock_data(
                            selected_symbol,
                            selected_interval,
                            selected_period
                        )

                        if data.empty:
                            st.error(f"No data found for {selected_symbol}")
                        else:
                            # Update session state
                            st.session_state.symbol = selected_symbol
                            st.session_state.interval = selected_interval
                            st.session_state.period = selected_period
                            st.session_state.data = data
                            st.session_state.model_trained = False
                            st.session_state.predictions = []

                            st.success(f"Data loaded for {selected_symbol}")

                            # Train models
                            self._train_models(data)

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            st.markdown("---")
            st.markdown("### 📊 Quick Actions")

            # Model retraining
            if st.button("🔄 Retrain Models", use_container_width=True):
                if not st.session_state.data.empty:
                    with st.spinner("Retraining models..."):
                        self._train_models(st.session_state.data)
                        st.success("Models retrained successfully!")

            # Download data
            if not st.session_state.data.empty:
                csv = st.session_state.data.to_csv()
                st.download_button(
                    label="📥 Download Data",
                    data=csv,
                    file_name=f"{st.session_state.symbol}_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            st.markdown("---")
            st.markdown("### ℹ️ About")
            st.info("""
            This dashboard provides AI-powered stock analysis using:
            - **Random Forest** for baseline predictions
            - **XGBoost** for enhanced signal detection
            - **LSTM** for temporal pattern learning

            Predictions are aggregated using an ensemble approach.
            """)

    def _render_welcome_screen(self):
        """Render welcome screen with sample dashboard"""

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label="Live Price", value="$150.25", delta="+2.5%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label="Market Cap", value="$2.5T", delta="+1.8%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label="Volume", value="45.2M", delta="-3.2%")
            st.markdown('</div>', unsafe_allow_html=True)

        # Sample chart
        st.markdown("### Sample Analysis")

        # Create sample data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.normal(150, 5, 30).cumsum(),
            'High': np.random.normal(155, 5, 30).cumsum(),
            'Low': np.random.normal(145, 5, 30).cumsum(),
            'Close': np.random.normal(152, 5, 30).cumsum(),
            'Volume': np.random.randint(1000000, 5000000, 30)
        }, index=dates)

        # Create sample chart
        fig = self.viz_engine.create_main_chart(sample_data, show_predictions=False)
        st.plotly_chart(fig, use_container_width=True)

    def _render_dashboard(self):
        """Render main dashboard"""

        data = st.session_state.data
        symbol = st.session_state.symbol

        # 1. Display stock header
        self._render_stock_header()

        # 2. Top metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            # Live quote
            quote = self.data_engine.get_live_quote(symbol)
            if quote:
                price = quote.get('price', 0)
                change = quote.get('change', 0)
                change_pct = quote.get('change_percent', 0)

                delta_color = "normal"
                delta_prefix = "+" if change >= 0 else ""

                st.metric(
                    label="Live Price",
                    value=f"${price:.2f}",
                    delta=f"{delta_prefix}{change_pct:.2f}%",
                    delta_color=delta_color
                )

        with col2:
            # Daily range
            if not data.empty:
                day_high = data['High'].iloc[-1] if 'High' in data.columns else 0
                day_low = data['Low'].iloc[-1] if 'Low' in data.columns else 0
                st.metric(
                    label="Daily Range",
                    value=f"${day_low:.2f} - ${day_high:.2f}"
                )

        with col3:
            # Volume
            if not data.empty:
                volume = data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
                avg_volume = data['Volume'].mean() if len(data) > 0 else 0
                volume_ratio = volume / avg_volume if avg_volume > 0 else 1

                volume_delta = f"{volume_ratio:.1f}x avg"
                if volume_ratio > 1.5:
                    volume_delta = f"📈 {volume_ratio:.1f}x avg"
                elif volume_ratio < 0.7:
                    volume_delta = f"📉 {volume_ratio:.1f}x avg"

                st.metric(
                    label="Volume",
                    value=f"{volume:,.0f}",
                    delta=volume_delta
                )

        with col4:
            # RSI if available
            if 'RSI' in data.columns and not data.empty:
                rsi = data['RSI'].iloc[-1]
                if rsi > 70:
                    rsi_status = "⚠️ Overbought"
                    rsi_color = "inverse"
                elif rsi < 30:
                    rsi_status = "⚠️ Oversold"
                    rsi_color = "inverse"
                else:
                    rsi_status = "Normal"
                    rsi_color = "normal"

                st.metric(
                    label="RSI",
                    value=f"{rsi:.1f}",
                    delta=rsi_status,
                    delta_color=rsi_color
                )

        with col5:
            # Prediction status
            if st.session_state.predictions:
                latest_pred = st.session_state.predictions[-1]
                direction = latest_pred.get('direction', 'neutral')
                confidence = latest_pred.get('confidence', 0) * 100

                if direction == 'bullish':
                    direction_icon = "📈"
                    delta_color = "normal"
                elif direction == 'bearish':
                    direction_icon = "📉"
                    delta_color = "inverse"
                else:
                    direction_icon = "➡️"
                    delta_color = "off"

                st.metric(
                    label="AI Signal",
                    value=f"{direction_icon} {direction.upper()}",
                    delta=f"{confidence:.1f}% conf",
                    delta_color=delta_color
                )

        # Main chart
        st.markdown("### 📊 Interactive Analysis Chart")

        # Create chart with predictions
        fig = self.viz_engine.create_main_chart(
            data,
            st.session_state.predictions,
            show_predictions=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Bottom panels
        col_left, col_right = st.columns([2, 1])

        with col_left:
            # AI Insights Panel
            st.markdown("### 🧠 AI Insights")

            if st.session_state.predictions:
                latest_pred = st.session_state.predictions[-1]
                self._render_ai_insights(latest_pred)

            # Technical Indicators
            st.markdown("### 📈 Technical Indicators")
            self._render_technical_indicators(data)

        with col_right:
            # Model Performance
            st.markdown("### ⚙️ Model Performance")

            # Get model insights
            model_insights = self.model_engine.get_model_insights()

            # Create gauge for latest prediction
            if st.session_state.predictions:
                latest_pred = st.session_state.predictions[-1]
                gauge_fig = self.viz_engine.create_performance_gauge(
                    latest_pred.get('confidence', 0.5),
                    latest_pred.get('direction', 'neutral')
                )
                st.plotly_chart(gauge_fig, use_container_width=True)

            # Model comparison chart
            if model_insights.get('model_performance'):
                comp_fig = self.viz_engine.create_model_comparison(model_insights)
                st.plotly_chart(comp_fig, use_container_width=True)

            # Feature importance
            if (model_insights.get('feature_importance') and
                    'rf' in model_insights['feature_importance']):

                # Get feature names from data
                feature_names = self.feature_engine.feature_columns
                rf_importances = model_insights['feature_importance']['rf']

                if len(feature_names) == len(rf_importances):
                    importance_fig = self.viz_engine.create_feature_importance_chart(
                        feature_names, rf_importances
                    )
                    st.plotly_chart(importance_fig, use_container_width=True)

    def _render_stock_header(self):
        """Render stock header with name and details"""

        stock_info = st.session_state.stock_info

        # Create header container
        st.markdown('<div class="stock-header">', unsafe_allow_html=True)

        # Create columns for logo and info
        col_logo, col_info = st.columns([1, 4])

        with col_logo:
            # Display logo or placeholder
            if stock_info.get('logo_url'):
                try:
                    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
                    st.image(stock_info['logo_url'], width=80)
                    st.markdown('</div>', unsafe_allow_html=True)
                except:
                    st.markdown("""
                    <div class="logo-container">
                        <span style="font-size: 3rem;">🏢</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="logo-container">
                    <span style="font-size: 3rem;">📊</span>
                </div>
                """, unsafe_allow_html=True)

        with col_info:
            # Company name and symbol
            st.markdown(
                f'<div class="company-name">{stock_info.get("name", st.session_state.symbol)} ({st.session_state.symbol})</div>',
                unsafe_allow_html=True)

            # Company details row
            col_details1, col_details2, col_details3 = st.columns(3)

            with col_details1:
                if stock_info.get('sector') != 'N/A':
                    st.markdown(f'<div class="company-details"><strong>Sector:</strong> {stock_info["sector"]}</div>',
                                unsafe_allow_html=True)
                if stock_info.get('industry') != 'N/A':
                    st.markdown(
                        f'<div class="company-details"><strong>Industry:</strong> {stock_info["industry"]}</div>',
                        unsafe_allow_html=True)

            with col_details2:
                if stock_info.get('exchange') != 'N/A':
                    st.markdown(
                        f'<div class="company-details"><strong>Exchange:</strong> {stock_info["exchange"]}</div>',
                        unsafe_allow_html=True)
                if stock_info.get('currency'):
                    st.markdown(
                        f'<div class="company-details"><strong>Currency:</strong> {stock_info["currency"]}</div>',
                        unsafe_allow_html=True)

            with col_details3:
                # Key metrics
                if stock_info.get('market_cap', 0) > 0:
                    market_cap_formatted = self._format_market_cap(stock_info['market_cap'])
                    st.markdown(
                        f'<div class="company-details"><strong>Market Cap:</strong> {market_cap_formatted}</div>',
                        unsafe_allow_html=True)

                if stock_info.get('pe_ratio') != 'N/A' and isinstance(stock_info['pe_ratio'], (int, float)):
                    st.markdown(
                        f'<div class="company-details"><strong>P/E Ratio:</strong> {stock_info["pe_ratio"]:.1f}</div>',
                        unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Display company overview in expander
        if stock_info.get('summary') and stock_info['summary'] != 'No description available.':
            with st.expander("📋 Company Overview", expanded=False):
                st.write(stock_info['summary'])

                # Display additional info in columns
                col_info1, col_info2 = st.columns(2)

                with col_info1:
                    if stock_info.get('website') != 'N/A':
                        st.markdown(f"**Website:** [{stock_info['website']}]({stock_info['website']})")

                    if stock_info.get('employees') != 'N/A':
                        st.markdown(f"**Employees:** {stock_info['employees']:,}")

                with col_info2:
                    if stock_info.get('beta') != 'N/A' and isinstance(stock_info['beta'], (int, float)):
                        st.markdown(f"**Beta:** {stock_info['beta']:.2f}")

                    if stock_info.get('dividend_yield', 0) > 0:
                        st.markdown(f"**Dividend Yield:** {stock_info['dividend_yield']:.2f}%")

                # 52-week range
                if stock_info.get('fifty_two_week_low', 0) > 0 and stock_info.get('fifty_two_week_high', 0) > 0:
                    st.markdown(
                        f"**52-Week Range:** ${stock_info['fifty_two_week_low']:.2f} - ${stock_info['fifty_two_week_high']:.2f}")

        st.markdown("---")

    def _format_market_cap(self, market_cap):
        """Format market cap to B/M/K"""
        if market_cap >= 1_000_000_000_000:
            return f"${market_cap / 1_000_000_000_000:.1f}T"
        elif market_cap >= 1_000_000_000:
            return f"${market_cap / 1_000_000_000:.1f}B"
        elif market_cap >= 1_000_000:
            return f"${market_cap / 1_000_000:.1f}M"
        else:
            return f"${market_cap:,.0f}"

    def _train_models(self, data: pd.DataFrame):
        """Train models on current data"""
        try:
            # Generate features
            with st.spinner("Generating features..."):
                X, y = self.feature_engine.generate_features(data)

            if X.empty:
                st.warning("Insufficient data for model training")
                return

            # Train models
            with st.spinner("Training AI models..."):
                self.model_engine.train_models(X, y, retrain=True)
                st.session_state.model_trained = True

            # Generate predictions
            with st.spinner("Generating predictions..."):
                predictions = []
                for i in range(len(X)):
                    if i >= 20:  # Only predict after we have enough history
                        X_subset = X.iloc[:i + 1]
                        pred = self.model_engine.predict(X_subset)
                        predictions.append(pred)

                st.session_state.predictions = predictions

            st.success("Models trained and predictions generated!")

        except Exception as e:
            st.error(f"Error training models: {str(e)}")

    def _render_ai_insights(self, prediction: Dict):
        """Render AI insights panel"""

        direction = prediction.get('direction', 'neutral')
        confidence = prediction.get('confidence', 0)
        bullish_prob = prediction.get('bullish_prob', 0.5)

        # Determine CSS class
        if direction == 'bullish':
            css_class = 'prediction-bullish'
            icon = "📈"
            sentiment = "Positive"
            color = "#26a69a"
        elif direction == 'bearish':
            css_class = 'prediction-bearish'
            icon = "📉"
            sentiment = "Negative"
            color = "#ef5350"
        else:
            css_class = 'prediction-neutral'
            icon = "➡️"
            sentiment = "Neutral"
            color = "#78909c"

        st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 3])

        with col1:
            st.markdown(f"### {icon}")

        with col2:
            st.markdown(f"**AI Outlook:** <span style='color:{color}; font-weight:bold;'>{direction.upper()}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"**Confidence:** {confidence:.1%}")
            st.markdown(f"**Bullish Probability:** {bullish_prob:.1%}")
            st.markdown(f"**Sentiment:** {sentiment}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Model-specific predictions
        if 'model_predictions' in prediction:
            st.markdown("**Model Consensus:**")

            model_preds = prediction['model_predictions']
            model_probs = prediction.get('model_probabilities', {})
            model_weights = prediction.get('model_weights', {})

            cols = st.columns(len(model_preds))

            for idx, (model_name, pred) in enumerate(model_preds.items()):
                with cols[idx]:
                    prob = model_probs.get(model_name, 0.5)
                    weight = model_weights.get(model_name, 0.33)

                    pred_text = "Bull" if pred == 1 else "Bear"
                    pred_color = "green" if pred == 1 else "red"

                    st.metric(
                        label=f"{model_name.upper()}",
                        value=pred_text,
                        delta=f"{prob:.1%}",
                        delta_color="normal"
                    )
                    st.caption(f"Weight: {weight:.1%}")

    def _render_technical_indicators(self, data: pd.DataFrame):
        """Render technical indicators panel"""

        if data.empty:
            return

        # Create columns for indicators
        col1, col2 = st.columns(2)

        with col1:
            # Moving averages
            if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                sma_20 = data['SMA_20'].iloc[-1]
                sma_50 = data['SMA_50'].iloc[-1]
                current_price = data['Close'].iloc[-1]

                st.markdown("**Moving Averages:**")
                st.write(f"SMA 20: ${sma_20:.2f}")
                st.write(f"SMA 50: ${sma_50:.2f}")

                # Golden/Death cross
                if len(data) > 1:
                    prev_sma_20 = data['SMA_20'].iloc[-2] if len(data) > 1 else sma_20
                    prev_sma_50 = data['SMA_50'].iloc[-2] if len(data) > 1 else sma_50

                    if sma_20 > sma_50 and prev_sma_20 <= prev_sma_50:
                        st.success("🟢 Golden Cross Detected (Bullish Signal)")
                    elif sma_20 < sma_50 and prev_sma_20 >= prev_sma_50:
                        st.error("🔴 Death Cross Detected (Bearish Signal)")
                    elif sma_20 > sma_50:
                        st.info("🟡 Uptrend (SMA20 > SMA50)")
                    else:
                        st.info("🟡 Downtrend (SMA20 < SMA50)")

            # MACD
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                macd = data['MACD'].iloc[-1]
                signal = data['MACD_Signal'].iloc[-1]

                st.markdown("**MACD:**")
                st.write(f"MACD: {macd:.4f}")
                st.write(f"Signal: {signal:.4f}")

                if macd > signal:
                    st.success("MACD above signal line (Bullish)")
                else:
                    st.warning("MACD below signal line (Bearish)")

        with col2:
            # RSI
            if 'RSI' in data.columns:
                rsi = data['RSI'].iloc[-1]

                st.markdown("**RSI:**")

                # RSI gauge
                if rsi > 70:
                    st.error(f"RSI: {rsi:.1f} (Overbought - Consider selling)")
                elif rsi < 30:
                    st.success(f"RSI: {rsi:.1f} (Oversold - Consider buying)")
                else:
                    st.info(f"RSI: {rsi:.1f} (Neutral)")

            # Bollinger Bands
            if 'BB_High' in data.columns and 'BB_Low' in data.columns:
                bb_high = data['BB_High'].iloc[-1]
                bb_low = data['BB_Low'].iloc[-1]
                current_price = data['Close'].iloc[-1]
                bb_position = (current_price - bb_low) / (bb_high - bb_low) if bb_high != bb_low else 0.5

                st.markdown("**Bollinger Bands:**")
                st.write(f"Position: {bb_position:.1%}")

                if current_price > bb_high:
                    st.warning("Price above upper band (Overbought)")
                elif current_price < bb_low:
                    st.warning("Price below lower band (Oversold)")
                else:
                    st.success("Price within normal range")

            # Volume analysis
            if 'Volume_Ratio' in data.columns:
                volume_ratio = data['Volume_Ratio'].iloc[-1]

                st.markdown("**Volume Analysis:**")
                st.write(f"Volume Ratio: {volume_ratio:.2f}x")

                if volume_ratio > 1.5:
                    st.info("High volume - Strong interest")
                elif volume_ratio < 0.5:
                    st.info("Low volume - Weak interest")
                else:
                    st.info("Normal volume")


def main():
    """Main application entry point"""

    # Initialize dashboard
    dashboard = StockIntelligenceDashboard()

    # Run the dashboard
    dashboard.run()


if __name__ == "__main__":
    main()
