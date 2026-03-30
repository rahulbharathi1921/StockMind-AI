# 📈 AI Stock Intelligence Dashboard

An advanced AI-powered stock market analysis dashboard built with Streamlit, featuring real-time data fetching, machine learning predictions, and interactive visualizations.

## 🚀 Features

### 📊 Real-Time Stock Data
- Live stock quotes and historical data from Yahoo Finance
- Support for multiple time intervals (1m, 5m, 1h, 1d, 1wk, 1mo)
- Popular stock symbols with company information
- Custom symbol input for any stock or ETF

### 🤖 AI-Powered Predictions
- **Random Forest Classifier** for baseline predictions
- **XGBoost** for enhanced signal detection
- Ensemble model approach with dynamic weight adjustment
- Confidence scores and probability estimates
- Bullish/Bearish/Neutral signal classification

### 📈 Technical Analysis
- **Moving Averages**: SMA 20, SMA 50, EMA 12, EMA 26
- **Momentum Indicators**: RSI, MACD, Stochastic Oscillator
- **Volatility Indicators**: Bollinger Bands, ATR
- **Trend Indicators**: ADX, CCI
- **Volume Analysis**: OBV, Volume Ratio, VPT

### 🎨 Interactive Visualizations
- Candlestick charts with prediction overlays
- Volume analysis with color-coded bars
- RSI indicator with overbought/oversold zones
- Model performance comparison charts
- Feature importance analysis
- Confidence gauge charts

### 📊 Performance Metrics
- Total return and annual return
- Sharpe ratio and Sortino ratio
- Maximum drawdown analysis
- Win rate and profit factor
- Calmar ratio

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-stock-intelligence-dashboard.git
cd ai-stock-intelligence-dashboard
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser**
Navigate to `http://localhost:8501`

## 📁 Project Structure

```
ai-stock-intelligence-dashboard/
├── app.py                 # Main Streamlit dashboard application
├── data_engine.py         # Yahoo Finance data fetching engine
├── feature_engine.py      # Feature engineering for ML models
├── model_engine.py        # Multi-model prediction engine
├── visualization.py       # Interactive visualization engine
├── utils.py              # Utility functions and helpers
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

## 🎯 Usage Guide

### 1. Select a Stock
- Choose from popular stocks in the dropdown menu
- Or enter a custom symbol (e.g., `BTC-USD`, `ETH-USD`, `NFLX`)

### 2. Configure Parameters
- **Time Interval**: Select data granularity (1m, 5m, 1h, 1d, 1wk, 1mo)
- **Time Period**: Choose historical data range (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y)

### 3. Fetch & Analyze
- Click "Fetch & Analyze" to load data and train models
- The dashboard will automatically:
  - Fetch stock information and company details
  - Generate technical indicators
  - Train AI models on historical data
  - Generate predictions for the entire dataset

### 4. Interpret Results
- **AI Signal**: Overall market outlook (Bullish/Bearish/Neutral)
- **Confidence**: How confident the model is in its prediction
- **Model Consensus**: Individual model predictions and weights
- **Technical Indicators**: Key technical analysis metrics
- **Feature Importance**: Which features drive predictions

## 🔧 Technical Details

### Machine Learning Models

#### Random Forest Classifier
- 100 estimators with max depth of 10
- Balanced class weights for imbalanced data
- Feature importance analysis

#### XGBoost Classifier
- 100 estimators with learning rate 0.1
- Subsample and column sampling for regularization
- Log loss evaluation metric

#### Ensemble Approach
- Dynamic weight adjustment based on model performance
- Weighted probability averaging
- Confidence-based signal classification

### Feature Engineering

The system generates 50+ features including:

1. **Price Features**: Returns, momentum, rolling statistics, Z-scores
2. **Technical Indicators**: SMA, EMA, MACD, RSI, Bollinger Bands
3. **Volume Features**: Volume ratio, OBV, VPT
4. **Statistical Features**: Skewness, kurtosis, Hurst exponent, autocorrelation
5. **Temporal Features**: Day of week, month, quarter, cyclical encoding

### Data Sources

- **Yahoo Finance**: Real-time and historical stock data
- **No API keys required**: Uses the free `yfinance` library

## ⚠️ Disclaimer

**This tool is for educational and research purposes only.**

- Stock market predictions are inherently uncertain
- Past performance does not guarantee future results
- Always consult with a financial advisor before making investment decisions
- The authors are not responsible for any financial losses

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Improvement
- Add more ML models (LSTM, Transformer-based)
- Implement backtesting framework
- Add portfolio analysis features
- Enhance visualization options
- Add sentiment analysis from news

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - For the amazing web framework
- [Yahoo Finance](https://finance.yahoo.com/) - For providing free stock data
- [Plotly](https://plotly.com/) - For interactive visualizations
- [scikit-learn](https://scikit-learn.org/) - For machine learning tools
- [XGBoost](https://xgboost.readthedocs.io/) - For gradient boosting
- [TA-Lib](https://github.com/bukosaboin/ta) - For technical analysis indicators

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yourusername/ai-stock-intelligence-dashboard/issues) page
2. Create a new issue with detailed information
3. Include error messages and steps to reproduce

---

**Made with ❤️ using Python and Streamlit**
