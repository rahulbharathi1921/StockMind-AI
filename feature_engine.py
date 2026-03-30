import pandas as pd
import numpy as np
import ta
from typing import Tuple, List, Optional, Dict
from scipy import stats


class MarketIntelligenceEngine:
    """Advanced feature engineering for stock market data"""

    def __init__(self, lookback_periods: List[int] = [5, 10, 20, 50]):
        self.lookback_periods = lookback_periods
        self.feature_columns = []

    def generate_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate comprehensive feature set from OHLCV data

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if data.empty or len(data) < max(self.lookback_periods):
            return pd.DataFrame(), pd.Series(dtype=float)

        df = data.copy()

        # 1. Price-based features
        df = self._add_price_features(df)

        # 2. Technical indicators
        df = self._add_technical_indicators(df)

        # 3. Volume features
        df = self._add_volume_features(df)

        # 4. Statistical features
        df = self._add_statistical_features(df)

        # 5. Temporal features
        df = self._add_temporal_features(df)

        # 6. Target variable (next period return direction)
        df['Target'] = self._create_target(df)

        # Drop NaN values created by indicators
        df = df.dropna()

        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=float)

        # Separate features and target
        feature_cols = [col for col in df.columns if
                        col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Returns']]
        self.feature_columns = feature_cols

        return df[feature_cols], df['Target']

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""

        # Basic price transformations
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Price momentum
        for period in self.lookback_periods:
            df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
            df[f'ROC_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)

            # Rolling statistics
            df[f'Rolling_Mean_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'Rolling_Std_{period}'] = df['Close'].rolling(window=period).std()
            df[f'Rolling_Min_{period}'] = df['Close'].rolling(window=period).min()
            df[f'Rolling_Max_{period}'] = df['Close'].rolling(window=period).max()

            # Z-score
            df[f'Z_Score_{period}'] = (df['Close'] - df[f'Rolling_Mean_{period}']) / df[
                f'Rolling_Std_{period}'].replace(0, 1e-10)

        # Price position within daily range
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, 1e-10)

        # Gap features
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using TA library"""

        try:
            # Moving averages
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
            df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)

            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Diff'] = macd.macd_diff()

            # RSI
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
            df['BB_High'] = bollinger.bollinger_hband()
            df['BB_Low'] = bollinger.bollinger_lband()
            df['BB_Width'] = bollinger.bollinger_wband()
            df['BB_Position'] = (df['Close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low']).replace(0, 1e-10)

            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()

            # Average True Range (volatility)
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)

            # Commodity Channel Index
            df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)

            # ADX (trend strength)
            df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)

        except Exception as e:
            print(f"Error calculating technical indicators: {e}")

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""

        # Volume indicators
        df['Volume_Change'] = df['Volume'].pct_change()

        # Volume moving averages
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20'].replace(0, 1e-10)

        # On-Balance Volume
        df['OBV'] = (np.sign(df['Returns'].fillna(0)) * df['Volume']).cumsum()
        df['OBV_Change'] = df['OBV'].pct_change()

        # Volume-price trend
        df['VPT'] = (df['Returns'] * df['Volume']).cumsum()

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""

        # Skewness and Kurtosis
        for period in [5, 10, 20]:
            if len(df) >= period:
                df[f'Skewness_{period}'] = df['Returns'].rolling(window=period).skew()
                df[f'Kurtosis_{period}'] = df['Returns'].rolling(window=period).kurt()
            else:
                df[f'Skewness_{period}'] = np.nan
                df[f'Kurtosis_{period}'] = np.nan

        # Hurst exponent (rough calculation)
        for period in [20, 50]:
            if len(df) >= period:
                df[f'Hurst_{period}'] = df['Returns'].rolling(window=period).apply(
                    lambda x: self._estimate_hurst(x) if len(x) == period else np.nan
                )
            else:
                df[f'Hurst_{period}'] = np.nan

        # Autocorrelation
        for lag in [1, 2, 3, 5]:
            if len(df) >= 20:
                df[f'Autocorr_{lag}'] = df['Returns'].rolling(window=20).apply(
                    lambda x: x.autocorr(lag=lag) if len(x) == 20 else np.nan
                )
            else:
                df[f'Autocorr_{lag}'] = np.nan

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features"""

        if isinstance(df.index, pd.DatetimeIndex):
            # Time-based features
            df['Day_of_Week'] = df.index.dayofweek
            df['Month'] = df.index.month
            df['Quarter'] = df.index.quarter
            df['Year'] = df.index.year

            # Cyclical encoding for day of week
            df['Day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['Day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

            # Month cyclical encoding
            df['Month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

        return df

    def _create_target(self, df: pd.DataFrame, forward_periods: int = 1) -> pd.Series:
        """Create target variable for prediction"""

        if len(df) <= forward_periods:
            return pd.Series([np.nan] * len(df), index=df.index)

        # Forward return (next period)
        forward_return = df['Close'].shift(-forward_periods) / df['Close'] - 1

        # Binary classification: 1 if positive return, 0 otherwise
        target = (forward_return > 0).astype(int)

        # Shift back to align with features
        target = target.shift(forward_periods)

        return target

    def _estimate_hurst(self, series: pd.Series) -> float:
        """Estimate Hurst exponent for time series"""
        try:
            if len(series) < 20:
                return 0.5

            # Simplified Hurst estimation
            lags = range(2, min(20, len(series) // 2))
            tau = []
            for lag in lags:
                tau.append(np.std(np.subtract(series[lag:], series[:-lag])))

            if len(tau) < 2:
                return 0.5

            # Fit to log-log plot
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        except:
            return 0.5

    def get_feature_importance_template(self) -> Dict[str, str]:
        """Get feature importance categories"""
        return {
            'price': ['Returns', 'Log_Returns', 'Momentum_*', 'ROC_*', 'Rolling_*', 'Z_Score_*'],
            'technical': ['SMA_*', 'EMA_*', 'MACD*', 'RSI', 'BB_*', 'Stoch_*', 'ATR', 'CCI', 'ADX'],
            'volume': ['Volume_*', 'OBV*', 'VPT'],
            'statistical': ['Skewness_*', 'Kurtosis_*', 'Hurst_*', 'Autocorr_*'],
            'temporal': ['Day_*', 'Month_*', 'Quarter', 'Year']
        }
