import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
import pickle
from typing import Any, Dict, List, Optional
import os


def calculate_returns(series: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate percentage returns"""
    return series.pct_change(periods=periods)


def calculate_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling volatility"""
    returns = calculate_returns(series)
    return returns.rolling(window=window).std()


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    if returns.std() == 0:
        return 0.0

    return np.sqrt(252) * excess_returns.mean() / returns.std()


def calculate_max_drawdown(prices: pd.Series) -> Dict[str, float]:
    """Calculate maximum drawdown"""
    try:
        cumulative_returns = (1 + prices.pct_change()).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_dd = drawdown.min()
        max_dd_duration = (drawdown == 0).astype(int).groupby((drawdown != 0).cumsum()).sum().max()

        return {
            'max_drawdown': abs(max_dd) if pd.notnull(max_dd) else 0,
            'max_drawdown_pct': abs(max_dd * 100) if pd.notnull(max_dd) else 0,
            'max_drawdown_duration': max_dd_duration if pd.notnull(max_dd_duration) else 0
        }
    except:
        return {
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'max_drawdown_duration': 0
        }


def create_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from datetime index"""
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    df = df.copy()
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    df['is_year_start'] = df.index.is_year_start.astype(int)
    df['is_year_end'] = df.index.is_year_end.astype(int)

    return df


def normalize_data(data: pd.Series) -> pd.Series:
    """Normalize data to 0-1 range"""
    if data.min() == data.max():
        return pd.Series([0.5] * len(data), index=data.index)
    return (data - data.min()) / (data.max() - data.min())


def detect_outliers(data: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Detect outliers using z-score method"""
    z_scores = np.abs((data - data.mean()) / data.std())
    return z_scores > threshold


def calculate_correlation_matrix(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Calculate correlation matrix for specified columns"""
    if not columns:
        return pd.DataFrame()

    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        return pd.DataFrame()

    return df[valid_columns].corr()


def save_to_cache(key: str, data: Any, cache_dir: str = 'cache'):
    """Save data to cache"""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = hashlib.md5(key.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")

        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error saving to cache: {e}")


def load_from_cache(key: str, cache_dir: str = 'cache', max_age_hours: int = 24) -> Optional[Any]:
    """Load data from cache with expiration"""
    try:
        cache_key = hashlib.md5(key.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")

        if not os.path.exists(cache_path):
            return None

        # Check cache age
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
        if cache_age.total_seconds() > max_age_hours * 3600:
            os.remove(cache_path)
            return None

        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except:
        return None


def format_currency(value: float) -> str:
    """Format value as currency"""
    if value is None or np.isnan(value):
        return "$0.00"

    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    elif abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    else:
        return f"${value:.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    if value is None or np.isnan(value):
        return "0.00%"
    return f"{value:.2f}%"


def create_performance_report(data: pd.DataFrame, initial_investment: float = 10000) -> Dict:
    """Create comprehensive performance report"""
    if data.empty or 'Returns' not in data.columns:
        return {}

    returns = data['Returns'].dropna()

    if len(returns) < 2:
        return {}

    try:
        # Calculate metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = calculate_sharpe_ratio(returns)
        max_dd_info = calculate_max_drawdown(data['Close'] if 'Close' in data.columns else returns)

        # Calculate Sortino ratio
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = (annual_return - 0.02) / downside_deviation if downside_deviation > 0 else 0

        # Calculate Calmar ratio
        calmar_ratio = annual_return / max_dd_info['max_drawdown'] if max_dd_info['max_drawdown'] > 0 else 0

        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_dd_info['max_drawdown'],
            'max_drawdown_pct': max_dd_info['max_drawdown_pct'],
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(returns),
            'positive_trades': (returns > 0).sum(),
            'negative_trades': (returns < 0).sum()
        }
    except Exception as e:
        print(f"Error creating performance report: {e}")
        return {}
