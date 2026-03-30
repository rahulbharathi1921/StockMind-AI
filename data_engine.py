import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class YahooFinanceEngine:
    """Advanced Yahoo Finance data engine with intelligent caching"""

    def __init__(self):
        self.cache = {}
        self.cache_timeout = {
            '1m': 60,  # 1 minute
            '5m': 300,  # 5 minutes
            '1h': 3600,  # 1 hour
            '1d': 86400,  # 1 day
            '1wk': 604800,  # 1 week
        }
        self.rate_limit_delay = 0.5

    def get_stock_data(self, symbol: str, interval: str = '1d',
                       period: str = '1mo', retry_count: int = 3) -> pd.DataFrame:
        """
        Fetch stock data with intelligent caching and retry logic

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Data interval ('1m', '5m', '1h', '1d', '1wk', '1mo')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y')
            retry_count: Number of retry attempts

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{interval}_{period}"

        # Check cache
        if cache_key in self.cache:
            cached_time, data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_timeout.get(interval, 300):
                return data.copy()

        # Fetch from Yahoo Finance
        for attempt in range(retry_count):
            try:
                ticker = yf.Ticker(symbol)

                # Handle different intervals
                if interval in ['1m', '5m', '15m', '30m', '90m']:
                    # Intraday data
                    data = ticker.history(interval=interval, period='60d')
                else:
                    # Daily/weekly/monthly data
                    data = ticker.history(interval=interval, period=period)

                if data.empty:
                    raise ValueError(f"No data found for {symbol}")

                # Clean and prepare data
                data = self._clean_data(data)

                # Cache the data
                self.cache[cache_key] = (time.time(), data.copy())

                # Rate limiting
                time.sleep(self.rate_limit_delay)

                return data

            except Exception as e:
                if attempt == retry_count - 1:
                    print(f"Failed to fetch data for {symbol}: {str(e)}")
                    return pd.DataFrame()
                time.sleep(2 ** attempt)  # Exponential backoff

        return pd.DataFrame()

    def get_stock_info(self, symbol: str) -> Dict:
        """Get detailed stock/company information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extract key information
            stock_info = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'short_name': info.get('shortName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'exchange': info.get('exchange', 'N/A'),
                'currency': info.get('currency', 'USD'),
                'country': info.get('country', 'US'),
                'website': info.get('website', 'N/A'),
                'summary': info.get('longBusinessSummary', 'No description available.'),
                'employees': info.get('fullTimeEmployees', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'beta': info.get('beta', 'N/A'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'average_volume': info.get('averageVolume', 0),
                'volume': info.get('volume', 0)
            }

            # Clean up the summary if it's too long
            if len(stock_info['summary']) > 200:
                stock_info['summary'] = stock_info['summary'][:200] + '...'

            # Try to get logo URL
            try:
                # Construct logo URL from website
                if stock_info['website'] != 'N/A':
                    logo_url = f"https://logo.clearbit.com/{stock_info['website'].replace('https://', '').replace('http://', '').split('/')[0]}"
                    stock_info['logo_url'] = logo_url
                else:
                    # Fallback to placeholder
                    stock_info['logo_url'] = ''
            except:
                stock_info['logo_url'] = ''

            return stock_info

        except Exception as e:
            print(f"Error fetching stock info for {symbol}: {e}")
            # Return minimal info
            return {
                'symbol': symbol,
                'name': symbol,
                'short_name': symbol,
                'sector': 'N/A',
                'industry': 'N/A',
                'exchange': 'N/A',
                'currency': 'USD',
                'logo_url': '',
                'summary': 'No information available',
                'market_cap': 0,
                'pe_ratio': 'N/A',
                'dividend_yield': 0
            }

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the data"""
        if data.empty:
            return data

        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                data[col] = np.nan

        # Sort by index
        data = data.sort_index()

        # Fill missing values
        data = data.ffill().bfill()

        # Calculate returns
        data['Returns'] = data['Close'].pct_change()

        return data

    def get_multiple_stocks(self, symbols: List[str], interval: str = '1d',
                            period: str = '1mo') -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks"""
        results = {}
        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, interval, period)
                results[symbol] = data
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                results[symbol] = pd.DataFrame()

        return results

    def get_live_quote(self, symbol: str) -> Dict:
        """Get live quote information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get the actual current data
            current_data = self.get_stock_data(symbol, '1d', '5d')

            if not current_data.empty:
                last_close = current_data['Close'].iloc[-1]
                prev_close = current_data['Close'].iloc[-2] if len(current_data) > 1 else last_close

                quote = {
                    'symbol': symbol,
                    'price': info.get('currentPrice', info.get('regularMarketPrice', last_close)),
                    'change': last_close - prev_close,
                    'change_percent': ((last_close - prev_close) / prev_close * 100) if prev_close > 0 else 0,
                    'volume': current_data['Volume'].iloc[-1] if 'Volume' in current_data.columns else 0,
                    'market_cap': info.get('marketCap', 0),
                    'day_high': current_data['High'].iloc[-1] if 'High' in current_data.columns else last_close,
                    'day_low': current_data['Low'].iloc[-1] if 'Low' in current_data.columns else last_close,
                    'previous_close': prev_close,
                    'open': current_data['Open'].iloc[-1] if 'Open' in current_data.columns else last_close
                }
                return quote
            else:
                return {
                    'symbol': symbol,
                    'price': 0,
                    'change': 0,
                    'change_percent': 0,
                    'volume': 0,
                    'market_cap': 0,
                    'day_high': 0,
                    'day_low': 0,
                    'previous_close': 0,
                    'open': 0
                }
        except Exception as e:
            print(f"Error getting live quote for {symbol}: {e}")
            return {
                'symbol': symbol,
                'price': 0,
                'change': 0,
                'change_percent': 0,
                'volume': 0,
                'market_cap': 0,
                'day_high': 0,
                'day_low': 0,
                'previous_close': 0,
                'open': 0
            }

    def get_top_symbols(self, category: str = 'most_active') -> List[str]:
        """Get top symbols by category"""
        try:
            if category == 'most_active':
                return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'JPM', 'JNJ']
            elif category == 'gainers':
                return ['TSLA', 'AMD', 'NFLX', 'PYPL', 'ADBE']
            elif category == 'losers':
                return ['INTC', 'CSCO', 'ORCL', 'IBM', 'T']
            else:
                return ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI']
        except:
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
