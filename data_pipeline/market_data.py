"""
Market Data Pipeline
Fetches historical stock data from yfinance for NSE-listed Indian stocks
and calculates technical indicators using the 'ta' library.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta


def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetches historical stock data for the given ticker.
    Ensures `.NS` is appended if it's an Indian stock without it.
    """
    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        ticker = f"{ticker}.NS"

    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}. It may be delisted or invalid.")

    # Drop rows with NaN in critical columns
    df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators using the 'ta' library.
    """
    # RSI (14-period)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

    # MACD
    macd_indicator = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_12_26_9'] = macd_indicator.macd()
    df['MACDs_12_26_9'] = macd_indicator.macd_signal()
    df['MACDh_12_26_9'] = macd_indicator.macd_diff()

    # Moving Averages
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()

    # Volatility (ATR - Average True Range)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)

    # Momentum (Rate of Change)
    df['ROC'] = ta.momentum.roc(df['Close'], window=10)

    # Price Change %
    df['Price_Change_Pct'] = df['Close'].pct_change() * 100

    return df


def create_labels(
    df: pd.DataFrame,
    future_days: int = 5,
    buy_threshold: float = 2.0,
    sell_threshold: float = -2.0,
) -> pd.DataFrame:
    """
    Dynamically creates labels (BUY/HOLD/SELL) based on future returns.

    Args:
        future_days: Number of days in the future to look for price movement.
        buy_threshold: Percentage return threshold to classify as BUY.
        sell_threshold: Percentage return threshold to classify as SELL.
    """
    # Calculate future return percentage
    df['Future_Close'] = df['Close'].shift(-future_days)
    df['Future_Return'] = ((df['Future_Close'] - df['Close']) / df['Close']) * 100

    def classify_return(ret):
        if pd.isna(ret):
            return np.nan
        if ret >= buy_threshold:
            return 2  # BUY
        elif ret <= sell_threshold:
            return 0  # SELL
        else:
            return 1  # HOLD

    df['Target'] = df['Future_Return'].apply(classify_return)

    return df


def get_processed_data(ticker: str, period: str = "2y", future_days: int = 5) -> pd.DataFrame:
    """
    Main pipeline function to fetch data, calculate indicators, and create labels.
    """
    df = fetch_stock_data(ticker, period)
    df = calculate_technical_indicators(df)
    df = create_labels(df, future_days)

    # Drop NaNs that were created by shifting and rolling windows
    df.dropna(inplace=True)
    return df
