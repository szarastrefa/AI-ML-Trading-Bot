# -*- coding: utf-8 -*-
"""
Custom Technical Analysis Library
Replacement for pandas-ta with essential indicators
Optimized for trading bot performance
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class CustomTA:
    """
    Custom Technical Analysis library with essential indicators
    Replaces pandas-ta dependency with lightweight implementations
    """
    
    @staticmethod
    def sma(series: pd.Series, length: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=length).mean()
    
    @staticmethod
    def ema(series: pd.Series, length: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=length, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, length: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = CustomTA.ema(series, fast)
        ema_slow = CustomTA.ema(series, slow)
        macd_line = ema_fast - ema_slow
        macd_signal = CustomTA.ema(macd_line, signal)
        macd_histogram = macd_line - macd_signal
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': macd_signal,
            'Histogram': macd_histogram
        }, index=series.index)
    
    @staticmethod
    def bollinger_bands(series: pd.Series, length: int = 20, std: float = 2) -> pd.DataFrame:
        """Bollinger Bands"""
        sma = CustomTA.sma(series, length)
        std_dev = series.rolling(window=length).std()
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        return pd.DataFrame({
            'BB_Upper': upper,
            'BB_Middle': sma,
            'BB_Lower': lower,
            'BB_Width': (upper - lower) / sma,
            'BB_Percent': (series - lower) / (upper - lower)
        }, index=series.index)
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            'Stoch_K': k_percent,
            'Stoch_D': d_percent
        }, index=close.index)
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=length).mean()
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = 0
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def ad_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Accumulation/Distribution Line"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero
        ad = (clv * volume).cumsum()
        return ad
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=length).max()
        lowest_low = low.rolling(window=length).min()
        
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=length).mean()
        mean_deviation = typical_price.rolling(window=length).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    @staticmethod
    def momentum(series: pd.Series, length: int = 10) -> pd.Series:
        """Price Momentum"""
        return series.diff(length)
    
    @staticmethod
    def roc(series: pd.Series, length: int = 10) -> pd.Series:
        """Rate of Change"""
        return ((series / series.shift(length)) - 1) * 100
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    @staticmethod
    def fibonacci_retracement(high_price: float, low_price: float) -> dict:
        """Calculate Fibonacci Retracement Levels"""
        diff = high_price - low_price
        
        return {
            '0.0%': high_price,
            '23.6%': high_price - diff * 0.236,
            '38.2%': high_price - diff * 0.382,
            '50.0%': high_price - diff * 0.500,
            '61.8%': high_price - diff * 0.618,
            '78.6%': high_price - diff * 0.786,
            '100.0%': low_price
        }
    
    @staticmethod
    def fibonacci_extension(high_price: float, low_price: float, retrace_price: float) -> dict:
        """Calculate Fibonacci Extension Levels"""
        diff = high_price - low_price
        
        return {
            '0.0%': retrace_price,
            '61.8%': retrace_price + diff * 0.618,
            '100.0%': retrace_price + diff * 1.000,
            '161.8%': retrace_price + diff * 1.618,
            '261.8%': retrace_price + diff * 2.618
        }
    
    @staticmethod
    def pivot_points(high: float, low: float, close: float) -> dict:
        """Calculate Pivot Points"""
        pivot = (high + low + close) / 3
        
        return {
            'PP': pivot,
            'R1': 2 * pivot - low,
            'R2': pivot + (high - low),
            'R3': high + 2 * (pivot - low),
            'S1': 2 * pivot - high,
            'S2': pivot - (high - low),
            'S3': low - 2 * (high - pivot)
        }
    
    @classmethod
    def add_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Add all available indicators to DataFrame"""
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("DataFrame must contain OHLC columns")
        
        result_df = df.copy()
        
        # Price-based indicators
        result_df['SMA_10'] = cls.sma(df['close'], 10)
        result_df['SMA_20'] = cls.sma(df['close'], 20)
        result_df['SMA_50'] = cls.sma(df['close'], 50)
        
        result_df['EMA_10'] = cls.ema(df['close'], 10)
        result_df['EMA_20'] = cls.ema(df['close'], 20)
        result_df['EMA_50'] = cls.ema(df['close'], 50)
        
        # Oscillators
        result_df['RSI_14'] = cls.rsi(df['close'], 14)
        result_df['RSI_7'] = cls.rsi(df['close'], 7)
        
        # MACD
        macd_data = cls.macd(df['close'])
        result_df = pd.concat([result_df, macd_data], axis=1)
        
        # Bollinger Bands
        bb_data = cls.bollinger_bands(df['close'])
        result_df = pd.concat([result_df, bb_data], axis=1)
        
        # Stochastic
        stoch_data = cls.stochastic(df['high'], df['low'], df['close'])
        result_df = pd.concat([result_df, stoch_data], axis=1)
        
        # ATR
        result_df['ATR_14'] = cls.atr(df['high'], df['low'], df['close'], 14)
        
        # Volume indicators (if volume available)
        if 'volume' in df.columns:
            result_df['OBV'] = cls.obv(df['close'], df['volume'])
            result_df['AD_Line'] = cls.ad_line(df['high'], df['low'], df['close'], df['volume'])
            result_df['VWAP'] = cls.vwap(df['high'], df['low'], df['close'], df['volume'])
        
        # Other indicators
        result_df['Williams_R'] = cls.williams_r(df['high'], df['low'], df['close'])
        result_df['CCI'] = cls.cci(df['high'], df['low'], df['close'])
        result_df['Momentum'] = cls.momentum(df['close'])
        result_df['ROC'] = cls.roc(df['close'])
        
        # Price patterns
        result_df['Returns'] = df['close'].pct_change()
        result_df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
        result_df['High_Low_Ratio'] = df['high'] / df['low']
        result_df['Price_Change'] = df['close'] - df['open']
        
        logger.info(f"✅ Added {len(result_df.columns) - len(df.columns)} technical indicators")
        
        return result_df

# Convenience functions for compatibility
def sma(series, length):
    return CustomTA.sma(series, length)

def ema(series, length):
    return CustomTA.ema(series, length)

def rsi(series, length=14):
    return CustomTA.rsi(series, length)

def macd(series, fast=12, slow=26, signal=9):
    return CustomTA.macd(series, fast, slow, signal)

def bollinger_bands(series, length=20, std=2):
    return CustomTA.bollinger_bands(series, length, std)

# Initialize custom TA
custom_ta = CustomTA()

logger.info("📈 Custom Technical Analysis library loaded successfully!")