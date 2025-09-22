"""
TA-Lib Stable Trading Strategy
Working implementation with comprehensive error handling and fallbacks
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Safe TA-Lib import with fallback
try:
    import talib
    TALIB_AVAILABLE = True
    print("? TA-Lib imported successfully")
except ImportError as e:
    print(f"?? TA-Lib not available: {e}")
    TALIB_AVAILABLE = False
    # Create mock talib for fallback
    class MockTALib:
        @staticmethod
        def SMA(data, timeperiod=20):
            return pd.Series(data).rolling(timeperiod).mean().values
        @staticmethod  
        def EMA(data, timeperiod=20):
            return pd.Series(data).ewm(span=timeperiod).mean().values
        @staticmethod
        def RSI(data, timeperiod=14):
            delta = pd.Series(data).diff()
            gain = (delta.where(delta > 0, 0)).rolling(timeperiod).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(timeperiod).mean()
            rs = gain / loss
            return (100 - (100 / (1 + rs))).values
        @staticmethod
        def MACD(data, fastperiod=12, slowperiod=26, signalperiod=9):
            exp1 = pd.Series(data).ewm(span=fastperiod).mean()
            exp2 = pd.Series(data).ewm(span=slowperiod).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=signalperiod).mean()
            histogram = macd - signal
            return macd.values, signal.values, histogram.values
        @staticmethod
        def BBANDS(data, timeperiod=20, nbdevup=2, nbdevdn=2):
            sma = pd.Series(data).rolling(timeperiod).mean()
            std = pd.Series(data).rolling(timeperiod).std()
            upper = sma + (std * nbdevup)
            lower = sma - (std * nbdevdn)
            return upper.values, sma.values, lower.values
        @staticmethod
        def ATR(high, low, close, timeperiod=14):
            h_l = pd.Series(high) - pd.Series(low)
            h_c = np.abs(pd.Series(high) - pd.Series(close).shift())
            l_c = np.abs(pd.Series(low) - pd.Series(close).shift())
            tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
            return tr.rolling(timeperiod).mean().values
    
    talib = MockTALib()

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class TALibStableStrategy(BaseStrategy):
    """
    Stable TA-Lib trading strategy with comprehensive error handling
    Works with or without TA-Lib installed
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        self.name = "TA-Lib Stable Strategy"
        self.version = "2.1.0"
        self.talib_available = TALIB_AVAILABLE
        
        # Strategy parameters
        self.sma_periods = [20, 50]
        self.ema_periods = [12, 26]
        self.rsi_period = 14
        self.bb_period = 20
        self.atr_period = 14
        
        logger.info(f"? {self.name} v{self.version} initialized")
        logger.info(f"?? TA-Lib available: {self.talib_available}")
        
    async def analyze(self, symbol: str, timeframe: str = "H1") -> Dict[str, Any]:
        """
        Main analysis function with comprehensive error handling
        """
        try:
            self._increment_analysis_count()
            start_time = datetime.utcnow()
            
            logger.info(f"?? Starting analysis: {symbol} {timeframe}")
            
            # Get market data (mock for now, implement your data source)
            df = await self._get_market_data(symbol, timeframe)
            
            if df is None or len(df) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                return self._create_empty_signal("Insufficient market data")
            
            # Apply technical indicators
            df_with_indicators = self._apply_indicators(df)
            
            # Generate trading signal
            signal = self._generate_signal(df_with_indicators, symbol)
            
            # Add metadata
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            signal.update({
                "symbol": symbol,
                "timeframe": timeframe,
                "execution_time": execution_time,
                "strategy": self.name,
                "version": self.version,
                "timestamp": datetime.utcnow().isoformat(),
                "data_points": len(df),
                "talib_used": self.talib_available
            })
            
            logger.info(f"? Analysis completed: {signal['signal']} ({signal['confidence']:.1f}%)")
            return signal
            
        except Exception as e:
            logger.error(f"? Analysis failed for {symbol}: {str(e)}")
            return self._create_empty_signal(f"Analysis error: {str(e)}")
    
    def _apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply technical indicators using TA-Lib or fallback implementations"""
        try:
            logger.debug(f"Applying indicators to {len(df)} data points")
            
            # Prepare data arrays
            high = df['high'].values.astype(np.float64)
            low = df['low'].values.astype(np.float64)
            close = df['close'].values.astype(np.float64)
            
            # Moving Averages
            df['SMA_20'] = talib.SMA(close, timeperiod=20)
            df['SMA_50'] = talib.SMA(close, timeperiod=50)
            df['EMA_12'] = talib.EMA(close, timeperiod=12)
            df['EMA_26'] = talib.EMA(close, timeperiod=26)
            
            # RSI
            df['RSI'] = talib.RSI(close, timeperiod=14)
            
            # MACD
            macd, signal, histogram = talib.MACD(close)
            df['MACD'] = macd
            df['MACD_Signal'] = signal  
            df['MACD_Histogram'] = histogram
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            df['BB_Upper'] = bb_upper
            df['BB_Middle'] = bb_middle
            df['BB_Lower'] = bb_lower
            
            # ATR (Average True Range)
            df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
            
            # Calculate additional derived indicators
            df = self._calculate_derived_indicators(df)
            
            logger.debug(f"? Applied {len([c for c in df.columns if c not in ['open','high','low','close','volume']])} indicators")
            return df
            
        except Exception as e:
            logger.error(f"? Error applying indicators: {str(e)}")
            return df
    
    def _calculate_derived_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived and composite indicators"""
        try:
            # Trend Direction
            df['Trend_Direction'] = np.where(df['SMA_20'] > df['SMA_50'], 1, 
                                           np.where(df['SMA_20'] < df['SMA_50'], -1, 0))
            
            # Price position relative to Bollinger Bands
            df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # MACD Signal
            df['MACD_Signal_Direction'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
            
            # RSI Levels
            df['RSI_Overbought'] = df['RSI'] > 70
            df['RSI_Oversold'] = df['RSI'] < 30
            
            # Volatility percentile
            df['ATR_Percentile'] = df['ATR'].rolling(100).rank(pct=True) * 100
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating derived indicators: {str(e)}")
            return df
    
    def _generate_signal(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Generate trading signal based on technical analysis"""
        try:
            latest = df.iloc[-1]
            current_price = float(latest['close'])
            
            # Initialize scoring system
            bullish_score = 0
            bearish_score = 0
            signal_strength = 0
            
            # Trend Analysis (40% weight)
            if latest['Trend_Direction'] == 1:  # Uptrend
                bullish_score += 2
            elif latest['Trend_Direction'] == -1:  # Downtrend
                bearish_score += 2
            
            if latest['EMA_12'] > latest['EMA_26']:  # Short EMA above long EMA
                bullish_score += 1
            else:
                bearish_score += 1
            
            # Momentum Analysis (30% weight)  
            rsi = latest['RSI']
            if pd.notna(rsi):
                if rsi < 30:  # Oversold - potential buy
                    bullish_score += 2
                elif rsi > 70:  # Overbought - potential sell
                    bearish_score += 2
                elif rsi > 50:  # Above midline
                    bullish_score += 1
                else:  # Below midline
                    bearish_score += 1
            
            # MACD Analysis (20% weight)
            if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
                if latest['MACD_Signal_Direction'] == 1:  # MACD above signal
                    bullish_score += 1
                else:  # MACD below signal
                    bearish_score += 1
            
            # Bollinger Bands Analysis (10% weight)
            bb_pos = latest['BB_Position']
            if pd.notna(bb_pos):
                if bb_pos < 0.2:  # Near lower band
                    bullish_score += 1
                elif bb_pos > 0.8:  # Near upper band
                    bearish_score += 1
            
            # Determine final signal
            total_score = bullish_score + bearish_score
            if total_score == 0:
                signal_direction = "HOLD"
                confidence = 0.0
            else:
                if bullish_score > bearish_score:
                    signal_direction = "BUY"
                    confidence = (bullish_score / total_score) * 100
                elif bearish_score > bullish_score:
                    signal_direction = "SELL"
                    confidence = (bearish_score / total_score) * 100
                else:
                    signal_direction = "HOLD"
                    confidence = 50.0
            
            # Risk Management - Calculate stop loss and take profit
            atr = latest['ATR'] if pd.notna(latest['ATR']) else current_price * 0.01
            
            if signal_direction == "BUY":
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 3)
            elif signal_direction == "SELL":
                stop_loss = current_price + (atr * 2)  
                take_profit = current_price - (atr * 3)
            else:
                stop_loss = current_price
                take_profit = current_price
            
            # Calculate risk-reward ratio
            if signal_direction != "HOLD":
                risk = abs(current_price - stop_loss)
                reward = abs(take_profit - current_price)
                rr_ratio = reward / risk if risk > 0 else 1.0
            else:
                rr_ratio = 1.0
            
            return {
                "signal": signal_direction,
                "confidence": round(confidence, 2),
                "entry_price": round(current_price, 5),
                "stop_loss": round(stop_loss, 5),
                "take_profit": round(take_profit, 5),
                "risk_reward_ratio": round(rr_ratio, 2),
                "analysis": {
                    "bullish_score": bullish_score,
                    "bearish_score": bearish_score,
                    "rsi": round(float(latest['RSI']), 2) if pd.notna(latest['RSI']) else None,
                    "trend_direction": int(latest['Trend_Direction']),
                    "macd_direction": int(latest['MACD_Signal_Direction']) if pd.notna(latest['MACD_Signal_Direction']) else None,
                    "bb_position": round(float(latest['BB_Position']), 3) if pd.notna(latest['BB_Position']) else None,
                    "atr": round(float(atr), 5)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return self._create_empty_signal(f"Signal generation error: {str(e)}")
    
    async def _get_market_data(self, symbol: str, timeframe: str, periods: int = 200) -> Optional[pd.DataFrame]:
        """
        Get market data - implement your data source here
        For now, generates realistic mock data for testing
        """
        try:
            logger.info(f"Getting market data for {symbol} {timeframe} ({periods} periods)")
            
            # Generate realistic mock OHLCV data for testing
            # In production, replace with real data source (MT5, CCXT, etc.)
            
            np.random.seed(hash(symbol) % 1000)  # Consistent data for same symbol
            
            # Base price around typical forex levels
            if 'USD' in symbol:
                base_price = 1.1000 if 'EUR' in symbol else 1.3000
            else:
                base_price = 50000 if 'BTC' in symbol else 100.0
            
            # Generate price series with realistic movement
            returns = np.random.normal(0, 0.001, periods)  # 0.1% average movement
            price_series = base_price * np.exp(np.cumsum(returns))
            
            # Create OHLCV data
            dates = pd.date_range(end=datetime.utcnow(), periods=periods, freq='1H')
            
            df = pd.DataFrame(index=dates)
            df['close'] = price_series
            df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
            
            # Generate high/low with realistic spreads
            volatility = np.random.uniform(0.0005, 0.002, periods)
            df['high'] = np.maximum(df['open'], df['close']) + (price_series * volatility)
            df['low'] = np.minimum(df['open'], df['close']) - (price_series * volatility)
            
            # Generate volume
            df['volume'] = np.random.randint(1000, 10000, periods)
            
            # Ensure data integrity
            df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
            df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
            
            logger.info(f"? Generated {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error generating market data: {str(e)}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Enhanced performance statistics"""
        base_stats = super().get_performance_stats()
        
        base_stats.update({
            "technical_features": {
                "indicators": ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands", "ATR"],
                "timeframes_supported": ["M1", "M5", "M15", "H1", "H4", "D1"],
                "signal_types": ["BUY", "SELL", "HOLD"],
                "talib_available": self.talib_available,
                "fallback_mode": not self.talib_available
            },
            "risk_management": {
                "stop_loss": "ATR-based (2x ATR)",
                "take_profit": "ATR-based (3x ATR)", 
                "risk_reward_target": 1.5,
                "position_sizing": "configurable"
            }
        })
        
        return base_stats

# Test function for standalone usage
if __name__ == "__main__":
    import asyncio
    
    async def test_strategy():
        print("?? Testing TA-Lib Stable Strategy...")
        
        config = {}
        strategy = TALibStableStrategy(config)
        
        # Test analysis
        result = await strategy.analyze("EURUSD", "H1")
        
        print(f"?? Signal: {result['signal']}")
        print(f"?? Confidence: {result['confidence']}%")
        print(f"?? Entry: {result['entry_price']}")
        print(f"?? Stop Loss: {result['stop_loss']}")  
        print(f"?? Take Profit: {result['take_profit']}")
        print(f"?? Risk/Reward: {result['risk_reward_ratio']}")
        
        # Test performance stats
        stats = strategy.get_performance_stats()
        print(f"?? Strategy: {stats['strategy_info']['name']}")
        print(f"?? Analysis Count: {stats['strategy_info']['analysis_count']}")
        print(f"? Test completed successfully!")
    
    # Run test
    asyncio.run(test_strategy())
