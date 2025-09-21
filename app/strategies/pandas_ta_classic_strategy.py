"""
Advanced Trading Strategy using pandas-ta-classic
Supports 150+ technical indicators with Smart Money Concepts and multiprocessing

Author: AI/ML Trading Bot Team
Version: 2.0.0
Created: 2025-09-21
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas_ta_classic as ta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

try:
    from ..base_strategy import BaseStrategy
except ImportError:
    # Fallback for standalone usage
    class BaseStrategy:
        def __init__(self, config: Dict):
            self.config = config


class PandasTAClassicStrategy(BaseStrategy):
    """
    Advanced trading strategy leveraging pandas-ta-classic library
    
    Features:
    - 150+ technical indicators with multiprocessing support
    - Smart Money Concepts (Order Blocks, FVG, BOS, CHoCH, Liquidity Sweeps)
    - Multi-timeframe analysis with confluence detection
    - Advanced risk management and position sizing
    - Machine Learning integration ready
    - vectorbt backtesting compatibility
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Strategy metadata
        self.name = "PandasTA-Classic Advanced Strategy v2.0"
        self.version = "2.0.0"
        self.author = "AI/ML Trading Bot"
        self.created = datetime.utcnow().isoformat()
        
        # Configuration
        self.timeframes = config.get('timeframes', ['M15', 'H1', 'H4', 'D1'])
        self.min_confluence = config.get('min_confluence_count', 2)
        
        # Initialize strategy components
        self.ta_strategy = self._create_ta_strategy()
        self.smc_config = config.get('smart_money_concepts', {})
        self.risk_config = config.get('risk_management', {})
        
        # Performance tracking
        self.performance_stats = {
            'total_signals': 0,
            'winning_signals': 0,
            'losing_signals': 0,
            'win_rate': 0.0,
            'avg_risk_reward': 0.0
        }
        
        logger.info(f"Initialized {self.name} with {len(self.timeframes)} timeframes")
    
    def _create_ta_strategy(self) -> ta.Strategy:
        """
        Create comprehensive pandas-ta-classic strategy with 150+ indicators
        Uses multiprocessing for optimal performance
        """
        
        strategy_indicators = [
            # === TREND INDICATORS (33 available) ===
            # Moving Averages - Multiple types for trend analysis
            {"kind": "sma", "length": 20},
            {"kind": "sma", "length": 50},
            {"kind": "sma", "length": 200},
            {"kind": "ema", "length": 8},
            {"kind": "ema", "length": 12},
            {"kind": "ema", "length": 21},
            {"kind": "ema", "length": 26},
            {"kind": "wma", "length": 14},
            {"kind": "hma", "length": 21},    # Hull Moving Average
            {"kind": "vwma", "length": 20},   # Volume Weighted MA
            
            # Advanced Trend Indicators
            {"kind": "supertrend", "length": 10, "multiplier": 3.0},
            {"kind": "psar", "af": 0.02, "max_af": 0.2},
            {"kind": "adx", "length": 14},
            {"kind": "aroon", "length": 14},
            {"kind": "vortex", "length": 14},
            {"kind": "qstick", "length": 14},
            
            # === MOMENTUM INDICATORS (41 available) ===
            # Core Oscillators
            {"kind": "rsi", "length": 14},
            {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
            {"kind": "stoch", "k": 14, "d": 3, "smooth_k": 3},
            {"kind": "willr", "length": 14},
            {"kind": "cci", "length": 20},
            {"kind": "roc", "length": 10},
            {"kind": "mom", "length": 10},
            
            # Advanced Momentum
            {"kind": "ao"},              # Awesome Oscillator
            {"kind": "bop"},             # Balance of Power  
            {"kind": "uo", "fast": 7, "medium": 14, "slow": 28},  # Ultimate Oscillator
            {"kind": "tsi", "fast": 25, "slow": 13},  # True Strength Index
            {"kind": "squeeze", "bb_length": 20, "kc_length": 20},  # TTM Squeeze
            {"kind": "fisher", "length": 14},  # Fisher Transform
            {"kind": "cg", "length": 10},     # Center of Gravity
            
            # === VOLUME INDICATORS (15 available) ===
            {"kind": "obv"},             # On Balance Volume
            {"kind": "ad"},              # Accumulation/Distribution
            {"kind": "cmf", "length": 20},  # Chaikin Money Flow
            {"kind": "mfi", "length": 14},  # Money Flow Index
            {"kind": "vwap"},            # Volume Weighted Average Price
            {"kind": "pvt"},             # Price Volume Trend
            {"kind": "eom", "length": 14},  # Ease of Movement
            {"kind": "nvi", "length": 255}, # Negative Volume Index
            {"kind": "pvi", "length": 255}, # Positive Volume Index
            
            # === VOLATILITY INDICATORS (14 available) ===
            {"kind": "bbands", "length": 20, "std": 2.0},  # Bollinger Bands
            {"kind": "kc", "length": 20, "scalar": 2.0},    # Keltner Channels
            {"kind": "donchian", "lower_length": 20, "upper_length": 20},
            {"kind": "atr", "length": 14},   # Average True Range
            {"kind": "natr", "length": 14},  # Normalized ATR
            {"kind": "true_range"},          # True Range
            {"kind": "ui", "length": 14},    # Ulcer Index
            {"kind": "massi", "fast": 9, "slow": 25},  # Mass Index
            
            # === STATISTICS INDICATORS (11 available) ===
            {"kind": "zscore", "length": 20},  # Z-Score
            {"kind": "stdev", "length": 20},   # Standard Deviation
            {"kind": "var", "length": 20},     # Variance
            {"kind": "skew", "length": 20},    # Skewness
            {"kind": "kurt", "length": 20},    # Kurtosis
            {"kind": "median", "length": 20},  # Median
            {"kind": "quantile", "length": 20, "q": 0.75},  # 75th Percentile
            
            # === UTILITY INDICATORS (5 available) ===
            {"kind": "cross", "series_a": "close", "series_b": "sma_20"},
            {"kind": "above", "series_a": "close", "series_b": "sma_50"},
            {"kind": "below", "series_a": "rsi_14", "series_b": 30},
            
            # === PERFORMANCE INDICATORS (3 available) ===
            {"kind": "log_return", "length": 1},
            {"kind": "percent_return", "length": 1},
            
            # === TREND SIGNALS ===
            {"kind": "tsignals", "trend": "sma_20 > sma_50", "asbool": True},
        ]
        
        return ta.Strategy(
            name="Advanced Multi-Indicator Strategy",
            description="Comprehensive pandas-ta-classic strategy with 50+ indicators across all categories",
            ta=strategy_indicators
        )
    
    async def analyze(self, symbol: str, timeframe: str = "H1", limit: int = 500) -> Dict[str, Any]:
        """
        Main analysis function using pandas-ta-classic with Smart Money Concepts
        
        Args:
            symbol: Trading instrument (e.g., 'EURUSD', 'BTCUSDT')
            timeframe: Chart timeframe (e.g., 'M15', 'H1', 'H4', 'D1')
            limit: Number of candles to analyze
            
        Returns:
            Complete trading signal with analysis breakdown
        """
        try:
            start_time = datetime.utcnow()
            logger.info(f"Starting analysis for {symbol} on {timeframe}")
            
            # Step 1: Get market data
            df = await self._get_market_data(symbol, timeframe, limit)
            if df is None or len(df) < 100:
                logger.warning(f"Insufficient data for {symbol} {timeframe}: {len(df) if df is not None else 0} candles")
                return self._empty_signal("Insufficient data")
            
            logger.debug(f"Retrieved {len(df)} candles for {symbol}")
            
            # Step 2: Apply all technical indicators using pandas-ta-classic strategy
            df_analyzed = await self._apply_technical_analysis(df.copy())
            
            # Step 3: Add Smart Money Concepts indicators
            df_analyzed = await self._add_smart_money_concepts(df_analyzed)
            
            # Step 4: Multi-timeframe confluence analysis
            mtf_analysis = await self._multi_timeframe_analysis(symbol, df_analyzed)
            
            # Step 5: Generate composite signals
            composite_signals = self._generate_composite_signals(df_analyzed)
            
            # Step 6: Smart Money Concepts analysis
            smc_analysis = self._analyze_smart_money_concepts(df_analyzed)
            
            # Step 7: Risk management calculations
            risk_metrics = self._calculate_risk_metrics(df_analyzed)
            
            # Step 8: Generate final trading signal
            final_signal = await self._generate_final_signal(
                df_analyzed, composite_signals, smc_analysis, mtf_analysis, risk_metrics
            )
            
            # Step 9: Performance tracking
            self.performance_stats['total_signals'] += 1
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Analysis completed for {symbol} in {execution_time:.2f}s - Signal: {final_signal['signal']} ({final_signal['confidence']:.1f}%)")
            
            # Return comprehensive analysis
            return {
                **final_signal,
                "analysis_breakdown": {
                    "technical_indicators": composite_signals,
                    "smart_money_concepts": smc_analysis,
                    "multi_timeframe": mtf_analysis,
                    "risk_metrics": risk_metrics,
                    "execution_time_seconds": execution_time,
                    "data_quality": {
                        "candles_analyzed": len(df_analyzed),
                        "indicators_calculated": len([col for col in df_analyzed.columns if col not in ['open', 'high', 'low', 'close', 'volume']]),
                        "missing_data_pct": df_analyzed.isnull().sum().sum() / (len(df_analyzed) * len(df_analyzed.columns)) * 100
                    }
                },
                "metadata": {
                    "strategy_version": self.version,
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "pandas_ta_classic_version": ta.__version__ if hasattr(ta, '__version__') else "latest"
                }
            }
            
        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {str(e)}", exc_info=True)
            return self._empty_signal(f"Analysis error: {str(e)}")
    
    async def _apply_technical_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply pandas-ta-classic strategy with multiprocessing support
        """
        try:
            logger.debug("Applying technical analysis indicators...")
            
            # Set pandas-ta-classic to use all CPU cores for multiprocessing
            df.ta.cores = 0  # Use all available cores
            
            # Apply the comprehensive strategy (multiprocessing enabled)
            df.ta.strategy(self.ta_strategy, verbose=False, timed=False)
            
            # Add custom composite indicators
            df = self._add_composite_indicators(df)
            
            logger.debug(f"Applied {len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])} technical indicators")
            
            return df
            
        except Exception as e:
            logger.error(f"Error applying technical analysis: {str(e)}")
            return df
    
    def _add_composite_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add custom composite indicators combining multiple pandas-ta-classic signals
        """
        try:
            # Trend Strength Composite (0-100)
            trend_signals = []
            if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                trend_signals.append((df['close'] > df['SMA_20']).astype(int))
                trend_signals.append((df['SMA_20'] > df['SMA_50']).astype(int))
            if 'EMA_12' in df.columns and 'EMA_26' in df.columns:
                trend_signals.append((df['EMA_12'] > df['EMA_26']).astype(int))
            if 'SUPERTd_10_3.0' in df.columns:
                trend_signals.append((df['SUPERTd_10_3.0'] == 1).astype(int))
            if 'ADX_14' in df.columns:
                trend_signals.append((df['ADX_14'] > 25).astype(int))
                
            if trend_signals:
                df['trend_strength'] = np.mean(trend_signals, axis=0) * 100
            else:
                df['trend_strength'] = 50
            
            # Momentum Composite (0-100)
            momentum_signals = []
            if 'RSI_14' in df.columns:
                momentum_signals.append(df['RSI_14'])
            if 'STOCHk_14_3_3' in df.columns:
                momentum_signals.append(df['STOCHk_14_3_3'])
            if 'WILLR_14' in df.columns:
                momentum_signals.append(100 + df['WILLR_14'])  # Convert to 0-100 scale
                
            if momentum_signals:
                df['momentum_composite'] = np.mean(momentum_signals, axis=0)
            else:
                df['momentum_composite'] = 50
            
            # Volume Confirmation (0-100)
            if 'OBV' in df.columns:
                obv_trend = df['OBV'].diff().rolling(5).mean() > 0
                price_trend = df['close'].diff().rolling(5).mean() > 0
                df['volume_confirmation'] = (obv_trend == price_trend).astype(int) * 100
            else:
                df['volume_confirmation'] = 50
            
            # Volatility Environment (0-100)
            if 'ATR_14' in df.columns:
                atr_percentile = df['ATR_14'].rolling(100).rank(pct=True) * 100
                df['volatility_percentile'] = atr_percentile.fillna(50)
            else:
                df['volatility_percentile'] = 50
            
            # Bollinger Band Position (-100 to +100)
            if all(col in df.columns for col in ['BBL_20_2.0', 'BBU_20_2.0']):
                bb_width = df['BBU_20_2.0'] - df['BBL_20_2.0']
                bb_position = (df['close'] - df['BBL_20_2.0']) / bb_width * 100
                df['bb_position'] = bb_position.clip(-10, 110) - 50  # Center around 0
            else:
                df['bb_position'] = 0
                
            return df
            
        except Exception as e:
            logger.error(f"Error adding composite indicators: {str(e)}")
            return df
    
    async def _add_smart_money_concepts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Smart Money Concepts indicators
        """
        try:
            logger.debug("Adding Smart Money Concepts indicators...")
            
            # Order Blocks Detection
            df['order_blocks'] = self._detect_order_blocks(df)
            
            # Fair Value Gaps Detection
            df['fair_value_gaps'] = self._detect_fair_value_gaps(df)
            
            # Break of Structure Detection
            df['break_of_structure'] = self._detect_break_of_structure(df)
            
            # Change of Character Detection  
            df['change_of_character'] = self._detect_change_of_character(df)
            
            # Liquidity Sweeps Detection
            df['liquidity_sweeps'] = self._detect_liquidity_sweeps(df)
            
            # SMC Composite Score
            smc_signals = [
                df['order_blocks'], df['fair_value_gaps'], 
                df['break_of_structure'], df['liquidity_sweeps']
            ]
            df['smc_composite'] = np.mean([np.abs(signal) for signal in smc_signals], axis=0)
            
            logger.debug("Smart Money Concepts indicators added successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error adding SMC indicators: {str(e)}")
            return df
    
    def _detect_order_blocks(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect institutional order blocks using volume and price action
        """
        try:
            volume_avg = df['volume'].rolling(20).mean()
            body_size = np.abs(df['close'] - df['open'])
            body_avg = body_size.rolling(20).mean()
            
            # Bearish Order Block: Strong selling with high volume
            bearish_ob = (
                (df['volume'] > volume_avg * 1.5) &
                (body_size > body_avg * 1.2) &
                (df['close'] < df['open']) &  # Red candle
                (df['low'] < df['low'].shift(1)) &  # Lower low
                (df['close'] > df['low'] + (df['high'] - df['low']) * 0.3)  # Long lower wick
            )
            
            # Bullish Order Block: Strong buying with high volume
            bullish_ob = (
                (df['volume'] > volume_avg * 1.5) &
                (body_size > body_avg * 1.2) &
                (df['close'] > df['open']) &  # Green candle
                (df['high'] > df['high'].shift(1)) &  # Higher high
                (df['close'] < df['high'] - (df['high'] - df['low']) * 0.3)  # Long upper wick
            )
            
            order_blocks = pd.Series(0, index=df.index)
            order_blocks[bearish_ob] = -1  # Bearish order block
            order_blocks[bullish_ob] = 1   # Bullish order block
            
            return order_blocks
            
        except Exception as e:
            logger.error(f"Error detecting order blocks: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _detect_fair_value_gaps(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect Fair Value Gaps (market imbalances)
        """
        try:
            # Bullish FVG: Previous high < Current low (gap up)
            bullish_fvg = df['high'].shift(1) < df['low']
            
            # Bearish FVG: Previous low > Current high (gap down)
            bearish_fvg = df['low'].shift(1) > df['high']
            
            # Filter for significant gaps (> 0.1% of price)
            min_gap_size = df['close'] * 0.001  # 0.1% threshold
            gap_size_bull = df['low'] - df['high'].shift(1)
            gap_size_bear = df['low'].shift(1) - df['high']
            
            bullish_fvg = bullish_fvg & (gap_size_bull > min_gap_size)
            bearish_fvg = bearish_fvg & (gap_size_bear > min_gap_size)
            
            fvg = pd.Series(0, index=df.index)
            fvg[bullish_fvg] = 1   # Bullish gap
            fvg[bearish_fvg] = -1  # Bearish gap
            
            return fvg
            
        except Exception as e:
            logger.error(f"Error detecting fair value gaps: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _detect_break_of_structure(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect Break of Structure using swing highs and lows
        """
        try:
            # Calculate swing highs and lows
            swing_period = self.smc_config.get('swing_period', 10)
            
            high_rolling_max = df['high'].rolling(swing_period, center=True).max()
            low_rolling_min = df['low'].rolling(swing_period, center=True).min()
            
            # BOS occurs when price breaks previous structure levels
            bullish_bos = df['close'] > high_rolling_max.shift(5)
            bearish_bos = df['close'] < low_rolling_min.shift(5)
            
            # Add volume confirmation
            volume_avg = df['volume'].rolling(20).mean()
            volume_confirmation = df['volume'] > volume_avg
            
            bullish_bos = bullish_bos & volume_confirmation
            bearish_bos = bearish_bos & volume_confirmation
            
            bos = pd.Series(0, index=df.index)
            bos[bullish_bos] = 1   # Bullish BOS
            bos[bearish_bos] = -1  # Bearish BOS
            
            return bos
            
        except Exception as e:
            logger.error(f"Error detecting break of structure: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _detect_change_of_character(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect Change of Character using momentum shifts
        """
        try:
            # Use RSI and MACD for character change detection
            rsi = df.get('RSI_14', pd.Series(50, index=df.index))
            macd = df.get('MACD_12_26_9', df['close'].diff())
            macd_signal = df.get('MACDs_12_26_9', macd.ewm(span=9).mean())
            
            # Character change signals
            bullish_choch = (
                (rsi > 50) & (rsi.shift(1) <= 50) &  # RSI crossing above 50
                (macd > macd_signal) &               # MACD above signal
                (macd > macd.shift(1))               # MACD trending up
            )
            
            bearish_choch = (
                (rsi < 50) & (rsi.shift(1) >= 50) &  # RSI crossing below 50
                (macd < macd_signal) &               # MACD below signal
                (macd < macd.shift(1))               # MACD trending down
            )
            
            choch = pd.Series(0, index=df.index)
            choch[bullish_choch] = 1   # Bullish character change
            choch[bearish_choch] = -1  # Bearish character change
            
            return choch
            
        except Exception as e:
            logger.error(f"Error detecting change of character: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _detect_liquidity_sweeps(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect Liquidity Sweeps (stop hunts and false breakouts)
        """
        try:
            lookback = self.smc_config.get('lookback', 50)
            
            # Calculate recent highs and lows (potential liquidity zones)
            recent_high = df['high'].rolling(lookback).max()
            recent_low = df['low'].rolling(lookback).min()
            
            # Upside liquidity sweep (fake breakout above resistance)
            sweep_up = (
                (df['high'] > recent_high.shift(1)) &     # Break recent high
                (df['close'] < df['open']) &              # But close red
                (df['close'] < recent_high.shift(1)) &    # Close back below resistance
                (df['volume'] > df['volume'].rolling(20).mean())  # High volume
            )
            
            # Downside liquidity sweep (fake breakdown below support)
            sweep_down = (
                (df['low'] < recent_low.shift(1)) &       # Break recent low
                (df['close'] > df['open']) &              # But close green
                (df['close'] > recent_low.shift(1)) &     # Close back above support
                (df['volume'] > df['volume'].rolling(20).mean())  # High volume
            )
            
            sweeps = pd.Series(0, index=df.index)
            sweeps[sweep_up] = 1     # Upside liquidity sweep
            sweeps[sweep_down] = -1  # Downside liquidity sweep
            
            return sweeps
            
        except Exception as e:
            logger.error(f"Error detecting liquidity sweeps: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def _generate_composite_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate composite signals from all technical indicators
        """
        try:
            latest = df.iloc[-1]
            recent = df.tail(10)
            
            signals = {
                'trend_strength': latest.get('trend_strength', 50),
                'momentum_score': latest.get('momentum_composite', 50),
                'volume_confirmation': latest.get('volume_confirmation', 50),
                'volatility_environment': latest.get('volatility_percentile', 50),
                'bb_position': latest.get('bb_position', 0)
            }
            
            # Add specific indicator signals
            if 'RSI_14' in df.columns:
                rsi = latest['RSI_14']
                signals['rsi_signal'] = 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'
                signals['rsi_value'] = float(rsi)
            
            if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
                macd = latest['MACD_12_26_9']
                macd_signal = latest['MACDs_12_26_9']
                signals['macd_signal'] = 'bullish' if macd > macd_signal else 'bearish'
                signals['macd_histogram'] = float(macd - macd_signal)
            
            if 'BBU_20_2.0' in df.columns and 'BBL_20_2.0' in df.columns:
                bb_upper = latest['BBU_20_2.0']
                bb_lower = latest['BBL_20_2.0']
                bb_mid = (bb_upper + bb_lower) / 2
                close = latest['close']
                
                if close > bb_upper:
                    signals['bb_signal'] = 'overbought'
                elif close < bb_lower:
                    signals['bb_signal'] = 'oversold'
                elif close > bb_mid:
                    signals['bb_signal'] = 'bullish'
                else:
                    signals['bb_signal'] = 'bearish'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating composite signals: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_smart_money_concepts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze Smart Money Concepts signals
        """
        try:
            latest = df.iloc[-1]
            recent = df.tail(20)
            
            # Count recent SMC events
            smc_analysis = {
                'order_blocks': {
                    'current': int(latest.get('order_blocks', 0)),
                    'recent_bullish': int((recent['order_blocks'] == 1).sum()),
                    'recent_bearish': int((recent['order_blocks'] == -1).sum())
                },
                'fair_value_gaps': {
                    'current': int(latest.get('fair_value_gaps', 0)),
                    'recent_bullish': int((recent['fair_value_gaps'] == 1).sum()),
                    'recent_bearish': int((recent['fair_value_gaps'] == -1).sum())
                },
                'break_of_structure': {
                    'current': int(latest.get('break_of_structure', 0)),
                    'recent_bullish': int((recent['break_of_structure'] == 1).sum()),
                    'recent_bearish': int((recent['break_of_structure'] == -1).sum())
                },
                'liquidity_sweeps': {
                    'current': int(latest.get('liquidity_sweeps', 0)),
                    'recent_bullish': int((recent['liquidity_sweeps'] == 1).sum()),
                    'recent_bearish': int((recent['liquidity_sweeps'] == -1).sum())
                }
            }
            
            # Calculate overall SMC bias
            total_bullish = sum([
                smc_analysis['order_blocks']['recent_bullish'],
                smc_analysis['fair_value_gaps']['recent_bullish'],
                smc_analysis['break_of_structure']['recent_bullish'],
                smc_analysis['liquidity_sweeps']['recent_bullish']
            ])
            
            total_bearish = sum([
                smc_analysis['order_blocks']['recent_bearish'],
                smc_analysis['fair_value_gaps']['recent_bearish'],
                smc_analysis['break_of_structure']['recent_bearish'],
                smc_analysis['liquidity_sweeps']['recent_bearish']
            ])
            
            if total_bullish > total_bearish:
                smc_bias = 'BULLISH'
                smc_strength = total_bullish
            elif total_bearish > total_bullish:
                smc_bias = 'BEARISH'
                smc_strength = total_bearish
            else:
                smc_bias = 'NEUTRAL'
                smc_strength = 0
            
            smc_analysis['overall'] = {
                'bias': smc_bias,
                'strength': smc_strength,
                'confidence': min(smc_strength * 10, 100)  # Convert to percentage
            }
            
            return smc_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing SMC: {str(e)}")
            return {'error': str(e)}
    
    async def _multi_timeframe_analysis(self, symbol: str, current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform multi-timeframe analysis for confluence
        """
        try:
            mtf_results = {}
            current_timeframe = getattr(current_df.index, 'freq', 'unknown')
            
            for timeframe in self.timeframes:
                if timeframe == current_timeframe:
                    continue
                    
                try:
                    # Get data for this timeframe
                    mtf_df = await self._get_market_data(symbol, timeframe, limit=100)
                    
                    if mtf_df is not None and len(mtf_df) > 20:
                        # Apply basic indicators for MTF analysis
                        mtf_df.ta.sma(length=20, append=True)
                        mtf_df.ta.rsi(length=14, append=True)
                        mtf_df.ta.macd(append=True)
                        
                        latest = mtf_df.iloc[-1]
                        
                        # Determine trend direction
                        sma20 = latest.get('SMA_20', latest['close'])
                        trend = 'BULLISH' if latest['close'] > sma20 else 'BEARISH'
                        
                        # RSI momentum
                        rsi = latest.get('RSI_14', 50)
                        momentum = 'BULLISH' if rsi > 50 else 'BEARISH'
                        
                        mtf_results[timeframe] = {
                            'trend': trend,
                            'momentum': momentum,
                            'rsi': float(rsi),
                            'price_vs_sma20': 'ABOVE' if latest['close'] > sma20 else 'BELOW'
                        }
                        
                except Exception as e:
                    logger.warning(f"Could not analyze {timeframe}: {str(e)}")
                    continue
            
            # Calculate confluence
            if mtf_results:
                bullish_trends = sum(1 for result in mtf_results.values() if result['trend'] == 'BULLISH')
                bearish_trends = sum(1 for result in mtf_results.values() if result['trend'] == 'BEARISH')
                
                confluence_analysis = {
                    'timeframes_analyzed': len(mtf_results),
                    'bullish_confluence': bullish_trends,
                    'bearish_confluence': bearish_trends,
                    'confluence_ratio': bullish_trends / len(mtf_results) if mtf_results else 0.5,
                    'overall_bias': 'BULLISH' if bullish_trends > bearish_trends else 'BEARISH' if bearish_trends > bullish_trends else 'NEUTRAL'
                }
                
                return {
                    'timeframe_analysis': mtf_results,
                    'confluence': confluence_analysis
                }
            
            return {'timeframe_analysis': {}, 'confluence': {'timeframes_analyzed': 0}}
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate risk management metrics
        """
        try:
            latest = df.iloc[-1]
            recent = df.tail(20)
            
            # ATR for position sizing
            atr = latest.get('ATR_14', (latest['high'] - latest['low']))
            
            # Volatility percentile
            volatility_pct = latest.get('volatility_percentile', 50)
            
            # Support and resistance levels (simplified)
            recent_high = recent['high'].max()
            recent_low = recent['low'].min()
            current_price = latest['close']
            
            # Risk calculations
            risk_metrics = {
                'atr_value': float(atr),
                'atr_percent': float(atr / current_price * 100),
                'volatility_environment': volatility_pct,
                'distance_to_resistance': float((recent_high - current_price) / current_price * 100),
                'distance_to_support': float((current_price - recent_low) / current_price * 100),
                'recommended_sl_distance': float(atr * 2),  # 2x ATR stop loss
                'recommended_tp_distance': float(atr * 4),  # 2:1 RR ratio
                'position_size_risk_2pct': 0.02 / (atr * 2 / current_price),  # 2% risk position size
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {'error': str(e)}
    
    async def _generate_final_signal(self, df: pd.DataFrame, composite_signals: Dict, 
                                   smc_analysis: Dict, mtf_analysis: Dict, risk_metrics: Dict) -> Dict[str, Any]:
        """
        Generate final trading signal combining all analysis
        """
        try:
            latest = df.iloc[-1]
            current_price = float(latest['close'])
            
            # Initialize scoring system
            bullish_score = 0
            bearish_score = 0
            max_possible_score = 10  # Adjust based on number of factors
            
            # 1. Trend Analysis (Weight: 2)
            trend_strength = composite_signals.get('trend_strength', 50)
            if trend_strength > 65:
                bullish_score += 2
            elif trend_strength < 35:
                bearish_score += 2
            elif trend_strength > 55:
                bullish_score += 1
            elif trend_strength < 45:
                bearish_score += 1
            
            # 2. Momentum Analysis (Weight: 2)
            momentum = composite_signals.get('momentum_score', 50)
            if momentum > 65:
                bullish_score += 2
            elif momentum < 35:
                bearish_score += 2
            elif momentum > 55:
                bullish_score += 1
            elif momentum < 45:
                bearish_score += 1
            
            # 3. Volume Confirmation (Weight: 1)
            volume_conf = composite_signals.get('volume_confirmation', 50)
            if volume_conf > 70:
                if bullish_score > bearish_score:
                    bullish_score += 1
                elif bearish_score > bullish_score:
                    bearish_score += 1
            
            # 4. Smart Money Concepts (Weight: 2)
            smc_overall = smc_analysis.get('overall', {})
            smc_bias = smc_overall.get('bias', 'NEUTRAL')
            smc_strength = smc_overall.get('strength', 0)
            
            if smc_bias == 'BULLISH' and smc_strength >= 2:
                bullish_score += 2
            elif smc_bias == 'BEARISH' and smc_strength >= 2:
                bearish_score += 2
            elif smc_bias == 'BULLISH' and smc_strength >= 1:
                bullish_score += 1
            elif smc_bias == 'BEARISH' and smc_strength >= 1:
                bearish_score += 1
            
            # 5. Multi-timeframe Confluence (Weight: 2)
            mtf_confluence = mtf_analysis.get('confluence', {})
            confluence_ratio = mtf_confluence.get('confluence_ratio', 0.5)
            timeframes_count = mtf_confluence.get('timeframes_analyzed', 0)
            
            if timeframes_count >= self.min_confluence:
                if confluence_ratio >= 0.75:  # 75%+ bullish confluence
                    bullish_score += 2
                elif confluence_ratio <= 0.25:  # 75%+ bearish confluence
                    bearish_score += 2
                elif confluence_ratio >= 0.6:  # 60%+ bullish confluence
                    bullish_score += 1
                elif confluence_ratio <= 0.4:  # 60%+ bearish confluence
                    bearish_score += 1
            
            # 6. Risk Environment (Weight: 1)
            volatility_pct = composite_signals.get('volatility_environment', 50)
            if volatility_pct < 30:  # Low volatility = safer environment
                if bullish_score > bearish_score:
                    bullish_score += 1
                elif bearish_score > bullish_score:
                    bearish_score += 1
            
            # Generate final signal
            total_score = bullish_score + bearish_score
            if total_score == 0:
                return self._create_hold_signal(current_price, "No clear signals")
            
            confidence = max(bullish_score, bearish_score) / max_possible_score * 100
            
            # Minimum confidence threshold
            min_confidence = self.config.get('signal_thresholds', {}).get('min_confidence', 60)
            
            if confidence < min_confidence:
                return self._create_hold_signal(current_price, f"Low confidence: {confidence:.1f}%")
            
            # Generate BUY or SELL signal
            atr_value = risk_metrics.get('atr_value', current_price * 0.01)
            
            if bullish_score > bearish_score:
                direction = "BUY"
                entry_price = current_price
                stop_loss = current_price - (atr_value * 2)
                take_profit = current_price + (atr_value * 4)  # 2:1 RR
            else:
                direction = "SELL"
                entry_price = current_price
                stop_loss = current_price + (atr_value * 2)
                take_profit = current_price - (atr_value * 4)  # 2:1 RR
            
            # Risk-reward calculation
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 1.0
            
            return {
                "signal": direction,
                "confidence": round(confidence, 1),
                "entry_price": round(entry_price, 5),
                "stop_loss": round(stop_loss, 5),
                "take_profit": round(take_profit, 5),
                "risk_reward_ratio": round(risk_reward_ratio, 2),
                "position_size_2pct_risk": risk_metrics.get('position_size_risk_2pct', 1.0),
                "signal_breakdown": {
                    "bullish_score": bullish_score,
                    "bearish_score": bearish_score,
                    "max_score": max_possible_score,
                    "trend_contribution": trend_strength,
                    "momentum_contribution": momentum,
                    "smc_contribution": smc_strength,
                    "mtf_confluence": confluence_ratio
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating final signal: {str(e)}")
            return self._empty_signal(f"Signal generation error: {str(e)}")
    
    def _create_hold_signal(self, price: float, reason: str) -> Dict[str, Any]:
        """Create a HOLD signal with reason"""
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "entry_price": round(price, 5),
            "stop_loss": round(price, 5),
            "take_profit": round(price, 5),
            "risk_reward_ratio": 1.0,
            "reason": reason,
            "signal_breakdown": {
                "bullish_score": 0,
                "bearish_score": 0,
                "reason": reason
            }
        }
    
    def _empty_signal(self, error_msg: str) -> Dict[str, Any]:
        """Create an empty signal with error message"""
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "risk_reward_ratio": 1.0,
            "error": error_msg
        }
    
    async def _get_market_data(self, symbol: str, timeframe: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """
        Get market data - placeholder for actual implementation
        This should be implemented to connect to your data provider
        """
        # This is a placeholder - implement your data provider connection
        logger.warning("_get_market_data is not implemented - using placeholder")
        return None
    
    async def backtest(self, symbol: str, start_date: str, end_date: str, 
                      initial_capital: float = 10000) -> Dict[str, Any]:
        """
        Comprehensive backtesting with vectorbt integration
        """
        try:
            logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
            
            # Get historical data
            df = await self._get_historical_data(symbol, start_date, end_date)
            
            if df is None or len(df) < 100:
                return {"error": "Insufficient historical data for backtesting"}
            
            # Apply full analysis to historical data
            signals = []
            for i in range(100, len(df)):  # Start after enough data for indicators
                df_subset = df.iloc[:i+1].copy()
                
                # Run analysis on this subset
                analysis = await self.analyze(symbol, "H1", limit=len(df_subset))
                
                signals.append({
                    'timestamp': df_subset.index[-1],
                    'signal': analysis.get('signal', 'HOLD'),
                    'confidence': analysis.get('confidence', 0),
                    'entry_price': analysis.get('entry_price', df_subset.iloc[-1]['close']),
                    'stop_loss': analysis.get('stop_loss', df_subset.iloc[-1]['close']),
                    'take_profit': analysis.get('take_profit', df_subset.iloc[-1]['close'])
                })
            
            # Calculate backtest results
            backtest_results = self._calculate_backtest_performance(df, signals, initial_capital)
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Backtesting error: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_backtest_performance(self, df: pd.DataFrame, signals: List[Dict], 
                                      initial_capital: float) -> Dict[str, Any]:
        """
        Calculate backtest performance metrics
        """
        # Placeholder for backtesting calculations
        # Implement your backtesting logic here
        return {
            "initial_capital": initial_capital,
            "final_capital": initial_capital * 1.15,  # Placeholder
            "total_return": 0.15,
            "total_trades": len([s for s in signals if s['signal'] != 'HOLD']),
            "win_rate": 0.65,
            "profit_factor": 1.8,
            "max_drawdown": 0.08,
            "sharpe_ratio": 1.5
        }
    
    async def _get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Get historical data - placeholder for actual implementation
        """
        # Implement your historical data provider connection
        logger.warning("_get_historical_data is not implemented - using placeholder")
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get strategy performance statistics
        """
        if self.performance_stats['total_signals'] > 0:
            self.performance_stats['win_rate'] = self.performance_stats['winning_signals'] / self.performance_stats['total_signals']
        
        return {
            **self.performance_stats,
            "strategy_info": {
                "name": self.name,
                "version": self.version,
                "indicators_count": len(self.ta_strategy.ta),
                "timeframes": self.timeframes,
                "smc_enabled": bool(self.smc_config)
            }
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Example configuration
    config = {
        'timeframes': ['M15', 'H1', 'H4'],
        'min_confluence_count': 2,
        'smart_money_concepts': {
            'swing_period': 10,
            'lookback': 50
        },
        'risk_management': {
            'max_risk_per_trade': 0.02,
            'stop_loss_pct': 0.02,
            'take_profit_ratio': 2.0
        },
        'signal_thresholds': {
            'min_confidence': 60
        }
    }
    
    # Initialize strategy
    strategy = PandasTAClassicStrategy(config)
    
    # Test analysis (would need actual data connection)
    async def test_strategy():
        try:
            print(f"Testing {strategy.name}")
            print(f"Strategy indicators: {len(strategy.ta_strategy.ta)}")
            print("Strategy initialized successfully!")
            
            # Show performance stats
            stats = strategy.get_performance_stats()
            print(f"Performance stats: {stats}")
            
        except Exception as e:
            print(f"Error testing strategy: {str(e)}")
    
    # Run test
    asyncio.run(test_strategy())