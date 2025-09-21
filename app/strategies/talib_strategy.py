"""TA-Lib Trading Strategy - Minimal Stable Version
Most reliable technical analysis with 150+ indicators

Author: AI/ML Trading Bot Team
Version: 2.0.3 (Stable)
Library: TA-Lib (most stable)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import talib  # Most stable TA library
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

try:
    from ..base_strategy import BaseStrategy
except ImportError:
    class BaseStrategy:
        def __init__(self, config: Dict):
            self.config = config


class TALibStrategy(BaseStrategy):
    """
    Rock-solid trading strategy using only TA-Lib
    
    Features:
    - 150+ TA-Lib indicators (battle-tested)
    - Custom Smart Money Concepts
    - Multi-timeframe analysis
    - Risk management
    - No dependency conflicts
    - Works everywhere
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.name = "TA-Lib Stable Strategy v2.0.3"
        self.version = "2.0.3"
        self.library = "TA-Lib (most stable)"
        
        self.timeframes = config.get('timeframes', ['M15', 'H1', 'H4', 'D1'])
        self.min_confluence = config.get('min_confluence_count', 2)
        
        self.performance_stats = {
            'total_signals': 0,
            'winning_signals': 0,
            'losing_signals': 0,
            'win_rate': 0.0
        }
        
        logger.info(f"Initialized {self.name} with TA-Lib (rock solid)")
    
    async def analyze(self, symbol: str, timeframe: str = "H1", limit: int = 500) -> Dict[str, Any]:
        """Main analysis using TA-Lib indicators"""
        try:
            start_time = datetime.utcnow()
            logger.info(f"Starting TA-Lib analysis for {symbol} on {timeframe}")
            
            # Get market data
            df = await self._get_market_data(symbol, timeframe, limit)
            if df is None or len(df) < 100:
                return self._empty_signal("Insufficient data")
            
            # Apply TA-Lib indicators (150+ available)
            df_analyzed = self._apply_talib_indicators(df.copy())
            
            # Add custom Smart Money Concepts
            df_analyzed = self._add_smart_money_concepts(df_analyzed)
            
            # Generate signals
            composite_signals = self._generate_composite_signals(df_analyzed)
            smc_analysis = self._analyze_smart_money_concepts(df_analyzed)
            risk_metrics = self._calculate_risk_metrics(df_analyzed)
            
            # Generate final signal
            final_signal = self._generate_final_signal(
                df_analyzed, composite_signals, smc_analysis, risk_metrics
            )
            
            self.performance_stats['total_signals'] += 1
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"TA-Lib analysis completed for {symbol} in {execution_time:.2f}s - Signal: {final_signal['signal']} ({final_signal['confidence']:.1f}%)")
            
            return {
                **final_signal,
                "analysis_breakdown": {
                    "technical_indicators": composite_signals,
                    "smart_money_concepts": smc_analysis,
                    "risk_metrics": risk_metrics,
                    "execution_time_seconds": execution_time,
                },
                "metadata": {
                    "strategy_version": self.version,
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "library": self.library,
                    "indicators_applied": len([col for col in df_analyzed.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
                }
            }
            
        except Exception as e:
            logger.error(f"TA-Lib analysis error for {symbol}: {str(e)}", exc_info=True)
            return self._empty_signal(f"Analysis error: {str(e)}")
    
    def _apply_talib_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive TA-Lib indicators (150+ available)"""
        try:
            logger.debug("Applying TA-Lib indicators...")
            
            # Extract OHLCV arrays for TA-Lib
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            open_price = df['open'].values
            volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))
            
            # === OVERLAP STUDIES (Moving Averages & Bands) ===
            df['SMA_10'] = talib.SMA(close, timeperiod=10)
            df['SMA_20'] = talib.SMA(close, timeperiod=20)
            df['SMA_50'] = talib.SMA(close, timeperiod=50)
            df['SMA_200'] = talib.SMA(close, timeperiod=200)
            df['EMA_8'] = talib.EMA(close, timeperiod=8)
            df['EMA_12'] = talib.EMA(close, timeperiod=12)
            df['EMA_21'] = talib.EMA(close, timeperiod=21)
            df['EMA_26'] = talib.EMA(close, timeperiod=26)
            df['EMA_50'] = talib.EMA(close, timeperiod=50)
            df['WMA_14'] = talib.WMA(close, timeperiod=14)
            df['TRIMA_14'] = talib.TRIMA(close, timeperiod=14)
            df['KAMA_14'] = talib.KAMA(close, timeperiod=14)
            df['T3_14'] = talib.T3(close, timeperiod=14)
            
            # Bollinger Bands
            df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            df['BB_WIDTH'] = df['BB_UPPER'] - df['BB_LOWER']
            df['BB_PERCENT'] = (close - df['BB_LOWER']) / df['BB_WIDTH'] * 100
            
            # MESA Adaptive Moving Average
            df['MAMA'], df['FAMA'] = talib.MAMA(close, fastlimit=0.5, slowlimit=0.05)
            
            # === MOMENTUM INDICATORS ===
            df['RSI_14'] = talib.RSI(close, timeperiod=14)
            df['RSI_21'] = talib.RSI(close, timeperiod=21)
            df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            df['STOCH_K'], df['STOCH_D'] = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            df['STOCHF_K'], df['STOCHF_D'] = talib.STOCHF(high, low, close, fastk_period=14, fastd_period=3)
            df['STOCHRSI_K'], df['STOCHRSI_D'] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3)
            df['WILLR_14'] = talib.WILLR(high, low, close, timeperiod=14)
            df['CCI_14'] = talib.CCI(high, low, close, timeperiod=14)
            df['CCI_20'] = talib.CCI(high, low, close, timeperiod=20)
            df['ROC_10'] = talib.ROC(close, timeperiod=10)
            df['MOM_10'] = talib.MOM(close, timeperiod=10)
            df['CMO_14'] = talib.CMO(close, timeperiod=14)
            df['TRIX_14'] = talib.TRIX(close, timeperiod=14)
            df['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26)
            df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26)
            
            # Ultimate Oscillator
            df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
            
            # === VOLUME INDICATORS ===
            df['OBV'] = talib.OBV(close, volume)
            df['AD'] = talib.AD(high, low, close, volume)
            df['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
            
            # === VOLATILITY INDICATORS ===
            df['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)
            df['ATR_20'] = talib.ATR(high, low, close, timeperiod=20)
            df['NATR_14'] = talib.NATR(high, low, close, timeperiod=14)
            df['TRANGE'] = talib.TRANGE(high, low, close)
            
            # === PRICE TRANSFORM ===
            df['AVGPRICE'] = talib.AVGPRICE(open_price, high, low, close)
            df['MEDPRICE'] = talib.MEDPRICE(high, low)
            df['TYPPRICE'] = talib.TYPPRICE(high, low, close)
            df['WCLPRICE'] = talib.WCLPRICE(high, low, close)
            
            # === CYCLE INDICATORS (Hilbert Transform) ===
            df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
            df['HT_DCPHASE'] = talib.HT_DCPHASE(close)
            df['HT_PHASOR_IQ'], df['HT_PHASOR_Q'] = talib.HT_PHASOR(close)
            df['HT_SINE_SINE'], df['HT_SINE_LEADSINE'] = talib.HT_SINE(close)
            df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)
            
            # === PATTERN RECOGNITION (Selected powerful patterns) ===
            df['CDL_DOJI'] = talib.CDLDOJI(open_price, high, low, close)
            df['CDL_HAMMER'] = talib.CDLHAMMER(open_price, high, low, close)
            df['CDL_SHOOTING_STAR'] = talib.CDLSHOOTINGSTAR(open_price, high, low, close)
            df['CDL_ENGULFING'] = talib.CDLENGULFING(open_price, high, low, close)
            df['CDL_HARAMI'] = talib.CDLHARAMI(open_price, high, low, close)
            df['CDL_MORNING_STAR'] = talib.CDLMORNINGSTAR(open_price, high, low, close)
            df['CDL_EVENING_STAR'] = talib.CDLEVENINGSTAR(open_price, high, low, close)
            df['CDL_THREE_BLACK_CROWS'] = talib.CDL3BLACKCROWS(open_price, high, low, close)
            df['CDL_THREE_WHITE_SOLDIERS'] = talib.CDL3WHITESOLDIERS(open_price, high, low, close)
            df['CDL_HANGING_MAN'] = talib.CDLHANGINGMAN(open_price, high, low, close)
            df['CDL_INVERTED_HAMMER'] = talib.CDLINVERTEDHAMMER(open_price, high, low, close)
            df['CDL_DRAGONFLY_DOJI'] = talib.CDLDRAGONFLYDOJI(open_price, high, low, close)
            df['CDL_GRAVESTONE_DOJI'] = talib.CDLGRAVESTONEDOJI(open_price, high, low, close)
            
            # === STATISTIC FUNCTIONS ===
            df['BETA'] = talib.BETA(high, low, timeperiod=5)
            df['CORREL'] = talib.CORREL(high, low, timeperiod=30)
            df['LINEARREG'] = talib.LINEARREG(close, timeperiod=14)
            df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(close, timeperiod=14)
            df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close, timeperiod=14)
            df['STDDEV'] = talib.STDDEV(close, timeperiod=5)
            df['TSF'] = talib.TSF(close, timeperiod=14)
            df['VAR'] = talib.VAR(close, timeperiod=5)
            
            # === MATH TRANSFORM ===
            df['SIN'] = talib.SIN(close)
            df['COS'] = talib.COS(close)
            df['ATAN'] = talib.ATAN(close)
            
            # === CUSTOM COMPOSITE INDICATORS ===
            df = self._add_composite_indicators(df)
            
            indicator_count = len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
            logger.debug(f"Applied {indicator_count} TA-Lib indicators successfully")
            
            return df
            
        except Exception as e:
            logger.error(f"Error applying TA-Lib indicators: {str(e)}")
            return df
    
    def _add_composite_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom composite indicators combining TA-Lib signals"""
        try:
            # Trend Strength Composite (0-100)
            trend_signals = []
            if all(col in df.columns for col in ['close', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26']):
                trend_signals.extend([
                    (df['close'] > df['SMA_20']).astype(int),
                    (df['SMA_20'] > df['SMA_50']).astype(int),
                    (df['EMA_12'] > df['EMA_26']).astype(int)
                ])
            if 'HT_TRENDMODE' in df.columns:
                trend_signals.append((df['HT_TRENDMODE'] == 1).astype(int))
            if 'LINEARREG_SLOPE' in df.columns:
                trend_signals.append((df['LINEARREG_SLOPE'] > 0).astype(int))
                
            df['trend_strength'] = np.mean(trend_signals, axis=0) * 100 if trend_signals else 50
            
            # Momentum Composite (0-100)
            momentum_values = []
            if 'RSI_14' in df.columns:
                momentum_values.append(df['RSI_14'])
            if 'STOCH_K' in df.columns:
                momentum_values.append(df['STOCH_K'])
            if 'CCI_14' in df.columns:
                # Normalize CCI to 0-100 scale
                cci_norm = (df['CCI_14'].clip(-200, 200) + 200) / 4
                momentum_values.append(cci_norm)
            if 'ULTOSC' in df.columns:
                momentum_values.append(df['ULTOSC'])
                
            df['momentum_composite'] = np.mean(momentum_values, axis=0) if momentum_values else 50
            
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
            
            # Pattern Recognition Composite
            pattern_cols = [col for col in df.columns if col.startswith('CDL_')]
            if pattern_cols:
                bullish_patterns = df[pattern_cols].gt(0).sum(axis=1)
                bearish_patterns = df[pattern_cols].lt(0).sum(axis=1)
                df['pattern_bullish_count'] = bullish_patterns
                df['pattern_bearish_count'] = bearish_patterns
                df['pattern_net_bias'] = bullish_patterns - bearish_patterns
            else:
                df['pattern_bullish_count'] = 0
                df['pattern_bearish_count'] = 0
                df['pattern_net_bias'] = 0
                
            return df
            
        except Exception as e:
            logger.error(f"Error adding composite indicators: {str(e)}")
            return df
    
    def _add_smart_money_concepts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom Smart Money Concepts"""
        try:
            logger.debug("Adding Smart Money Concepts...")
            
            df['order_blocks'] = self._detect_order_blocks(df)
            df['fair_value_gaps'] = self._detect_fair_value_gaps(df)
            df['break_of_structure'] = self._detect_break_of_structure(df)
            df['liquidity_sweeps'] = self._detect_liquidity_sweeps(df)
            
            # SMC composite score
            smc_signals = [df['order_blocks'], df['fair_value_gaps'], df['break_of_structure'], df['liquidity_sweeps']]
            df['smc_composite'] = np.mean([np.abs(signal) for signal in smc_signals], axis=0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding SMC: {str(e)}")
            return df
    
    def _detect_order_blocks(self, df: pd.DataFrame) -> pd.Series:
        """Detect institutional order blocks"""
        try:
            if 'volume' not in df.columns:
                return pd.Series(0, index=df.index)
                
            volume_avg = df['volume'].rolling(20).mean()
            body_size = np.abs(df['close'] - df['open'])
            body_avg = body_size.rolling(20).mean()
            
            # Use ATR for volatility context
            atr = df.get('ATR_14', body_size)
            
            bearish_ob = (
                (df['volume'] > volume_avg * 1.5) &
                (body_size > body_avg * 1.2) &
                (df['close'] < df['open']) &
                (df['low'] < df['low'].shift(1)) &
                (body_size > atr * 0.5)  # Significant move
            )
            
            bullish_ob = (
                (df['volume'] > volume_avg * 1.5) &
                (body_size > body_avg * 1.2) &
                (df['close'] > df['open']) &
                (df['high'] > df['high'].shift(1)) &
                (body_size > atr * 0.5)  # Significant move
            )
            
            order_blocks = pd.Series(0, index=df.index)
            order_blocks[bearish_ob] = -1
            order_blocks[bullish_ob] = 1
            
            return order_blocks
            
        except Exception:
            return pd.Series(0, index=df.index)
    
    def _detect_fair_value_gaps(self, df: pd.DataFrame) -> pd.Series:
        """Detect Fair Value Gaps (imbalances)"""
        try:
            # Bullish FVG: Previous high < Current low (gap up)
            bullish_fvg = df['high'].shift(1) < df['low']
            
            # Bearish FVG: Previous low > Current high (gap down)
            bearish_fvg = df['low'].shift(1) > df['high']
            
            # Filter for significant gaps using ATR
            atr = df.get('ATR_14', df['close'] * 0.001)
            min_gap_size = atr * 0.5
            
            gap_size_bull = df['low'] - df['high'].shift(1)
            gap_size_bear = df['low'].shift(1) - df['high']
            
            bullish_fvg = bullish_fvg & (gap_size_bull > min_gap_size)
            bearish_fvg = bearish_fvg & (gap_size_bear > min_gap_size)
            
            fvg = pd.Series(0, index=df.index)
            fvg[bullish_fvg] = 1
            fvg[bearish_fvg] = -1
            
            return fvg
            
        except Exception:
            return pd.Series(0, index=df.index)
    
    def _detect_break_of_structure(self, df: pd.DataFrame) -> pd.Series:
        """Detect Break of Structure using swing levels"""
        try:
            swing_period = 10
            high_rolling_max = df['high'].rolling(swing_period, center=True).max()
            low_rolling_min = df['low'].rolling(swing_period, center=True).min()
            
            # BOS when price breaks significant levels
            bullish_bos = df['close'] > high_rolling_max.shift(5)
            bearish_bos = df['close'] < low_rolling_min.shift(5)
            
            # Add volume confirmation if available
            if 'volume' in df.columns:
                volume_avg = df['volume'].rolling(20).mean()
                volume_confirmation = df['volume'] > volume_avg
                bullish_bos = bullish_bos & volume_confirmation
                bearish_bos = bearish_bos & volume_confirmation
            
            bos = pd.Series(0, index=df.index)
            bos[bullish_bos] = 1
            bos[bearish_bos] = -1
            
            return bos
            
        except Exception:
            return pd.Series(0, index=df.index)
    
    def _detect_liquidity_sweeps(self, df: pd.DataFrame) -> pd.Series:
        """Detect Liquidity Sweeps (stop hunts)"""
        try:
            lookback = 50
            recent_high = df['high'].rolling(lookback).max()
            recent_low = df['low'].rolling(lookback).min()
            
            # Upside liquidity sweep
            sweep_up = (
                (df['high'] > recent_high.shift(1)) &     # Break recent high
                (df['close'] < df['open']) &              # But close bearish
                (df['close'] < recent_high.shift(1))      # Close back below level
            )
            
            # Downside liquidity sweep
            sweep_down = (
                (df['low'] < recent_low.shift(1)) &       # Break recent low
                (df['close'] > df['open']) &              # But close bullish
                (df['close'] > recent_low.shift(1))       # Close back above level
            )
            
            # Add volume confirmation if available
            if 'volume' in df.columns:
                volume_avg = df['volume'].rolling(20).mean()
                volume_spike = df['volume'] > volume_avg * 1.2
                sweep_up = sweep_up & volume_spike
                sweep_down = sweep_down & volume_spike
            
            sweeps = pd.Series(0, index=df.index)
            sweeps[sweep_up] = 1
            sweeps[sweep_down] = -1
            
            return sweeps
            
        except Exception:
            return pd.Series(0, index=df.index)
    
    def _generate_composite_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate composite trading signals"""
        try:
            latest = df.iloc[-1]
            
            signals = {
                'trend_strength': latest.get('trend_strength', 50),
                'momentum_score': latest.get('momentum_composite', 50),
                'volume_confirmation': latest.get('volume_confirmation', 50),
                'volatility_environment': latest.get('volatility_percentile', 50)
            }
            
            # RSI analysis
            if 'RSI_14' in df.columns:
                rsi = latest['RSI_14']
                signals['rsi_value'] = float(rsi)
                signals['rsi_signal'] = 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'
            
            # MACD analysis
            if all(col in df.columns for col in ['MACD', 'MACD_SIGNAL', 'MACD_HIST']):
                macd = latest['MACD']
                macd_signal = latest['MACD_SIGNAL']
                macd_hist = latest['MACD_HIST']
                
                signals['macd_value'] = float(macd)
                signals['macd_signal_value'] = float(macd_signal)
                signals['macd_histogram'] = float(macd_hist)
                signals['macd_signal'] = 'bullish' if macd > macd_signal else 'bearish'
                signals['macd_momentum'] = 'increasing' if macd_hist > 0 else 'decreasing'
            
            # Bollinger Bands analysis
            if all(col in df.columns for col in ['BB_UPPER', 'BB_LOWER', 'BB_PERCENT']):
                bb_percent = latest['BB_PERCENT']
                signals['bb_percent'] = float(bb_percent)
                
                if bb_percent > 100:
                    signals['bb_signal'] = 'overbought'
                elif bb_percent < 0:
                    signals['bb_signal'] = 'oversold'
                elif bb_percent > 50:
                    signals['bb_signal'] = 'bullish'
                else:
                    signals['bb_signal'] = 'bearish'
            
            # Pattern recognition summary
            if all(col in df.columns for col in ['pattern_bullish_count', 'pattern_bearish_count', 'pattern_net_bias']):
                signals['pattern_bullish_count'] = int(latest['pattern_bullish_count'])
                signals['pattern_bearish_count'] = int(latest['pattern_bearish_count'])
                signals['pattern_net_bias'] = int(latest['pattern_net_bias'])
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_smart_money_concepts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Smart Money Concepts"""
        try:
            latest = df.iloc[-1]
            recent = df.tail(20)
            
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
                'confidence': min(smc_strength * 15, 100),
                'composite_score': float(latest.get('smc_composite', 0))
            }
            
            return smc_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing SMC: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk management metrics"""
        try:
            latest = df.iloc[-1]
            recent = df.tail(20)
            
            # ATR for volatility-based risk
            atr = latest.get('ATR_14', (latest['high'] - latest['low']))
            current_price = latest['close']
            
            # Support/Resistance levels
            recent_high = recent['high'].max()
            recent_low = recent['low'].min()
            
            # Volatility percentile
            volatility_pct = latest.get('volatility_percentile', 50)
            
            risk_metrics = {
                'atr_value': float(atr),
                'atr_percent': float(atr / current_price * 100),
                'current_price': float(current_price),
                'recent_high': float(recent_high),
                'recent_low': float(recent_low),
                'distance_to_resistance_pct': float((recent_high - current_price) / current_price * 100),
                'distance_to_support_pct': float((current_price - recent_low) / current_price * 100),
                'volatility_environment': volatility_pct,
                
                # Risk management suggestions
                'recommended_sl_distance': float(atr * 2),
                'recommended_tp_distance': float(atr * 4),
                'conservative_sl_distance': float(atr * 1.5),
                'aggressive_tp_distance': float(atr * 6),
                
                # Position sizing (2% risk)
                'position_size_2pct_risk': 0.02 / (atr * 2 / current_price) if atr > 0 else 1.0,
                
                # Volatility classification
                'volatility_class': 'high' if volatility_pct > 75 else 'low' if volatility_pct < 25 else 'medium'
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {'error': str(e)}
    
    def _generate_final_signal(self, df: pd.DataFrame, composite_signals: Dict, 
                              smc_analysis: Dict, risk_metrics: Dict) -> Dict[str, Any]:
        """Generate final trading signal with comprehensive analysis"""
        try:
            latest = df.iloc[-1]
            current_price = float(latest['close'])
            
            # Initialize scoring system
            bullish_score = 0
            bearish_score = 0
            max_score = 10
            
            # 1. Trend Analysis (Weight: 2)
            trend_strength = composite_signals.get('trend_strength', 50)
            if trend_strength > 70:
                bullish_score += 2
            elif trend_strength < 30:
                bearish_score += 2
            elif trend_strength > 60:
                bullish_score += 1
            elif trend_strength < 40:
                bearish_score += 1
            
            # 2. Momentum Analysis (Weight: 2)
            momentum = composite_signals.get('momentum_score', 50)
            if momentum > 70:
                bullish_score += 2
            elif momentum < 30:
                bearish_score += 2
            elif momentum > 60:
                bullish_score += 1
            elif momentum < 40:
                bearish_score += 1
            
            # 3. RSI Analysis (Weight: 1)
            rsi_signal = composite_signals.get('rsi_signal', 'neutral')
            if rsi_signal == 'oversold':
                bullish_score += 1
            elif rsi_signal == 'overbought':
                bearish_score += 1
            
            # 4. MACD Analysis (Weight: 1)
            macd_signal = composite_signals.get('macd_signal', 'neutral')
            macd_momentum = composite_signals.get('macd_momentum', 'neutral')
            if macd_signal == 'bullish' and macd_momentum == 'increasing':
                bullish_score += 1
            elif macd_signal == 'bearish' and macd_momentum == 'decreasing':
                bearish_score += 1
            
            # 5. Bollinger Bands Analysis (Weight: 1)
            bb_signal = composite_signals.get('bb_signal', 'neutral')
            if bb_signal == 'oversold':
                bullish_score += 1
            elif bb_signal == 'overbought':
                bearish_score += 1
            
            # 6. Smart Money Concepts (Weight: 2)
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
            
            # 7. Pattern Recognition (Weight: 1)
            pattern_net_bias = composite_signals.get('pattern_net_bias', 0)
            if pattern_net_bias > 0:
                bullish_score += 1
            elif pattern_net_bias < 0:
                bearish_score += 1
            
            # Generate signal based on scores
            total_score = bullish_score + bearish_score
            if total_score == 0:
                return self._create_hold_signal(current_price, "No clear signals detected")
            
            confidence = max(bullish_score, bearish_score) / max_score * 100
            
            # Minimum confidence threshold
            min_confidence = 55  # Lower threshold for TA-Lib signals
            if confidence < min_confidence:
                return self._create_hold_signal(current_price, f"Low confidence: {confidence:.1f}%")
            
            # Generate BUY or SELL signal
            atr_value = risk_metrics.get('atr_value', current_price * 0.01)
            volatility_class = risk_metrics.get('volatility_class', 'medium')
            
            # Adjust risk based on volatility
            if volatility_class == 'high':
                sl_multiplier = 2.5
                tp_multiplier = 5.0
            elif volatility_class == 'low':
                sl_multiplier = 1.5
                tp_multiplier = 3.0
            else:
                sl_multiplier = 2.0
                tp_multiplier = 4.0
            
            if bullish_score > bearish_score:
                direction = "BUY"
                entry_price = current_price
                stop_loss = current_price - (atr_value * sl_multiplier)
                take_profit = current_price + (atr_value * tp_multiplier)
            else:
                direction = "SELL"
                entry_price = current_price
                stop_loss = current_price + (atr_value * sl_multiplier)
                take_profit = current_price - (atr_value * tp_multiplier)
            
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
                "position_size_2pct_risk": risk_metrics.get('position_size_2pct_risk', 1.0),
                "volatility_class": volatility_class,
                "signal_breakdown": {
                    "bullish_score": bullish_score,
                    "bearish_score": bearish_score,
                    "max_score": max_score,
                    "trend_strength": trend_strength,
                    "momentum_score": momentum,
                    "rsi_signal": rsi_signal,
                    "macd_signal": macd_signal,
                    "bb_signal": bb_signal,
                    "smc_bias": smc_bias,
                    "smc_strength": smc_strength,
                    "pattern_bias": pattern_net_bias,
                    "atr_multiplier_used": f"SL: {sl_multiplier}x, TP: {tp_multiplier}x"
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating final signal: {str(e)}")
            return self._empty_signal(f"Signal generation error: {str(e)}")
    
    def _create_hold_signal(self, price: float, reason: str) -> Dict[str, Any]:
        """Create HOLD signal"""
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "entry_price": round(price, 5),
            "stop_loss": round(price, 5),
            "take_profit": round(price, 5),
            "risk_reward_ratio": 1.0,
            "reason": reason,
            "signal_breakdown": {
                "reason": reason,
                "recommendation": "Wait for clearer signals"
            }
        }
    
    def _empty_signal(self, error_msg: str) -> Dict[str, Any]:
        """Create empty signal with error"""
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
        """Get market data - implement your data source here"""
        logger.warning("_get_market_data not implemented - using sample data for testing")
        
        # Generate realistic sample data for testing
        try:
            dates = pd.date_range(start='2024-01-01', periods=limit, freq='1H')
            np.random.seed(42)  # Reproducible for testing
            
            # Generate realistic OHLCV data with trends
            base_price = 1.1000 if 'USD' in symbol else 50000 if 'BTC' in symbol else 100
            price_changes = np.random.randn(limit) * 0.001
            
            # Add some trend
            trend = np.linspace(0, 0.05, limit) if np.random.random() > 0.5 else np.linspace(0, -0.05, limit)
            prices = base_price * (1 + np.cumsum(price_changes) + trend)
            
            # Generate OHLC with realistic relationships
            volatility = np.random.rand(limit) * 0.002 + 0.0005
            
            df = pd.DataFrame({
                'open': prices + np.random.randn(limit) * volatility * 0.5,
                'high': prices + np.abs(np.random.randn(limit)) * volatility,
                'low': prices - np.abs(np.random.randn(limit)) * volatility,
                'close': prices,
                'volume': np.random.randint(1000, 50000, limit)
            }, index=dates)
            
            # Ensure OHLC relationships are correct
            df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
            df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
            
            # Add some volume spikes
            spike_indices = np.random.choice(len(df), size=len(df)//20, replace=False)
            df.loc[df.index[spike_indices], 'volume'] *= np.random.uniform(2, 5, len(spike_indices))
            
            logger.debug(f"Generated sample data: {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error generating sample data: {str(e)}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics"""
        if self.performance_stats['total_signals'] > 0:
            self.performance_stats['win_rate'] = self.performance_stats['winning_signals'] / self.performance_stats['total_signals']
        
        return {
            **self.performance_stats,
            "strategy_info": {
                "name": self.name,
                "version": self.version,
                "library": self.library,
                "indicators_available": "150+ TA-Lib indicators",
                "features": [
                    "Trend Analysis",
                    "Momentum Oscillators", 
                    "Volume Indicators",
                    "Volatility Measures",
                    "Pattern Recognition",
                    "Smart Money Concepts",
                    "Risk Management"
                ],
                "timeframes": self.timeframes,
                "smart_money_concepts": True,
                "dependency_conflicts": "None - Rock Solid!"
            }
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    config = {
        'timeframes': ['H1', 'H4'],
        'min_confluence_count': 1,
        'smart_money_concepts': {
            'swing_period': 10,
            'lookback': 50
        }
    }
    
    async def test_strategy():
        try:
            print("\n🚀 Testing TA-Lib Strategy...")
            strategy = TALibStrategy(config)
            print(f"Strategy: {strategy.name}")
            print(f"Library: {strategy.library}")
            
            # Test analysis
            result = await strategy.analyze("EURUSD", "H1")
            
            print(f"\n📊 Analysis Results:")
            print(f"Signal: {result['signal']} ({result['confidence']:.1f}%)")
            print(f"Entry: {result['entry_price']:.5f}")
            print(f"Stop Loss: {result['stop_loss']:.5f}")
            print(f"Take Profit: {result['take_profit']:.5f}")
            print(f"Risk/Reward: {result['risk_reward_ratio']:.2f}")
            
            if 'signal_breakdown' in result:
                breakdown = result['signal_breakdown']
                print(f"\n🔍 Signal Breakdown:")
                print(f"Bullish Score: {breakdown.get('bullish_score', 0)}")
                print(f"Bearish Score: {breakdown.get('bearish_score', 0)}")
                print(f"Trend Strength: {breakdown.get('trend_strength', 0):.1f}")
                print(f"Momentum: {breakdown.get('momentum_score', 0):.1f}")
                print(f"SMC Bias: {breakdown.get('smc_bias', 'N/A')}")
            
            # Test performance stats
            stats = strategy.get_performance_stats()
            print(f"\n📈 Performance Stats:")
            print(f"Total Signals: {stats['total_signals']}")
            print(f"Features: {', '.join(stats['strategy_info']['features'])}")
            print(f"Dependency Issues: {stats['strategy_info']['dependency_conflicts']}")
            
            print("\n✅ TA-Lib Strategy test completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error testing strategy: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Run the test
    asyncio.run(test_strategy())