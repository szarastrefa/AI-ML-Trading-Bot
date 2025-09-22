"""
Smart Money Concepts Strategy
Implementation of institutional trading concepts including:
- Order Blocks (OB)
- Fair Value Gaps (FVG) 
- Break of Structure (BOS)
- Liquidity Sweeps
- Market Structure Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Safe TA-Lib import
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class SmartMoneyStrategy(BaseStrategy):
    """
    Advanced Smart Money Concepts (SMC) Strategy
    Based on institutional trading principles
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        self.name = "Smart Money Concepts Strategy"
        self.version = "2.1.0"
        
        # SMC Parameters
        self.swing_period = self.config.get('swing_period', 10)
        self.lookback_periods = self.config.get('lookback_periods', 50)
        self.ob_threshold = self.config.get('order_block_threshold', 0.002)  # 0.2%
        self.fvg_threshold = self.config.get('fvg_threshold', 0.001)  # 0.1%
        self.liquidity_threshold = self.config.get('liquidity_threshold', 0.0015)  # 0.15%
        
        logger.info(f"✅ {self.name} v{self.version} initialized")
        logger.info(f"📊 SMC Parameters: swing={self.swing_period}, lookback={self.lookback_periods}")
        
    async def analyze(self, symbol: str, timeframe: str = "H1") -> Dict[str, Any]:
        """
        Main SMC analysis function
        """
        try:
            self._increment_analysis_count()
            start_time = datetime.utcnow()
            
            logger.info(f"📊 Starting SMC analysis: {symbol} {timeframe}")
            
            # Get market data
            df = await self._get_market_data(symbol, timeframe)
            
            if df is None or len(df) < self.lookback_periods:
                logger.warning(f"Insufficient data for SMC analysis: {symbol}")
                return self._create_empty_signal("Insufficient market data")
            
            # Apply SMC analysis components
            df_analyzed = self._apply_smc_analysis(df)
            
            # Generate SMC signal
            signal = self._generate_smc_signal(df_analyzed, symbol)
            
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
                "smc_analysis": True
            })
            
            logger.info(f"✅ SMC Analysis completed: {signal['signal']} ({signal['confidence']:.1f}%)")
            return signal
            
        except Exception as e:
            logger.error(f"❌ SMC Analysis failed for {symbol}: {str(e)}")
            return self._create_empty_signal(f"SMC Analysis error: {str(e)}")
    
    def _apply_smc_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Smart Money Concepts analysis components
        """
        try:
            logger.debug(f"Applying SMC analysis to {len(df)} data points")
            
            # 1. Market Structure Analysis
            df = self._identify_market_structure(df)
            
            # 2. Order Blocks Detection
            df = self._detect_order_blocks(df)
            
            # 3. Fair Value Gaps Detection
            df = self._detect_fair_value_gaps(df)
            
            # 4. Break of Structure Analysis
            df = self._analyze_break_of_structure(df)
            
            # 5. Liquidity Analysis
            df = self._analyze_liquidity(df)
            
            # 6. SMC Confluence Score
            df = self._calculate_smc_confluence(df)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error in SMC analysis: {str(e)}")
            return df
    
    def _identify_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify market structure: Higher Highs/Lower Lows, trend direction
        """
        try:
            # Identify swing highs and lows
            df['swing_high'] = (df['high'] > df['high'].shift(self.swing_period)) & \
                              (df['high'] > df['high'].shift(-self.swing_period))
            
            df['swing_low'] = (df['low'] < df['low'].shift(self.swing_period)) & \
                             (df['low'] < df['low'].shift(-self.swing_period))
            
            # Track swing points
            df['swing_high_price'] = np.where(df['swing_high'], df['high'], np.nan)
            df['swing_low_price'] = np.where(df['swing_low'], df['low'], np.nan)
            
            # Forward fill swing points
            df['last_swing_high'] = df['swing_high_price'].fillna(method='ffill')
            df['last_swing_low'] = df['swing_low_price'].fillna(method='ffill')
            
            # Market structure bias
            df['market_structure'] = 0  # 0=ranging, 1=bullish, -1=bearish
            
            for i in range(self.swing_period, len(df)):
                recent_highs = df['swing_high_price'].iloc[i-self.swing_period:i].dropna()
                recent_lows = df['swing_low_price'].iloc[i-self.swing_period:i].dropna()
                
                if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                    # Higher highs and higher lows = bullish structure
                    if (recent_highs.iloc[-1] > recent_highs.iloc[-2] and 
                        recent_lows.iloc[-1] > recent_lows.iloc[-2]):
                        df.at[df.index[i], 'market_structure'] = 1
                    
                    # Lower highs and lower lows = bearish structure  
                    elif (recent_highs.iloc[-1] < recent_highs.iloc[-2] and 
                          recent_lows.iloc[-1] < recent_lows.iloc[-2]):
                        df.at[df.index[i], 'market_structure'] = -1
            
            return df
            
        except Exception as e:
            logger.error(f"Error in market structure analysis: {str(e)}")
            return df
    
    def _detect_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Order Blocks - areas where smart money accumulated positions
        """
        try:
            df['bullish_ob'] = False
            df['bearish_ob'] = False
            df['ob_strength'] = 0.0
            
            for i in range(10, len(df) - 5):
                current_candle = df.iloc[i]
                
                # Bullish Order Block:
                # Last bearish candle before strong bullish move
                if (current_candle['close'] > current_candle['open'] and  # Current is bullish
                    df.iloc[i-1]['close'] < df.iloc[i-1]['open'] and     # Previous is bearish
                    current_candle['close'] > df.iloc[i-1]['high']):      # Gap up
                    
                    # Check for strong move (price appreciation)
                    price_move = (current_candle['high'] - df.iloc[i-1]['low']) / df.iloc[i-1]['low']
                    
                    if price_move > self.ob_threshold:
                        df.at[df.index[i-1], 'bullish_ob'] = True
                        df.at[df.index[i-1], 'ob_strength'] = price_move
                
                # Bearish Order Block:
                # Last bullish candle before strong bearish move
                if (current_candle['close'] < current_candle['open'] and  # Current is bearish
                    df.iloc[i-1]['close'] > df.iloc[i-1]['open'] and     # Previous is bullish
                    current_candle['close'] < df.iloc[i-1]['low']):       # Gap down
                    
                    # Check for strong move (price depreciation)
                    price_move = (df.iloc[i-1]['high'] - current_candle['low']) / df.iloc[i-1]['high']
                    
                    if price_move > self.ob_threshold:
                        df.at[df.index[i-1], 'bearish_ob'] = True
                        df.at[df.index[i-1], 'ob_strength'] = price_move
            
            return df
            
        except Exception as e:
            logger.error(f"Error in order blocks detection: {str(e)}")
            return df
    
    def _detect_fair_value_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Fair Value Gaps (FVG) - price inefficiencies
        """
        try:
            df['bullish_fvg'] = False
            df['bearish_fvg'] = False
            df['fvg_strength'] = 0.0
            
            for i in range(2, len(df)):
                candle1 = df.iloc[i-2]  # First candle
                candle2 = df.iloc[i-1]  # Middle candle  
                candle3 = df.iloc[i]    # Current candle
                
                # Bullish FVG: Gap between candle1 high and candle3 low
                if (candle1['high'] < candle3['low'] and
                    candle2['close'] > candle2['open']):  # Middle candle is bullish
                    
                    gap_size = (candle3['low'] - candle1['high']) / candle1['high']
                    
                    if gap_size > self.fvg_threshold:
                        df.at[df.index[i-1], 'bullish_fvg'] = True
                        df.at[df.index[i-1], 'fvg_strength'] = gap_size
                
                # Bearish FVG: Gap between candle1 low and candle3 high
                if (candle1['low'] > candle3['high'] and
                    candle2['close'] < candle2['open']):  # Middle candle is bearish
                    
                    gap_size = (candle1['low'] - candle3['high']) / candle1['low']
                    
                    if gap_size > self.fvg_threshold:
                        df.at[df.index[i-1], 'bearish_fvg'] = True
                        df.at[df.index[i-1], 'fvg_strength'] = gap_size
            
            return df
            
        except Exception as e:
            logger.error(f"Error in FVG detection: {str(e)}")
            return df
    
    def _analyze_break_of_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze Break of Structure (BOS) - trend change confirmations
        """
        try:
            df['bos_bullish'] = False
            df['bos_bearish'] = False
            df['bos_strength'] = 0.0
            
            for i in range(20, len(df)):
                current_price = df.iloc[i]['close']
                
                # Look for recent swing highs/lows
                recent_data = df.iloc[i-20:i]
                
                # Bullish BOS: Breaking above recent swing high
                recent_swing_highs = recent_data[recent_data['swing_high'] == True]
                if len(recent_swing_highs) > 0:
                    highest_swing = recent_swing_highs['high'].max()
                    
                    if current_price > highest_swing:
                        strength = (current_price - highest_swing) / highest_swing
                        df.at[df.index[i], 'bos_bullish'] = True
                        df.at[df.index[i], 'bos_strength'] = strength
                
                # Bearish BOS: Breaking below recent swing low
                recent_swing_lows = recent_data[recent_data['swing_low'] == True]
                if len(recent_swing_lows) > 0:
                    lowest_swing = recent_swing_lows['low'].min()
                    
                    if current_price < lowest_swing:
                        strength = (lowest_swing - current_price) / lowest_swing
                        df.at[df.index[i], 'bos_bearish'] = True
                        df.at[df.index[i], 'bos_strength'] = strength
            
            return df
            
        except Exception as e:
            logger.error(f"Error in BOS analysis: {str(e)}")
            return df
    
    def _analyze_liquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze liquidity zones and potential sweeps
        """
        try:
            df['liquidity_grab_high'] = False
            df['liquidity_grab_low'] = False
            df['liquidity_strength'] = 0.0
            
            for i in range(10, len(df) - 1):
                current = df.iloc[i]
                next_candle = df.iloc[i + 1]
                
                # High liquidity grab: Wick above previous highs then reversal
                recent_highs = df.iloc[i-10:i]['high'].max()
                
                if (current['high'] > recent_highs and  # New high
                    current['close'] < current['open'] and  # Bearish close
                    next_candle['close'] < current['low']):  # Continued bearish
                    
                    strength = (current['high'] - recent_highs) / recent_highs
                    if strength > self.liquidity_threshold:
                        df.at[df.index[i], 'liquidity_grab_high'] = True
                        df.at[df.index[i], 'liquidity_strength'] = strength
                
                # Low liquidity grab: Wick below previous lows then reversal
                recent_lows = df.iloc[i-10:i]['low'].min()
                
                if (current['low'] < recent_lows and  # New low
                    current['close'] > current['open'] and  # Bullish close
                    next_candle['close'] > current['high']):  # Continued bullish
                    
                    strength = (recent_lows - current['low']) / recent_lows
                    if strength > self.liquidity_threshold:
                        df.at[df.index[i], 'liquidity_grab_low'] = True
                        df.at[df.index[i], 'liquidity_strength'] = strength
            
            return df
            
        except Exception as e:
            logger.error(f"Error in liquidity analysis: {str(e)}")
            return df
    
    def _calculate_smc_confluence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate SMC confluence score based on multiple factors
        """
        try:
            df['smc_bullish_score'] = 0
            df['smc_bearish_score'] = 0
            df['smc_confluence'] = 0
            
            for i in range(len(df)):
                bullish_score = 0
                bearish_score = 0
                
                # Market Structure
                if df.iloc[i]['market_structure'] == 1:
                    bullish_score += 2
                elif df.iloc[i]['market_structure'] == -1:
                    bearish_score += 2
                
                # Order Blocks
                if df.iloc[i]['bullish_ob']:
                    bullish_score += 3
                if df.iloc[i]['bearish_ob']:
                    bearish_score += 3
                
                # Fair Value Gaps
                if df.iloc[i]['bullish_fvg']:
                    bullish_score += 2
                if df.iloc[i]['bearish_fvg']:
                    bearish_score += 2
                
                # Break of Structure
                if df.iloc[i]['bos_bullish']:
                    bullish_score += 3
                if df.iloc[i]['bos_bearish']:
                    bearish_score += 3
                
                # Liquidity Grabs (contrarian signals)
                if df.iloc[i]['liquidity_grab_high']:
                    bearish_score += 2  # High grab suggests bearish reversal
                if df.iloc[i]['liquidity_grab_low']:
                    bullish_score += 2  # Low grab suggests bullish reversal
                
                df.at[df.index[i], 'smc_bullish_score'] = bullish_score
                df.at[df.index[i], 'smc_bearish_score'] = bearish_score
                df.at[df.index[i], 'smc_confluence'] = bullish_score - bearish_score
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating SMC confluence: {str(e)}")
            return df
    
    def _generate_smc_signal(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Generate trading signal based on SMC analysis
        """
        try:
            latest = df.iloc[-1]
            recent = df.iloc[-10:]  # Last 10 candles for trend analysis
            
            current_price = float(latest['close'])
            
            # SMC Signal Logic
            bullish_score = int(latest['smc_bullish_score'])
            bearish_score = int(latest['smc_bearish_score'])
            confluence = int(latest['smc_confluence'])
            
            # Additional confirmations
            recent_bullish_avg = recent['smc_bullish_score'].mean()
            recent_bearish_avg = recent['smc_bearish_score'].mean()
            
            # Signal determination
            if confluence >= 3 and bullish_score >= bearish_score:
                signal_direction = "BUY"
                confidence = min(95, 60 + (bullish_score * 5) + (recent_bullish_avg * 2))
            elif confluence <= -3 and bearish_score >= bullish_score:
                signal_direction = "SELL"
                confidence = min(95, 60 + (bearish_score * 5) + (recent_bearish_avg * 2))
            elif abs(confluence) <= 1:
                signal_direction = "HOLD"
                confidence = 50.0
            else:
                signal_direction = "HOLD"
                confidence = 50.0 - abs(confluence)
            
            # Risk Management using ATR if available
            atr_value = current_price * 0.01  # Default 1% if no ATR
            
            if TALIB_AVAILABLE:
                try:
                    high_data = df['high'].values.astype(np.float64)
                    low_data = df['low'].values.astype(np.float64)
                    close_data = df['close'].values.astype(np.float64)
                    atr_array = talib.ATR(high_data, low_data, close_data, timeperiod=14)
                    if not np.isnan(atr_array[-1]):
                        atr_value = float(atr_array[-1])
                except:
                    pass  # Use default ATR
            
            # Stop Loss and Take Profit
            if signal_direction == "BUY":
                stop_loss = current_price - (atr_value * 2.0)
                take_profit = current_price + (atr_value * 3.0)
            elif signal_direction == "SELL":
                stop_loss = current_price + (atr_value * 2.0)
                take_profit = current_price - (atr_value * 3.0)
            else:
                stop_loss = current_price
                take_profit = current_price
            
            # Risk-Reward Ratio
            if signal_direction != "HOLD":
                risk = abs(current_price - stop_loss)
                reward = abs(take_profit - current_price)
                rr_ratio = reward / risk if risk > 0 else 1.0
            else:
                rr_ratio = 1.0
            
            # SMC Analysis Summary
            smc_analysis = {
                "market_structure": int(latest['market_structure']),
                "bullish_order_blocks": bool(latest['bullish_ob']),
                "bearish_order_blocks": bool(latest['bearish_ob']),
                "bullish_fvg": bool(latest['bullish_fvg']),
                "bearish_fvg": bool(latest['bearish_fvg']),
                "bos_bullish": bool(latest['bos_bullish']),
                "bos_bearish": bool(latest['bos_bearish']),
                "liquidity_grab_high": bool(latest['liquidity_grab_high']),
                "liquidity_grab_low": bool(latest['liquidity_grab_low']),
                "bullish_score": bullish_score,
                "bearish_score": bearish_score,
                "confluence": confluence,
                "recent_trend": "bullish" if recent_bullish_avg > recent_bearish_avg else "bearish" if recent_bearish_avg > recent_bullish_avg else "neutral"
            }
            
            return {
                "signal": signal_direction,
                "confidence": round(confidence, 2),
                "entry_price": round(current_price, 5),
                "stop_loss": round(stop_loss, 5),
                "take_profit": round(take_profit, 5),
                "risk_reward_ratio": round(rr_ratio, 2),
                "smc_analysis": smc_analysis,
                "atr_value": round(atr_value, 5)
            }
            
        except Exception as e:
            logger.error(f"Error generating SMC signal: {str(e)}")
            return self._create_empty_signal(f"SMC signal generation error: {str(e)}")
    
    async def _get_market_data(self, symbol: str, timeframe: str, periods: int = 200) -> Optional[pd.DataFrame]:
        """
        Get market data for SMC analysis
        Enhanced mock data with realistic price patterns
        """
        try:
            logger.info(f"Getting market data for SMC analysis: {symbol} {timeframe} ({periods} periods)")
            
            # Generate realistic OHLCV data with SMC patterns
            np.random.seed(hash(symbol) % 1000)
            
            # Base price
            if 'USD' in symbol:
                base_price = 1.1000 if 'EUR' in symbol else 1.3000
            else:
                base_price = 50000 if 'BTC' in symbol else 100.0
            
            # Generate price series with trending behavior
            trend_strength = np.random.uniform(-0.0002, 0.0002)
            volatility = np.random.uniform(0.0008, 0.0015)
            
            # Create realistic price movements with trends and reversals
            price_changes = []
            for i in range(periods):
                # Add trend component
                trend_component = trend_strength
                
                # Add mean reversion
                if i > 20:
                    recent_change = np.mean(price_changes[-10:])
                    mean_reversion = -recent_change * 0.1
                else:
                    mean_reversion = 0
                
                # Random component
                random_component = np.random.normal(0, volatility)
                
                # Combine components
                total_change = trend_component + mean_reversion + random_component
                price_changes.append(total_change)
            
            # Calculate price series
            price_series = base_price * np.exp(np.cumsum(price_changes))
            
            # Create OHLCV data
            dates = pd.date_range(end=datetime.utcnow(), periods=periods, freq='1H')
            
            df = pd.DataFrame(index=dates)
            df['close'] = price_series
            df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
            
            # Generate realistic high/low with wicks
            candle_ranges = np.random.uniform(0.0003, 0.001, periods) * price_series
            wick_sizes = np.random.uniform(0.0001, 0.0005, periods) * price_series
            
            for i in range(len(df)):
                open_price = df.iloc[i]['open']
                close_price = df.iloc[i]['close']
                
                body_high = max(open_price, close_price)
                body_low = min(open_price, close_price)
                
                # Add wicks
                df.at[df.index[i], 'high'] = body_high + wick_sizes[i]
                df.at[df.index[i], 'low'] = body_low - wick_sizes[i]
            
            # Generate volume
            df['volume'] = np.random.randint(1000, 10000, periods)
            
            # Ensure data integrity
            df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
            df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
            
            logger.info(f"✅ Generated {len(df)} data points for SMC analysis")
            return df
            
        except Exception as e:
            logger.error(f"Error generating market data for SMC: {str(e)}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Enhanced performance statistics for SMC strategy
        """
        base_stats = super().get_performance_stats()
        
        base_stats.update({
            "strategy_type": "Smart Money Concepts",
            "smc_features": {
                "market_structure": "Higher Highs/Lower Lows analysis",
                "order_blocks": "Institutional accumulation zones",
                "fair_value_gaps": "Price inefficiency detection",
                "break_of_structure": "Trend change confirmation",
                "liquidity_analysis": "Stop hunt detection",
                "confluence_scoring": "Multi-factor signal strength"
            },
            "parameters": {
                "swing_period": self.swing_period,
                "lookback_periods": self.lookback_periods,
                "order_block_threshold": f"{self.ob_threshold*100:.2f}%",
                "fvg_threshold": f"{self.fvg_threshold*100:.2f}%",
                "liquidity_threshold": f"{self.liquidity_threshold*100:.2f}%"
            },
            "institutional_focus": {
                "smart_money_tracking": True,
                "retail_trap_detection": True,
                "liquidity_sweep_analysis": True,
                "market_maker_behavior": True
            }
        })
        
        return base_stats

# Test function for standalone usage
if __name__ == "__main__":
    import asyncio
    
    async def test_smc_strategy():
        print("🧪 Testing Smart Money Concepts Strategy...")
        
        config = {
            'swing_period': 10,
            'lookback_periods': 100,
            'order_block_threshold': 0.002,
            'fvg_threshold': 0.001,
            'liquidity_threshold': 0.0015
        }
        
        strategy = SmartMoneyStrategy(config)
        
        # Test analysis
        result = await strategy.analyze("EURUSD", "H1")
        
        print(f"📊 Signal: {result['signal']}")
        print(f"📈 Confidence: {result['confidence']}%")
        print(f"💰 Entry: {result['entry_price']}")
        print(f"🛑 Stop Loss: {result['stop_loss']}")
        print(f"🎯 Take Profit: {result['take_profit']}")
        print(f"⚖️ Risk/Reward: {result['risk_reward_ratio']}")
        
        # SMC Analysis details
        smc = result.get('smc_analysis', {})
        print(f"\n🧠 SMC Analysis:")
        print(f"   Market Structure: {smc.get('market_structure', 'N/A')}")
        print(f"   Bullish Score: {smc.get('bullish_score', 0)}")
        print(f"   Bearish Score: {smc.get('bearish_score', 0)}")
        print(f"   Confluence: {smc.get('confluence', 0)}")
        print(f"   Order Blocks: Bull={smc.get('bullish_order_blocks', False)}, Bear={smc.get('bearish_order_blocks', False)}")
        print(f"   Fair Value Gaps: Bull={smc.get('bullish_fvg', False)}, Bear={smc.get('bearish_fvg', False)}")
        print(f"   Break of Structure: Bull={smc.get('bos_bullish', False)}, Bear={smc.get('bos_bearish', False)}")
        print(f"   Liquidity Grabs: High={smc.get('liquidity_grab_high', False)}, Low={smc.get('liquidity_grab_low', False)}")
        
        # Test performance stats
        stats = strategy.get_performance_stats()
        print(f"\n📈 Strategy: {stats['strategy_info']['name']}")
        print(f"🔢 Analysis Count: {stats['strategy_info']['analysis_count']}")
        print(f"✅ SMC Test completed successfully!")
    
    # Run test
    asyncio.run(test_smc_strategy())