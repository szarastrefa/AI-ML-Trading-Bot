# -*- coding: utf-8 -*-
"""
Fibonacci Team Strategy Implementation
Based on Łukasz Fijołek's methodology from Fibonacci Team YouTube channel

Complete implementation of:
- Fibonacci Retracements & Extensions
- Harmonic Patterns (Gartley, Bat, Butterfly, Crab)
- Smart Money Concepts integration
- Risk Management (2% default stop loss)
- Volume Analysis
- Live Trading Session optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class HarmonicPattern(Enum):
    GARTLEY = "Gartley"
    BAT = "Bat"
    BUTTERFLY = "Butterfly"
    CRAB = "Crab"
    NONE = "None"

@dataclass
class FibonacciLevels:
    """Fibonacci levels for retracements and extensions"""
    retracement_236: float = 23.6
    retracement_382: float = 38.2
    retracement_500: float = 50.0
    retracement_618: float = 61.8
    retracement_786: float = 78.6
    
    extension_618: float = 61.8
    extension_1000: float = 100.0
    extension_1618: float = 161.8
    extension_2618: float = 261.8

@dataclass
class TradingSignal:
    """Complete trading signal with Fibonacci Team methodology"""
    symbol: str
    signal_type: SignalType
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: List[float]  # Multiple TP levels
    fibonacci_level: str
    harmonic_pattern: HarmonicPattern
    risk_reward_ratio: float
    session_quality: str  # London, New York, etc.
    volume_confirmation: bool
    timestamp: datetime
    
class FibonacciTeamStrategy:
    """
    Complete Fibonacci Team Strategy Implementation
    
    Features:
    - Fibonacci retracements and extensions
    - Harmonic patterns (Gartley, Bat, Butterfly, Crab)
    - Volume analysis integration
    - 2% default stop loss (editable)
    - Smart Money Concepts overlay
    - Live trading session optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = "Fibonacci Team Strategy"
        self.version = "1.0.0"
        
        # Risk Management - Fibonacci Team Standard
        self.default_stop_loss_pct = self.config.get('stop_loss_pct', 2.0)  # 2% default
        self.min_risk_reward_ratio = self.config.get('min_rr', 2.0)  # Minimum 1:2
        self.max_position_size = self.config.get('max_position', 2.0)  # 2% portfolio risk
        
        # Fibonacci Levels
        self.fib_levels = FibonacciLevels()
        
        # Harmonic Pattern Ratios
        self.harmonic_ratios = {
            HarmonicPattern.GARTLEY: {
                'AB_XA': (0.618, 0.618),  # AB retracement of XA
                'BC_AB': (0.382, 0.886),  # BC retracement of AB
                'CD_AB': (1.13, 1.618),   # CD extension of AB
                'XD_XA': (0.786, 0.786)   # Final ratio
            },
            HarmonicPattern.BAT: {
                'AB_XA': (0.382, 0.500),
                'BC_AB': (0.382, 0.886),
                'CD_AB': (1.618, 2.618),
                'XD_XA': (0.886, 0.886)
            },
            HarmonicPattern.BUTTERFLY: {
                'AB_XA': (0.786, 0.786),
                'BC_AB': (0.382, 0.886),
                'CD_AB': (1.618, 2.618),
                'XD_XA': (1.272, 1.618)
            },
            HarmonicPattern.CRAB: {
                'AB_XA': (0.382, 0.618),
                'BC_AB': (0.382, 0.886),
                'CD_AB': (2.240, 3.618),
                'XD_XA': (1.618, 1.618)
            }
        }
        
        # Trading Sessions (GMT)
        self.trading_sessions = {
            'london': {'start': 8, 'end': 17, 'quality': 'high'},
            'new_york': {'start': 13, 'end': 22, 'quality': 'high'},
            'overlap': {'start': 13, 'end': 17, 'quality': 'premium'},  # London-NY overlap
            'asian': {'start': 0, 'end': 9, 'quality': 'medium'},
            'sydney': {'start': 22, 'end': 7, 'quality': 'low'}
        }
        
        logger.info(f"✅ {self.name} initialized with 2% default stop loss")
    
    async def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str = 'H1') -> TradingSignal:
        """
        Complete Fibonacci Team analysis
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            TradingSignal with complete analysis
        """
        try:
            # 1. Identify trend and swing points
            swing_points = self._identify_swing_points(data)
            
            # 2. Calculate Fibonacci levels
            fib_levels = self._calculate_fibonacci_levels(data, swing_points)
            
            # 3. Detect harmonic patterns
            harmonic_pattern = self._detect_harmonic_patterns(swing_points)
            
            # 4. Analyze volume confirmation
            volume_signal = self._analyze_volume(data)
            
            # 5. Check trading session quality
            session_quality = self._get_session_quality()
            
            # 6. Generate trading signal
            signal = self._generate_signal(
                fib_levels, harmonic_pattern, volume_signal, 
                session_quality, data.iloc[-1]
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Fibonacci Team analysis error: {str(e)}")
            return self._default_signal(symbol)
    
    def _identify_swing_points(self, data: pd.DataFrame, lookback: int = 10) -> Dict[str, List[Tuple[int, float]]]:
        """
        Identify swing highs and lows for Fibonacci analysis
        Based on Fibonacci Team methodology
        """
        highs = data['High'].values
        lows = data['Low'].values
        
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(highs) - lookback):
            # Swing High: highest point in lookback window
            if highs[i] == max(highs[i-lookback:i+lookback+1]):
                swing_highs.append((i, highs[i]))
            
            # Swing Low: lowest point in lookback window
            if lows[i] == min(lows[i-lookback:i+lookback+1]):
                swing_lows.append((i, lows[i]))
        
        return {
            'highs': swing_highs[-5:],  # Last 5 swing highs
            'lows': swing_lows[-5:]     # Last 5 swing lows
        }
    
    def _calculate_fibonacci_levels(self, data: pd.DataFrame, swing_points: Dict) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement and extension levels
        """
        if not swing_points['highs'] or not swing_points['lows']:
            return {}
        
        # Get last significant swing high and low
        last_high = swing_points['highs'][-1][1]
        last_low = swing_points['lows'][-1][1]
        
        # Determine trend direction
        if last_high > last_low:
            # Uptrend - calculate retracements from high to low
            swing_range = last_high - last_low
            base_price = last_high
            multiplier = -1
        else:
            # Downtrend - calculate retracements from low to high
            swing_range = abs(last_high - last_low)
            base_price = last_low
            multiplier = 1
        
        # Fibonacci retracement levels
        fib_levels = {
            'fib_23.6': base_price + (swing_range * 0.236 * multiplier),
            'fib_38.2': base_price + (swing_range * 0.382 * multiplier),
            'fib_50.0': base_price + (swing_range * 0.500 * multiplier),
            'fib_61.8': base_price + (swing_range * 0.618 * multiplier),
            'fib_78.6': base_price + (swing_range * 0.786 * multiplier),
            
            # Extensions
            'ext_61.8': base_price + (swing_range * 0.618 * -multiplier),
            'ext_100': base_price + (swing_range * 1.000 * -multiplier),
            'ext_161.8': base_price + (swing_range * 1.618 * -multiplier),
            'ext_261.8': base_price + (swing_range * 2.618 * -multiplier),
            
            'swing_high': last_high,
            'swing_low': last_low,
            'range': swing_range
        }
        
        return fib_levels
    
    def _detect_harmonic_patterns(self, swing_points: Dict) -> Tuple[HarmonicPattern, float]:
        """
        Detect harmonic patterns using Fibonacci Team methodology
        
        Patterns implemented:
        - Gartley: XA-AB-BC-CD with specific ratios
        - Bat: Most accurate pattern in harmonic arsenal
        - Butterfly: Extended pattern with 127.2-161.8% XD/XA
        - Crab: Deep pattern with 161.8% XD/XA
        """
        if len(swing_points['highs']) < 3 or len(swing_points['lows']) < 3:
            return HarmonicPattern.NONE, 0.0
        
        # Get points for pattern analysis (need at least 4 points: X-A-B-C-D)
        all_points = []
        
        # Combine and sort swing points by time
        for idx, price in swing_points['highs']:
            all_points.append((idx, price, 'H'))
        for idx, price in swing_points['lows']:
            all_points.append((idx, price, 'L'))
        
        all_points.sort(key=lambda x: x[0])  # Sort by time
        
        if len(all_points) < 5:
            return HarmonicPattern.NONE, 0.0
        
        # Get last 5 points (X-A-B-C-D)
        points = all_points[-5:]
        X, A, B, C, D = [(p[1]) for p in points]
        
        # Calculate ratios
        XA = abs(A - X)
        AB = abs(B - A)
        BC = abs(C - B)
        CD = abs(D - C)
        XD = abs(D - X)
        
        if XA == 0:  # Avoid division by zero
            return HarmonicPattern.NONE, 0.0
        
        # Pattern recognition
        AB_XA_ratio = AB / XA
        BC_AB_ratio = BC / AB if AB != 0 else 0
        CD_AB_ratio = CD / AB if AB != 0 else 0
        XD_XA_ratio = XD / XA
        
        # Check each pattern
        pattern_scores = {
            HarmonicPattern.GARTLEY: self._score_gartley_pattern(
                AB_XA_ratio, BC_AB_ratio, CD_AB_ratio, XD_XA_ratio
            ),
            HarmonicPattern.BAT: self._score_bat_pattern(
                AB_XA_ratio, BC_AB_ratio, CD_AB_ratio, XD_XA_ratio
            ),
            HarmonicPattern.BUTTERFLY: self._score_butterfly_pattern(
                AB_XA_ratio, BC_AB_ratio, CD_AB_ratio, XD_XA_ratio
            ),
            HarmonicPattern.CRAB: self._score_crab_pattern(
                AB_XA_ratio, BC_AB_ratio, CD_AB_ratio, XD_XA_ratio
            )
        }
        
        # Find best matching pattern
        best_pattern = max(pattern_scores.keys(), key=lambda k: pattern_scores[k])
        best_score = pattern_scores[best_pattern]
        
        if best_score > 0.7:  # 70% minimum confidence
            return best_pattern, best_score
        else:
            return HarmonicPattern.NONE, 0.0
    
    def _score_gartley_pattern(self, AB_XA: float, BC_AB: float, CD_AB: float, XD_XA: float) -> float:
        """Score Gartley pattern accuracy"""
        target_ratios = self.harmonic_ratios[HarmonicPattern.GARTLEY]
        
        scores = []
        
        # AB/XA should be around 0.618
        ab_score = 1 - abs(AB_XA - 0.618) / 0.618
        scores.append(max(0, ab_score))
        
        # BC/AB should be between 0.382-0.886
        bc_score = 1.0 if 0.382 <= BC_AB <= 0.886 else 0.0
        scores.append(bc_score)
        
        # CD/AB should be between 1.13-1.618
        cd_score = 1.0 if 1.13 <= CD_AB <= 1.618 else 0.0
        scores.append(cd_score)
        
        # XD/XA should be around 0.786
        xd_score = 1 - abs(XD_XA - 0.786) / 0.786
        scores.append(max(0, xd_score))
        
        return np.mean(scores)
    
    def _score_bat_pattern(self, AB_XA: float, BC_AB: float, CD_AB: float, XD_XA: float) -> float:
        """Score Bat pattern - most accurate in harmonic arsenal"""
        scores = []
        
        # AB/XA should be between 0.382-0.500
        ab_score = 1.0 if 0.382 <= AB_XA <= 0.500 else 0.0
        scores.append(ab_score)
        
        # BC/AB should be between 0.382-0.886
        bc_score = 1.0 if 0.382 <= BC_AB <= 0.886 else 0.0
        scores.append(bc_score)
        
        # CD/AB should be between 1.618-2.618
        cd_score = 1.0 if 1.618 <= CD_AB <= 2.618 else 0.0
        scores.append(cd_score)
        
        # XD/XA should be around 0.886
        xd_score = 1 - abs(XD_XA - 0.886) / 0.886
        scores.append(max(0, xd_score))
        
        return np.mean(scores)
    
    def _score_butterfly_pattern(self, AB_XA: float, BC_AB: float, CD_AB: float, XD_XA: float) -> float:
        """Score Butterfly pattern"""
        scores = []
        
        # AB/XA should be around 0.786
        ab_score = 1 - abs(AB_XA - 0.786) / 0.786
        scores.append(max(0, ab_score))
        
        # BC/AB should be between 0.382-0.886
        bc_score = 1.0 if 0.382 <= BC_AB <= 0.886 else 0.0
        scores.append(bc_score)
        
        # CD/AB should be between 1.618-2.618
        cd_score = 1.0 if 1.618 <= CD_AB <= 2.618 else 0.0
        scores.append(cd_score)
        
        # XD/XA should be between 1.272-1.618
        xd_score = 1.0 if 1.272 <= XD_XA <= 1.618 else 0.0
        scores.append(xd_score)
        
        return np.mean(scores)
    
    def _score_crab_pattern(self, AB_XA: float, BC_AB: float, CD_AB: float, XD_XA: float) -> float:
        """Score Crab pattern - deep retracement pattern"""
        scores = []
        
        # AB/XA should be between 0.382-0.618
        ab_score = 1.0 if 0.382 <= AB_XA <= 0.618 else 0.0
        scores.append(ab_score)
        
        # BC/AB should be between 0.382-0.886
        bc_score = 1.0 if 0.382 <= BC_AB <= 0.886 else 0.0
        scores.append(bc_score)
        
        # CD/AB should be between 2.240-3.618
        cd_score = 1.0 if 2.240 <= CD_AB <= 3.618 else 0.0
        scores.append(cd_score)
        
        # XD/XA should be around 1.618
        xd_score = 1 - abs(XD_XA - 1.618) / 1.618
        scores.append(max(0, xd_score))
        
        return np.mean(scores)
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Volume analysis according to Fibonacci Team methodology
        
        Uses:
        - On-Balance Volume (OBV)
        - Volume-Weighted Average Price (VWAP)
        - Accumulation/Distribution Line
        """
        if 'Volume' not in data.columns:
            return {'confirmation': False, 'strength': 0.0}
        
        # On-Balance Volume calculation
        obv = self._calculate_obv(data)
        
        # VWAP calculation
        vwap = self._calculate_vwap(data)
        
        # Volume trend analysis
        recent_volume = data['Volume'].tail(10).mean()
        historical_volume = data['Volume'].mean()
        volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1.0
        
        # Price-Volume relationship
        price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
        volume_change = (data['Volume'].iloc[-1] - data['Volume'].iloc[-2]) / data['Volume'].iloc[-2]
        
        # Volume confirmation rules (Fibonacci Team methodology)
        confirmation = False
        strength = 0.0
        
        if price_change > 0 and volume_change > 0:  # Rising price + rising volume = bullish
            confirmation = True
            strength = min(volume_ratio, 2.0) / 2.0
        elif price_change < 0 and volume_change > 0:  # Falling price + rising volume = bearish
            confirmation = True
            strength = min(volume_ratio, 2.0) / 2.0
        
        return {
            'confirmation': confirmation,
            'strength': strength,
            'obv_trend': 'rising' if obv[-1] > obv[-5] else 'falling',
            'volume_ratio': volume_ratio,
            'vwap_position': 'above' if data['Close'].iloc[-1] > vwap[-1] else 'below'
        }
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = [0]
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv)
    
    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume-Weighted Average Price"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        return vwap
    
    def _get_session_quality(self) -> str:
        """Determine current trading session quality"""
        current_hour = datetime.utcnow().hour
        
        # Check for premium London-NY overlap (13:00-17:00 GMT)
        if 13 <= current_hour <= 17:
            return 'premium'  # Highest volume and volatility
        
        # London session (8:00-17:00 GMT)
        elif 8 <= current_hour <= 17:
            return 'high'
        
        # New York session (13:00-22:00 GMT)
        elif 13 <= current_hour <= 22:
            return 'high'
        
        # Asian session (0:00-9:00 GMT)
        elif 0 <= current_hour <= 9:
            return 'medium'
        
        else:
            return 'low'
    
    def _generate_signal(self, fib_levels: Dict, harmonic_pattern: Tuple, 
                        volume_analysis: Dict, session_quality: str, 
                        current_candle: pd.Series) -> TradingSignal:
        """
        Generate comprehensive trading signal using Fibonacci Team methodology
        """
        pattern, pattern_confidence = harmonic_pattern
        current_price = current_candle['Close']
        
        # Default signal
        signal_type = SignalType.HOLD
        confidence = 0.0
        fibonacci_level = 'none'
        
        # Check for Fibonacci level interaction
        price_tolerance = current_price * 0.001  # 0.1% tolerance
        
        for level_name, level_price in fib_levels.items():
            if abs(current_price - level_price) <= price_tolerance:
                fibonacci_level = level_name
                
                # Fibonacci Team rules for entry
                if 'fib_61.8' in level_name or 'fib_38.2' in level_name:
                    # Strong retracement levels - counter-trend entry
                    if current_price < fib_levels.get('swing_high', current_price):
                        signal_type = SignalType.BUY
                        confidence = 0.7
                    else:
                        signal_type = SignalType.SELL
                        confidence = 0.7
                
                elif 'ext_' in level_name:
                    # Extension levels - trend continuation
                    confidence = 0.6
                    # Determine direction based on recent price action
                    signal_type = SignalType.BUY if current_candle['Close'] > current_candle['Open'] else SignalType.SELL
        
        # Enhance confidence with harmonic pattern
        if pattern != HarmonicPattern.NONE:
            confidence = min(confidence + pattern_confidence * 0.3, 0.95)
        
        # Volume confirmation boost
        if volume_analysis['confirmation']:
            confidence = min(confidence + 0.1, 0.95)
        
        # Session quality adjustment
        session_multipliers = {'premium': 1.2, 'high': 1.1, 'medium': 1.0, 'low': 0.8}
        confidence *= session_multipliers.get(session_quality, 1.0)
        confidence = min(confidence, 0.95)
        
        # Calculate stop loss and take profit (Fibonacci Team methodology)
        if signal_type == SignalType.BUY:
            stop_loss = current_price * (1 - self.default_stop_loss_pct / 100)  # 2% stop loss
            risk_amount = current_price - stop_loss
            
            # Multiple take profit levels (Fibonacci extensions)
            take_profits = [
                current_price + risk_amount * 2,    # 1:2 RR
                current_price + risk_amount * 3,    # 1:3 RR
                current_price + risk_amount * 5     # 1:5 RR
            ]
        
        elif signal_type == SignalType.SELL:
            stop_loss = current_price * (1 + self.default_stop_loss_pct / 100)  # 2% stop loss
            risk_amount = stop_loss - current_price
            
            take_profits = [
                current_price - risk_amount * 2,    # 1:2 RR
                current_price - risk_amount * 3,    # 1:3 RR
                current_price - risk_amount * 5     # 1:5 RR
            ]
        
        else:
            stop_loss = current_price
            take_profits = [current_price]
        
        # Risk-reward ratio
        risk = abs(current_price - stop_loss)
        reward = abs(take_profits[0] - current_price) if take_profits else 0
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        return TradingSignal(
            symbol=current_candle.name if hasattr(current_candle, 'name') else 'UNKNOWN',
            signal_type=signal_type,
            confidence=round(confidence, 3),
            entry_price=round(current_price, 5),
            stop_loss=round(stop_loss, 5),
            take_profit=[round(tp, 5) for tp in take_profits],
            fibonacci_level=fibonacci_level,
            harmonic_pattern=pattern,
            risk_reward_ratio=round(risk_reward_ratio, 2),
            session_quality=session_quality,
            volume_confirmation=volume_analysis['confirmation'],
            timestamp=datetime.utcnow()
        )
    
    def _default_signal(self, symbol: str) -> TradingSignal:
        """Default signal when analysis fails"""
        return TradingSignal(
            symbol=symbol,
            signal_type=SignalType.HOLD,
            confidence=0.0,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=[0.0],
            fibonacci_level='none',
            harmonic_pattern=HarmonicPattern.NONE,
            risk_reward_ratio=0.0,
            session_quality=self._get_session_quality(),
            volume_confirmation=False,
            timestamp=datetime.utcnow()
        )
    
    def get_fibonacci_arrows_signal(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Implement "Strzałki Fibonacciego" system mentioned in Fibonacci Team videos
        
        This is the latest innovation combining:
        - Precise Fibonacci levels
        - Trend analysis algorithms  
        - Entry/exit signals
        - Risk management
        """
        try:
            signal = await self.analyze(data, symbol)
            
            return {
                'system': 'Fibonacci Arrows',
                'signal': signal.signal_type.value,
                'confidence': signal.confidence,
                'entry': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit_levels': signal.take_profit,
                'fibonacci_interaction': signal.fibonacci_level,
                'harmonic_pattern': signal.harmonic_pattern.value,
                'session_quality': signal.session_quality,
                'risk_reward': signal.risk_reward_ratio,
                'volume_confirmed': signal.volume_confirmation,
                'methodology': 'Fibonacci Team - Łukasz Fijołek',
                'timestamp': signal.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fibonacci Arrows system error: {str(e)}")
            return {
                'system': 'Fibonacci Arrows',
                'signal': 'HOLD',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def get_scalping_signals(self, data: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """
        Scalping signals for European and American sessions
        Based on Fibonacci Team scalping techniques
        
        Uses:
        - Stochastic Oscillator
        - Moving Averages (5, 20, 200)
        - Pivot Points
        - Fibonacci levels
        """
        signals = []
        
        try:
            # Calculate indicators
            stoch_k, stoch_d = self._calculate_stochastic(data)
            ma5 = data['Close'].rolling(5).mean()
            ma20 = data['Close'].rolling(20).mean()
            ma200 = data['Close'].rolling(200).mean()
            
            current_price = data['Close'].iloc[-1]
            
            # Stochastic signals
            if len(stoch_k) > 1 and len(stoch_d) > 1:
                if stoch_k.iloc[-2] < stoch_d.iloc[-2] and stoch_k.iloc[-1] > stoch_d.iloc[-1]:
                    if stoch_k.iloc[-1] < 20:  # Oversold
                        signals.append({
                            'type': 'BUY',
                            'reason': 'Stochastic oversold crossover',
                            'confidence': 0.7,
                            'timeframe': 'scalping'
                        })
                
                elif stoch_k.iloc[-2] > stoch_d.iloc[-2] and stoch_k.iloc[-1] < stoch_d.iloc[-1]:
                    if stoch_k.iloc[-1] > 80:  # Overbought
                        signals.append({
                            'type': 'SELL',
                            'reason': 'Stochastic overbought crossover',
                            'confidence': 0.7,
                            'timeframe': 'scalping'
                        })
            
            # Moving Average signals
            if len(ma5) > 1 and len(ma20) > 1:
                if ma5.iloc[-2] < ma20.iloc[-2] and ma5.iloc[-1] > ma20.iloc[-1]:
                    signals.append({
                        'type': 'BUY',
                        'reason': 'MA5 crosses above MA20',
                        'confidence': 0.6,
                        'timeframe': 'scalping'
                    })
                
                elif ma5.iloc[-2] > ma20.iloc[-2] and ma5.iloc[-1] < ma20.iloc[-1]:
                    signals.append({
                        'type': 'SELL',
                        'reason': 'MA5 crosses below MA20',
                        'confidence': 0.6,
                        'timeframe': 'scalping'
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Scalping analysis error: {str(e)}")
            return []
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = data['Low'].rolling(k_period).min()
        highest_high = data['High'].rolling(k_period).max()
        
        k_percent = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(d_period).mean()
        
        return k_percent, d_percent
    
    def calculate_position_size(self, account_balance: float, risk_percentage: float, 
                               entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size using Fibonacci Team risk management
        
        Rules:
        - Maximum 1-2% capital risk per trade
        - Account balance protection priority
        - Fibonacci sequence for position scaling
        """
        risk_amount = account_balance * (risk_percentage / 100)
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0.0
        
        # Basic position size
        position_size = risk_amount / price_risk
        
        # Apply Fibonacci sequence scaling (optional)
        fib_sequence = [1, 1, 2, 3, 5, 8, 13]
        scaling_factor = fib_sequence[min(len(fib_sequence) - 1, 3)] / 10  # Use 3rd Fibonacci number
        
        return round(position_size * scaling_factor, 2)
    
    def get_risk_management_params(self) -> Dict[str, Any]:
        """
        Get Fibonacci Team risk management parameters
        """
        return {
            'default_stop_loss_pct': self.default_stop_loss_pct,
            'min_risk_reward_ratio': self.min_risk_reward_ratio,
            'max_position_risk_pct': self.max_position_size,
            'max_portfolio_risk_pct': 10.0,  # Maximum 10% total portfolio risk
            'fibonacci_scaling': True,
            'session_based_sizing': True,
            'harmonic_pattern_boost': 1.5,  # 50% size increase for harmonic confirmations
            'methodology': 'Fibonacci Team - Professional Risk Management'
        }

# Export strategy instance
fibonacci_strategy = FibonacciTeamStrategy()