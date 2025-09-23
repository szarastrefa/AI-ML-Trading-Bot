# -*- coding: utf-8 -*-
"""
Smart Money Concepts (SMC) Strategy Implementation

Implements institutional trading concepts:
- Order Blocks (Supply/Demand zones)
- Fair Value Gaps (FVG)
- Break of Structure (BOS)
- Change of Character (CHoCH)
- Liquidity Sweeps
- Market Structure Analysis
- Institutional Order Flow
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class StructureType(Enum):
    HIGHER_HIGH = "HH"
    HIGHER_LOW = "HL"
    LOWER_HIGH = "LH"
    LOWER_LOW = "LL"
    EQUAL_HIGH = "EQH"
    EQUAL_LOW = "EQL"

class OrderBlockType(Enum):
    BULLISH = "Bullish OB"
    BEARISH = "Bearish OB"
    BREAKER = "Breaker Block"

class TrendDirection(Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    RANGING = "Ranging"

@dataclass
class OrderBlock:
    """Order Block structure"""
    ob_type: OrderBlockType
    high: float
    low: float
    time_created: datetime
    mitigation_count: int = 0
    is_active: bool = True
    origin_candle_index: int = 0
    
@dataclass
class FairValueGap:
    """Fair Value Gap structure"""
    gap_type: str  # "Bullish" or "Bearish"
    high: float
    low: float
    time_created: datetime
    is_filled: bool = False
    fill_percentage: float = 0.0
    
@dataclass
class LiquiditySweep:
    """Liquidity Sweep detection"""
    sweep_type: str  # "Buy Side" or "Sell Side"
    swept_level: float
    sweep_time: datetime
    follow_through: bool = False
    
class SmartMoneyStrategy:
    """
    Smart Money Concepts (SMC) Strategy
    
    Implements institutional trading methodology:
    - Market structure analysis
    - Order block identification
    - Fair value gap detection
    - Liquidity sweep recognition
    - Break of structure confirmation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = "Smart Money Concepts"
        self.version = "1.0.0"
        
        # SMC Parameters
        self.structure_lookback = self.config.get('structure_lookback', 10)
        self.order_block_threshold = self.config.get('ob_threshold', 0.002)  # 0.2%
        self.fvg_threshold = self.config.get('fvg_threshold', 0.001)  # 0.1%
        self.liquidity_threshold = self.config.get('liquidity_threshold', 0.0015)  # 0.15%
        
        # State tracking
        self.order_blocks: List[OrderBlock] = []
        self.fair_value_gaps: List[FairValueGap] = []
        self.liquidity_sweeps: List[LiquiditySweep] = []
        self.market_structure: List[Tuple[int, float, StructureType]] = []
        
        logger.info(f"✅ {self.name} initialized with institutional parameters")
    
    async def analyze(self, data: pd.DataFrame, symbol: str, timeframe: str = 'H1') -> Dict[str, Any]:
        """
        Complete Smart Money Concepts analysis
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            Complete SMC analysis with signals
        """
        try:
            # 1. Analyze market structure
            market_structure = self._analyze_market_structure(data)
            
            # 2. Identify order blocks
            order_blocks = self._identify_order_blocks(data)
            
            # 3. Detect fair value gaps
            fair_value_gaps = self._detect_fair_value_gaps(data)
            
            # 4. Find liquidity sweeps
            liquidity_sweeps = self._detect_liquidity_sweeps(data)
            
            # 5. Determine trend direction
            trend_direction = self._determine_trend_direction(market_structure)
            
            # 6. Generate trading signal
            signal = self._generate_smc_signal(
                data, market_structure, order_blocks, 
                fair_value_gaps, liquidity_sweeps, trend_direction
            )
            
            return {
                'strategy': 'Smart Money Concepts',
                'signal': signal,
                'market_structure': market_structure,
                'order_blocks': order_blocks,
                'fair_value_gaps': fair_value_gaps,
                'liquidity_sweeps': liquidity_sweeps,
                'trend_direction': trend_direction.value,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Smart Money analysis error: {str(e)}")
            return self._default_smc_analysis(symbol)
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze market structure for Higher Highs, Higher Lows, etc.
        """
        structure_points = []
        highs = data['High'].values
        lows = data['Low'].values
        times = data.index
        
        # Find swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(self.structure_lookback, len(highs) - self.structure_lookback):
            # Swing High
            if highs[i] == max(highs[i-self.structure_lookback:i+self.structure_lookback+1]):
                swing_highs.append((i, highs[i], times[i]))
            
            # Swing Low
            if lows[i] == min(lows[i-self.structure_lookback:i+self.structure_lookback+1]):
                swing_lows.append((i, lows[i], times[i]))
        
        # Classify structure
        for i, (idx, price, time) in enumerate(swing_highs[1:], 1):
            prev_high = swing_highs[i-1][1]
            
            if price > prev_high * 1.001:  # 0.1% threshold
                structure_type = StructureType.HIGHER_HIGH
            elif price < prev_high * 0.999:
                structure_type = StructureType.LOWER_HIGH
            else:
                structure_type = StructureType.EQUAL_HIGH
            
            structure_points.append({
                'type': structure_type.value,
                'price': price,
                'time': time,
                'index': idx,
                'point_type': 'high'
            })
        
        for i, (idx, price, time) in enumerate(swing_lows[1:], 1):
            prev_low = swing_lows[i-1][1]
            
            if price > prev_low * 1.001:
                structure_type = StructureType.HIGHER_LOW
            elif price < prev_low * 0.999:
                structure_type = StructureType.LOWER_LOW
            else:
                structure_type = StructureType.EQUAL_LOW
            
            structure_points.append({
                'type': structure_type.value,
                'price': price,
                'time': time,
                'index': idx,
                'point_type': 'low'
            })
        
        return sorted(structure_points, key=lambda x: x['index'])
    
    def _identify_order_blocks(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify Order Blocks (institutional supply/demand zones)
        
        Order Block = Last candle before strong impulsive move
        """
        order_blocks = []
        
        for i in range(20, len(data) - 5):  # Look for patterns
            current_candle = data.iloc[i]
            
            # Look for strong bullish impulse after current candle
            next_5_candles = data.iloc[i+1:i+6]
            bullish_impulse = (
                next_5_candles['Close'].iloc[-1] > current_candle['High'] * (1 + self.order_block_threshold)
            )
            
            # Look for strong bearish impulse after current candle
            bearish_impulse = (
                next_5_candles['Close'].iloc[-1] < current_candle['Low'] * (1 - self.order_block_threshold)
            )
            
            if bullish_impulse:
                # Bearish order block (last supply before bullish move)
                order_blocks.append({
                    'type': OrderBlockType.BEARISH.value,
                    'high': current_candle['High'],
                    'low': current_candle['Low'],
                    'open': current_candle['Open'],
                    'close': current_candle['Close'],
                    'time': current_candle.name,
                    'index': i,
                    'is_active': True,
                    'mitigation_count': 0
                })
            
            elif bearish_impulse:
                # Bullish order block (last demand before bearish move)
                order_blocks.append({
                    'type': OrderBlockType.BULLISH.value,
                    'high': current_candle['High'],
                    'low': current_candle['Low'],
                    'open': current_candle['Open'],
                    'close': current_candle['Close'],
                    'time': current_candle.name,
                    'index': i,
                    'is_active': True,
                    'mitigation_count': 0
                })
        
        # Remove old/invalid order blocks
        current_price = data['Close'].iloc[-1]
        active_blocks = []
        
        for ob in order_blocks:
            # Check if order block is still valid (not fully mitigated)
            if ob['type'] == OrderBlockType.BULLISH.value:
                if current_price >= ob['low']:  # Price touched demand zone
                    ob['mitigation_count'] += 1
                    if current_price <= ob['high']:  # Still in zone
                        ob['is_active'] = True
                        active_blocks.append(ob)
            
            elif ob['type'] == OrderBlockType.BEARISH.value:
                if current_price <= ob['high']:  # Price touched supply zone
                    ob['mitigation_count'] += 1
                    if current_price >= ob['low']:  # Still in zone
                        ob['is_active'] = True
                        active_blocks.append(ob)
        
        return active_blocks[-10:]  # Keep last 10 active order blocks
    
    def _detect_fair_value_gaps(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect Fair Value Gaps (price imbalances)
        
        FVG = Gap between candle wicks that hasn't been filled
        """
        fvgs = []
        
        for i in range(2, len(data)):
            candle1 = data.iloc[i-2]  # First candle
            candle2 = data.iloc[i-1]  # Middle candle (creates gap)
            candle3 = data.iloc[i]    # Third candle
            
            # Bullish FVG: gap between candle1 high and candle3 low
            if candle1['High'] < candle3['Low']:
                gap_size = candle3['Low'] - candle1['High']
                if gap_size > candle2['Close'] * self.fvg_threshold:
                    fvgs.append({
                        'type': 'Bullish FVG',
                        'high': candle3['Low'],
                        'low': candle1['High'],
                        'gap_size': gap_size,
                        'time_created': candle2.name,
                        'index': i-1,
                        'is_filled': False,
                        'fill_percentage': 0.0
                    })
            
            # Bearish FVG: gap between candle1 low and candle3 high
            elif candle1['Low'] > candle3['High']:
                gap_size = candle1['Low'] - candle3['High']
                if gap_size > candle2['Close'] * self.fvg_threshold:
                    fvgs.append({
                        'type': 'Bearish FVG',
                        'high': candle1['Low'],
                        'low': candle3['High'],
                        'gap_size': gap_size,
                        'time_created': candle2.name,
                        'index': i-1,
                        'is_filled': False,
                        'fill_percentage': 0.0
                    })
        
        # Check for filled FVGs
        current_price = data['Close'].iloc[-1]
        
        for fvg in fvgs:
            if not fvg['is_filled']:
                if fvg['low'] <= current_price <= fvg['high']:
                    # Calculate fill percentage
                    gap_range = fvg['high'] - fvg['low']
                    if fvg['type'] == 'Bullish FVG':
                        fill_amount = current_price - fvg['low']
                    else:
                        fill_amount = fvg['high'] - current_price
                    
                    fvg['fill_percentage'] = min(100, (fill_amount / gap_range) * 100)
                    
                    if fvg['fill_percentage'] >= 50:  # 50% fill threshold
                        fvg['is_filled'] = True
        
        return [fvg for fvg in fvgs if not fvg['is_filled']][-20:]  # Keep last 20 unfilled FVGs
    
    def _detect_liquidity_sweeps(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect liquidity sweeps (stop hunts)
        
        Liquidity Sweep = Price breaks previous high/low then reverses
        """
        sweeps = []
        
        # Find recent swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(10, len(data) - 10):
            if data['High'].iloc[i] == data['High'].iloc[i-10:i+11].max():
                swing_highs.append((i, data['High'].iloc[i]))
            
            if data['Low'].iloc[i] == data['Low'].iloc[i-10:i+11].min():
                swing_lows.append((i, data['Low'].iloc[i]))
        
        # Check for liquidity sweeps
        for i in range(len(data) - 20, len(data)):
            current_candle = data.iloc[i]
            
            # Buy-side liquidity sweep (sweep above previous high)
            for swing_idx, swing_high in swing_highs:
                if swing_idx < i - 5:  # Must be at least 5 candles ago
                    sweep_threshold = swing_high * (1 + self.liquidity_threshold)
                    
                    if current_candle['High'] > sweep_threshold:
                        # Check for reversal after sweep
                        next_candles = data.iloc[i+1:i+4] if i+4 < len(data) else data.iloc[i+1:]
                        if len(next_candles) > 0 and next_candles['Close'].iloc[-1] < current_candle['Close']:
                            sweeps.append({
                                'type': 'Buy Side Liquidity Sweep',
                                'swept_level': swing_high,
                                'sweep_high': current_candle['High'],
                                'sweep_time': current_candle.name,
                                'follow_through': True,
                                'index': i
                            })
            
            # Sell-side liquidity sweep (sweep below previous low)
            for swing_idx, swing_low in swing_lows:
                if swing_idx < i - 5:
                    sweep_threshold = swing_low * (1 - self.liquidity_threshold)
                    
                    if current_candle['Low'] < sweep_threshold:
                        # Check for reversal after sweep
                        next_candles = data.iloc[i+1:i+4] if i+4 < len(data) else data.iloc[i+1:]
                        if len(next_candles) > 0 and next_candles['Close'].iloc[-1] > current_candle['Close']:
                            sweeps.append({
                                'type': 'Sell Side Liquidity Sweep',
                                'swept_level': swing_low,
                                'sweep_low': current_candle['Low'],
                                'sweep_time': current_candle.name,
                                'follow_through': True,
                                'index': i
                            })
        
        return sweeps[-10:]  # Keep last 10 sweeps
    
    def _determine_trend_direction(self, market_structure: List[Dict]) -> TrendDirection:
        """
        Determine overall trend direction from market structure
        """
        if not market_structure:
            return TrendDirection.RANGING
        
        recent_structure = market_structure[-6:]  # Last 6 structure points
        
        hh_count = sum(1 for s in recent_structure if s['type'] == 'HH')
        hl_count = sum(1 for s in recent_structure if s['type'] == 'HL')
        lh_count = sum(1 for s in recent_structure if s['type'] == 'LH')
        ll_count = sum(1 for s in recent_structure if s['type'] == 'LL')
        
        bullish_signals = hh_count + hl_count
        bearish_signals = lh_count + ll_count
        
        if bullish_signals > bearish_signals * 1.5:
            return TrendDirection.BULLISH
        elif bearish_signals > bullish_signals * 1.5:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.RANGING
    
    def _generate_smc_signal(self, data: pd.DataFrame, market_structure: List[Dict],
                            order_blocks: List[Dict], fair_value_gaps: List[Dict],
                            liquidity_sweeps: List[Dict], trend_direction: TrendDirection) -> Dict[str, Any]:
        """
        Generate trading signal based on Smart Money Concepts
        """
        current_price = data['Close'].iloc[-1]
        current_time = data.index[-1]
        
        signal = {
            'type': 'HOLD',
            'confidence': 0.0,
            'entry_price': current_price,
            'stop_loss': current_price,
            'take_profit': current_price,
            'reasoning': [],
            'smc_confluence': []
        }
        
        confidence_factors = []
        
        # 1. Order Block signals
        for ob in order_blocks:
            if ob['is_active']:
                if (ob['low'] <= current_price <= ob['high']):
                    if ob['type'] == OrderBlockType.BULLISH.value and trend_direction == TrendDirection.BULLISH:
                        signal['type'] = 'BUY'
                        signal['stop_loss'] = ob['low'] * 0.999
                        signal['take_profit'] = current_price * 1.02
                        confidence_factors.append(0.3)
                        signal['reasoning'].append(f"Bullish Order Block interaction at {ob['low']:.5f}")
                        signal['smc_confluence'].append('Bullish OB')
                    
                    elif ob['type'] == OrderBlockType.BEARISH.value and trend_direction == TrendDirection.BEARISH:
                        signal['type'] = 'SELL'
                        signal['stop_loss'] = ob['high'] * 1.001
                        signal['take_profit'] = current_price * 0.98
                        confidence_factors.append(0.3)
                        signal['reasoning'].append(f"Bearish Order Block interaction at {ob['high']:.5f}")
                        signal['smc_confluence'].append('Bearish OB')
        
        # 2. Fair Value Gap signals
        for fvg in fair_value_gaps:
            if fvg['low'] <= current_price <= fvg['high']:
                if fvg['type'] == 'Bullish FVG' and trend_direction != TrendDirection.BEARISH:
                    confidence_factors.append(0.2)
                    signal['reasoning'].append(f"Price in Bullish FVG zone")
                    signal['smc_confluence'].append('Bullish FVG')
                
                elif fvg['type'] == 'Bearish FVG' and trend_direction != TrendDirection.BULLISH:
                    confidence_factors.append(0.2)
                    signal['reasoning'].append(f"Price in Bearish FVG zone")
                    signal['smc_confluence'].append('Bearish FVG')
        
        # 3. Liquidity Sweep signals
        recent_sweeps = [s for s in liquidity_sweeps if s['index'] >= len(data) - 10]
        for sweep in recent_sweeps:
            if sweep['follow_through']:
                if sweep['type'] == 'Buy Side Liquidity Sweep':
                    if trend_direction == TrendDirection.BEARISH:
                        confidence_factors.append(0.25)
                        signal['reasoning'].append("Buy-side liquidity swept - bearish continuation")
                        signal['smc_confluence'].append('BSL Sweep')
                
                elif sweep['type'] == 'Sell Side Liquidity Sweep':
                    if trend_direction == TrendDirection.BULLISH:
                        confidence_factors.append(0.25)
                        signal['reasoning'].append("Sell-side liquidity swept - bullish continuation")
                        signal['smc_confluence'].append('SSL Sweep')
        
        # 4. Market Structure confirmation
        if market_structure:
            recent_structure = market_structure[-2:]
            if len(recent_structure) >= 2:
                if (recent_structure[-1]['type'] == 'HH' and recent_structure[-2]['type'] == 'HL'):
                    confidence_factors.append(0.15)
                    signal['reasoning'].append("Bullish market structure (HH + HL)")
                    signal['smc_confluence'].append('Bullish MS')
                
                elif (recent_structure[-1]['type'] == 'LL' and recent_structure[-2]['type'] == 'LH'):
                    confidence_factors.append(0.15)
                    signal['reasoning'].append("Bearish market structure (LL + LH)")
                    signal['smc_confluence'].append('Bearish MS')
        
        # Calculate final confidence
        signal['confidence'] = min(sum(confidence_factors), 0.95)
        
        # Adjust for trend alignment
        if signal['type'] != 'HOLD':
            if ((signal['type'] == 'BUY' and trend_direction == TrendDirection.BULLISH) or
                (signal['type'] == 'SELL' and trend_direction == TrendDirection.BEARISH)):
                signal['confidence'] *= 1.2  # 20% boost for trend alignment
            elif ((signal['type'] == 'BUY' and trend_direction == TrendDirection.BEARISH) or
                  (signal['type'] == 'SELL' and trend_direction == TrendDirection.BULLISH)):
                signal['confidence'] *= 0.7  # 30% reduction for counter-trend
        
        signal['confidence'] = min(signal['confidence'], 0.95)
        
        return signal
    
    def _default_smc_analysis(self, symbol: str) -> Dict[str, Any]:
        """Default analysis when SMC analysis fails"""
        return {
            'strategy': 'Smart Money Concepts',
            'signal': {
                'type': 'HOLD',
                'confidence': 0.0,
                'entry_price': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'reasoning': ['Analysis failed'],
                'smc_confluence': []
            },
            'market_structure': [],
            'order_blocks': [],
            'fair_value_gaps': [],
            'liquidity_sweeps': [],
            'trend_direction': TrendDirection.RANGING.value,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_institutional_levels(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Get key institutional levels (order blocks, FVGs, liquidity zones)
        """
        try:
            order_blocks = self._identify_order_blocks(data)
            fair_value_gaps = self._detect_fair_value_gaps(data)
            
            return {
                'bullish_order_blocks': [ob['low'] for ob in order_blocks if ob['type'] == 'Bullish OB'],
                'bearish_order_blocks': [ob['high'] for ob in order_blocks if ob['type'] == 'Bearish OB'],
                'bullish_fvg_zones': [(fvg['low'], fvg['high']) for fvg in fair_value_gaps if fvg['type'] == 'Bullish FVG'],
                'bearish_fvg_zones': [(fvg['low'], fvg['high']) for fvg in fair_value_gaps if fvg['type'] == 'Bearish FVG'],
                'liquidity_levels': self._get_liquidity_levels(data)
            }
        
        except Exception as e:
            logger.error(f"Institutional levels error: {str(e)}")
            return {
                'bullish_order_blocks': [],
                'bearish_order_blocks': [],
                'bullish_fvg_zones': [],
                'bearish_fvg_zones': [],
                'liquidity_levels': []
            }
    
    def _get_liquidity_levels(self, data: pd.DataFrame) -> List[float]:
        """Get liquidity pool levels (previous highs/lows)"""
        liquidity_levels = []
        
        # Find swing highs and lows
        for i in range(20, len(data) - 5):
            if data['High'].iloc[i] == data['High'].iloc[i-20:i+6].max():
                liquidity_levels.append(data['High'].iloc[i])
            
            if data['Low'].iloc[i] == data['Low'].iloc[i-20:i+6].min():
                liquidity_levels.append(data['Low'].iloc[i])
        
        return sorted(set(liquidity_levels), reverse=True)[:10]  # Top 10 levels

# Export strategy instance
smart_money_strategy = SmartMoneyStrategy()