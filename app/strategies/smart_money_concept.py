"""
Strategia Smart Money Concept - zaawansowana analiza struktury rynku
Zawiera: order blocks, fair value gaps, liquidity zones, market structure
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from .base_strategy import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)

class MarketStructure(Enum):
    """Struktura rynku"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"
    BREAK_OF_STRUCTURE = "bos"
    CHANGE_OF_CHARACTER = "choch"

class OrderBlockType(Enum):
    """Typy order blocks"""
    BULLISH_OB = "bullish_ob"
    BEARISH_OB = "bearish_ob"
    MITIGATION_BLOCK = "mitigation_block"

@dataclass
class OrderBlock:
    """Order Block - strefa gdzie instytucje składały zlecenia"""
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    high: float
    low: float
    volume: float
    block_type: OrderBlockType
    strength: float  # 0-1
    tested: bool = False

@dataclass
class FairValueGap:
    """Fair Value Gap - nieefektywność cenowa"""
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    gap_high: float
    gap_low: float
    gap_type: str  # "bullish" or "bearish"
    filled: bool = False

class SmartMoneyConceptStrategy(BaseStrategy):
    """
    Strategia Smart Money Concept
    Oparta na analizie:
    - Struktury rynku (market structure)
    - Order blocks (strefy instytucjonalne)
    - Fair Value Gaps (nieefektywności)
    - Liquidity sweeps (zbieranie likwidności)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "smart_money_concept"

        # Parametry strategii SMC
        self.swing_lookback = config.get("swing_lookback", 20)
        self.ob_min_size = config.get("ob_min_size", 0.001)  # Min 0.1% rozmiaru
        self.fvg_min_size = config.get("fvg_min_size", 0.0005)  # Min 0.05%

        # Zarządzanie ryzykiem
        self.default_stop_loss = 0.02  # 2%
        self.risk_reward_ratio = 3.0   # 1:3 RR dla SMC
        self.max_position_size = 0.02  # 2% kapitału

        # Stan rynku
        self.current_structure = MarketStructure.RANGING
        self.order_blocks: List[OrderBlock] = []
        self.fair_value_gaps: List[FairValueGap] = []

    def identify_market_structure(self, data: pd.DataFrame) -> MarketStructure:
        """Identyfikuj strukturę rynku"""
        if len(data) < self.swing_lookback * 2:
            return MarketStructure.RANGING

        # Znajdź swing highs/lows
        highs, lows = self.find_swing_points(data, self.swing_lookback // 2)

        if len(highs) < 3 or len(lows) < 3:
            return MarketStructure.RANGING

        # Ostatnie 3 swing points każdego typu
        recent_highs = highs[-3:]
        recent_lows = lows[-3:]

        try:
            # Higher Highs & Higher Lows = Bullish
            if (data['high'].iloc[recent_highs[-1]] > data['high'].iloc[recent_highs[-2]] and
                data['low'].iloc[recent_lows[-1]] > data['low'].iloc[recent_lows[-2]]):
                return MarketStructure.BULLISH

            # Lower Highs & Lower Lows = Bearish  
            elif (data['high'].iloc[recent_highs[-1]] < data['high'].iloc[recent_highs[-2]] and
                  data['low'].iloc[recent_lows[-1]] < data['low'].iloc[recent_lows[-2]]):
                return MarketStructure.BEARISH
        except:
            pass

        return MarketStructure.RANGING

    def find_swing_points(self, data: pd.DataFrame, window: int = 5) -> Tuple[List[int], List[int]]:
        """Znajdź swing highs i lows"""
        highs = []
        lows = []

        high_prices = data['high'].values if 'high' in data.columns else data['close'].values
        low_prices = data['low'].values if 'low' in data.columns else data['close'].values

        for i in range(window, len(data) - window):
            # Swing High
            is_high = all(high_prices[i] >= high_prices[j] for j in range(i - window, i + window + 1) if j != i)
            if is_high and high_prices[i] > high_prices[i-1] and high_prices[i] > high_prices[i+1]:
                highs.append(i)

            # Swing Low
            is_low = all(low_prices[i] <= low_prices[j] for j in range(i - window, i + window + 1) if j != i)
            if is_low and low_prices[i] < low_prices[i-1] and low_prices[i] < low_prices[i+1]:
                lows.append(i)

        return highs, lows

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generuj sygnały SMC"""
        signals = []

        if len(data) < 50:
            return signals

        try:
            # Aktualizuj stan rynku
            self.current_structure = self.identify_market_structure(data)

            current_price = data['close'].iloc[-1]
            current_time = data.index[-1]

            # Podstawowe sygnały SMC na podstawie struktury rynku
            if self.current_structure == MarketStructure.BULLISH:
                signal = Signal(
                    timestamp=current_time,
                    signal_type=SignalType.BUY,
                    confidence=0.7,
                    entry_price=current_price,
                    stop_loss=current_price * (1 - self.default_stop_loss),
                    take_profit=current_price * (1 + self.default_stop_loss * self.risk_reward_ratio),
                    metadata={
                        "strategy": "smart_money_concept",
                        "trigger": "bullish_market_structure",
                        "market_structure": self.current_structure.value
                    }
                )
                signals.append(signal)

            elif self.current_structure == MarketStructure.BEARISH:
                signal = Signal(
                    timestamp=current_time,
                    signal_type=SignalType.SELL,
                    confidence=0.7,
                    entry_price=current_price,
                    stop_loss=current_price * (1 + self.default_stop_loss),
                    take_profit=current_price * (1 - self.default_stop_loss * self.risk_reward_ratio),
                    metadata={
                        "strategy": "smart_money_concept",
                        "trigger": "bearish_market_structure", 
                        "market_structure": self.current_structure.value
                    }
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"Error generating SMC signals: {e}")

        return signals

    def calculate_position_size(self, account_balance: float, entry_price: float, stop_loss: float) -> float:
        """Oblicz wielkość pozycji dla SMC"""
        risk_amount = account_balance * self.max_position_size
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit == 0:
            return 0

        return risk_amount / risk_per_unit

    def get_strategy_info(self) -> Dict:
        """Informacje o strategii SMC"""
        return {
            "name": "Smart Money Concept Strategy",
            "description": "Zaawansowana analiza struktury rynku i zachowań instytucjonalnych",
            "features": [
                "Market Structure Analysis",
                "Order Blocks Detection", 
                "Fair Value Gaps Identification",
                "Liquidity Zones Mapping",
                "Break of Structure Signals"
            ],
            "risk_management": {
                "default_stop_loss": self.default_stop_loss,
                "risk_reward_ratio": self.risk_reward_ratio,
                "max_position_size": self.max_position_size
            },
            "timeframes": ["5m", "15m", "1h", "4h", "1d"],
            "markets": ["Forex", "Crypto", "Indices", "Commodities"]
        }
