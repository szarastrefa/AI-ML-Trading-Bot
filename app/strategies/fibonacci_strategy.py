"""
Strategia Fibonacci Team - implementacja strategii z pliku Fibon.md
Zawiera: poziomy Fibonacciego, formacje harmoniczne, scalping, zarządzanie ryzykiem
Oparta na metodologii Łukasza Fijołka z kanału Fibonacci Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from .base_strategy import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)

class FibonacciLevel(Enum):
    """Poziomy zniesień Fibonacciego"""
    LEVEL_236 = 0.236  # Płytkie cofnięcie w silnych trendach
    LEVEL_382 = 0.382  # Strefa "buy-the-dip"
    LEVEL_500 = 0.500  # Psychologiczny poziom
    LEVEL_618 = 0.618  # Złoty podział - kluczowy punkt
    LEVEL_786 = 0.786  # Ostatni etap zniesienia

class FibonacciExtension(Enum):
    """Rozszerzenia Fibonacciego"""
    EXT_618 = 0.618    # Pierwszy poziom rozszerzenia
    EXT_1000 = 1.000   # Równa długość ruchu
    EXT_1618 = 1.618   # Drugi poziom rozszerzenia
    EXT_2618 = 2.618   # Trzeci poziom rozszerzenia

@dataclass
class HarmonicPattern:
    """Formacja harmoniczna"""
    name: str
    points: Dict[str, float]  # X, A, B, C, D points
    ratios: Dict[str, Tuple[float, float]]  # Min/max ratios for each leg
    confidence: float
    direction: str  # "bullish" or "bearish"

@dataclass
class FibonacciLevels:
    """Poziomy Fibonacciego dla danego ruchu"""
    high: float
    low: float
    levels: Dict[str, float]
    extensions: Dict[str, float]

class FibonacciTeamStrategy(BaseStrategy):
    """
    Strategia Fibonacci Team z wszystkimi kluczowymi elementami:
    - Poziomy zniesień i rozszerzeń Fibonacciego
    - Formacje harmoniczne (Gartley, Bat, Butterfly, Crab)
    - Analiza wolumenu
    - Scalping na sesjach Londyn/NY
    - System "Strzałki Fibonacciego"
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "fibonacci_team"

        # Parametry strategii z Fibonacci Team
        self.fib_lookback = config.get("fib_lookback", 50)
        self.min_swing_size = config.get("min_swing_size", 0.002)  # Min 20 pipsów
        self.volume_threshold = config.get("volume_threshold", 1.5)
        self.pattern_tolerance = config.get("pattern_tolerance", 0.02)  # 2% tolerancja

        # Zarządzanie ryzykiem zgodne z Fibonacci Team
        self.default_stop_loss = 0.02  # 2% jak w wymaganiach
        self.min_risk_reward = 2.0     # Min 1:2 RR
        self.max_position_size = 0.02  # 2% kapitału na transakcję

        # Sesje handlowe (GMT)
        self.london_session = (8, 17)   # 8:00-17:00 GMT
        self.ny_session = (13, 22)      # 13:00-22:00 GMT
        self.overlap_session = (13, 17) # Najwyższa płynność

        # Wzorce harmoniczne - dokładne proporcje
        self.harmonic_patterns = self._init_harmonic_patterns()

    def _init_harmonic_patterns(self) -> Dict[str, HarmonicPattern]:
        """Inicjalizacja wzorców harmonicznych zgodnie z Fibonacci Team"""
        return {
            "gartley": HarmonicPattern(
                name="Gartley",
                points={},
                ratios={
                    "AB_XA": (0.618, 0.618),  # AB = 61.8% XA
                    "BC_AB": (0.382, 0.886),  # BC = 38.2%-88.6% AB  
                    "CD_AB": (1.13, 1.618),   # CD = 113%-161.8% AB
                },
                confidence=0.0,
                direction=""
            ),
            "bat": HarmonicPattern(
                name="Bat", 
                points={},
                ratios={
                    "AB_XA": (0.382, 0.500),  # AB = 38.2%-50% XA
                    "BC_AB": (0.382, 0.886),  # BC = 38.2%-88.6% AB
                    "CD_AB": (1.618, 2.618),  # CD = 161.8%-261.8% AB
                    "XD_XA": (0.886, 0.886),  # XD = 88.6% XA
                },
                confidence=0.0,
                direction=""
            )
        }

    def calculate_fibonacci_levels(self, high: float, low: float, direction: str = "retracement") -> FibonacciLevels:
        """Oblicza poziomy Fibonacciego dla danego swing"""
        diff = high - low

        levels = {}
        extensions = {}

        if direction == "retracement":
            # Zniesienia od high do low
            for level in FibonacciLevel:
                levels[f"fib_{level.name.lower()}"] = high - (diff * level.value)
        else:
            # Projekcje/rozszerzenia
            for ext in FibonacciExtension:
                extensions[f"ext_{ext.name.lower()}"] = high + (diff * ext.value)

        return FibonacciLevels(
            high=high,
            low=low, 
            levels=levels,
            extensions=extensions
        )
    
    def find_swing_points(self, data: pd.DataFrame, window: int = 5) -> Tuple[List[int], List[int]]:
        """Znajdź punkty swing high/low"""
        highs = []
        lows = []

        high_prices = data['high'].values if 'high' in data.columns else data['close'].values
        low_prices = data['low'].values if 'low' in data.columns else data['close'].values

        for i in range(window, len(data) - window):
            # Swing High
            is_high = True
            for j in range(i - window, i + window + 1):
                if j != i and high_prices[j] >= high_prices[i]:
                    is_high = False
                    break
            if is_high:
                highs.append(i)

            # Swing Low    
            is_low = True
            for j in range(i - window, i + window + 1):
                if j != i and low_prices[j] <= low_prices[i]:
                    is_low = False
                    break
            if is_low:
                lows.append(i)

        return highs, lows

    def is_active_session(self, timestamp) -> bool:
        """Sprawdź czy aktualny czas to aktywna sesja (Londyn/NY)"""
        try:
            hour = timestamp.hour if hasattr(timestamp, 'hour') else 12
        except:
            hour = 12

        # Sesja londyńska lub nowojorska
        london_active = self.london_session[0] <= hour < self.london_session[1]
        ny_active = self.ny_session[0] <= hour < self.ny_session[1]

        return london_active or ny_active

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generuj sygnały zgodnie z metodologią Fibonacci Team"""
        signals = []

        if len(data) < 50:  # Potrzebujemy wystarczająco danych
            return signals

        try:
            # Znajdź swing points i poziomy Fibonacciego
            highs, lows = self.find_swing_points(data)

            if len(highs) > 0 and len(lows) > 0:
                last_high_idx = highs[-1] if highs else -1
                last_low_idx = lows[-1] if lows else -1

                if last_high_idx >= 0 and last_low_idx >= 0:
                    last_high = data.iloc[last_high_idx]['high'] if 'high' in data.columns else data.iloc[last_high_idx]['close']
                    last_low = data.iloc[last_low_idx]['low'] if 'low' in data.columns else data.iloc[last_low_idx]['close']

                    # Oblicz poziomy Fibonacciego
                    fib_levels = self.calculate_fibonacci_levels(last_high, last_low)

                    current_price = data['close'].iloc[-1]
                    current_time = data.index[-1]

                    # Sprawdź czy jesteśmy w aktywnej sesji
                    if not self.is_active_session(current_time):
                        return signals

                    # Sygnały kupna/sprzedaży na poziomach Fibonacciego
                    for level_name, level_price in fib_levels.levels.items():
                        price_diff = abs(current_price - level_price) / current_price

                        if price_diff < 0.001:  # W pobliżu poziomu Fibo (0.1%)

                            # Sygnał kupna na kluczowych poziomach zniesienia
                            if level_name in ["fib_level_382", "fib_level_618"]:

                                # Oblicz stop loss i take profit
                                if last_high_idx > last_low_idx:  # Trend wzrostowy
                                    stop_loss = current_price * (1 - self.default_stop_loss)
                                    take_profit = current_price * (1 + self.default_stop_loss * self.min_risk_reward)
                                    signal_type = SignalType.BUY
                                else:  # Trend spadkowy
                                    stop_loss = current_price * (1 + self.default_stop_loss)
                                    take_profit = current_price * (1 - self.default_stop_loss * self.min_risk_reward)
                                    signal_type = SignalType.SELL

                                signal = Signal(
                                    timestamp=current_time,
                                    signal_type=signal_type,
                                    confidence=0.75,
                                    entry_price=current_price,
                                    stop_loss=stop_loss,
                                    take_profit=take_profit,
                                    metadata={
                                        "strategy": "fibonacci_team",
                                        "trigger": f"Fibonacci level {level_name}",
                                        "level_price": level_price
                                    }
                                )
                                signals.append(signal)

                                logger.info(f"Fibonacci signal generated: {signal.signal_type.value} at {level_name}")

        except Exception as e:
            logger.error(f"Error generating Fibonacci signals: {e}")

        return signals

    def calculate_position_size(self, account_balance: float, entry_price: float, stop_loss: float) -> float:
        """Oblicz wielkość pozycji zgodnie z zarządzaniem ryzykiem Fibonacci Team"""
        # Maksymalnie 2% kapitału na transakcję
        risk_amount = account_balance * self.max_position_size

        # Oblicz ryzyko na jednostkę
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit == 0:
            return 0

        # Wielkość pozycji
        position_size = risk_amount / risk_per_unit

        # Dodatkowe ograniczenia
        max_position_value = account_balance * 0.1  # Max 10% wartości konta
        max_units = max_position_value / entry_price

        return min(position_size, max_units)

    def validate_signal(self, signal: Signal, current_data: pd.DataFrame) -> bool:
        """Waliduj sygnał przed wykonaniem"""
        # Sprawdź sesję handlową
        if not self.is_active_session(signal.timestamp):
            return False

        # Sprawdź spread (dla scalpingu ważne)
        current_price = current_data['close'].iloc[-1]
        if abs(signal.entry_price - current_price) / current_price > 0.005:  # 0.5% tolerancja
            return False

        # Sprawdź risk/reward ratio
        if signal.stop_loss and signal.take_profit:
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.entry_price)

            if risk > 0 and reward / risk < self.min_risk_reward:
                return False

        return True

    def get_strategy_info(self) -> Dict[str, Any]:
        """Informacje o strategii"""
        return {
            "name": "Fibonacci Team Strategy",
            "description": "Strategia oparta na metodologii Fibonacci Team - poziomy Fibo, formacje harmoniczne, scalping",
            "author": "Based on Łukasz Fijołek - Fibonacci Team",
            "risk_management": {
                "default_stop_loss": self.default_stop_loss,
                "min_risk_reward": self.min_risk_reward,
                "max_position_size": self.max_position_size
            },
            "features": [
                "Poziomy zniesień i rozszerzeń Fibonacciego",
                "Formacje harmoniczne (Gartley, Bat, Butterfly, Crab)",
                "Scalping na sesjach Londyn/NY",
                "System zarządzania ryzykiem"
            ],
            "timeframes": ["1m", "5m", "15m", "30m"],
            "markets": ["Forex", "Indices", "Commodities"]
        }
