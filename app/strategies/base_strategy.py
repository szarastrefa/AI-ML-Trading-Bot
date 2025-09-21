"""
Bazowa klasa dla wszystkich strategii tradingowych
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

class SignalType(Enum):
    """Typy sygnałów"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"

@dataclass
class Signal:
    """Sygnał tradingowy"""
    timestamp: pd.Timestamp
    signal_type: SignalType
    confidence: float  # 0.0 - 1.0
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None

class BaseStrategy(ABC):
    """Bazowa klasa dla wszystkich strategii"""

    def __init__(self, config: dict):
        self.config = config
        self.name = "base_strategy"

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generuj sygnały tradingowe"""
        pass

    @abstractmethod
    def calculate_position_size(self, account_balance: float, entry_price: float, stop_loss: float) -> float:
        """Oblicz wielkość pozycji"""
        pass

    def validate_signal(self, signal: Signal, current_data: pd.DataFrame) -> bool:
        """Waliduj sygnał przed wykonaniem"""
        return True

    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """Zwróć informacje o strategii"""
        pass
