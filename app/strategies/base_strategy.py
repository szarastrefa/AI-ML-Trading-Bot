"""
Base Strategy Class for AI/ML Trading Bot
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "BaseStrategy"
        self.version = "1.0.0"
    
    @abstractmethod
    async def analyze(self, symbol: str, timeframe: str = "H1", limit: int = 500) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signal
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'BTCUSDT')
            timeframe: Chart timeframe
            limit: Number of candles to analyze
            
        Returns:
            Trading signal with analysis
        """
        pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics"""
        return {
            "name": self.name,
            "version": self.version,
            "total_signals": 0,
            "win_rate": 0.0
        }
