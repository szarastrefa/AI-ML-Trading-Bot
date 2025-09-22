"""
Base Strategy Class for AI/ML Trading Bot
Foundation for all trading strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    Provides common interface and utilities
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = "BaseStrategy"
        self.version = "1.0.0"
        self.created_at = datetime.utcnow()
        self.analysis_count = 0
        
    @abstractmethod
    async def analyze(self, symbol: str, timeframe: str = "H1") -> Dict[str, Any]:
        """
        Abstract method for market analysis
        Must be implemented by all strategy subclasses
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'BTCUSDT')
            timeframe: Chart timeframe (e.g., 'H1', 'D1')
            
        Returns:
            Dict containing trading signal and analysis
        """
        pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics"""
        return {
            "strategy_info": {
                "name": self.name,
                "version": self.version,
                "created_at": self.created_at.isoformat(),
                "analysis_count": self.analysis_count
            }
        }
    
    def _increment_analysis_count(self):
        """Internal method to track analysis calls"""
        self.analysis_count += 1
    
    def _create_empty_signal(self, error_msg: str = "") -> Dict[str, Any]:
        """Create empty/error signal"""
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "risk_reward_ratio": 1.0,
            "error": error_msg,
            "timestamp": datetime.utcnow().isoformat()
        }
