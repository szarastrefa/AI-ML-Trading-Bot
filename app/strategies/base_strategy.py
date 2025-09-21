"""Base Strategy Class for AI/ML Trading Bot v2.1
Research-based abstract interface for all trading strategies

Author: AI/ML Trading Bot Team
Version: 2.1.0 (Research-Enhanced)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies
    Research-based design principles applied for maximum stability
    
    All trading strategies should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base strategy
        
        Args:
            config: Strategy configuration dictionary
        """
        self.config = config
        self.name = "BaseStrategy"
        self.version = "2.1.0"
        self.created = datetime.utcnow().isoformat()
        
        # Enhanced performance tracking
        self.performance_stats = {
            'total_signals': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'winning_signals': 0,
            'losing_signals': 0,
            'win_rate': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'avg_execution_time': 0.0,
            'sharpe_ratio': 0.0
        }
    
    @abstractmethod
    async def analyze(self, symbol: str, timeframe: str = "H1") -> Dict[str, Any]:
        """
        Abstract method for market analysis - must be implemented by subclasses
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'BTCUSDT')
            timeframe: Chart timeframe (e.g., 'M15', 'H1', 'H4', 'D1')
            
        Returns:
            Dict containing:
            - signal: 'BUY', 'SELL', or 'HOLD'
            - confidence: Confidence level (0-100)
            - entry_price: Recommended entry price
            - stop_loss: Recommended stop loss price
            - take_profit: Recommended take profit price
            - risk_reward_ratio: Risk to reward ratio
            - analysis: Detailed analysis breakdown
            - metadata: Strategy metadata and timing
        """
        pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy performance statistics
        
        Returns:
            Dictionary with performance metrics and strategy information
        """
        # Calculate derived metrics
        total_signals = self.performance_stats['total_signals']
        successful_analyses = self.performance_stats['successful_analyses']
        
        success_rate = (successful_analyses / total_signals * 100) if total_signals > 0 else 0
        win_rate = (self.performance_stats['winning_signals'] / total_signals * 100) if total_signals > 0 else 0
        
        return {
            **self.performance_stats,
            "derived_metrics": {
                "success_rate_percent": round(success_rate, 2),
                "win_rate_percent": round(win_rate, 2),
                "total_trades": total_signals,
                "reliability_score": round((success_rate + win_rate) / 2, 2)
            },
            "strategy_info": {
                "name": self.name,
                "version": self.version,
                "created": self.created,
                "last_updated": datetime.utcnow().isoformat()
            }
        }
    
    def update_performance(self, signal_result: Dict[str, Any], actual_result: Optional[Dict[str, Any]] = None):
        """
        Update strategy performance statistics
        
        Args:
            signal_result: The signal that was generated
            actual_result: The actual trading result (optional)
        """
        self.performance_stats['total_signals'] += 1
        
        if actual_result:
            profit = actual_result.get('profit', 0)
            if profit > 0:
                self.performance_stats['winning_signals'] += 1
                self.performance_stats['total_return'] += profit
            else:
                self.performance_stats['losing_signals'] += 1
                self.performance_stats['total_return'] += profit
            
            # Update win rate
            total = self.performance_stats['total_signals']
            if total > 0:
                self.performance_stats['win_rate'] = (
                    self.performance_stats['winning_signals'] / total
                )
    
    def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported trading symbols
        
        Returns:
            List of supported symbols
        """
        return self.config.get('supported_symbols', [
            # Forex Major Pairs
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
            # Forex Minor Pairs  
            'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY',
            # Cryptocurrency
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'XRPUSDT', 'BNBUSDT', 'SOLUSDT',
            # Commodities
            'XAUUSD', 'XAGUSD', 'USOIL', 'UKOIL',
            # Stock Indices
            'SPX500', 'NAS100', 'GER40', 'UK100', 'JPN225'
        ])
    
    def get_supported_timeframes(self) -> List[str]:
        """
        Get list of supported timeframes
        
        Returns:
            List of supported timeframes
        """
        return self.config.get('supported_timeframes', [
            'M1', 'M5', 'M15', 'M30', 'H1', 'H2', 'H4', 'H6', 'H12', 'D1', 'W1', 'MN1'
        ])
    
    def get_strategy_features(self) -> Dict[str, Any]:
        """
        Get strategy features and capabilities
        
        Returns:
            Dictionary describing strategy features
        """
        return {
            "technical_analysis": "Available",
            "smart_money_concepts": "Supported",
            "multi_timeframe": "Enabled",
            "risk_management": "Integrated",
            "pattern_recognition": "Advanced",
            "volume_analysis": "Professional",
            "backtesting_ready": True,
            "real_time_analysis": True
        }
    
    def validate_config(self) -> bool:
        """
        Validate strategy configuration
        
        Returns:
            True if configuration is valid, False otherwise
        """
        if not isinstance(self.config, dict):
            return False
        
        # Check for required configuration keys
        required_keys = ['timeframes']
        for key in required_keys:
            if key not in self.config:
                return False
        
        return True
    
    def get_risk_parameters(self) -> Dict[str, float]:
        """
        Get risk management parameters
        
        Returns:
            Dictionary with risk parameters
        """
        return self.config.get('risk_management', {
            'max_risk_per_trade': 0.02,  # 2% per trade
            'max_portfolio_risk': 0.10,  # 10% total portfolio
            'stop_loss_multiplier': 2.0,  # 2x ATR
            'take_profit_multiplier': 3.0,  # 3x ATR (1.5:1 RR)
            'position_size_method': 'fixed_risk',
            'max_correlation': 0.7,  # Max correlation between trades
            'max_drawdown_limit': 0.15  # 15% max drawdown
        })
    
    async def backtest(self, symbol: str, start_date: str, end_date: str, 
                      initial_capital: float = 10000) -> Dict[str, Any]:
        """
        Backtest the strategy (optional implementation)
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Initial capital for backtesting
            
        Returns:
            Backtesting results
        """
        return {
            "error": "Backtesting not implemented for this strategy",
            "message": "Override the backtest method to implement backtesting",
            "recommendation": "Use specialized backtesting frameworks like vectorbt or backtrader"
        }
    
    def _empty_signal(self, error_msg: str) -> Dict[str, Any]:
        """
        Create an empty/error signal
        
        Args:
            error_msg: Error message
            
        Returns:
            Empty signal with error message
        """
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "risk_reward_ratio": 1.0,
            "error": error_msg,
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": {},
            "metadata": {
                "strategy": self.name,
                "version": self.version
            }
        }
    
    def _validate_input(self, symbol: str, timeframe: str) -> bool:
        """
        Validate input parameters
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            
        Returns:
            True if valid, False otherwise
        """
        # Validate symbol
        supported_symbols = self.get_supported_symbols()
        if symbol not in supported_symbols:
            return False
        
        # Validate timeframe
        supported_timeframes = self.get_supported_timeframes()
        if timeframe not in supported_timeframes:
            return False
        
        return True
    
    def calculate_position_size(self, account_balance: float, risk_amount: float, 
                               stop_loss_distance: float, price: float) -> float:
        """
        Calculate position size based on risk management rules
        
        Args:
            account_balance: Current account balance
            risk_amount: Amount to risk (in account currency)
            stop_loss_distance: Distance to stop loss in price units
            price: Current price of the instrument
            
        Returns:
            Position size in units
        """
        try:
            if stop_loss_distance <= 0 or price <= 0:
                return 0.0
            
            # Calculate position size using fixed risk method
            position_size = risk_amount / stop_loss_distance
            
            # Apply maximum position size limits (e.g., 5% of balance)
            max_position_value = account_balance * 0.05
            max_position_size = max_position_value / price
            
            return min(position_size, max_position_size)
            
        except Exception:
            return 0.0
    
    def __str__(self) -> str:
        """String representation of the strategy"""
        return f"{self.name} v{self.version}"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}', created='{self.created}')"


class StrategyError(Exception):
    """
    Custom exception for strategy-related errors
    """
    def __init__(self, message: str, strategy_name: str = None):
        self.message = message
        self.strategy_name = strategy_name
        super().__init__(self.message)


class StrategyConfigError(StrategyError):
    """
    Exception raised for strategy configuration errors
    """
    pass


class StrategyAnalysisError(StrategyError):
    """
    Exception raised for strategy analysis errors
    """
    pass


class StrategyDataError(StrategyError):
    """
    Exception raised for strategy data-related errors
    """
    pass


# Strategy Registry for managing multiple strategies
class StrategyRegistry:
    """
    Registry for managing multiple trading strategies
    """
    
    def __init__(self):
        self._strategies: Dict[str, BaseStrategy] = {}
    
    def register(self, name: str, strategy: BaseStrategy):
        """Register a strategy"""
        self._strategies[name] = strategy
    
    def get(self, name: str) -> Optional[BaseStrategy]:
        """Get a strategy by name"""
        return self._strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """List all registered strategy names"""
        return list(self._strategies.keys())
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance stats for all strategies"""
        return {
            name: strategy.get_performance_stats() 
            for name, strategy in self._strategies.items()
        }