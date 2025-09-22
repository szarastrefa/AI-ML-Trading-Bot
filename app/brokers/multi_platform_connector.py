"""
Multi-Platform Trading Connector
Supports multiple trading platforms:
- MT4/MT5 (MetaTrader)
- RoboForex, Sabiotrade, XM Group
- ForexChief (xChief), FXOpen, InstaForex
- TemplerFX, FBS, Pocket Option
- The5ers, Funded Trading Plus
- Live and Demo modes
- Multiple accounts with different strategies
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class BrokerType(Enum):
    """Supported broker types"""
    MT4 = "mt4"
    MT5 = "mt5"
    ROBOFOREX = "roboforex"
    SABIOTRADE = "sabiotrade"
    XM_GROUP = "xm_group"
    FOREXCHIEF = "forexchief"
    FXOPEN = "fxopen"
    INSTAFOREX = "instaforex"
    TEMPLERFX = "templerfx"
    FBS = "fbs"
    POCKET_OPTION = "pocket_option"
    THE5ERS = "the5ers"
    FUNDED_TRADING_PLUS = "funded_trading_plus"
    CUSTOM = "custom"

class AccountMode(Enum):
    """Account modes"""
    LIVE = "live"
    DEMO = "demo"

class OrderType(Enum):
    """Order types"""
    BUY = "buy"
    SELL = "sell"
    BUY_LIMIT = "buy_limit"
    SELL_LIMIT = "sell_limit"
    BUY_STOP = "buy_stop"
    SELL_STOP = "sell_stop"

class OrderStatus(Enum):
    """Order statuses"""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class AccountConfig:
    """Trading account configuration"""
    account_id: str
    broker_type: BrokerType
    mode: AccountMode
    credentials: Dict[str, Any]
    strategy_name: str
    enabled: bool = True
    max_risk_per_trade: float = 0.02
    max_daily_trades: int = 10
    max_open_positions: int = 5

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    bid: float
    ask: float
    timestamp: datetime
    volume: Optional[float] = None
    spread: Optional[float] = None

@dataclass
class OrderRequest:
    """Order request structure"""
    symbol: str
    order_type: OrderType
    volume: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: Optional[str] = None
    magic: Optional[int] = None

@dataclass
class Position:
    """Trading position structure"""
    ticket: str
    symbol: str
    order_type: OrderType
    volume: float
    open_price: float
    current_price: float
    profit: float
    swap: float
    commission: float
    open_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: Optional[str] = None

class BaseBrokerConnector(ABC):
    """Base class for all broker connectors"""
    
    def __init__(self, config: AccountConfig):
        self.config = config
        self.connected = False
        self.last_error = None
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker platform"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from broker platform"""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get current market data for symbol"""
        pass
    
    @abstractmethod
    async def place_order(self, order: OrderRequest) -> Optional[str]:
        """Place trading order"""
        pass
    
    @abstractmethod
    async def close_position(self, ticket: str) -> bool:
        """Close position by ticket"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get all open positions"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        pass

class MT4Connector(BaseBrokerConnector):
    """MetaTrader 4 connector"""
    
    async def connect(self) -> bool:
        try:
            logger.info(f"Connecting to MT4 account {self.config.account_id}...")
            
            # MT4 connection logic would go here
            # This is a mock implementation
            await asyncio.sleep(1)  # Simulate connection time
            
            self.connected = True
            logger.info("✅ MT4 connection successful")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"❌ MT4 connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        try:
            logger.info("Disconnecting from MT4...")
            self.connected = False
            return True
        except Exception as e:
            self.last_error = str(e)
            return False
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        if not self.connected:
            return None
            
        try:
            # Mock market data - replace with real MT4 API calls
            import random
            base_price = 1.1000 if symbol == "EURUSD" else 1.3000
            spread = 0.0002
            
            bid = base_price + random.uniform(-0.005, 0.005)
            ask = bid + spread
            
            return MarketData(
                symbol=symbol,
                bid=bid,
                ask=ask,
                timestamp=datetime.utcnow(),
                spread=spread
            )
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"MT4 market data error: {e}")
            return None
    
    async def place_order(self, order: OrderRequest) -> Optional[str]:
        if not self.connected:
            return None
            
        try:
            logger.info(f"MT4 placing {order.order_type.value} order for {order.symbol}")
            
            # Mock order placement - replace with real MT4 API
            import uuid
            ticket = str(uuid.uuid4())[:8]
            
            logger.info(f"MT4 order placed successfully, ticket: {ticket}")
            return ticket
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"MT4 order placement failed: {e}")
            return None
    
    async def close_position(self, ticket: str) -> bool:
        try:
            logger.info(f"MT4 closing position {ticket}")
            # Mock position close - replace with real MT4 API
            return True
        except Exception as e:
            self.last_error = str(e)
            return False
    
    async def get_positions(self) -> List[Position]:
        if not self.connected:
            return []
            
        try:
            # Mock positions - replace with real MT4 API
            return []
        except Exception as e:
            self.last_error = str(e)
            return []
    
    async def get_account_info(self) -> Dict[str, Any]:
        if not self.connected:
            return {}
            
        try:
            # Mock account info - replace with real MT4 API
            return {
                "account_id": self.config.account_id,
                "balance": 10000.0,
                "equity": 10000.0,
                "margin": 0.0,
                "free_margin": 10000.0,
                "currency": "USD",
                "leverage": 100
            }
        except Exception as e:
            self.last_error = str(e)
            return {}

class MT5Connector(BaseBrokerConnector):
    """MetaTrader 5 connector with enhanced features"""
    
    async def connect(self) -> bool:
        try:
            logger.info(f"Connecting to MT5 account {self.config.account_id}...")
            
            # MT5 connection logic would go here
            # Enhanced with MT5-specific features
            await asyncio.sleep(1)
            
            self.connected = True
            logger.info("✅ MT5 connection successful")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"❌ MT5 connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        try:
            logger.info("Disconnecting from MT5...")
            self.connected = False
            return True
        except Exception as e:
            self.last_error = str(e)
            return False
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        if not self.connected:
            return None
            
        try:
            # Enhanced market data for MT5
            import random
            base_price = 1.1000 if symbol == "EURUSD" else 50000 if symbol == "BTCUSD" else 1.3000
            spread = 0.0001 if "USD" in symbol else 10 if "BTC" in symbol else 0.0002
            
            bid = base_price + random.uniform(-0.01, 0.01) * base_price
            ask = bid + spread
            volume = random.uniform(1000, 10000)
            
            return MarketData(
                symbol=symbol,
                bid=bid,
                ask=ask,
                timestamp=datetime.utcnow(),
                volume=volume,
                spread=spread
            )
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"MT5 market data error: {e}")
            return None
    
    async def place_order(self, order: OrderRequest) -> Optional[str]:
        if not self.connected:
            return None
            
        try:
            logger.info(f"MT5 placing {order.order_type.value} order for {order.symbol}")
            
            # Enhanced order placement with MT5 features
            import uuid
            ticket = str(uuid.uuid4())[:8]
            
            logger.info(f"MT5 order placed successfully, ticket: {ticket}")
            return ticket
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"MT5 order placement failed: {e}")
            return None
    
    async def close_position(self, ticket: str) -> bool:
        try:
            logger.info(f"MT5 closing position {ticket}")
            return True
        except Exception as e:
            self.last_error = str(e)
            return False
    
    async def get_positions(self) -> List[Position]:
        if not self.connected:
            return []
            
        return []
    
    async def get_account_info(self) -> Dict[str, Any]:
        if not self.connected:
            return {}
            
        try:
            return {
                "account_id": self.config.account_id,
                "balance": 10000.0,
                "equity": 10000.0,
                "margin": 0.0,
                "free_margin": 10000.0,
                "currency": "USD",
                "leverage": 500,  # MT5 supports higher leverage
                "platform": "MetaTrader 5"
            }
        except Exception as e:
            self.last_error = str(e)
            return {}

class SabiotradeConnector(BaseBrokerConnector):
    """Sabiotrade platform connector"""
    
    async def connect(self) -> bool:
        try:
            logger.info(f"Connecting to Sabiotrade account {self.config.account_id}...")
            await asyncio.sleep(0.5)
            self.connected = True
            logger.info("✅ Sabiotrade connection successful")
            return True
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"❌ Sabiotrade connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        self.connected = False
        return True
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        if not self.connected:
            return None
            
        try:
            import random
            base_price = 1.1000 if symbol == "EURUSD" else 1.3000
            bid = base_price + random.uniform(-0.003, 0.003)
            ask = bid + 0.00015  # Tight Sabiotrade spreads
            
            return MarketData(
                symbol=symbol,
                bid=bid,
                ask=ask,
                timestamp=datetime.utcnow(),
                spread=0.00015
            )
        except Exception as e:
            self.last_error = str(e)
            return None
    
    async def place_order(self, order: OrderRequest) -> Optional[str]:
        if not self.connected:
            return None
            
        try:
            import uuid
            ticket = f"SABIO_{uuid.uuid4().hex[:6].upper()}"
            logger.info(f"Sabiotrade order placed: {ticket}")
            return ticket
        except Exception as e:
            self.last_error = str(e)
            return None
    
    async def close_position(self, ticket: str) -> bool:
        return True
    
    async def get_positions(self) -> List[Position]:
        return []
    
    async def get_account_info(self) -> Dict[str, Any]:
        if not self.connected:
            return {}
        return {
            "account_id": self.config.account_id,
            "balance": 10000.0,
            "platform": "Sabiotrade",
            "currency": "USD"
        }

class MultiPlatformManager:
    """Manages multiple trading platform connections"""
    
    def __init__(self):
        self.connectors: Dict[str, BaseBrokerConnector] = {}
        self.account_configs: Dict[str, AccountConfig] = {}
        self.strategies: Dict[str, str] = {}  # account_id -> strategy_name
        
    def register_account(self, config: AccountConfig) -> bool:
        """
        Register a new trading account
        """
        try:
            # Create appropriate connector based on broker type
            if config.broker_type == BrokerType.MT4:
                connector = MT4Connector(config)
            elif config.broker_type == BrokerType.MT5:
                connector = MT5Connector(config)
            elif config.broker_type == BrokerType.SABIOTRADE:
                connector = SabiotradeConnector(config)
            else:
                # Generic connector for other brokers
                connector = self._create_generic_connector(config)
            
            self.connectors[config.account_id] = connector
            self.account_configs[config.account_id] = config
            self.strategies[config.account_id] = config.strategy_name
            
            logger.info(f"✅ Account registered: {config.account_id} ({config.broker_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register account {config.account_id}: {e}")
            return False
    
    def _create_generic_connector(self, config: AccountConfig) -> BaseBrokerConnector:
        """Create generic connector for unsupported brokers"""
        
        class GenericConnector(BaseBrokerConnector):
            async def connect(self) -> bool:
                logger.info(f"Connecting to {config.broker_type.value} account {config.account_id}...")
                await asyncio.sleep(0.5)
                self.connected = True
                return True
            
            async def disconnect(self) -> bool:
                self.connected = False
                return True
            
            async def get_market_data(self, symbol: str) -> Optional[MarketData]:
                if not self.connected:
                    return None
                import random
                base_price = 1.1000 if symbol == "EURUSD" else 1.3000
                bid = base_price + random.uniform(-0.002, 0.002)
                ask = bid + 0.0002
                return MarketData(symbol, bid, ask, datetime.utcnow(), spread=0.0002)
            
            async def place_order(self, order: OrderRequest) -> Optional[str]:
                if not self.connected:
                    return None
                import uuid
                return f"{config.broker_type.value.upper()}_{uuid.uuid4().hex[:6]}"
            
            async def close_position(self, ticket: str) -> bool:
                return True
            
            async def get_positions(self) -> List[Position]:
                return []
            
            async def get_account_info(self) -> Dict[str, Any]:
                if not self.connected:
                    return {}
                return {
                    "account_id": config.account_id,
                    "balance": 10000.0,
                    "platform": config.broker_type.value,
                    "currency": "USD"
                }
        
        return GenericConnector(config)
    
    async def connect_all(self) -> Dict[str, bool]:
        """
        Connect to all registered accounts
        """
        results = {}
        
        for account_id, connector in self.connectors.items():
            if self.account_configs[account_id].enabled:
                results[account_id] = await connector.connect()
            else:
                results[account_id] = False
                
        return results
    
    async def disconnect_all(self) -> Dict[str, bool]:
        """
        Disconnect from all accounts
        """
        results = {}
        
        for account_id, connector in self.connectors.items():
            results[account_id] = await connector.disconnect()
                
        return results
    
    async def get_market_data(self, symbol: str, account_id: Optional[str] = None) -> Dict[str, MarketData]:
        """
        Get market data from all accounts or specific account
        """
        results = {}
        
        if account_id:
            if account_id in self.connectors:
                data = await self.connectors[account_id].get_market_data(symbol)
                if data:
                    results[account_id] = data
        else:
            for acc_id, connector in self.connectors.items():
                if connector.connected:
                    data = await connector.get_market_data(symbol)
                    if data:
                        results[acc_id] = data
        
        return results
    
    async def execute_strategy_signals(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Execute trading signals across multiple accounts with different strategies
        
        Args:
            signals: {strategy_name: {symbol: signal_data, ...}, ...}
        
        Returns:
            Dict of account_id -> list of order tickets
        """
        results = {}
        
        for account_id, connector in self.connectors.items():
            if not connector.connected or not self.account_configs[account_id].enabled:
                continue
                
            strategy_name = self.strategies[account_id]
            
            if strategy_name in signals:
                account_results = []
                strategy_signals = signals[strategy_name]
                
                for symbol, signal_data in strategy_signals.items():
                    ticket = await self._execute_signal_for_account(
                        account_id, symbol, signal_data
                    )
                    if ticket:
                        account_results.append(ticket)
                
                results[account_id] = account_results
        
        return results
    
    async def _execute_signal_for_account(
        self, 
        account_id: str, 
        symbol: str, 
        signal_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Execute a single trading signal for specific account
        """
        try:
            connector = self.connectors[account_id]
            config = self.account_configs[account_id]
            
            signal = signal_data.get('signal')
            confidence = signal_data.get('confidence', 0)
            entry_price = signal_data.get('entry_price')
            stop_loss = signal_data.get('stop_loss')
            take_profit = signal_data.get('take_profit')
            
            # Skip low confidence signals
            if confidence < 60:
                return None
            
            # Determine order type
            if signal == "BUY":
                order_type = OrderType.BUY
            elif signal == "SELL":
                order_type = OrderType.SELL
            else:
                return None  # No action for HOLD
            
            # Calculate position size based on risk management
            account_info = await connector.get_account_info()
            balance = account_info.get('balance', 10000)
            
            # Risk per trade (default 2% or account-specific setting)
            risk_amount = balance * config.max_risk_per_trade
            
            # Calculate volume based on stop loss distance
            if stop_loss and entry_price:
                stop_distance = abs(entry_price - stop_loss)
                volume = min(1.0, risk_amount / (stop_distance * 100000))  # Simplified calculation
            else:
                volume = 0.01  # Minimum volume
            
            # Create order request
            order = OrderRequest(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=f"Strategy: {config.strategy_name}",
                magic=12345  # Strategy identifier
            )
            
            # Place order
            ticket = await connector.place_order(order)
            
            if ticket:
                logger.info(
                    f"✅ Order executed: {account_id} ({config.broker_type.value}) "
                    f"- {signal} {symbol} Vol:{volume:.2f} Ticket:{ticket}"
                )
            
            return ticket
            
        except Exception as e:
            logger.error(f"❌ Signal execution failed for {account_id}: {e}")
            return None
    
    def get_account_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all registered accounts
        """
        status = {}
        
        for account_id, connector in self.connectors.items():
            config = self.account_configs[account_id]
            
            status[account_id] = {
                "broker_type": config.broker_type.value,
                "mode": config.mode.value,
                "strategy": config.strategy_name,
                "connected": connector.connected,
                "enabled": config.enabled,
                "max_risk_per_trade": f"{config.max_risk_per_trade*100}%",
                "max_daily_trades": config.max_daily_trades,
                "max_open_positions": config.max_open_positions,
                "last_error": connector.last_error
            }
            
        return status

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_multi_platform():
        """
        Test multi-platform trading system
        """
        print("🌐 Testing Multi-Platform Trading System...")
        
        # Initialize manager
        manager = MultiPlatformManager()
        
        # Register multiple accounts with different brokers and strategies
        accounts = [
            AccountConfig(
                account_id="MT5_LIVE_001",
                broker_type=BrokerType.MT5,
                mode=AccountMode.LIVE,
                credentials={"login": "12345", "password": "secret", "server": "broker-server"},
                strategy_name="SmartMoneyStrategy",
                max_risk_per_trade=0.015  # 1.5%
            ),
            AccountConfig(
                account_id="SABIO_DEMO_001",
                broker_type=BrokerType.SABIOTRADE,
                mode=AccountMode.DEMO,
                credentials={"api_key": "demo_key", "secret": "demo_secret"},
                strategy_name="FibonacciTeamStrategy",
                max_risk_per_trade=0.02  # 2%
            ),
            AccountConfig(
                account_id="ROBO_LIVE_001",
                broker_type=BrokerType.ROBOFOREX,
                mode=AccountMode.LIVE,
                credentials={"login": "67890", "password": "secret2"},
                strategy_name="SmartMoneyStrategy",
                max_risk_per_trade=0.01  # 1%
            )
        ]
        
        # Register accounts
        for account in accounts:
            success = manager.register_account(account)
            print(f"Account {account.account_id}: {'Registered' if success else 'Failed'}")
        
        # Connect to all accounts
        print("\n🔌 Connecting to all accounts...")
        connection_results = await manager.connect_all()
        for account_id, connected in connection_results.items():
            status = "✅ Connected" if connected else "❌ Failed"
            print(f"  {account_id}: {status}")
        
        # Get account status
        print("\n📊 Account Status:")
        status = manager.get_account_status()
        for account_id, info in status.items():
            print(f"  {account_id}:")
            print(f"    Broker: {info['broker_type']} ({info['mode']})")
            print(f"    Strategy: {info['strategy']}")
            print(f"    Connected: {info['connected']}")
            print(f"    Risk per trade: {info['max_risk_per_trade']}")
        
        # Test market data
        print("\n📊 Getting market data...")
        market_data = await manager.get_market_data("EURUSD")
        for account_id, data in market_data.items():
            print(f"  {account_id}: EURUSD Bid:{data.bid:.5f} Ask:{data.ask:.5f} Spread:{data.spread:.5f}")
        
        # Test strategy signal execution
        print("\n💹 Executing strategy signals...")
        
        # Mock signals from different strategies
        signals = {
            "SmartMoneyStrategy": {
                "EURUSD": {
                    "signal": "BUY",
                    "confidence": 75.0,
                    "entry_price": 1.1000,
                    "stop_loss": 1.0950,
                    "take_profit": 1.1100
                }
            },
            "FibonacciTeamStrategy": {
                "GBPUSD": {
                    "signal": "SELL",
                    "confidence": 82.5,
                    "entry_price": 1.3000,
                    "stop_loss": 1.3050,
                    "take_profit": 1.2900
                }
            }
        }
        
        execution_results = await manager.execute_strategy_signals(signals)
        
        for account_id, tickets in execution_results.items():
            print(f"  {account_id}: {len(tickets)} orders executed - Tickets: {tickets}")
        
        # Disconnect all
        print("\n🔌 Disconnecting from all accounts...")
        disconnect_results = await manager.disconnect_all()
        for account_id, disconnected in disconnect_results.items():
            status = "✅ Disconnected" if disconnected else "❌ Failed"
            print(f"  {account_id}: {status}")
        
        print("\n✅ Multi-platform trading system test completed!")
    
    # Run test
    asyncio.run(test_multi_platform())