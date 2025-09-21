"""
Serwis brokerski - obsługa platform tradingowych
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Reprezentacja pozycji"""
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    pnl: float

@dataclass
class AccountInfo:
    """Informacje o koncie"""
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    currency: str

class BaseBroker(ABC):
    """Bazowa klasa dla wszystkich brokerów"""

    def __init__(self, config: dict, demo_mode: bool = True):
        self.config = config
        self.demo_mode = demo_mode
        self.is_connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Nawiąż połączenie z brokerem"""
        pass

    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """Pobierz informacje o koncie"""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Pobierz otwarte pozycje"""
        pass

class DemoBroker(BaseBroker):
    """Demo broker do testowania"""

    async def connect(self) -> bool:
        """Połączenie demo"""
        self.is_connected = True
        logger.info("Demo broker connected")
        return True

    async def get_account_info(self) -> AccountInfo:
        """Demo informacje o koncie"""
        return AccountInfo(
            balance=10000.0,
            equity=10250.0,
            margin=500.0,
            free_margin=9750.0,
            margin_level=2050.0,
            currency="USD"
        )

    async def get_positions(self) -> List[Position]:
        """Demo pozycje"""
        return [
            Position(
                symbol="EURUSD",
                side="buy",
                size=0.1,
                entry_price=1.0950,
                current_price=1.0975,
                pnl=25.0
            )
        ]

class BrokerService:
    """Główny serwis zarządzania brokerami"""

    def __init__(self):
        self.brokers: Dict[str, BaseBroker] = {}
        self.active_broker: Optional[str] = None

    async def add_broker(self, name: str, broker_config: dict, demo_mode: bool = True) -> bool:
        """Dodaj brokera do systemu"""
        try:
            # W pełnej wersji tutaj byłby BrokerFactory
            broker = DemoBroker(broker_config, demo_mode)

            if await broker.connect():
                self.brokers[name] = broker
                if not self.active_broker:
                    self.active_broker = name
                logger.info(f"Broker {name} added successfully")
                return True
            else:
                logger.error(f"Failed to connect to broker {name}")
                return False

        except Exception as e:
            logger.error(f"Error adding broker {name}: {e}")
            return False

    async def get_current_broker(self) -> Optional[BaseBroker]:
        """Pobierz aktualnego brokera"""
        if self.active_broker and self.active_broker in self.brokers:
            return self.brokers[self.active_broker]
        return None

    async def get_account_info(self) -> Optional[AccountInfo]:
        """Pobierz informacje o koncie z aktywnego brokera"""
        broker = await self.get_current_broker()
        if broker:
            return await broker.get_account_info()
        return None

    async def get_positions(self) -> List[Position]:
        """Pobierz pozycje z aktywnego brokera"""
        broker = await self.get_current_broker()
        if broker:
            return await broker.get_positions()
        return []

# Instancja globalna serwisu brokerskiego
broker_service = BrokerService()
