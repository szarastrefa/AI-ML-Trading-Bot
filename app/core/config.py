"""
AI/ML Trading Bot - Główna konfiguracja systemu
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class BrokerConfig:
    """Konfiguracja brokerów"""
    name: str
    api_url: str
    demo_mode: bool = True
    timeout: int = 30
    max_retries: int = 3

@dataclass 
class DatabaseConfig:
    """Konfiguracja bazy danych"""
    url: str = "sqlite:///data/trading_bot.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20

@dataclass
class RiskManagementConfig:
    """Konfiguracja zarządzania ryzykiem"""
    max_position_size: float = 0.02  # 2% kapitału na transakcję
    default_stop_loss: float = 0.02  # 2% stop loss
    max_daily_loss: float = 0.05     # 5% maksymalnej straty dziennej
    max_drawdown: float = 0.15       # 15% maksymalnego drawdown
    min_risk_reward_ratio: float = 2.0  # Min 1:2 risk/reward

@dataclass
class MLConfig:
    """Konfiguracja modeli ML"""
    model_save_path: str = "data/models/"
    retrain_interval_hours: int = 24
    feature_window: int = 100
    prediction_horizon: int = 5
    ensemble_models: bool = True

class TradingBotConfig:
    """Główna konfiguracja Trading Bot"""
    
    # Podstawowe ustawienia
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    ENV: str = os.getenv("ENV", "development")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    
    # Ścieżki
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = DATA_DIR / "logs"
    MODELS_DIR = DATA_DIR / "models"
    
    # Serwer
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Baza danych
    DATABASE = DatabaseConfig()
    
    # Zarządzanie ryzykiem
    RISK_MANAGEMENT = RiskManagementConfig()
    
    # Machine Learning
    ML_CONFIG = MLConfig()
    
    # Brokerzy obsługiwani przez system
    SUPPORTED_BROKERS = {
        "mt4": BrokerConfig(
            name="MetaTrader 4",
            api_url="localhost:9090"
        ),
        "mt5": BrokerConfig(
            name="MetaTrader 5", 
            api_url="localhost:9091"
        ),
        "roboforex": BrokerConfig(
            name="RoboForex",
            api_url="https://api.roboforex.com"
        ),
        "sabiotrade": BrokerConfig(
            name="SabioTrade",
            api_url="https://api.sabiotrade.com"
        ),
        "xm_group": BrokerConfig(
            name="XM Group",
            api_url="https://api.xmglobal.com"
        ),
        "forexchief": BrokerConfig(
            name="ForexChief (xChief)",
            api_url="https://api.forexchief.com"
        ),
        "fxopen": BrokerConfig(
            name="FXOpen",
            api_url="https://api.fxopen.com"
        ),
        "instaforex": BrokerConfig(
            name="InstaForex",
            api_url="https://api.instaforex.com"
        ),
        "templerfx": BrokerConfig(
            name="TemplerFX",
            api_url="https://api.templerfx.com"
        ),
        "fbs": BrokerConfig(
            name="FBS",
            api_url="https://api.fbs.com"
        ),
        "pocket_option": BrokerConfig(
            name="Pocket Option",
            api_url="https://api.po.market"
        ),
        "the5ers": BrokerConfig(
            name="The5ers",
            api_url="https://api.the5ers.com"
        ),
        "funded_trading_plus": BrokerConfig(
            name="Funded Trading Plus",
            api_url="https://api.fundedtradingplus.com"
        )
    }
    
    # Strategie dostępne w systemie
    AVAILABLE_STRATEGIES = [
        "smart_money_concept",
        "fibonacci_strategy", 
        "scalping_strategy",
        "ml_ensemble_strategy"
    ]
    
    # Interwały czasowe
    TIMEFRAMES = {
        "1m": "1 minuta",
        "5m": "5 minut", 
        "15m": "15 minut",
        "30m": "30 minut",
        "1h": "1 godzina",
        "4h": "4 godziny", 
        "1d": "1 dzień",
        "1w": "1 tydzień"
    }
    
    @classmethod
    def create_directories(cls):
        """Tworzy potrzebne katalogi"""
        for directory in [cls.DATA_DIR, cls.LOGS_DIR, cls.MODELS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

# Instancja globalnej konfiguracji
config = TradingBotConfig()
config.create_directories()