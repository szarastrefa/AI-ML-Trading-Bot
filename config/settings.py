"""
Konfiguracja brokerów i parametrów tradingowych
"""

# Konfiguracja brokerów
BROKER_CONFIGS = {
    "mt5": {
        "enabled": True,
        "name": "MetaTrader 5",
        "server": "MetaQuotes-Demo",
        "demo_mode": True
    },
    "roboforex": {
        "enabled": True,
        "name": "RoboForex",
        "api_url": "https://api.roboforex.com",
        "demo_mode": True
    },
    "sabiotrade": {
        "enabled": True,
        "name": "SabioTrade",
        "api_url": "https://api.sabiotrade.com",
        "sandbox": True
    },
    "xm_group": {
        "enabled": True,
        "name": "XM Group",
        "api_url": "https://api.xmglobal.com",
        "demo_mode": True
    },
    "forexchief": {
        "enabled": True,
        "name": "ForexChief (xChief)",
        "api_url": "https://api.forexchief.com",
        "demo_mode": True
    },
    "fxopen": {
        "enabled": True,
        "name": "FXOpen",
        "api_url": "https://api.fxopen.com",
        "demo_mode": True
    },
    "instaforex": {
        "enabled": True,
        "name": "InstaForex",
        "api_url": "https://api.instaforex.com",
        "demo_mode": True
    },
    "templerfx": {
        "enabled": True,
        "name": "TemplerFX",
        "api_url": "https://api.templerfx.com",
        "sandbox": True
    },
    "fbs": {
        "enabled": True,
        "name": "FBS",
        "api_url": "https://api.fbs.com",
        "demo_mode": True
    },
    "pocket_option": {
        "enabled": True,
        "name": "Pocket Option",
        "api_url": "https://api.po.market",
        "demo": True
    },
    "the5ers": {
        "enabled": True,
        "name": "The5ers",
        "api_url": "https://api.the5ers.com",
        "sandbox": True
    },
    "funded_trading_plus": {
        "enabled": True,
        "name": "Funded Trading Plus",
        "api_url": "https://api.fundedtradingplus.com",
        "sandbox": True
    }
}

# Zarządzanie ryzykiem
RISK_MANAGEMENT = {
    "max_risk_per_trade": 0.02,  # 2% kapitału na transakcję
    "stop_loss_pct": 0.02,       # 2% stop loss
    "take_profit_ratio": 2.0,    # Risk/Reward 1:2
    "max_daily_loss": 0.06,      # 6% max strata dzienna
    "max_drawdown": 0.20         # 20% max drawdown
}

# Sesje tradingowe
TRADING_SESSIONS = {
    "london": {"start": 8, "end": 17},
    "newyork": {"start": 13, "end": 22},
    "overlap": {"start": 13, "end": 17}
}

# Strategie
STRATEGIES = {
    "smart_money_concept": {
        "enabled": True,
        "timeframes": ["15m", "1h", "4h"],
        "risk_reward": 3.0
    },
    "fibonacci_team": {
        "enabled": True,
        "timeframes": ["5m", "15m", "1h"],
        "risk_reward": 2.0
    }
}