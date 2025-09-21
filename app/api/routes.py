"""
API Routes dla Trading Bot
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any

router = APIRouter()

# Mock data for demo
@router.get("/brokers")
async def get_brokers():
    """Pobierz listę dostępnych brokerów"""
    return {
        "brokers": ["demo", "mt5", "roboforex"],
        "active_broker": "demo",
        "supported": ["mt4", "mt5", "roboforex", "sabiotrade", "xm_group", "forexchief", "fxopen", "instaforex", "templerfx", "fbs", "pocket_option", "the5ers", "funded_trading_plus"]
    }

@router.get("/account")
async def get_account_info():
    """Pobierz informacje o koncie"""
    return {
        "balance": 10000.0,
        "equity": 10250.0,
        "margin": 500.0,
        "free_margin": 9750.0,
        "margin_level": 2050.0,
        "currency": "USD"
    }

@router.get("/positions")
async def get_positions():
    """Pobierz otwarte pozycje"""
    return {
        "positions": [
            {
                "symbol": "EURUSD",
                "side": "buy",
                "size": 0.1,
                "entry_price": 1.0950,
                "current_price": 1.0975,
                "pnl": 25.0,
                "timestamp": "2024-01-21T10:30:00Z"
            }
        ]
    }

@router.get("/strategies")
async def get_strategies():
    """Pobierz dostępne strategie"""
    return {
        "available_strategies": ["fibonacci_team", "smart_money_concept", "scalping_strategy"],
        "active_strategies": ["fibonacci_team"]
    }

@router.post("/strategies/{strategy_name}/start")
async def start_strategy(strategy_name: str, strategy_config: dict = None):
    """Uruchom strategię"""
    return {"message": f"Strategy {strategy_name} started", "status": "success"}

@router.post("/strategies/{strategy_name}/stop") 
async def stop_strategy(strategy_name: str):
    """Zatrzymaj strategię"""
    return {"message": f"Strategy {strategy_name} stopped", "status": "success"}

@router.get("/analytics/performance")
async def get_performance_analytics():
    """Pobierz analityki wydajności"""
    return {
        "total_trades": 45,
        "win_rate": 67.5,
        "profit_factor": 1.85,
        "total_return": 12.5,
        "max_drawdown": -3.2
    }

@router.get("/analytics/pnl")
async def get_pnl_data(period: str = "1w"):
    """Pobierz dane P&L"""
    return {
        "period": period,
        "data": [
            {"date": "2024-01-15", "pnl": 100},
            {"date": "2024-01-16", "pnl": 150},
            {"date": "2024-01-17", "pnl": 120},
            {"date": "2024-01-18", "pnl": 200},
            {"date": "2024-01-19", "pnl": 180},
            {"date": "2024-01-20", "pnl": 250},
            {"date": "2024-01-21", "pnl": 300}
        ]
    }
