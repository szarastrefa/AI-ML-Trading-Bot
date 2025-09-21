"""
AI/ML Trading Bot - Main FastAPI Application (Fixed)
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create required directories
def create_directories():
    """Create required application directories"""
    dirs = [
        "data/logs",
        "data/models", 
        "data/historical",
        "data/backtest",
        "data/live",
        "data/cache",
        "logs"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    logger.info("Directories created successfully")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting AI/ML Trading Bot v2.0...")
    create_directories()
    logger.info("All services started successfully")
    yield
    # Shutdown
    logger.info("Shutting down AI/ML Trading Bot...")

# Create FastAPI application
app = FastAPI(
    title="AI/ML Trading Bot",
    description="Advanced Trading Bot with AI/ML, pandas_ta, and Smart Money Concepts",
    version="2.0.1",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI/ML Trading Bot API v2.0.1 (Fixed)",
        "version": "2.0.1",
        "status": "running",
        "features": [
            "pandas_ta indicators (stable)",
            "Smart Money Concepts",
            "Multi-timeframe analysis",
            "TensorFlow 2.15 compatible"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        import pandas_ta as ta
        import tensorflow as tf
        import pandas as pd
        import numpy as np
        
        return {
            "status": "healthy",
            "timestamp": pd.Timestamp.now().isoformat(),
            "dependencies": {
                "pandas_ta": getattr(ta, 'version', 'stable'),
                "tensorflow": tf.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__
            },
            "environment": os.getenv("ENV", "development")
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

@app.get("/info")
async def app_info():
    """Application information"""
    return {
        "name": "AI/ML Trading Bot",
        "version": "2.0.1",
        "description": "Fixed version with stable dependencies",
        "supported_indicators": "130+ via pandas_ta",
        "smart_money_concepts": True,
        "machine_learning": "TensorFlow 2.15.0",
        "supported_brokers": ["roboforex", "binance", "ccxt"],
        "timeframes": ["M1", "M5", "M15", "H1", "H4", "D1"],
        "strategies": ["pandas_ta_advanced", "smart_money_concepts"]
    }

@app.post("/api/v1/analyze")
async def analyze_symbol(
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    strategy: str = "pandas_ta_classic"
):
    """
    Analyze trading symbol and generate signal
    """
    try:
        # Import strategy
        from app.strategies.pandas_ta_classic_strategy import PandasTAClassicStrategy
        
        # Initialize strategy
        config = {
            'timeframes': ['M15', 'H1', 'H4'],
            'min_confluence_count': 2,
            'smart_money_concepts': {
                'swing_period': 10,
                'lookback': 50
            },
            'risk_management': {
                'max_risk_per_trade': 0.02
            }
        }
        
        strategy_instance = PandasTAClassicStrategy(config)
        
        # Run analysis (will use placeholder data for now)
        result = await strategy_instance.analyze(symbol, timeframe)
        
        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": "pandas_ta_classic_v2.0.1",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "Analysis failed - check logs for details"
            }
        )

@app.get("/api/v1/performance")
async def get_performance():
    """Get strategy performance statistics"""
    try:
        from app.strategies.pandas_ta_classic_strategy import PandasTAClassicStrategy
        
        config = {'timeframes': ['H1'], 'min_confluence_count': 1}
        strategy = PandasTAClassicStrategy(config)
        
        stats = strategy.get_performance_stats()
        
        return {
            "success": True,
            "performance": stats,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

if __name__ == "__main__":
    # Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    logger.info(f"Starting server on {HOST}:{PORT} (debug={DEBUG})")
    
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info"
    )
