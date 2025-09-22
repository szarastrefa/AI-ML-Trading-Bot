"""
AI/ML Trading Bot - Main Application (FIXED VERSION)
All import and structural issues resolved
"""

import os
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

# Add app directory to Python path for imports
sys.path.insert(0, '/app')

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create required directories"""
    directories = [
        "data/logs", "data/models", "data/historical", 
        "data/backtest", "data/live", "data/cache", "logs", "tmp"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info("? Directory structure created")

def verify_dependencies():
    """Verify all critical dependencies"""
    try:
        import numpy as np
        import pandas as pd
        logger.info(f"? NumPy: {np.__version__}")
        logger.info(f"? Pandas: {pd.__version__}")
        
        try:
            import talib
            logger.info("? TA-Lib: Available and working")
            
            # Test TA-Lib functionality
            test_data = np.array([100.0, 101.0, 102.0, 101.5, 103.0], dtype=np.float64)
            sma_result = talib.SMA(test_data, timeperiod=3)
            logger.info(f"? TA-Lib test SMA: {sma_result[-1]:.2f}")
            
        except ImportError:
            logger.warning("?? TA-Lib not available - using fallback implementations")
        except Exception as e:
            logger.warning(f"?? TA-Lib test failed: {e} - using fallback")
        
        # Test strategy import
        try:
            from app.strategies.talib_stable_strategy import TALibStableStrategy
            logger.info("? Strategy import successful")
            
            # Test strategy instantiation
            strategy = TALibStableStrategy({})
            logger.info(f"? Strategy created: {strategy.name}")
            
        except ImportError as e:
            logger.error(f"? Strategy import failed: {e}")
            raise
        except Exception as e:
            logger.error(f"? Strategy creation failed: {e}")
            raise
        
        logger.info("?? All dependency verification completed successfully")
        
    except Exception as e:
        logger.error(f"? Dependency verification failed: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("?? Starting AI/ML Trading Bot v2.1 (FIXED)")
    
    try:
        create_directories()
        verify_dependencies()
        logger.info("? System initialization completed successfully")
    except Exception as e:
        logger.error(f"? System initialization failed: {str(e)}")
        raise
    
    yield
    
    logger.info("?? Shutting down AI/ML Trading Bot...")

# Create FastAPI application
app = FastAPI(
    title="AI/ML Trading Bot",
    description="Fixed Trading Bot with TA-Lib Support",
    version="2.1.0",
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
    """Root endpoint with system information"""
    return {
        "message": "AI/ML Trading Bot v2.1 - FIXED & OPERATIONAL",
        "version": "2.1.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "features": {
            "technical_analysis": "TA-Lib with fallback",
            "strategies": ["TALibStableStrategy"],
            "data_sources": "Mock data (implement real sources)",
            "risk_management": "ATR-based position sizing"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        # System checks
        import numpy as np
        import pandas as pd
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "dependencies": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "fastapi": "working"
            },
            "system": {
                "python_path": sys.path[0],
                "working_directory": os.getcwd()
            }
        }
        
        # TA-Lib check
        try:
            import talib
            test_data = np.array([100.0, 101.0, 102.0, 101.5, 103.0], dtype=np.float64)
            sma_result = talib.SMA(test_data, timeperiod=3)
            health_data["dependencies"]["talib"] = "working"
            health_data["talib_test"] = f"SMA(3) = {sma_result[-1]:.2f}"
        except ImportError:
            health_data["dependencies"]["talib"] = "not_available_using_fallback"
        except Exception as e:
            health_data["dependencies"]["talib"] = f"error: {str(e)}"
        
        # Strategy check
        try:
            from app.strategies.talib_stable_strategy import TALibStableStrategy
            strategy = TALibStableStrategy({})
            health_data["strategy"] = {
                "name": strategy.name,
                "version": strategy.version,
                "status": "ready"
            }
        except Exception as e:
            health_data["strategy"] = {"error": str(e)}
        
        return health_data
        
    except Exception as e:
        logger.error(f"? Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.post("/api/v1/analyze")
async def analyze_symbol(
    symbol: str = Query("EURUSD", description="Trading symbol"),
    timeframe: str = Query("H1", description="Chart timeframe")
):
    """
    Analyze trading symbol and generate signals
    """
    try:
        logger.info(f"?? Starting analysis: {symbol} {timeframe}")
        
        # Import and initialize strategy
        from app.strategies.talib_stable_strategy import TALibStableStrategy
        
        config = {
            "risk_management": {
                "max_risk_per_trade": 0.02,
                "stop_loss_atr_multiplier": 2.0,
                "take_profit_atr_multiplier": 3.0
            }
        }
        
        strategy = TALibStableStrategy(config)
        
        # Perform analysis
        start_time = datetime.utcnow()
        result = await strategy.analyze(symbol, timeframe)
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"? Analysis completed in {execution_time:.3f}s: {result.get('signal', 'UNKNOWN')}")
        
        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy.name,
            "execution_time": execution_time,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"? Analysis failed for {symbol}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/api/v1/strategy/info")
async def get_strategy_info():
    """Get detailed strategy information"""
    try:
        from app.strategies.talib_stable_strategy import TALibStableStrategy
        
        strategy = TALibStableStrategy({})
        stats = strategy.get_performance_stats()
        
        return {
            "success": True,
            "strategy_info": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"? Strategy info failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/v1/indicators/test")
async def test_indicators():
    """Test TA-Lib indicators functionality"""
    try:
        import numpy as np
        
        # Generate test data
        test_data = np.array([100.0, 101.0, 102.0, 101.5, 103.0, 102.8, 104.0, 103.5, 105.0, 104.2], dtype=np.float64)
        
        results = {}
        
        try:
            import talib
            results["sma"] = talib.SMA(test_data, timeperiod=3)[-1]
            results["ema"] = talib.EMA(test_data, timeperiod=3)[-1] 
            results["rsi"] = talib.RSI(test_data, timeperiod=3)[-1]
            results["talib_status"] = "working"
        except ImportError:
            # Use fallback implementations
            sma = np.convolve(test_data, np.ones(3)/3, mode='valid')
            results["sma"] = sma[-1] if len(sma) > 0 else None
            results["talib_status"] = "using_fallback"
        except Exception as e:
            results["talib_status"] = f"error: {str(e)}"
        
        return {
            "success": True,
            "test_data_length": len(test_data),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/v1/system/status")
async def system_status():
    """Get comprehensive system status"""
    try:
        from app.strategies.talib_stable_strategy import TALibStableStrategy
        
        # Create strategy instance
        strategy = TALibStableStrategy({})
        
        # Perform quick test analysis
        test_result = await strategy.analyze("TEST", "H1")
        
        return {
            "success": True,
            "system": {
                "status": "operational",
                "uptime": "running",
                "memory_usage": "normal"
            },
            "strategy": {
                "name": strategy.name,
                "version": strategy.version,
                "analysis_count": strategy.analysis_count
            },
            "test_analysis": {
                "signal": test_result.get("signal", "ERROR"),
                "confidence": test_result.get("confidence", 0),
                "execution_successful": "error" not in test_result
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

if __name__ == "__main__":
    # Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    logger.info(f"?? Starting AI/ML Trading Bot v2.1 on {HOST}:{PORT}")
    logger.info(f"?? Debug mode: {DEBUG}")
    logger.info(f"?? Working directory: {os.getcwd()}")
    
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info"
    )
