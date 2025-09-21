"""
AI/ML Trading Bot - Główna aplikacja FastAPI
"""

import asyncio
import logging
import logging.config
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

from app.core.config import config
from app.api.routes import router as api_router

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Zarządzanie cyklem życia aplikacji"""

    # Startup
    logger.info("Starting AI/ML Trading Bot...")

    # Utwórz katalogi
    config.create_directories()
    logger.info("Directories created")

    logger.info("All services started successfully")
    yield

    # Shutdown
    logger.info("Shutting down AI/ML Trading Bot...")
    logger.info("All services stopped")

# Utwórz aplikację FastAPI
app = FastAPI(
    title="AI/ML Trading Bot",
    description="Zaawansowany bot tradingowy z AI/ML obsługujący MT4/MT5, Forex, Crypto",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # W produkcji ograniczyć do konkretnych domen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Główny endpoint"""
    return {
        "message": "AI/ML Trading Bot API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": config.ENV
    }

@app.get("/info")
async def app_info():
    """Informacje o aplikacji"""
    return {
        "name": "AI/ML Trading Bot",
        "version": "1.0.0",
        "environment": config.ENV,
        "supported_brokers": list(config.SUPPORTED_BROKERS.keys()),
        "available_strategies": config.AVAILABLE_STRATEGIES,
        "timeframes": config.TIMEFRAMES
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG
    )
