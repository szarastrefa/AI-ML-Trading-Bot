# -*- coding: utf-8 -*-
"""
Database Models for Multi-Account Trading System
Supports multiple trading accounts, strategies, and ML models
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
import json
import logging
from typing import Dict, List, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

Base = declarative_base()

class AccountStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    ERROR = "error"

class OrderType(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    FAILED = "failed"

class TradingAccount(Base):
    """
    Trading Account model - supports multiple accounts per user
    Each account can run different strategies and ML models
    """
    __tablename__ = 'trading_accounts'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)  # User-friendly name
    platform = Column(String(50), nullable=False)  # MT4/MT5, Sabiotrade, etc.
    account_id = Column(String(100), nullable=False)  # Platform account ID
    api_key = Column(String(500))  # Encrypted API key
    api_secret = Column(String(500))  # Encrypted API secret
    server_url = Column(String(200))  # Platform server URL
    
    # Account settings
    balance = Column(Float, default=0.0)
    equity = Column(Float, default=0.0)
    margin = Column(Float, default=0.0)
    free_margin = Column(Float, default=0.0)
    
    # Risk management
    max_risk_per_trade = Column(Float, default=2.0)  # Percentage
    max_daily_trades = Column(Integer, default=10)
    max_open_positions = Column(Integer, default=5)
    
    # Status and settings
    status = Column(String(20), default=AccountStatus.ACTIVE.value)
    is_demo = Column(Boolean, default=True)
    auto_trading = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_sync = Column(DateTime)
    
    # Relationships
    strategies = relationship("AccountStrategy", back_populates="account")
    orders = relationship("TradingOrder", back_populates="account")
    ml_models = relationship("MLModel", back_populates="account")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'platform': self.platform,
            'account_id': self.account_id,
            'balance': self.balance,
            'equity': self.equity,
            'status': self.status,
            'is_demo': self.is_demo,
            'auto_trading': self.auto_trading,
            'max_risk_per_trade': self.max_risk_per_trade,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_sync': self.last_sync.isoformat() if self.last_sync else None
        }

class TradingStrategy(Base):
    """
    Trading Strategy definitions (Smart Money, Fibonacci Team, ML Ensemble)
    """
    __tablename__ = 'trading_strategies'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)  # Strategy name
    type = Column(String(50), nullable=False)  # "smart_money", "fibonacci_team", "ml_ensemble"
    description = Column(Text)
    
    # Strategy parameters (JSON)
    parameters = Column(JSON)  # Strategy-specific parameters
    
    # Performance tracking
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    total_profit = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    account_strategies = relationship("AccountStrategy", back_populates="strategy")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'description': self.description,
            'parameters': self.parameters,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'total_profit': self.total_profit,
            'is_active': self.is_active
        }

class AccountStrategy(Base):
    """
    Many-to-Many relationship between Accounts and Strategies
    Allows multiple strategies per account with different settings
    """
    __tablename__ = 'account_strategies'
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('trading_accounts.id'), nullable=False)
    strategy_id = Column(Integer, ForeignKey('trading_strategies.id'), nullable=False)
    
    # Strategy-specific settings for this account
    position_size = Column(Float, default=0.01)  # Lot size or position size
    stop_loss_pct = Column(Float, default=2.0)   # Stop loss percentage
    take_profit_pct = Column(Float, default=4.0) # Take profit percentage
    max_positions = Column(Integer, default=1)   # Max positions for this strategy
    
    # Performance for this account-strategy combination
    trades_count = Column(Integer, default=0)
    profit_loss = Column(Float, default=0.0)
    last_signal = Column(String(10))  # Last signal: BUY, SELL, HOLD
    last_signal_time = Column(DateTime)
    
    # Status
    is_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    account = relationship("TradingAccount", back_populates="strategies")
    strategy = relationship("TradingStrategy", back_populates="account_strategies")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'account_id': self.account_id,
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy.name if self.strategy else None,
            'position_size': self.position_size,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'trades_count': self.trades_count,
            'profit_loss': self.profit_loss,
            'last_signal': self.last_signal,
            'is_enabled': self.is_enabled
        }

class MLModel(Base):
    """
    Machine Learning Models for each account
    Supports RandomForest, LSTM, and custom models
    """
    __tablename__ = 'ml_models'
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('trading_accounts.id'), nullable=False)
    
    name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # "random_forest", "lstm", "ensemble"
    version = Column(String(20), default="1.0")
    
    # Model metadata
    features_count = Column(Integer)
    training_samples = Column(Integer)
    validation_accuracy = Column(Float)
    training_date = Column(DateTime)
    
    # Model parameters and performance
    hyperparameters = Column(JSON)  # Model hyperparameters
    performance_metrics = Column(JSON)  # Accuracy, precision, recall, etc.
    feature_importance = Column(JSON)  # Feature importance scores
    
    # Model file path (for saving/loading)
    model_path = Column(String(500))  # Path to saved model file
    scaler_path = Column(String(500))  # Path to saved scaler
    
    # Status
    is_active = Column(Boolean, default=True)
    is_trained = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_prediction = Column(DateTime)
    
    # Relationships
    account = relationship("TradingAccount", back_populates="ml_models")
    predictions = relationship("MLPrediction", back_populates="model")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'account_id': self.account_id,
            'name': self.name,
            'model_type': self.model_type,
            'version': self.version,
            'features_count': self.features_count,
            'training_samples': self.training_samples,
            'validation_accuracy': self.validation_accuracy,
            'is_active': self.is_active,
            'is_trained': self.is_trained,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'last_prediction': self.last_prediction.isoformat() if self.last_prediction else None,
            'performance_metrics': self.performance_metrics
        }

class MLPrediction(Base):
    """
    ML Model predictions history
    """
    __tablename__ = 'ml_predictions'
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey('ml_models.id'), nullable=False)
    
    symbol = Column(String(20), nullable=False)
    prediction = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float, nullable=False)
    
    # Input features used for prediction
    features = Column(JSON)
    
    # Actual result (if available)
    actual_result = Column(Float)  # Actual price change
    was_correct = Column(Boolean)  # Was prediction correct?
    
    # Timestamps
    prediction_time = Column(DateTime, default=datetime.utcnow)
    evaluation_time = Column(DateTime)  # When actual result was recorded
    
    # Relationships
    model = relationship("MLModel", back_populates="predictions")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'model_id': self.model_id,
            'symbol': self.symbol,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'actual_result': self.actual_result,
            'was_correct': self.was_correct,
            'prediction_time': self.prediction_time.isoformat(),
            'evaluation_time': self.evaluation_time.isoformat() if self.evaluation_time else None
        }

class TradingOrder(Base):
    """
    Trading Orders placed by the system
    """
    __tablename__ = 'trading_orders'
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey('trading_accounts.id'), nullable=False)
    
    # Order details
    symbol = Column(String(20), nullable=False)
    order_type = Column(String(10), nullable=False)  # BUY, SELL
    volume = Column(Float, nullable=False)
    
    # Prices
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    exit_price = Column(Float)
    
    # Order management
    platform_order_id = Column(String(100))  # Order ID from trading platform
    status = Column(String(20), default=OrderStatus.PENDING.value)
    
    # Strategy and ML info
    strategy_name = Column(String(100))
    ml_model_id = Column(Integer, ForeignKey('ml_models.id'))
    ml_confidence = Column(Float)
    
    # P&L
    profit_loss = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    swap = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime)
    closed_at = Column(DateTime)
    
    # Relationships
    account = relationship("TradingAccount", back_populates="orders")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'account_id': self.account_id,
            'symbol': self.symbol,
            'order_type': self.order_type,
            'volume': self.volume,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'exit_price': self.exit_price,
            'status': self.status,
            'strategy_name': self.strategy_name,
            'ml_confidence': self.ml_confidence,
            'profit_loss': self.profit_loss,
            'created_at': self.created_at.isoformat(),
            'executed_at': self.executed_at.isoformat() if self.executed_at else None,
            'closed_at': self.closed_at.isoformat() if self.closed_at else None
        }

class SystemLog(Base):
    """
    System logs for debugging and monitoring
    """
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    level = Column(String(10), nullable=False)  # INFO, WARNING, ERROR
    message = Column(Text, nullable=False)
    component = Column(String(50))  # ml_system, trading_engine, etc.
    account_id = Column(Integer, ForeignKey('trading_accounts.id'))
    
    # Additional context
    context = Column(JSON)  # Additional context data
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'level': self.level,
            'message': self.message,
            'component': self.component,
            'account_id': self.account_id,
            'context': self.context,
            'created_at': self.created_at.isoformat()
        }

# Database manager class
class DatabaseManager:
    """
    Database manager for multi-account trading system
    """
    
    def __init__(self, database_url: str = "sqlite:///trading_bot.db"):
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("✅ Database tables created")
        
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
        
    def init_default_strategies(self):
        """Initialize default trading strategies"""
        session = self.get_session()
        try:
            # Check if strategies already exist
            existing = session.query(TradingStrategy).count()
            if existing > 0:
                logger.info(f"📊 Found {existing} existing strategies")
                return
            
            # Create default strategies
            strategies = [
                TradingStrategy(
                    name="Smart Money Concepts",
                    type="smart_money",
                    description="Institutional trading concepts: Order Blocks, Fair Value Gaps, Break of Structure",
                    parameters={
                        "structure_lookback": 10,
                        "ob_threshold": 0.002,
                        "fvg_threshold": 0.001,
                        "liquidity_threshold": 0.0015
                    }
                ),
                TradingStrategy(
                    name="Fibonacci Team",
                    type="fibonacci_team", 
                    description="Fibonacci Team methodology: Harmonic patterns, 2% SL, Fibonacci levels",
                    parameters={
                        "default_stop_loss_pct": 2.0,
                        "min_risk_reward_ratio": 2.0,
                        "harmonic_patterns": ["Gartley", "Bat", "Butterfly", "Crab"]
                    }
                ),
                TradingStrategy(
                    name="ML Ensemble",
                    type="ml_ensemble",
                    description="Machine Learning ensemble: RandomForest + LSTM + Online Learning",
                    parameters={
                        "models": ["random_forest", "lstm"],
                        "ensemble_method": "voting",
                        "min_confidence": 0.7
                    }
                )
            ]
            
            for strategy in strategies:
                session.add(strategy)
            
            session.commit()
            logger.info("✅ Default strategies created")
            
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error creating default strategies: {e}")
        finally:
            session.close()

# Global database manager instance
db_manager = DatabaseManager()

logger.info("🗄️ Database models loaded successfully")