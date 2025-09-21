"""
Modern Configuration for AI/ML Trading Bot v2.0
with pandas-ta-classic, Smart Money Concepts, and Advanced ML Pipeline
"""
import os
from typing import Dict, List, Optional
from datetime import datetime

# =============================================================================
# TRADING CONFIGURATION
# =============================================================================

TRADING_CONFIG = {
    # Risk Management - Conservative but Profitable
    "risk_management": {
        "max_risk_per_trade": 0.02,        # 2% risk per trade
        "stop_loss_pct": 0.02,             # 2% stop loss
        "take_profit_ratio": 2.0,          # 1:2 minimum risk/reward
        "max_daily_loss": 0.06,            # 6% max daily drawdown
        "max_weekly_loss": 0.15,           # 15% max weekly drawdown
        "max_monthly_drawdown": 0.25,      # 25% max monthly drawdown
        "position_sizing": "kelly",         # kelly, fixed, percent_risk
        "max_open_positions": 3,           # Max concurrent positions
        "correlation_limit": 0.7,          # Max correlation between positions
        "portfolio_heat": 0.06             # Max 6% portfolio risk
    },
    
    # Modern Strategy Configuration - pandas-ta-classic
    "strategies": {
        "pandas_ta_classic_v2": {
            "enabled": True,
            "priority": 1,
            "timeframes": ["M15", "H1", "H4", "D1"],
            "multi_timeframe_confluence": True,
            "min_confluence_count": 2,
            
            # Technical Indicators (150+ available in pandas-ta-classic)
            "indicators": {
                # Trend Analysis
                "moving_averages": {
                    "sma_periods": [20, 50, 100, 200],
                    "ema_periods": [8, 12, 21, 26, 50],
                    "wma_periods": [14, 21],
                    "vwma_periods": [20],
                    "hma_period": 21,         # Hull Moving Average
                    "alma_period": 14,        # Arnaud Legoux MA
                    "kama_period": 14         # Kaufman Adaptive MA
                },
                
                "trend_indicators": {
                    "supertrend": {"length": 10, "multiplier": 3.0},
                    "parabolic_sar": {"af": 0.02, "max_af": 0.2},
                    "adx": {"length": 14, "lensig": 14},
                    "aroon": {"length": 14},
                    "vortex": {"length": 14}
                },
                
                # Momentum Oscillators
                "momentum": {
                    "rsi": {"length": 14, "overbought": 70, "oversold": 30},
                    "macd": {"fast": 12, "slow": 26, "signal": 9},
                    "stochastic": {"k": 14, "d": 3, "smooth_k": 3},
                    "williams_r": {"length": 14},
                    "cci": {"length": 20},
                    "roc": {"length": 10},
                    "awesome_oscillator": {},
                    "tsi": {"fast": 25, "slow": 13, "signal": 13},
                    "ultimate_oscillator": {"short": 7, "medium": 14, "long": 28}
                },
                
                # Volume Analysis
                "volume": {
                    "obv": {},                 # On Balance Volume
                    "ad": {},                  # Accumulation/Distribution
                    "cmf": {"length": 20},     # Chaikin Money Flow
                    "mfi": {"length": 14},     # Money Flow Index
                    "vwap": {"anchor": "D"},   # Volume Weighted Average Price
                    "pvt": {},                 # Price Volume Trend
                    "eom": {"length": 14},     # Ease of Movement
                    "kvo": {"fast": 34, "slow": 55, "signal": 13}  # Klinger Volume
                },
                
                # Volatility Indicators
                "volatility": {
                    "bollinger_bands": {"length": 20, "std": 2.0},
                    "keltner_channels": {"length": 20, "scalar": 2.0},
                    "donchian_channels": {"lower_length": 20, "upper_length": 20},
                    "atr": {"length": 14},
                    "natr": {"length": 14},    # Normalized ATR
                    "true_range": {},
                    "ulcer_index": {"length": 14}
                },
                
                # Smart Money Concepts (Custom Implementation)
                "smart_money_concepts": {
                    "order_blocks": {
                        "enabled": True,
                        "lookback": 20,
                        "volume_threshold": 1.5,
                        "body_size_threshold": 1.2,
                        "wick_ratio": 0.3
                    },
                    "fair_value_gaps": {
                        "enabled": True,
                        "min_gap_size": 0.1,
                        "max_age_periods": 10,
                        "require_volume_confirmation": True
                    },
                    "break_of_structure": {
                        "enabled": True,
                        "swing_period": 10,
                        "confirmation_periods": 2
                    },
                    "change_of_character": {
                        "enabled": True,
                        "momentum_threshold": 0.02,
                        "volume_confirmation": True
                    },
                    "liquidity_sweeps": {
                        "enabled": True,
                        "lookback": 50,
                        "threshold": 0.02,
                        "reversal_confirmation": True
                    }
                }
            },
            
            # Signal Generation Thresholds
            "signal_thresholds": {
                "trend_strength_min": 65,         # Minimum trend strength (0-100)
                "momentum_min": 60,               # Minimum momentum score
                "volume_confirmation_min": 70,    # Volume confirmation threshold
                "smc_strength_min": 2,            # Smart Money Concepts strength
                "confluence_min": 2,              # Minimum timeframe confluence
                "risk_reward_min": 1.5,           # Minimum R:R ratio
                "max_spread_pips": 3              # Max spread for entry
            },
            
            # Position Management
            "position_management": {
                "entry_method": "limit",          # market, limit, stop
                "partial_closes": [0.5, 0.3, 0.2], # Close portions at TP levels
                "trailing_stop": {
                    "enabled": True,
                    "trigger_ratio": 1.0,         # Start trailing at 1:1
                    "step_size": 0.5              # Trail by 0.5x ATR
                },
                "break_even": {
                    "enabled": True,
                    "trigger_ratio": 1.0          # Move SL to BE at 1:1
                }
            }
        },
        
        # Legacy strategies (disabled in favor of pandas-ta-classic)
        "fibonacci_team": {
            "enabled": False,
            "deprecated": True,
            "replacement": "pandas_ta_classic_v2",
            "note": "Integrated into modern pandas-ta-classic strategy"
        },
        
        "smart_money_concept_legacy": {
            "enabled": False,
            "deprecated": True,
            "replacement": "pandas_ta_classic_v2.smart_money_concepts",
            "note": "Enhanced SMC features now built into pandas-ta-classic strategy"
        }
    },
    
    # Trading Instruments with Specifications
    "instruments": {
        "forex": {
            "major_pairs": {
                "EURUSD": {"pip_value": 0.0001, "spread_avg": 1.2, "session_best": "london_ny"},
                "GBPUSD": {"pip_value": 0.0001, "spread_avg": 1.8, "session_best": "london"},
                "USDJPY": {"pip_value": 0.01, "spread_avg": 1.1, "session_best": "tokyo_london"},
                "USDCHF": {"pip_value": 0.0001, "spread_avg": 1.5, "session_best": "london"},
                "AUDUSD": {"pip_value": 0.0001, "spread_avg": 1.4, "session_best": "sydney_tokyo"},
                "USDCAD": {"pip_value": 0.0001, "spread_avg": 1.8, "session_best": "ny"},
                "NZDUSD": {"pip_value": 0.0001, "spread_avg": 2.1, "session_best": "sydney"}
            },
            "minor_pairs": {
                "EURGBP": {"pip_value": 0.0001, "spread_avg": 2.0},
                "EURJPY": {"pip_value": 0.01, "spread_avg": 1.8},
                "GBPJPY": {"pip_value": 0.01, "spread_avg": 2.5},
                "AUDJPY": {"pip_value": 0.01, "spread_avg": 2.2},
                "EURCHF": {"pip_value": 0.0001, "spread_avg": 2.1},
                "AUDCAD": {"pip_value": 0.0001, "spread_avg": 2.8}
            }
        },
        
        "crypto": {
            "major": {
                "BTCUSDT": {"min_notional": 10, "fee": 0.001},
                "ETHUSDT": {"min_notional": 10, "fee": 0.001},
                "BNBUSDT": {"min_notional": 10, "fee": 0.00075},
                "ADAUSDT": {"min_notional": 10, "fee": 0.001}
            },
            "altcoins": {
                "DOTUSDT": {"min_notional": 10, "fee": 0.001},
                "LINKUSDT": {"min_notional": 10, "fee": 0.001},
                "LTCUSDT": {"min_notional": 10, "fee": 0.001}
            }
        },
        
        "indices": {
            "US30": {"contract_size": 1, "pip_value": 1},
            "SPX500": {"contract_size": 1, "pip_value": 1},
            "NAS100": {"contract_size": 1, "pip_value": 1},
            "GER40": {"contract_size": 1, "pip_value": 1}
        }
    },
    
    # Optimized Trading Sessions
    "trading_sessions": {
        "sydney": {
            "start": "22:00", "end": "07:00", "timezone": "UTC",
            "volatility": "low", "best_pairs": ["AUDUSD", "NZDUSD", "AUDJPY"]
        },
        "tokyo": {
            "start": "00:00", "end": "09:00", "timezone": "UTC", 
            "volatility": "medium", "best_pairs": ["USDJPY", "AUDJPY", "GBPJPY"]
        },
        "london": {
            "start": "08:00", "end": "17:00", "timezone": "UTC",
            "volatility": "high", "best_pairs": ["GBPUSD", "EURGBP", "EURUSD"]
        },
        "newyork": {
            "start": "13:00", "end": "22:00", "timezone": "UTC",
            "volatility": "high", "best_pairs": ["EURUSD", "GBPUSD", "USDCAD"]
        },
        "london_newyork_overlap": {
            "start": "13:00", "end": "17:00", "timezone": "UTC",
            "volatility": "very_high", "best_pairs": ["EURUSD", "GBPUSD", "USDCHF"]
        }
    }
}

# =============================================================================
# BROKER CONFIGURATIONS
# =============================================================================

BROKER_CONFIGS = {
    "roboforex": {
        "enabled": True,
        "name": "RoboForex",
        "type": "mt5_alternative",  # Using alternatives since MT5 doesn't work on Linux
        "server": "RoboForex-Demo",
        "api_url": "https://api.roboforex.com",
        "supported_instruments": ["forex", "crypto", "indices", "commodities"],
        "features": ["copy_trading", "social_trading", "ea_hosting"],
        "min_deposit": 10,
        "max_leverage": {"forex": 2000, "crypto": 5, "indices": 500},
        "commission": {"forex": 0, "crypto": 0.1, "indices": 0}
    },
    
    "sabiotrade": {
        "enabled": True,
        "name": "SabioTrade", 
        "type": "api_native",
        "api_url": "https://api.sabiotrade.com",
        "sandbox": True,
        "supported_instruments": ["forex", "crypto"],
        "features": ["ai_signals", "social_trading", "copy_trading", "api_trading"],
        "specialties": ["ai_integration", "algorithmic_trading"]
    },
    
    "binance": {
        "enabled": True,
        "name": "Binance",
        "type": "ccxt",
        "exchange_id": "binance",
        "sandbox": True,
        "supported_instruments": ["crypto", "futures", "options"],
        "features": ["spot", "margin", "futures", "options"],
        "fee_structure": {
            "maker": 0.001, "taker": 0.001,
            "futures_maker": 0.0002, "futures_taker": 0.0004
        }
    },
    
    "bybit": {
        "enabled": True,
        "name": "Bybit", 
        "type": "ccxt",
        "exchange_id": "bybit",
        "sandbox": True,
        "supported_instruments": ["crypto", "derivatives", "futures"],
        "features": ["perpetual_swaps", "options", "spot"]
    }
}

# =============================================================================
# MACHINE LEARNING CONFIGURATION
# =============================================================================

ML_CONFIG = {
    "models": {
        "tensorflow_lstm_v2": {
            "enabled": True,
            "model_type": "sequence_prediction",
            "framework": "tensorflow",
            "version": "2.15.0",
            "architecture": {
                "sequence_length": 60,
                "features_count": 25,  # pandas-ta-classic indicators + price data
                "layers": [
                    {"type": "lstm", "units": 128, "dropout": 0.2, "recurrent_dropout": 0.1, "return_sequences": True},
                    {"type": "lstm", "units": 64, "dropout": 0.2, "recurrent_dropout": 0.1, "return_sequences": True},
                    {"type": "lstm", "units": 32, "dropout": 0.2, "recurrent_dropout": 0.1},
                    {"type": "dense", "units": 16, "activation": "relu"},
                    {"type": "dropout", "rate": 0.2},
                    {"type": "dense", "units": 3, "activation": "softmax"}  # BUY, SELL, HOLD
                ]
            },
            "training": {
                "epochs": 100,
                "batch_size": 64,
                "validation_split": 0.2,
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "loss": "categorical_crossentropy",
                "metrics": ["accuracy", "precision", "recall"],
                "callbacks": ["early_stopping", "reduce_lr_on_plateau", "model_checkpoint"]
            }
        },
        
        "xgboost_classifier_v2": {
            "enabled": True, 
            "model_type": "gradient_boosting_classification",
            "framework": "xgboost",
            "version": "2.0.3",
            "features": [
                # pandas-ta-classic indicators
                "trend_strength", "momentum_score", "volume_confirmation",
                "rsi_14", "macd_signal", "bb_position", "atr_normalized",
                "supertrend_signal", "adx_strength", "aroon_signal",
                # Smart Money Concepts
                "order_blocks_strength", "fvg_signal", "bos_signal", "liquidity_sweep",
                # Multi-timeframe
                "mtf_trend_alignment", "mtf_momentum_confluence"
            ],
            "parameters": {
                "n_estimators": 200,
                "max_depth": 8,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42
            },
            "target_classes": ["BUY", "SELL", "HOLD"],
            "class_weights": {"BUY": 1.2, "SELL": 1.2, "HOLD": 0.8}  # Favor trading signals
        },
        
        "ensemble_model_v2": {
            "enabled": True,
            "models": ["tensorflow_lstm_v2", "xgboost_classifier_v2"],
            "voting": "soft",
            "weights": [0.65, 0.35],  # LSTM gets more weight for sequence patterns
            "confidence_threshold": 0.7,  # Minimum confidence for signal generation
            "agreement_threshold": 0.6     # Minimum model agreement
        }
    },
    
    "feature_engineering": {
        "pandas_ta_classic_indicators": [
            # Trend indicators
            "sma_20", "sma_50", "ema_12", "ema_26", "supertrend", "adx", "psar",
            # Momentum indicators
            "rsi_14", "macd", "macd_signal", "stoch_k", "stoch_d", "williams_r", "cci",
            # Volume indicators  
            "obv", "ad", "cmf", "mfi", "vwap",
            # Volatility indicators
            "bb_upper", "bb_lower", "bb_percent", "atr", "keltner_upper", "keltner_lower"
        ],
        
        "smart_money_features": [
            "order_blocks_bull", "order_blocks_bear", "fair_value_gaps",
            "break_of_structure", "change_of_character", "liquidity_sweeps"
        ],
        
        "derived_features": [
            "price_position",      # Position within recent range
            "volume_ratio",        # Current vs average volume
            "volatility_rank",     # Current vs historical volatility
            "trend_strength",      # Composite trend strength
            "momentum_divergence",  # Price vs momentum divergence
            "support_resistance"   # Distance to key levels
        ],
        
        "lookback_periods": [5, 10, 20, 50, 100],
        "normalization": "robust_scaler",  # robust_scaler, standard_scaler, minmax_scaler
        "handle_missing": "forward_fill"    # forward_fill, interpolate, drop
    }
}

# =============================================================================
# SYSTEM CONFIGURATION 
# =============================================================================

SYSTEM_CONFIG = {
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": os.getenv("DEBUG", "False").lower() == "true",
        "reload": os.getenv("DEBUG", "False").lower() == "true",
        "workers": int(os.getenv("WORKERS", 1)),
        "cors": {
            "allow_origins": ["*"],
            "allow_methods": ["*"], 
            "allow_headers": ["*"]
        }
    },
    
    "database": {
        "postgresql": {
            "host": os.getenv("POSTGRES_HOST", "postgres"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "trading_bot"),
            "username": os.getenv("POSTGRES_USER", "trading_bot"), 
            "password": os.getenv("POSTGRES_PASSWORD", "trading_bot123"),
            "pool_size": 10,
            "max_overflow": 20,
            "echo": False
        },
        
        "redis": {
            "host": os.getenv("REDIS_HOST", "redis"),
            "port": int(os.getenv("REDIS_PORT", 6379)),
            "db": int(os.getenv("REDIS_DB", 0)),
            "decode_responses": True,
            "socket_keepalive": True
        }
    },
    
    "logging": {
        "level": os.getenv("LOG_LEVEL", "INFO"),
        "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
        "rotation": "00:00",  # Rotate at midnight
        "retention": "30 days",
        "compression": "zip",
        "backtrace": True,
        "diagnose": True
    }
}

# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

def get_config() -> Dict:
    """Get complete system configuration"""
    return {
        "trading": TRADING_CONFIG,
        "brokers": BROKER_CONFIGS,
        "ml": ML_CONFIG,
        "system": SYSTEM_CONFIG,
        "version": "2.0.0",
        "last_updated": datetime.utcnow().isoformat(),
        "pandas_ta_classic": True,
        "python_version": "3.11",
        "tensorflow_version": "2.15.0"
    }

# Validation
if __name__ == "__main__":
    config = get_config()
    print(f"Configuration loaded successfully")
    print(f"Version: {config['version']}")
    print(f"Strategies enabled: {len([s for s in config['trading']['strategies'].values() if s.get('enabled')])}")  
    print(f"Brokers configured: {len([b for b in config['brokers'].values() if b.get('enabled')])}")
    print(f"ML models enabled: {len([m for m in config['ml']['models'].values() if m.get('enabled')])}") 