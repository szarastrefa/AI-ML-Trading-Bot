# 🤖 AI/ML Trading Bot v2.0 - **pandas-ta-classic Edition**

**The most advanced AI-driven trading system with 150+ technical indicators, Smart Money Concepts, and Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![pandas-ta-classic](https://img.shields.io/badge/pandas--ta--classic-1.0+-green.svg)](https://github.com/xgboosted/pandas-ta-classic)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎆 **What's New in v2.0**

### 📊 **pandas-ta-classic Integration**
- **150+ Technical Indicators** with multiprocessing support
- **Python 3.9-3.13 compatibility** (optimized for 3.11)
- **Community maintained** - active development and support
- **vectorbt integration** - professional backtesting framework
- **TA-Lib compatibility** - fastest indicator calculations

### 🧠 **Smart Money Concepts (SMC)**
- **Order Blocks** detection with volume confirmation
- **Fair Value Gaps (FVG)** identification and tracking  
- **Break of Structure (BOS)** - market structure analysis
- **Change of Character (CHoCH)** - momentum shifts
- **Liquidity Sweeps** - stop hunt detection

### ⚡ **Performance Optimizations**
- **Python 3.11** - 25% faster execution
- **Numba JIT compilation** - accelerated numerical computations
- **Multiprocessing** - parallel indicator calculations
- **Cython extensions** - C-speed critical paths

---

## 🎯 **Key Features**

| Feature | Description | Status |
|---------|-------------|--------|
| **Technical Analysis** | 150+ indicators across 9 categories | ✅ Active |
| **Smart Money Concepts** | Professional SMC implementation | ✅ Active |
| **Multi-Timeframe Analysis** | Confluence detection across timeframes | ✅ Active |
| **Machine Learning** | TensorFlow 2.15 + XGBoost 2.0.3 | ✅ Active |
| **Risk Management** | Advanced position sizing & risk controls | ✅ Active |
| **Backtesting** | vectorbt integration with performance metrics | ✅ Active |
| **Real-time Trading** | Live market data and execution | ✅ Active |
| **Monitoring** | Prometheus + Grafana dashboards | ✅ Active |
| **API-First** | RESTful API with FastAPI | ✅ Active |
| **Multi-Broker** | 12+ broker integrations | ✅ Active |

---

## 🚀 **Quick Start**

### **Prerequisites**
- Docker & Docker Compose
- Git
- 4GB RAM minimum (8GB recommended)
- Internet connection for data feeds

### **1. Clone & Setup**
```bash
git clone https://github.com/szarastrefa/AI-ML-Trading-Bot.git
cd AI-ML-Trading-Bot

# Create required directories
mkdir -p volumes/{postgres,redis,models,cache,prometheus,grafana,jupyter,minio}

# Set permissions
sudo chown -R $USER:$USER volumes/
```

### **2. Start the System**
```bash
# Core services (Trading Bot + Database + Cache)
docker-compose up -d

# With monitoring (add Prometheus + Grafana)
docker-compose --profile monitoring up -d

# Full stack (add Jupyter + MinIO + Nginx)
docker-compose --profile monitoring --profile development --profile storage up -d
```

### **3. Verify Installation**
```bash
# Check all services
docker-compose ps

# View logs
docker-compose logs -f trading-bot

# Health checks
curl http://localhost:8000/health
```

### **4. Access Applications**
| Service | URL | Credentials |
|---------|-----|-------------|
| **Trading Bot API** | http://localhost:8000/docs | - |
| **Health Check** | http://localhost:8000/health | - |
| **Grafana Dashboard** | http://localhost:3000 | admin / trading_admin_2025 |
| **Prometheus Metrics** | http://localhost:9090 | - |
| **Jupyter Notebook** | http://localhost:8888 | Token: trading_jupyter_2025 |
| **MinIO Storage** | http://localhost:9001 | minio_admin / minio_secure_2025 |

---

## 📊 **Technical Indicators (pandas-ta-classic)**

### **Trend Indicators (33 available)**
```python
# Moving Averages
df.ta.sma(length=20)          # Simple Moving Average
df.ta.ema(length=12)          # Exponential Moving Average
df.ta.wma(length=14)          # Weighted Moving Average
df.ta.hma(length=21)          # Hull Moving Average
df.ta.vwma(length=20)         # Volume Weighted Moving Average

# Advanced Trend
df.ta.supertrend()            # SuperTrend
df.ta.psar()                  # Parabolic SAR
df.ta.adx(length=14)          # Average Directional Movement
df.ta.aroon(length=14)        # Aroon Oscillator
df.ta.vortex(length=14)       # Vortex Indicator
```

### **Momentum Indicators (41 available)**
```python
# Core Oscillators
df.ta.rsi(length=14)          # Relative Strength Index
df.ta.macd()                  # Moving Average Convergence Divergence
df.ta.stoch()                 # Stochastic Oscillator
df.ta.willr(length=14)        # Williams %R
df.ta.cci(length=20)          # Commodity Channel Index

# Advanced Momentum
df.ta.ao()                    # Awesome Oscillator
df.ta.uo()                    # Ultimate Oscillator
df.ta.tsi()                   # True Strength Index
df.ta.squeeze()               # TTM Squeeze
df.ta.fisher(length=14)       # Fisher Transform
```

### **Volume Indicators (15 available)**
```python
# Volume Analysis
df.ta.obv()                   # On Balance Volume
df.ta.ad()                    # Accumulation/Distribution
df.ta.cmf(length=20)          # Chaikin Money Flow
df.ta.mfi(length=14)          # Money Flow Index
df.ta.vwap()                  # Volume Weighted Average Price
df.ta.pvt()                   # Price Volume Trend
df.ta.eom(length=14)          # Ease of Movement
```

### **Volatility Indicators (14 available)**
```python
# Bands & Channels
df.ta.bbands(length=20, std=2.0)    # Bollinger Bands
df.ta.kc(length=20, scalar=2.0)     # Keltner Channels
df.ta.donchian()                     # Donchian Channels

# Volatility Measures
df.ta.atr(length=14)          # Average True Range
df.ta.natr(length=14)         # Normalized ATR
df.ta.true_range()            # True Range
df.ta.ui(length=14)           # Ulcer Index
```

### **Candlestick Patterns (64 available)**
```python
# Pattern Recognition (requires TA-Lib)
df.ta.cdl_doji()              # Doji
df.ta.cdl_hammer()            # Hammer
df.ta.cdl_engulfing()         # Engulfing Pattern
df.ta.cdl_harami()            # Harami Pattern
df.ta.cdl_morning_star()      # Morning Star
df.ta.cdl_evening_star()      # Evening Star

# Get all patterns
df = df.ta.cdl_pattern(name="all")
```

---

## 🧠 **Smart Money Concepts**

### **Order Blocks**
Institutional levels where large orders were placed:
```python
# Detection criteria:
# - High volume (1.5x average)
# - Large body size (1.2x average)  
# - Strong rejection (long wicks)
# - Price reaction from level

order_blocks = strategy.detect_order_blocks(df)
# Returns: 1 (Bullish OB), -1 (Bearish OB), 0 (None)
```

### **Fair Value Gaps (FVG)**
Market imbalances requiring price return:
```python
# Bullish FVG: Previous high < Current low
# Bearish FVG: Previous low > Current high
# Minimum gap size: 0.1% of price

fvg = strategy.detect_fair_value_gaps(df)
# Returns: 1 (Bullish Gap), -1 (Bearish Gap), 0 (None)
```

### **Break of Structure (BOS)**
Significant changes in market structure:
```python
# Criteria:
# - Break of swing highs/lows
# - Volume confirmation
# - Sustained movement

bos = strategy.detect_break_of_structure(df)
# Returns: 1 (Bullish BOS), -1 (Bearish BOS), 0 (None)
```

### **Liquidity Sweeps**
Stop hunts and false breakouts:
```python
# Detection:
# - Break recent high/low
# - Immediate reversal
# - High volume confirmation

sweeps = strategy.detect_liquidity_sweeps(df)
# Returns: 1 (Upside sweep), -1 (Downside sweep), 0 (None)
```

---

## 💰 **Trading Strategy Example**

### **Complete Strategy Implementation**
```python
import pandas_ta_classic as ta
from app.strategies.pandas_ta_classic_strategy import PandasTAClassicStrategy

# Configuration
config = {
    'timeframes': ['M15', 'H1', 'H4'],
    'min_confluence_count': 2,
    'smart_money_concepts': {
        'order_blocks': {'enabled': True, 'lookback': 20},
        'fair_value_gaps': {'enabled': True, 'min_gap_size': 0.1},
        'break_of_structure': {'enabled': True, 'swing_period': 10},
        'liquidity_sweeps': {'enabled': True, 'lookback': 50}
    },
    'risk_management': {
        'max_risk_per_trade': 0.02,  # 2% risk
        'stop_loss_pct': 0.02,       # 2% SL
        'take_profit_ratio': 2.0     # 1:2 RR
    }
}

# Initialize strategy
strategy = PandasTAClassicStrategy(config)

# Run analysis
result = await strategy.analyze('EURUSD', 'H1')

print(f"Signal: {result['signal']}")
print(f"Confidence: {result['confidence']}%")
print(f"Entry: {result['entry_price']}")
print(f"SL: {result['stop_loss']}")
print(f"TP: {result['take_profit']}")
print(f"R/R: {result['risk_reward_ratio']}")
```

### **Multi-Timeframe Strategy**
```python
# Create comprehensive strategy
advanced_strategy = ta.Strategy(
    name="Advanced SMC + Multi-TF",
    description="Smart Money Concepts with multi-timeframe confluence",
    ta=[
        # Trend Analysis
        {"kind": "sma", "length": 20},
        {"kind": "sma", "length": 50},
        {"kind": "ema", "length": 12},
        {"kind": "supertrend", "length": 10, "multiplier": 3.0},
        
        # Momentum
        {"kind": "rsi", "length": 14},
        {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
        {"kind": "stoch", "k": 14, "d": 3},
        
        # Volume
        {"kind": "obv"},
        {"kind": "vwap"},
        {"kind": "cmf", "length": 20},
        
        # Volatility
        {"kind": "bbands", "length": 20, "std": 2.0},
        {"kind": "atr", "length": 14},
        
        # Utility
        {"kind": "tsignals", "trend": "close > sma_50", "asbool": True}
    ]
)

# Apply to DataFrame with multiprocessing
df.ta.strategy(advanced_strategy, verbose=True)
```

---

## 📊 **Backtesting with vectorbt**

### **Simple Backtest**
```python
import vectorbt as vbt
import pandas_ta_classic as ta

# Get data
df = pd.DataFrame().ta.ticker("AAPL", period="1y")

# Generate signals
df['sma_20'] = df.ta.sma(20)
df['sma_50'] = df.ta.sma(50)
df['signal'] = df['sma_20'] > df['sma_50']

# Create entry/exit signals
signals = df.ta.tsignals(df['signal'], asbool=True)

# Run backtest
pf = vbt.Portfolio.from_signals(
    df['close'], 
    entries=signals['TS_Entries'], 
    exits=signals['TS_Exits'],
    init_cash=10000,
    fees=0.001,
    slippage=0.001
)

# Results
print(f"Total Return: {pf.total_return():.1%}")
print(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}")
print(f"Max Drawdown: {pf.max_drawdown():.1%}")
print(f"Win Rate: {pf.trades.win_rate:.1%}")
```

### **Advanced Backtest with SMC**
```python
# Run comprehensive backtest
backtest_result = await strategy.backtest(
    symbol='EURUSD',
    start_date='2023-01-01',
    end_date='2024-01-01',
    initial_capital=10000
)

print(f"Performance Summary:")
print(f"Total Return: {backtest_result['total_return']:.1%}")
print(f"Sharpe Ratio: {backtest_result['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {backtest_result['max_drawdown']:.1%}")
print(f"Total Trades: {backtest_result['total_trades']}")
print(f"Win Rate: {backtest_result['win_rate']:.1%}")
```

---

## 🔧 **Configuration**

### **Strategy Configuration**
```python
# config/settings.py
TRADING_CONFIG = {
    "strategies": {
        "pandas_ta_classic_v2": {
            "enabled": True,
            "timeframes": ["M15", "H1", "H4", "D1"],
            "indicators": {
                "moving_averages": {
                    "sma_periods": ,
                    "ema_periods": [8, 12, 21, 26, 50]
                },
                "momentum": {
                    "rsi": {"length": 14, "overbought": 70, "oversold": 30},
                    "macd": {"fast": 12, "slow": 26, "signal": 9},
                    "stochastic": {"k": 14, "d": 3, "smooth_k": 3}
                },
                "smart_money_concepts": {
                    "order_blocks": {"enabled": True, "lookback": 20},
                    "fair_value_gaps": {"enabled": True, "min_gap_size": 0.1}
                }
            }
        }
    }
}
```

### **Risk Management**
```python
"risk_management": {
    "max_risk_per_trade": 0.02,        # 2% per trade
    "stop_loss_pct": 0.02,             # 2% stop loss
    "take_profit_ratio": 2.0,          # 1:2 risk/reward
    "max_daily_loss": 0.06,            # 6% daily limit
    "max_open_positions": 3,           # Max concurrent trades
    "position_sizing": "kelly"          # Kelly criterion sizing
}
```

---

## 💻 **API Reference**

### **Trading Signals**
```bash
# Get trading signal
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "EURUSD",
    "timeframe": "H1",
    "strategy": "pandas_ta_classic_v2"
  }'
```

**Response:**
```json
{
  "signal": "BUY",
  "confidence": 85.2,
  "entry_price": 1.0950,
  "stop_loss": 1.0928,
  "take_profit": 1.0994,
  "risk_reward_ratio": 2.0,
  "analysis_breakdown": {
    "technical_indicators": {
      "trend_strength": 78.5,
      "momentum_score": 68.2,
      "volume_confirmation": 82.1
    },
    "smart_money_concepts": {
      "overall": {"bias": "BULLISH", "strength": 3, "confidence": 85}
    },
    "multi_timeframe": {
      "confluence": {"bullish_confluence": 3, "bearish_confluence": 0}
    }
  }
}
```

### **Backtesting**
```bash
# Run backtest
curl -X POST "http://localhost:8000/api/v1/backtest" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "initial_capital": 10000,
    "strategy": "pandas_ta_classic_v2"
  }'
```

### **Performance Metrics**
```bash
# Get strategy performance
curl "http://localhost:8000/api/v1/performance/pandas_ta_classic_v2"
```

---

## 📊 **Monitoring & Analytics**

### **Grafana Dashboards**
Access comprehensive dashboards at http://localhost:3000:

- **Trading Performance** - P&L, win rate, drawdown
- **Signal Analysis** - Signal accuracy, confidence distribution
- **Market Analysis** - Volatility, volume, correlation
- **System Health** - API latency, memory usage, error rates
- **Risk Management** - Position sizing, exposure, limits

### **Prometheus Metrics**
Key metrics available at http://localhost:9090:

```
# Trading metrics
trading_signals_total{strategy="pandas_ta_classic_v2"}
trading_pnl_total{symbol="EURUSD"}
trading_win_rate{timeframe="H1"}
trading_drawdown_current

# System metrics
api_request_duration_seconds
api_requests_total{endpoint="/analyze"}
memory_usage_bytes
cpu_usage_percent
```

---

## 🌐 **Broker Integration**

### **Supported Brokers**
| Broker | Type | Status | Features |
|--------|------|--------|---------|
| **RoboForex** | MT5 Alternative | ✅ | Demo/Live, Copy Trading |
| **SabioTrade** | API Native | ✅ | AI Signals, Social Trading |
| **Binance** | Crypto | ✅ | Spot, Futures, Options |
| **Bybit** | Crypto | ✅ | Derivatives, Perpetuals |
| **XM Global** | MT5 Alternative | ✅ | Multi-Asset Trading |
| **FXOpen** | MT5 Alternative | ✅ | ECN, Professional |

### **Configuration Example**
```python
# config/settings.py
BROKER_CONFIGS = {
    "roboforex": {
        "enabled": True,
        "type": "mt5_alternative",
        "server": "RoboForex-Demo",
        "supported_instruments": ["forex", "crypto", "indices"],
        "max_leverage": {"forex": 2000, "crypto": 5}
    },
    "binance": {
        "enabled": True,
        "type": "ccxt",
        "exchange_id": "binance",
        "sandbox": True,
        "fee_structure": {"maker": 0.001, "taker": 0.001}
    }
}
```

---

## 🚀 **Development**

### **Local Development Setup**
```bash
# Clone repository
git clone https://github.com/szarastrefa/AI-ML-Trading-Bot.git
cd AI-ML-Trading-Bot

# Create virtual environment (Python 3.11 recommended)
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export POSTGRES_HOST=localhost
export POSTGRES_DB=trading_bot_v2
export REDIS_HOST=localhost
export DEBUG=True

# Run application
python app/main.py
```

### **Testing**
```bash
# Run tests
pytest tests/ -v --cov=app

# Run specific test category
pytest tests/test_strategies/ -v
pytest tests/test_indicators/ -v
pytest tests/test_smc/ -v

# Generate coverage report
pytest --cov=app --cov-report=html
```

### **Code Quality**
```bash
# Format code
black app/ tests/ config/
isort app/ tests/ config/

# Lint
flake8 app/ tests/ config/

# Type checking
mypy app/
```

---

## 📚 **Documentation**

- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **pandas-ta-classic Guide**: https://github.com/xgboosted/pandas-ta-classic
- **Smart Money Concepts**: docs/smart_money_concepts.md
- **Machine Learning Pipeline**: docs/ml_pipeline.md
- **Broker Integration**: docs/broker_integration.md
- **Deployment Guide**: docs/deployment.md

---

## 🔍 **Troubleshooting**

### **Common Issues**

**1. pandas-ta-classic installation fails:**
```bash
# Install TA-Lib dependencies first
sudo apt-get install build-essential
pip install --upgrade pip wheel setuptools
pip install pandas-ta-classic
```

**2. Docker build errors:**
```bash
# Clean Docker cache
docker system prune -a
docker-compose build --no-cache
```

**3. Database connection issues:**
```bash
# Check PostgreSQL status
docker-compose ps postgres
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

**4. Memory issues:**
```bash
# Increase Docker memory limit to 4GB+
# Reduce number of parallel indicators
df.ta.cores = 2  # Limit CPU cores
```

### **Performance Optimization**
```python
# Optimize pandas-ta-classic performance
df.ta.cores = 0  # Use all CPU cores

# Use specific indicators instead of full strategy
df.ta.sma(20, append=True)
df.ta.rsi(14, append=True)
df.ta.macd(append=True)

# Enable Numba acceleration
import numba
numba.set_num_threads(4)
```

---

## 🤝 **Contributing**

### **Development Workflow**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run quality checks: `black`, `flake8`, `mypy`, `pytest`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open Pull Request

### **Contribution Guidelines**
- Follow PEP 8 coding standards
- Add tests for new features
- Update documentation
- Ensure backwards compatibility
- Add type hints

---

## 📄 **License**

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

## ⚠️ **Disclaimer**

**This software is for educational and research purposes only. Trading financial instruments involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Always test strategies on demo accounts before using real money.**

**The developers are not responsible for any financial losses incurred through the use of this software.**

---

## 📞 **Support & Community**

- **Issues**: [GitHub Issues](https://github.com/szarastrefa/AI-ML-Trading-Bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/szarastrefa/AI-ML-Trading-Bot/discussions)
- **Documentation**: [Wiki](https://github.com/szarastrefa/AI-ML-Trading-Bot/wiki)

---

**🎆 Built with pandas-ta-classic • 🐍 Python 3.11 • 🤖 AI/ML • 💹 Smart Money Concepts • ⚡ High Performance**

**⭐ If you find this project helpful, please give it a star!**