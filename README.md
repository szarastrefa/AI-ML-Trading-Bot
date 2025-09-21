# 🤖 AI/ML Trading Bot

**Zaawansowany bot tradingowy z Artificial Intelligence i Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14.0-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ **Funkcje**

### 🎯 **Trading Features**
- **12+ brokerów**: MetaTrader alternatives, RoboForex, XM Group, FXOpen, Binance, Bybit
- **Strategie AI/ML**: Smart Money Concept, Fibonacci Team, Custom strategies
- **Risk Management**: 2% SL, 1:2 RR, Max drawdown protection
- **Backtesting**: Historical data analysis z TensorFlow
- **Real-time signals**: WebSocket connections

### 🧠 **AI/ML Stack** 
- **TensorFlow 2.14.0** - Deep learning models (bez konfliktów zależności)
- **XGBoost 2.0.3** - Gradient boosting
- **pandas-ta 0.3.14b0** - Technical analysis (kompatybilne z Python 3.10)
- **scikit-learn 1.3.2** - Machine Learning
- **pandas 2.1.4** - Data processing

### 🔧 **Tech Stack**
- **Python 3.10** - Optymalna kompatybilność z bibliotekami ML/Trading
- **FastAPI** - Modern API framework
- **PostgreSQL** - Baza danych  
- **Redis** - Cache & message broker
- **Docker** - Konteneryzacja

## 🚀 **Quick Start**

### **1. Klonuj repozytorium**
```bash
git clone https://github.com/szarastrefa/AI-ML-Trading-Bot.git
cd AI-ML-Trading-Bot
```

### **2. Uruchom z Docker**
```bash
# Zbuduj i uruchom wszystkie serwisy
docker-compose up -d --build

# Sprawdź logi
docker-compose logs -f trading-bot

# Sprawdź status
docker-compose ps
```

### **3. Dostęp do API**
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **App Info**: http://localhost:8000/info

## 📊 **Konfiguracja Brokerów**

### **MetaTrader5 Alternative (Linux)**

⚠️ **UWAGA**: MetaTrader5 Python package działa **tylko na Windows**!

Dla **Linux/Docker** użyj alternatyw z `config/mt5_alternatives.yaml`:

```yaml
# 1. ZeroMQ Bridge (Zalecane)
zeromq_bridge:
  enabled: true
  host: "localhost"
  port: 5555

# 2. Alternative Brokers
alternative_brokers:
  roboforex:
    enabled: true
    api_type: "REST"
  
  binance:
    enabled: true
    testnet: true
```

### **Konfiguracja Brokerów**

Edytuj `config/settings.py`:

```python
BROKER_CONFIGS = {
    "roboforex": {
        "enabled": True,
        "demo_mode": True
    },
    "binance": {
        "enabled": True,
        "testnet": True,
        "api_key": "your_api_key",
        "api_secret": "your_secret"
    }
}
```

## 🔍 **Rozwiązane Problemy**

### ✅ **Konflikty Zależności**

**Problem**: 
```
tensorflow 2.13.0 depends on typing-extensions<4.6.0
fastapi 0.104.1 depends on typing-extensions>=4.8.0
```

**Rozwiązanie**:
- **TensorFlow 2.14.0** - nowsza wersja bez konfliktów
- **typing-extensions==4.8.0** - fixed version
- **Python 3.10** - optymalna kompatybilność

### ✅ **MetaTrader5 na Linux**

**Problem**: MetaTrader5 package tylko Windows

**Rozwiązanie**:
1. **ZeroMQ Bridge** - komunikacja MT5 ↔ Linux bot
2. **REST API Bridge** - HTTP API dla MT5  
3. **Alternative Brokers** - RoboForex, XM Group, FXOpen
4. **Crypto Exchanges** - Binance, Bybit z pełnym wsparciem Linux

## 📈 **Strategie Trading**

### **1. Smart Money Concept (SMC)**
```python
strategies = {
    "smart_money_concept": {
        "enabled": True,
        "timeframes": ["15m", "1h", "4h"],
        "risk_reward": 3.0
    }
}
```

### **2. Fibonacci Team**
```python
"fibonacci_team": {
    "enabled": True,
    "timeframes": ["5m", "15m", "1h"],
    "risk_reward": 2.0
}
```

## 🛠️ **Development**

### **Struktura Projekt**
```
AI-ML-Trading-Bot/
├── app/                    # FastAPI aplikacja
├── config/                 # Konfiguracje
│   ├── settings.py        # Broker configs
│   └── mt5_alternatives.yaml
├── scripts/               # Utility scripts  
├── data/                  # Data storage
├── Dockerfile            # Python 3.10
├── docker-compose.yml    # Full stack
└── requirements.txt      # Dependencies
```

### **Lokalne uruchomienie**
```bash
# Zainstaluj zależności
pip install -r requirements.txt

# Inicjalizuj bazę danych
python scripts/init_db.py

# Uruchom aplikację
python app/main.py
```

## 🔐 **Environment Variables**

```bash
# Database
DATABASE_URL=postgresql://trading:trading123@localhost:5432/trading_bot
REDIS_URL=redis://localhost:6379/0

# App
ENV=development
DEBUG=true
HOST=0.0.0.0
PORT=8000

# Trading
DEFAULT_BROKER=roboforex
```

## 📝 **API Endpoints**

```bash
# Health & Info
GET /health
GET /info
GET /

# Trading API
GET /api/v1/brokers
POST /api/v1/signals
GET /api/v1/history
POST /api/v1/backtest
```

## ⚡ **Performance**

- **Python 3.10**: ~25% szybszy niż 3.9
- **TensorFlow 2.14**: Optymalizacje dla trading
- **Redis cache**: Sub-millisecond response
- **PostgreSQL**: Miliony transakcji

## ⚠️ **Disclaimer**

**To oprogramowanie służy wyłącznie celom edukacyjnym. UŻYWAJ NA WŁASNE RYZYKO.**

## 🤝 **Contributing**

1. Fork the project
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`) 
5. Open Pull Request

## 📄 **License**

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 **Support**

- **Issues**: [GitHub Issues](https://github.com/szarastrefa/AI-ML-Trading-Bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/szarastrefa/AI-ML-Trading-Bot/discussions)

---

**⭐ Jeśli projekt Ci się podoba, zostaw gwiazdkę!**

Made with ❤️ by Trading Bot Team