# AI/ML Trading Bot

🤖 **Zaawansowany bot tradingowy z uczeniem maszynowym** obsługujący MT4/MT5, Forex, Crypto i wiele platform brokerskich.

## 🚀 Funkcje

### 📊 Obsługiwane Platformy
- **MetaTrader 4/5** - Pełne API
- **RoboForex** - Demo i Live
- **Sabiotrade** - Integracja API
- **XM Group** - Multi-account support
- **ForexChief (xChief)** - Automated trading
- **FXOpen** - Professional trading
- **InstaForex** - Social trading
- **TemplerFX** - Advanced features
- **FBS** - Global markets
- **Pocket Option** - Binary options
- **The5ers** - Funded accounts
- **Funded Trading Plus** - Prop trading

### 🧠 Strategie AI/ML
- **Smart Money Concept** - Analiza instytucjonalna (Order Blocks, FVG, Liquidity Sweeps)
- **Fibonacci Team** - Pełna implementacja strategii Łukasza Fijołka
- **Scalping** - High-frequency trading
- **RandomForest + LSTM** - Ensemble predictions
- **Online Learning** - Ciągłe dostrajanie modeli

### 📈 Zarządzanie Ryzykiem
- Stop loss domyślnie 2% (edytowalny)
- Risk/Reward minimum 1:2 (konfigurowalny)
- Maksymalny drawdown 15%
- Position sizing 2% kapitału na transakcję
- Daily loss limits

### 🌐 Web Interface
- **Real-time Dashboard** - Live P&L, pozycje, performance
- **Strategy Management** - Start/stop strategii z GUI
- **ML Model Manager** - Import/export wytrenowanych modeli
- **Performance Analytics** - Wykresy z wyborem okresu (1w, 1m, 3m, 1y, All)
- **Risk Configuration** - GUI do ustawień zarządzania ryzykiem

## 🛠️ Instalacja i Uruchomienie

### Docker (Zalecane)

```bash
# Sklonuj repozytorium
git clone https://github.com/szarastrefa/AI-ML-Trading-Bot.git
cd AI-ML-Trading-Bot

# Uruchom z Docker Compose
docker-compose up -d

# Sprawdź logi
docker-compose logs -f trading-bot
```

Aplikacja będzie dostępna na:
- **Web UI**: http://localhost:8080
- **API**: http://localhost:8000
- **Metabase**: http://localhost:3000

### Lokalna Instalacja

```bash
# Zainstaluj zależności Python
pip install -r requirements.txt

# Skonfiguruj bazę danych
export DATABASE_URL="postgresql://user:pass@localhost/trading_bot"
python scripts/init_db.py

# Uruchom aplikację
python -m app.main
```

## 📊 API Endpoints

### Broker Management
- `GET /api/v1/brokers` - Lista brokerów
- `POST /api/v1/brokers/{name}/connect` - Połącz z brokerem
- `GET /api/v1/account` - Informacje o koncie

### Trading Operations
- `POST /api/v1/orders` - Złóż zlecenie
- `GET /api/v1/positions` - Otwarte pozycje
- `DELETE /api/v1/orders/{id}` - Anuluj zlecenie

### ML Models
- `GET /api/v1/ml/models` - Informacje o modelach
- `POST /api/v1/ml/models/train` - Trenuj modele
- `GET /api/v1/ml/predictions` - Pobierz predykcje

## 🧪 Testowanie

```bash
# Testy jednostkowe
pytest tests/ -v

# Backtest strategii
python scripts/backtest_strategy.py --strategy fibonacci_team --symbol EURUSD --period 1y
```

## ⚠️ Disclaimer

**To oprogramowanie służy wyłącznie celom edukacyjnym. UŻYWAJ NA WŁASNE RYZYKO.**

## 📄 Licencja

MIT License

---

Made with ❤️ by Trading Bot Team