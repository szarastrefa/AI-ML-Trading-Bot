# 🚀 Quick Start Guide - AI/ML Trading Bot v3.0

**Get your Professional Trading System running in minutes!**

---

## 🎯 **Option 1: Docker (Recommended)**

### 1. **Clone Repository**
```bash
git clone https://github.com/szarastrefa/AI-ML-Trading-Bot.git
cd AI-ML-Trading-Bot
```

### 2. **Start with Docker Compose**
```bash
docker-compose up -d
```

### 3. **Access Dashboard**
🌐 **Web GUI**: http://localhost:8000

---

## 🐍 **Option 2: Local Python**

### 1. **Clone & Setup**
```bash
git clone https://github.com/szarastrefa/AI-ML-Trading-Bot.git
cd AI-ML-Trading-Bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Run Application**
```bash
cd app
python main.py
```

### 4. **Access Dashboard**
🌐 **Web GUI**: http://localhost:8000

---

## ✅ **Verification**

After startup, verify everything works:

1. **Health Check**: http://localhost:8000/health
2. **API Docs**: http://localhost:8000/docs
3. **Trading Analysis**: http://localhost:8000/api/v1/analyze?symbol=EURUSD

---

## 🎨 **Web GUI Features**

### 📈 **Dashboard Components**
- **Key Metrics** - Account Balance, Win Rate, Active Positions, AI Confidence
- **P&L Charts** - Interactive charts with 5 time periods (1W, 1M, 3M, 1Y, All)
- **Strategy Performance** - Real-time analytics for SMC, Fibonacci Team, ML Ensemble
- **Multi-Platform Status** - Live broker connection status

### 💼 **Position Management**
- **Live Positions Table** - Real-time P&L updates
- **Close Positions** - Individual and bulk position closure
- **Auto-refresh** - Updates every 30 seconds

### 🧠 **ML Model Manager**
- **Model Status** - RandomForest, LSTM, Ensemble models
- **Import/Export** - Drag & drop model upload/download
- **Training Controls** - Start/stop model training
- **Performance Metrics** - Accuracy and validation scores

### ⚖️ **Risk Management**
- **Stop Loss Slider** - 2% default (Fibonacci Team standard)
- **Risk Parameters** - Max risk per trade, daily trades, positions
- **Real-time Assessment** - Portfolio risk calculation
- **Settings Save** - Persistent risk configuration

---

## 🛠️ **Docker Management**

```bash
# View running containers
docker-compose ps

# View logs
docker logs ai-trading-bot-stable

# Restart service
docker-compose restart trading-bot

# Stop system
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

---

## 🔧 **Configuration**

### **Risk Management Settings**
- Default stop loss: **2%** (Fibonacci Team methodology)
- Max risk per trade: **1.5%** 
- Max daily trades: **10**
- Max open positions: **5**
- Risk/reward ratio: **2:1**

### **Trading Strategies**
- **Smart Money Concepts** - Order Blocks, FVG, BOS analysis
- **Fibonacci Team** - Based on Łukasz Fijolek's methodology
- **ML Ensemble** - RandomForest + LSTM predictions

### **Multi-Platform Support**
- MT5 Live, Sabiotrade, RoboForex, XM Group
- ForexChief, FXOpen + 7 more brokers
- Demo and live account modes

---

## 📊 **API Testing**

### **Test Trading Analysis**
```bash
# Smart Money Concepts analysis
curl "http://localhost:8000/api/v1/analyze?symbol=EURUSD&strategy=SmartMoney"

# Multiple symbols
curl "http://localhost:8000/api/v1/analyze?symbol=BTCUSD"
curl "http://localhost:8000/api/v1/analyze?symbol=XAUUSD"
```

### **Test Web GUI APIs**
```bash
# Get P&L chart data
curl "http://localhost:8000/api/v2/pnl/chart?period=1M"

# Get current positions
curl "http://localhost:8000/api/v2/positions/current"

# Get strategy performance
curl "http://localhost:8000/api/v2/strategies/performance"
```

---

## ⚡ **Performance Expectations**

- **API Response Time**: < 300ms
- **Chart Loading**: < 2 seconds
- **Memory Usage**: < 500MB
- **Auto-refresh**: Every 30 seconds
- **Browser Support**: Chrome 90+, Firefox 88+, Safari 14+

---

## 🏆 **What You Get**

✅ **Complete Professional Web GUI**  
✅ **Real-time P&L Charts** (5 periods)  
✅ **Position Management** (Live updates)  
✅ **ML Model Manager** (Import/Export)  
✅ **Risk Management Interface** (2% default stop loss)  
✅ **Multi-Platform Dashboard** (13+ brokers)  
✅ **Strategy Analytics** (SMC, Fibonacci, ML)  
✅ **Auto-refresh & Notifications**  
✅ **Responsive Design** (Mobile-friendly)  
✅ **Professional Trading System**  

---

## 🐛 **Troubleshooting**

### **Docker Issues**
```bash
# If containers fail to start
docker-compose down
docker system prune -f
docker-compose up -d --build
```

### **Port Already in Use**
```bash
# Change port in docker-compose.yml
# From: "8000:8000" 
# To: "8001:8000"
# Then access: http://localhost:8001
```

### **Permission Issues (Linux)**
```bash
sudo chown -R $USER:$USER .
sudo chmod +x scripts/start.sh
```

---

## 🚀 **Next Steps**

1. **Explore the Web GUI** - Test all features and components
2. **Review Configuration** - Check `config/trading_config.yaml`
3. **Test Trading Analysis** - Try different symbols and strategies
4. **Customize Risk Settings** - Adjust parameters via GUI
5. **Integration** - Connect real broker APIs when ready

---

## 📞 **Support**

- **GitHub Issues**: [Report problems](https://github.com/szarastrefa/AI-ML-Trading-Bot/issues)
- **Discussions**: [Ask questions](https://github.com/szarastrefa/AI-ML-Trading-Bot/discussions)
- **Documentation**: [Full docs](README.md)
- **Health Check**: http://localhost:8000/health

---

## ⚖️ **Disclaimer**

This software is for educational purposes only. Trading involves risk of loss. Always test with demo accounts before using real money.

---

**🏆 Congratulations! You now have a professional AI/ML trading system running!** 🎆