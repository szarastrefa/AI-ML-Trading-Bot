# 🚀 AI/ML Trading Bot v5.0 - DEPLOYMENT GUIDE

**Kompletny przewodnik wdrożenia systemu AI/ML Trading Bot v5.0 od development po enterprise production.**

## 🎯 **DEPLOYMENT SCENARIOS**

### **1. 💻 Local Development**
### **2. 🐳 Docker Standard**  
### **3. 🌍 Production Enterprise**
### **4. ☁️ Cloud Deployment**

---

## 💻 **1. LOCAL DEVELOPMENT SETUP**

### **Prerequisites:**
```bash
# System requirements
Python 3.11+
Node.js 18+ (dla frontend tools)
Git
8GB RAM (zalecane)
20GB free disk space
```

### **Installation Steps:**
```bash
# 1. Clone repository
git clone https://github.com/szarastrefa/AI-ML-Trading-Bot.git
cd AI-ML-Trading-Bot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create directories
mkdir -p data logs backups models config

# 5. Setup environment
cp .env.example .env
nano .env  # Edytuj konfiguracje

# 6. Initialize database
python -c "from app.database import init_db; init_db()"

# 7. Run development server
cd app
python main.py

# 8. Access application
# http://localhost:8000 - Panel Sterowania
# http://localhost:8000/docs - API Documentation
```

### **Development Commands:**
```bash
# Hot reload development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v

# Code formatting
black app/
flake8 app/

# Type checking
mypy app/
```

---

## 🐳 **2. DOCKER STANDARD DEPLOYMENT**

### **Quick Docker Setup:**
```bash
# 1. Clone and prepare
git clone https://github.com/szarastrefa/AI-ML-Trading-Bot.git
cd AI-ML-Trading-Bot

# 2. Create required directories
mkdir -p data logs backups models config
sudo chown -R $USER:$USER data logs backups models config

# 3. Environment setup
cp .env.example .env

# Generate secure keys
export SECRET_KEY=$(openssl rand -hex 32)
echo "SECRET_KEY=$SECRET_KEY" >> .env

# 4. Deploy system
docker-compose down --remove-orphans
docker-compose build --no-cache
docker-compose up -d

# 5. Wait for startup (60-90 seconds)
sleep 90

# 6. Check system health
curl -s http://localhost:8000/health | jq .
docker-compose ps
docker-compose logs trading-bot --tail=20
```

### **Docker Management:**
```bash
# View logs
docker-compose logs -f trading-bot

# Restart services
docker-compose restart

# Stop system
docker-compose down

# Update system
git pull origin main
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Cleanup (CAUTION: deletes all data)
docker-compose down -v
docker system prune -a
```

---

## 🌍 **3. PRODUCTION ENTERPRISE DEPLOYMENT**

### **Production Prerequisites:**
```bash
# Server requirements
Ubuntu 20.04+ lub CentOS 8+
16GB RAM (minimum), 32GB zalecane
4+ CPU cores
100GB+ SSD storage
Statyczne IP lub domain
Firewall configuration
```

### **Production Server Setup:**
```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Docker & Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 3. Install additional tools
sudo apt install -y nginx certbot python3-certbot-nginx htop iotop

# 4. Configure firewall
sudo ufw allow 22      # SSH
sudo ufw allow 80      # HTTP
sudo ufw allow 443     # HTTPS
sudo ufw allow 8000    # Trading Bot API
sudo ufw --force enable

# 5. Logout and login again (for Docker group)
exit
# SSH back in
```

### **Production Deployment:**
```bash
# 1. Clone and setup
git clone https://github.com/szarastrefa/AI-ML-Trading-Bot.git
cd AI-ML-Trading-Bot

# 2. Create production directories
sudo mkdir -p /opt/ai-trading-bot/{data,logs,backups,models,config,ssl}
sudo chown -R $USER:$USER /opt/ai-trading-bot
ln -sf /opt/ai-trading-bot/{data,logs,backups,models,config} .

# 3. Production environment configuration
cp .env.example .env

# Generate secure secrets
export SECRET_KEY=$(openssl rand -hex 32)
export POSTGRES_PASSWORD=$(openssl rand -hex 24)
export REDIS_PASSWORD=$(openssl rand -hex 16)

# Update .env file
sed -i "s/change-this-super-secret-key-in-production/$SECRET_KEY/g" .env
echo "POSTGRES_PASSWORD=$POSTGRES_PASSWORD" >> .env
echo "REDIS_PASSWORD=$REDIS_PASSWORD" >> .env
echo "GRAFANA_PASSWORD=admin$(openssl rand -hex 8)" >> .env

# 4. SSL Certificate (Let's Encrypt)
# Zastąp 'your-domain.com' swoim prawdziwym domain
export DOMAIN=your-domain.com
sudo certbot certonly --standalone -d $DOMAIN
sudo cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/$DOMAIN/privkey.pem nginx/ssl/key.pem
sudo chown $USER:$USER nginx/ssl/*.pem

# 5. Deploy production system
docker-compose -f docker-compose.prod.yml down --remove-orphans
docker-compose -f docker-compose.prod.yml build --no-cache
docker-compose -f docker-compose.prod.yml up -d

# 6. Deploy with monitoring (optional)
docker-compose -f docker-compose.prod.yml --profile monitoring up -d

# 7. Verify deployment
sleep 120  # Wait for all services
docker-compose -f docker-compose.prod.yml ps
curl -s https://$DOMAIN/health | jq .
```

### **Production Health Check:**
```bash
# System status
docker-compose -f docker-compose.prod.yml ps

# Service health
curl -s https://your-domain.com/health | jq .

# Database connectivity
docker-compose -f docker-compose.prod.yml exec postgres psql -U trading_user -d trading_bot -c "SELECT version();"

# Redis connectivity
docker-compose -f docker-compose.prod.yml exec redis redis-cli ping

# Resource usage
docker stats --no-stream

# Application logs
docker-compose -f docker-compose.prod.yml logs --tail=50 trading-bot
```

---

## 📈 **4. MONITORING & MAINTENANCE**

### **System Monitoring:**
```bash
# Real-time monitoring
watch -n 5 'docker stats --no-stream'

# Log monitoring
tail -f logs/system.log

# Health monitoring script
cat > monitor.sh << 'EOF'
#!/bin/bash
while true; do
    echo "=== $(date) ==="
    curl -s http://localhost:8000/health | jq -r '.status, .system_stats.accounts.total_balance, .system_stats.strategies.active_count'
    docker-compose ps --services --filter status=running | wc -l
    echo "Active services: $(docker-compose ps --services --filter status=running | wc -l)/4"
    echo
    sleep 300  # Check every 5 minutes
done
EOF
chmod +x monitor.sh
./monitor.sh
```

### **Backup Procedures:**
```bash
# 1. Database backup
cat > backup-db.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="./backups/db"
mkdir -p $BACKUP_DIR

# PostgreSQL backup
docker-compose -f docker-compose.prod.yml exec -T postgres pg_dump -U trading_user trading_bot > "$BACKUP_DIR/trading_bot_$DATE.sql"

# Compress backup
gzip "$BACKUP_DIR/trading_bot_$DATE.sql"

echo "Database backup created: $BACKUP_DIR/trading_bot_$DATE.sql.gz"
EOF
chmod +x backup-db.sh

# 2. Full system backup
cat > backup-full.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="./backups/full"
mkdir -p $BACKUP_DIR

# Create full backup
tar -czf "$BACKUP_DIR/ai-trading-bot-full_$DATE.tar.gz" \
    --exclude='./backups' \
    --exclude='./logs/*.log' \
    --exclude='.git' \
    --exclude='venv' \
    --exclude='__pycache__' \
    .

echo "Full backup created: $BACKUP_DIR/ai-trading-bot-full_$DATE.tar.gz"
EOF
chmod +x backup-full.sh

# 3. Setup automated backups (crontab)
echo "0 2 * * * /path/to/AI-ML-Trading-Bot/backup-db.sh" | crontab -
echo "0 6 * * 0 /path/to/AI-ML-Trading-Bot/backup-full.sh" | crontab -
```

### **Update Procedures:**
```bash
# 1. Controlled update
cat > update-system.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting AI/ML Trading Bot v5.0 Update..."

# Create backup before update
echo "💾 Creating backup..."
./backup-full.sh

# Stop system gracefully
echo "🛑 Stopping system..."
docker-compose -f docker-compose.prod.yml down

# Pull latest changes
echo "📲 Pulling latest changes..."
git fetch origin main
git reset --hard origin/main

# Rebuild and start
echo "🔧 Rebuilding system..."
docker-compose -f docker-compose.prod.yml build --no-cache

# Start services
echo "🚀 Starting services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait and check health
echo "🔍 Checking system health..."
sleep 120
curl -s http://localhost:8000/health | jq .

echo "✅ Update completed!"
EOF
chmod +x update-system.sh
```

---

## 🔐 **SECURITY CONFIGURATION**

### **SSL/TLS Setup:**
```bash
# 1. Generate self-signed certificate (development)
mkdir -p nginx/ssl
openssl req -x509 -newkey rsa:4096 \
    -keyout nginx/ssl/key.pem \
    -out nginx/ssl/cert.pem \
    -days 365 -nodes \
    -subj "/C=PL/ST=Pomorskie/L=Gdansk/O=AI Trading Bot/CN=localhost"

# 2. Let's Encrypt certificate (production)
# Zastąp 'your-domain.com' prawdziwym domain
export DOMAIN=your-domain.com
sudo certbot certonly --standalone -d $DOMAIN
sudo cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/$DOMAIN/privkey.pem nginx/ssl/key.pem

# 3. Auto-renewal setup
echo "0 2 * * * certbot renew --quiet && docker-compose -f docker-compose.prod.yml restart nginx" | sudo crontab -
```

### **Nginx Configuration:**
```bash
# Create nginx config
mkdir -p nginx
cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream trading_bot {
        server trading-bot:8000;
    }
    
    # SSL Configuration
    server {
        listen 443 ssl http2;
        server_name your-domain.com;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        
        # Trading Bot proxy
        location / {
            proxy_pass http://trading_bot;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # WebSocket support
        location /ws {
            proxy_pass http://trading_bot;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
    
    # HTTP to HTTPS redirect
    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }
}
EOF
```

### **Database Security:**
```bash
# PostgreSQL security
cat > config/postgres-security.sql << 'EOF'
-- Create trading user with limited privileges
CREATE USER trading_user WITH PASSWORD 'secure_password_change_me';
CREATE DATABASE trading_bot OWNER trading_user;
GRANT CONNECT ON DATABASE trading_bot TO trading_user;
GRANT USAGE ON SCHEMA public TO trading_user;
GRANT CREATE ON SCHEMA public TO trading_user;

-- Revoke public access
REVOKE ALL ON DATABASE trading_bot FROM public;
REVOKE ALL ON SCHEMA public FROM public;
EOF

# Redis security
cat > config/redis-prod.conf << 'EOF'
# Redis production configuration
bind 127.0.0.1
port 6379
requirepass your_redis_password_here
maxmemory 256mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data
EOF
```

---

## 🎨 **ACCESS & USAGE**

### **System URLs:**
```bash
# Development
http://localhost:8000          # Panel Sterowania
http://localhost:8000/docs     # API Documentation
http://localhost:8000/health   # System Health

# Production
https://your-domain.com        # Panel Sterowania (SSL)
https://your-domain.com/docs   # API Documentation
https://your-domain.com/health # System Health
http://your-domain.com:3000    # Grafana (jeśli włączone)
http://your-domain.com:9090    # Prometheus (jeśli włączone)
```

### **Pierwsze Logowanie:**
```
1. 🌐 Otwórz panel: https://your-domain.com
2. 💳 Sekcja: "Konta & Logowanie"
3. 🏦 Wybierz brokera: "MetaTrader 5"
4. 🛡️ Typ konta: "DEMO" (zalecane dla testów)
5. 👤 Podaj dane logowania
6. ✅ Klik: "Zaloguj i Połącz"
7. 📈 Aktywuj strategię: "Smart Money Concept v1"
8. 📊 Monitor: Dashboard dla live performance
```

---

## 🔧 **TROUBLESHOOTING**

### **Common Issues & Solutions:**

**1. Container Build Failures:**
```bash
# Clear Docker cache
docker system prune -a -f
docker volume prune -f

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

**2. TensorFlow Import Errors:**
```bash
# Check TensorFlow installation
docker-compose exec trading-bot python -c "import tensorflow as tf; print('TF Version:', tf.__version__)"

# Check environment variables
docker-compose exec trading-bot env | grep TF_

# Reset TensorFlow cache
docker-compose exec trading-bot rm -rf /root/.cache/tensorflow
docker-compose restart trading-bot
```

**3. Broker Connection Issues:**
```bash
# Test network connectivity
docker-compose exec trading-bot ping api.sabiotrade.com
docker-compose exec trading-bot ping 8.8.8.8

# Check API credentials
docker-compose logs trading-bot | grep -i auth
docker-compose logs trading-bot | grep -i error

# Verify environment variables
docker-compose exec trading-bot env | grep -E "(MT5|SABIO|ROBO)"
```

**4. Database Connection Problems:**
```bash
# Check PostgreSQL
docker-compose exec postgres pg_isready -U trading_user

# Check Redis
docker-compose exec redis redis-cli ping

# Reset databases
docker-compose down
docker volume rm $(docker volume ls -q | grep trading)
docker-compose up -d
```

**5. Performance Issues:**
```bash
# Monitor resource usage
docker stats --no-stream

# Check memory usage
docker-compose exec trading-bot free -h

# Monitor CPU usage
docker-compose exec trading-bot top

# Check disk space
docker-compose exec trading-bot df -h

# Optimize for production
docker-compose -f docker-compose.prod.yml up -d
```

### **Debugging Commands:**
```bash
# Access container shell
docker-compose exec trading-bot /bin/bash

# Check Python packages
docker-compose exec trading-bot pip list

# Test imports
docker-compose exec trading-bot python -c "import tensorflow, sklearn, pandas, numpy; print('All imports OK')"

# Check file permissions
docker-compose exec trading-bot ls -la /app

# Restart specific service
docker-compose restart trading-bot
```

---

## ☁️ **CLOUD DEPLOYMENT**

### **AWS Deployment:**
```bash
# 1. Launch EC2 instance
# - Instance type: t3.large (2 vCPU, 8GB RAM)
# - Storage: 50GB GP2 SSD
# - Security group: ports 22, 80, 443, 8000
# - Key pair: create and download

# 2. Connect to instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu

# 4. Deploy system
git clone https://github.com/szarastrefa/AI-ML-Trading-Bot.git
cd AI-ML-Trading-Bot
cp .env.example .env
# Edit .env with your settings

# 5. Deploy
docker-compose -f docker-compose.prod.yml up -d --build

# 6. Configure DNS
# Point your domain to EC2 public IP

# 7. Setup SSL
sudo certbot certonly --standalone -d your-domain.com
```

### **Google Cloud Platform:**
```bash
# 1. Create VM instance
gcloud compute instances create ai-trading-bot \
    --machine-type=e2-standard-4 \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --zone=europe-west1-b

# 2. SSH and setup
gcloud compute ssh ai-trading-bot --zone=europe-west1-b

# 3. Install and deploy (same as AWS steps 3-7)
```

### **Azure Deployment:**
```bash
# 1. Create VM
az vm create \
    --resource-group ai-trading-rg \
    --name ai-trading-vm \
    --image UbuntuLTS \
    --size Standard_D2s_v3 \
    --admin-username azureuser \
    --generate-ssh-keys

# 2. SSH and deploy (same process)
```

---

## 📊 **PERFORMANCE OPTIMIZATION**

### **System Tuning:**
```bash
# 1. Docker optimization
cat > /etc/docker/daemon.json << 'EOF'
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "storage-driver": "overlay2",
    "dns": ["8.8.8.8", "8.8.4.4"]
}
EOF
sudo systemctl restart docker

# 2. System optimization
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'net.core.rmem_max=134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max=134217728' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# 3. TensorFlow optimization
export TF_CPP_MIN_LOG_LEVEL=1
export OMP_NUM_THREADS=4
export TF_NUM_INTEROP_THREADS=2
export TF_NUM_INTRAOP_THREADS=4
```

### **Resource Monitoring:**
```bash
# Create monitoring script
cat > performance-monitor.sh << 'EOF'
#!/bin/bash
echo "AI/ML Trading Bot v5.0 - Performance Monitor"
echo "========================================="

while true; do
    clear
    echo "=== $(date) ==="
    
    # System resources
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//'
    
    echo "Memory Usage:"
    free -h | grep Mem | awk '{print "Used: " $3 "/" $2 " (" $3/$2*100 "%)"'}
    
    echo "Disk Usage:"
    df -h / | tail -1 | awk '{print "Used: " $3 "/" $2 " (" $5 "%)"'}
    
    # Docker stats
    echo "\nDocker Container Stats:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
    
    # Trading Bot health
    echo "\nTrading Bot Health:"
    curl -s http://localhost:8000/health | jq -r '.status, .system_stats.accounts.total_count, .system_stats.strategies.active_count'
    
    sleep 10
done
EOF
chmod +x performance-monitor.sh
```

---

## ⚠️ **PRODUCTION CHECKLIST**

### **Pre-Deployment:**
- [ ] 💻 System requirements verified
- [ ] 🐳 Docker & Docker Compose installed
- [ ] 🌐 Domain DNS configured
- [ ] 🔐 SSL certificates generated
- [ ] ⚙️ Environment variables configured
- [ ] 🔑 API keys and secrets set
- [ ] 💾 Backup procedures tested

### **Post-Deployment:**
- [ ] ✅ System health check passed
- [ ] 📈 All services running
- [ ] 🔗 Database connections working
- [ ] 🧠 ML models loading correctly
- [ ] 📊 Dashboard accessible
- [ ] 🔐 HTTPS working properly
- [ ] 🚨 Emergency stop tested
- [ ] 💾 Backup automation configured
- [ ] 📈 Monitoring alerts setup
- [ ] 📝 Documentation updated

### **Security Checklist:**
- [ ] 🔑 Strong passwords for all accounts
- [ ] 🔐 SSL/TLS certificates valid
- [ ] 🏠 Firewall properly configured
- [ ] 💳 API keys secured and not in code
- [ ] 🔄 Regular security updates scheduled
- [ ] 💾 Backup encryption enabled
- [ ] 📈 Monitoring and alerting active
- [ ] 👤 Access control implemented
- [ ] 📝 Security logs monitored
- [ ] 🔍 Regular security audits planned

---

## 🎆 **SUCCESS VERIFICATION**

### **Deployment Success Test:**
```bash
# Complete system test
cat > test-deployment.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing AI/ML Trading Bot v5.0 Deployment"

# 1. Health check
echo "1. System Health Check..."
HEALTH=$(curl -s http://localhost:8000/health | jq -r '.status')
if [ "$HEALTH" = "healthy" ]; then
    echo "✅ System is healthy"
else
    echo "❌ System health check failed"
    exit 1
fi

# 2. API connectivity
echo "2. API Connectivity Test..."
API_TEST=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/v5/dashboard)
if [ "$API_TEST" = "200" ]; then
    echo "✅ API is responding"
else
    echo "❌ API connectivity failed"
    exit 1
fi

# 3. Database connectivity
echo "3. Database Test..."
DB_TEST=$(docker-compose exec -T postgres pg_isready -U trading_user)
if [[ $DB_TEST == *"accepting connections"* ]]; then
    echo "✅ Database is connected"
else
    echo "❌ Database connection failed"
fi

# 4. ML models test
echo "4. ML Models Test..."
ML_TEST=$(curl -s http://localhost:8000/api/v5/ml-models | jq -r '.success')
if [ "$ML_TEST" = "true" ]; then
    echo "✅ ML models are loaded"
else
    echo "❌ ML models loading failed"
fi

echo "
🎉 DEPLOYMENT SUCCESSFUL!"
echo "Panel Sterowania: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
EOF
chmod +x test-deployment.sh
./test-deployment.sh
```

---

## 📞 **SUPPORT & KONTAKT**

### **Deployment Support:**
- **🐛 Issues**: [GitHub Issues](https://github.com/szarastrefa/AI-ML-Trading-Bot/issues)
- **💬 Discord**: [AI Trading Community](https://discord.gg/ai-trading)
- **📧 Email**: deployment-support@protonmail.com
- **📚 Wiki**: [Deployment Wiki](https://github.com/szarastrefa/AI-ML-Trading-Bot/wiki)

### **Professional Services:**
- **🌍 Custom Deployment**: Enterprise deployment service
- **🔧 Technical Support**: 24/7 technical support
- **🏆 Training**: Professional trading bot training
- **⚙️ Customization**: Custom strategy development

---

<div align="center">

**🚀 AI/ML Trading Bot v5.0 - PRODUCTION DEPLOYMENT SUCCESS!**

*Kompletny system AI/ML trading z profesjonalnym panelem sterowania*

**🎉 READY FOR PROFESSIONAL TRADING!**

</div>