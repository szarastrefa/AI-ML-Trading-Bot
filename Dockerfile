# AI/ML Trading Bot v5.0 - PROFESSIONAL CONTROL PANEL
# Optimized Dockerfile for production deployment

FROM python:3.10-slim as base

# Set environment variables for Python and TensorFlow optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# TensorFlow optimization environment variables
ENV TF_USE_LEGACY_KERAS=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    OMP_NUM_THREADS=4

# Create app user for security (non-root)
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install system dependencies (build stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    make \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y

# Upgrade pip to latest version
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first (better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy application code
COPY app/ ./app/
COPY config/ ./config/

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/logs /app/backups /app/models /app/tmp \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check for v5.0 professional panel
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8001

# Labels for container identification
LABEL version="5.0.0" \
      description="AI/ML Trading Bot v5.0 - Professional Control Panel" \
      maintainer="AI Trading Bot Team" \
      github="https://github.com/szarastrefa/AI-ML-Trading-Bot"

# Start command - Professional Control Panel v5.0
CMD ["python", "-u", "app/main.py"]

# =============================================================================
# BUILD INSTRUCTIONS:
# =============================================================================
# 
# Development build:
#   docker build -t ai-trading-bot:dev .
#   docker run -p 8000:8000 ai-trading-bot:dev
# 
# Production build:
#   docker build -t ai-trading-bot:v5.0 .
#   docker-compose up -d --build
# 
# Access:
#   - Control Panel: http://localhost:8000
#   - API Docs: http://localhost:8000/docs
#   - Health Check: http://localhost:8000/health
# 
# Features:
#   ✅ Professional Control Panel with 10 sections
#   ✅ Multi-broker authentication (13+ brokers)
#   ✅ 4 AI/ML Trading Strategies
#   ✅ 6 ML Models (TensorFlow + Scikit-learn)
#   ✅ Real-time monitoring and logging
#   ✅ Emergency controls and risk management
#   ✅ Production-ready deployment
# =============================================================================