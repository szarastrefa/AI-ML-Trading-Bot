# AI/ML Trading Bot - RESEARCH-BASED STABLE VERSION
# Python 3.10 + Debian Bullseye for maximum TA-Lib compatibility
FROM python:3.10-slim-bullseye

# Environment variables for stability
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies for TA-Lib compilation
# Research: TA-Lib needs specific build environment
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    unzip \
    libc6-dev \
    gcc \
    g++ \
    libssl-dev \
    libffi-dev \
    make \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source (most reliable method)
# Research: Source compilation ensures compatibility
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy requirements and install Python packages
COPY requirements.txt .

# Install Python dependencies (order matters for stability)
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY config/ ./config/ || true
COPY scripts/ ./scripts/ || true

# Create comprehensive directory structure
RUN mkdir -p \
    data/logs \
    data/models \
    data/historical \
    data/backtest \
    data/live \
    data/cache \
    logs \
    tmp

# Set proper permissions
RUN chmod -R 755 app/ && \
    chmod +x scripts/*.py scripts/*.sh 2>/dev/null || true && \
    chmod 777 tmp/

# Health check for Docker
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8080

# Use optimized startup
CMD ["python", "app/main.py"]