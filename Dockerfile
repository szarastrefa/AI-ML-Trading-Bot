# AI/ML Trading Bot - Python 3.11 + pandas-ta-classic optimized
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

WORKDIR /app

# Install system dependencies + TA-Lib compilation tools
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
    gfortran \
    libblas3 \
    libblas-dev \
    liblapack3 \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source for maximum compatibility with pandas-ta-classic
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create comprehensive data directory structure
RUN mkdir -p \
    data/logs \
    data/models \
    data/historical \
    data/backtest \
    data/live \
    data/cache \
    /tmp/numba_cache

# Set permissions for all scripts
RUN chmod +x scripts/*.py scripts/*.sh || true

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000 8080

# Use startup script for proper service orchestration
CMD ["bash", "scripts/start.sh"]