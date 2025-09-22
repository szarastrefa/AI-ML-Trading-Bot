# AI/ML Trading Bot - STABLE VERSION (Python 3.10 + Fixed)
FROM python:3.10-slim-bullseye

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies + curl for healthcheck
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
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Update ldconfig for TA-Lib
RUN ldconfig

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY config/ ./config/ 
COPY scripts/ ./scripts/

# Create directories
RUN mkdir -p data/{logs,models,historical,backtest,live,cache} logs

# Set permissions
RUN chmod +x scripts/*.sh || true

# Verify TA-Lib installation
RUN python -c "import talib; print(f'TA-Lib installed: {getattr(talib, \"__version__\", \"OK\")}')"

# Health check with curl
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000 8080

CMD ["python", "app/main.py"]
