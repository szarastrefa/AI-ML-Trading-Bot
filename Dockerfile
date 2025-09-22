# AI/ML Trading Bot v2.1 - SIMPLE WORKING VERSION
# Minimal approach - no verification steps that cause syntax errors
FROM python:3.10-slim-bullseye

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

# Install NumPy first (critical for TA-Lib)
RUN pip install --upgrade pip && pip install numpy==1.24.4

# Install TA-Lib from wheel (most reliable)
RUN pip install --find-links https://github.com/MrJbdu4/ta-lib-python/releases/latest/download/ TA-Lib

# Copy and install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY app/ ./app/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create directories
RUN mkdir -p data/{logs,models,historical,backtest,live,cache} logs tmp

# Simple test without f-strings
RUN python -c "import talib; import numpy; import pandas; print('All imports OK')"

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["bash", "scripts/start.sh"]