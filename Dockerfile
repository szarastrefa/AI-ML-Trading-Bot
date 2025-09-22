# AI/ML Trading Bot v2.1 - CONDA-BASED TA-LIB INSTALLATION
# Most reliable approach for TA-Lib compatibility
FROM continuumio/miniconda3:24.1.2-0

# Environment variables for stability
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DEFAULT_ENV=trading
ENV PATH=/opt/conda/envs/trading/bin:$PATH

WORKDIR /app

# Install system dependencies (minimal for conda approach)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment with Python 3.10
RUN conda create -n trading python=3.10 -y

# Activate environment for subsequent commands
SHELL ["conda", "run", "-n", "trading", "/bin/bash", "-c"]

# Install TA-Lib using conda-forge (pre-compiled, most reliable)
# This avoids all compilation issues with NumPy compatibility
RUN conda install -c conda-forge ta-lib=0.4.28 -y

# Install NumPy version compatible with TA-Lib via conda
RUN conda install -c conda-forge numpy=1.24.4 -y

# Verify conda installations - FIXED syntax
RUN python -c "import numpy; print('NumPy: ' + numpy.__version__)"
RUN python -c "import talib; print('TA-Lib: ' + str(getattr(talib, '__version__', 'installed')))"

# Copy requirements and install remaining Python packages via pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY config/ ./config/
COPY scripts/ ./scripts/

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

# Final verification of all critical components - FIXED syntax
RUN echo "=== FINAL VERIFICATION ===" && \
    python -c "import talib; import numpy; import pandas; import fastapi; print('✅ All critical imports successful')" && \
    python -c "import talib; print('✅ TA-Lib version: ' + str(getattr(talib, '__version__', 'installed')))" && \
    python -c "import numpy; print('✅ NumPy version: ' + numpy.__version__)" && \
    python -c "import pandas; print('✅ Pandas version: ' + pandas.__version__)" && \
    echo "=== SYSTEM READY ==="

# Health check with conda environment
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8080

# Use conda environment for startup
SHELL ["conda", "run", "-n", "trading", "/bin/bash", "-c"]
CMD ["python", "app/main.py"]