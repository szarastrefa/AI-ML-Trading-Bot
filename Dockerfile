# AI/ML Trading Bot - Dockerfile (Python 3.9 for MetaTrader5 compatibility)
FROM python:3.9-slim

# Ustaw zmienne środowiskowe
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Ustaw katalog roboczy
WORKDIR /app

# Zainstaluj systemowe zależności
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    unzip \
    libc6-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Skopiuj pliki requirements
COPY requirements.txt .

# Zaktualizuj pip i zainstaluj zależności Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj kod aplikacji
COPY app/ ./app/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Stwórz katalogi danych
RUN mkdir -p data/logs data/models data/historical

# Ustaw uprawnienia
RUN chmod +x scripts/*.py || true

# Expuj porty
EXPOSE 8000 8080

# Komenda startowa
CMD ["python", "app/main.py"]