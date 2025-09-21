#!/bin/bash

# AI/ML Trading Bot v2.0.1 - Fixed Startup Script
set -e

echo "🚀 Starting AI/ML Trading Bot v2.0.1 (Fixed Version)..."

# Set environment variables
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Create directories
mkdir -p data/{logs,models,historical,backtest,live,cache}
mkdir -p logs

# Test imports
echo "🔍 Testing critical imports..."
python -c "
import pandas_ta as ta
import tensorflow as tf
import pandas as pd
import numpy as np
print(f'✅ pandas_ta: {getattr(ta, \"version\", \"stable\")}')
print(f'✅ TensorFlow: {tf.__version__}')
print(f'✅ pandas: {pd.__version__}')
print(f'✅ numpy: {np.__version__}')
print('✅ All dependencies working!')
"

echo "🐍 Starting Python application..."
cd /app
exec python app/main.py
