#!/bin/bash
# Startup script for AI/ML Trading Bot

echo "?? Starting AI/ML Trading Bot v2.1..."
echo "?? Current directory: $(pwd)"
echo "?? Python path: $PYTHONPATH"

# Set Python path
export PYTHONPATH="/app:$PYTHONPATH"

# Create directories
mkdir -p data/{logs,models,historical,backtest,live,cache} logs tmp

# Test critical imports
echo "?? Testing imports..."
python -c "
import sys
sys.path.insert(0, '/app')

print('Testing core imports...')
import numpy as np
import pandas as pd
print(f'? NumPy: {np.__version__}')
print(f'? Pandas: {pd.__version__}')

try:
    import talib
    print('? TA-Lib: Available')
except ImportError:
    print('?? TA-Lib: Using fallback')

try:
    from app.strategies.talib_stable_strategy import TALibStableStrategy
    strategy = TALibStableStrategy({})
    print(f'? Strategy: {strategy.name}')
except Exception as e:
    print(f'? Strategy import error: {e}')
    exit(1)

print('? All imports successful!')
"

if [ $? -ne 0 ]; then
    echo "? Import test failed - check dependencies"
    exit 1
fi

echo "?? Starting main application..."
cd /app
exec python app/main.py
