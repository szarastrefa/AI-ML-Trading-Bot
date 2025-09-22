#!/bin/bash
# AI/ML Trading Bot v3.0 - Professional Web GUI Startup Script

echo "🚀 Starting AI/ML Trading Bot v3.0 - Professional Web GUI..."
echo "📝 Current directory: $(pwd)"
echo "🐍 Python version: $(python --version)"

# Set Python path
export PYTHONPATH="/app:$PYTHONPATH"

# Create required directories
echo "📁 Creating directories..."
mkdir -p data/{models,cache,logs} logs tmp

# Test core imports
echo "🧪 Testing core imports..."
python -c "
import sys
sys.path.insert(0, '/app')

print('🔍 Testing imports...')
try:
    import fastapi
    print(f'✅ FastAPI: {fastapi.__version__}')
except ImportError as e:
    print(f'❌ FastAPI import failed: {e}')
    exit(1)

try:
    import uvicorn
    print('✅ Uvicorn: Available')
except ImportError as e:
    print(f'❌ Uvicorn import failed: {e}')
    exit(1)

try:
    import pandas as pd
    print(f'✅ Pandas: {pd.__version__}')
except ImportError:
    print('⚠️ Pandas: Not available (optional)')

try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
except ImportError:
    print('⚠️ NumPy: Not available (optional)')

print('✅ Core imports successful!')
"

if [ $? -ne 0 ]; then
    echo "❌ Import test failed - check dependencies"
    exit 1
fi

echo "🌐 Starting Professional Web GUI..."
echo "📈 Dashboard will be available at: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "❤️ Health Check: http://localhost:8000/health"
echo "---------------------------------------------------"

# Change to app directory
cd /app

# Start the application
exec python app/main.py