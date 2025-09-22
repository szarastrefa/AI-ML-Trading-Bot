#!/bin/bash
# ============================================================================
# AI/ML Trading Bot - CRITICAL TA-LIB HOTFIX SCRIPT
# Addresses NumPy 2.0+ compatibility issues with TA-Lib 0.4.28
# ============================================================================

set -e
echo "🚑 CRITICAL TA-LIB HOTFIX - Starting..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Ensure we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found. Please run from project root."
    exit 1
fi

echo "🗑️ Cleaning existing containers and images..."
# Complete cleanup
docker-compose down -v 2>/dev/null || true
docker system prune -a -f
docker volume prune -f
docker network prune -f

echo "🔄 Pulling latest changes from GitHub..."
git pull origin main

echo "🐳 Building with conda-based approach (primary method)..."
echo "This will take 5-10 minutes for first build..."

# Try primary conda approach
echo "🚀 Starting build with conda-based Dockerfile..."
if docker-compose up -d --build; then
    echo "✅ SUCCESS: Conda-based build completed!"
    
    # Wait for services to start
    echo "⏳ Waiting for services to initialize..."
    sleep 60
    
    # Health check
    echo "🎯 Testing system health..."
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "✅ SUCCESS: System is healthy and responsive!"
        echo "
🎆 SYSTEM FULLY OPERATIONAL!"
        echo "🌐 API Documentation: http://localhost:8000/docs"
        echo "❤️ Health Check: http://localhost:8000/health"
        echo "📊 Strategy Info: http://localhost:8000/api/v1/strategy/info"
        echo "🔢 Available Indicators: http://localhost:8000/api/v1/indicators/available"
        exit 0
    else
        echo "⚠️ Conda build succeeded but health check failed. Trying alternative..."
    fi
else
    echo "⚠️ Conda build failed. Trying alternative approach..."
fi

echo "🔄 Switching to alternative pre-compiled wheel approach..."

# Backup current Dockerfile and switch to alternative
cp Dockerfile Dockerfile.conda.backup
cp Dockerfile.alternative Dockerfile

echo "🐳 Building with pre-compiled wheel approach..."
if docker-compose up -d --build; then
    echo "✅ Alternative build completed!"
    
    # Wait for services
    sleep 60
    
    # Health check
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "✅ SUCCESS: Alternative approach working!"
        echo "
🎆 SYSTEM OPERATIONAL WITH ALTERNATIVE METHOD!"
        echo "🌐 API Documentation: http://localhost:8000/docs"
        exit 0
    else
        echo "❌ Both approaches failed health check"
    fi
else
    echo "❌ Alternative build also failed"
fi

# Restore original Dockerfile
cp Dockerfile.conda.backup Dockerfile
rm -f Dockerfile.conda.backup

echo "❌ CRITICAL: Both TA-Lib installation methods failed."
echo "🔧 Troubleshooting steps:"
echo "1. Check Docker memory allocation (should be 4GB+)"
echo "2. Check internet connectivity for package downloads"
echo "3. Try manual build: docker build -t trading-bot ."
echo "4. Check logs: docker-compose logs trading-bot"
echo "
For support, please check:"
echo "- https://github.com/szarastrefa/AI-ML-Trading-Bot/issues"
echo "- System requirements in README.md"

exit 1