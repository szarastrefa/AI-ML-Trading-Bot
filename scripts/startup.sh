#!/bin/bash

# AI/ML Trading Bot v2.0 - Enhanced Startup Script
# pandas-ta-classic Edition with Smart Money Concepts
# Author: AI/ML Trading Bot Team
# Version: 2.0.0

set -euo pipefail

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# System Configuration
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR=/tmp/numba_cache

# Application Metadata
APP_NAME="AI/ML Trading Bot v2.0"
APP_VERSION="2.0.0"
STARTUP_TIMEOUT=300
HEALTH_CHECK_RETRIES=30

# Directory Paths
APP_DIR="/app"
LOGS_DIR="/app/logs"
DATA_DIR="/app/data"
MODELS_DIR="/app/models"
CACHE_DIR="/app/cache"

# Database Configuration
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_DB="${POSTGRES_DB:-trading_bot_v2}"
POSTGRES_USER="${POSTGRES_USER:-trading_bot}"
REDIS_HOST="${REDIS_HOST:-redis}"
REDIS_PORT="${REDIS_PORT:-6379}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${PURPLE}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    fi
}

# =============================================================================
# BANNER & SYSTEM INFO
# =============================================================================

display_banner() {
    cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════════════╗
║                          🤖 AI/ML Trading Bot v2.0                          ║
║                        pandas-ta-classic Edition                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🐍 Python 3.11     📊 150+ Indicators     🧠 Smart Money Concepts         ║
║  ⚡ TensorFlow 2.15  🚀 Multiprocessing     📈 Advanced ML Pipeline         ║
║  🔍 vectorbt Ready  💹 Real-time Trading   📊 Prometheus Monitoring         ║
╚══════════════════════════════════════════════════════════════════════════════╝

EOF
}

show_system_info() {
    log_info "🖥️  System Information:"
    echo "    - Hostname: $(hostname)"
    echo "    - OS: $(uname -s) $(uname -r)"
    echo "    - Architecture: $(uname -m)"
    echo "    - CPUs: $(nproc)"
    echo "    - Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
    echo
    
    log_info "🐍 Python Environment:"
    echo "    - Python Version: $(python --version 2>&1)"
    echo "    - pip Version: $(pip --version | cut -d' ' -f2)"
    echo "    - Working Directory: $(pwd)"
    echo
}

# =============================================================================
# DIRECTORY & ENVIRONMENT SETUP
# =============================================================================

setup_directories() {
    log_info "📁 Setting up application directories..."
    
    local dirs=(
        "${LOGS_DIR}"
        "${DATA_DIR}/historical"
        "${DATA_DIR}/live" 
        "${DATA_DIR}/backtest"
        "${DATA_DIR}/signals"
        "${MODELS_DIR}/tensorflow"
        "${MODELS_DIR}/xgboost"
        "${MODELS_DIR}/ensemble"
        "${CACHE_DIR}/indicators"
        "${CACHE_DIR}/market_data"
        "/tmp/numba_cache"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_debug "Created directory: $dir"
        fi
    done
    
    # Set permissions
    chmod 755 "${LOGS_DIR}" "${DATA_DIR}" "${MODELS_DIR}" "${CACHE_DIR}"
    chmod 777 "/tmp/numba_cache"
    
    log_success "✅ Directories setup completed"
}

optimize_performance() {
    log_info "⚡ Applying performance optimizations..."
    
    # Python optimizations
    export PYTHONOPTIMIZE=1
    
    # Numba JIT compilation
    export NUMBA_NUM_THREADS=$(nproc)
    export NUMBA_THREADING_LAYER="omp"
    export NUMBA_CACHE_DIR="/tmp/numba_cache"
    
    # TensorFlow optimizations
    export TF_CPP_MIN_LOG_LEVEL=2
    export TF_ENABLE_ONEDNN_OPTS=1
    
    # pandas-ta-classic multiprocessing
    export PANDAS_TA_CORES=0  # Use all cores
    
    log_success "✅ Performance optimizations applied"
}

# =============================================================================
# DEPENDENCY CHECKS
# =============================================================================

check_python_version() {
    log_info "🐍 Checking Python version..."
    
    local python_version
    python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    
    if [[ "$python_version" == "3.11" ]]; then
        log_success "✅ Python $python_version detected (Optimal)"
    elif [[ "$python_version" =~ ^3\.(9|10|12|13)$ ]]; then
        log_success "✅ Python $python_version detected (Compatible)"
    else
        log_error "❌ Python $python_version not supported. Requires Python 3.9-3.13"
        return 1
    fi
}

check_dependencies() {
    log_info "📦 Checking critical dependencies..."
    
    local critical_packages=(
        "pandas_ta_classic:pandas-ta-classic with 150+ indicators"
        "pandas:Data manipulation framework"
        "numpy:Numerical computing"
        "fastapi:Web framework"
        "sqlalchemy:Database ORM"
        "redis:Cache and message broker"
        "tensorflow:Machine learning framework"
        "xgboost:Gradient boosting"
    )
    
    local missing_packages=()
    
    for package_info in "${critical_packages[@]}"; do
        local package_name=${package_info%%:*}
        local package_desc=${package_info##*:}
        
        if python -c "import $package_name" 2>/dev/null; then
            local version
            version=$(python -c "import $package_name; print(getattr($package_name, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
            log_success "  ✅ $package_name ($version) - $package_desc"
        else
            log_error "  ❌ $package_name - $package_desc"
            missing_packages+=("$package_name")
        fi
    done
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        log_error "❌ Missing critical packages: ${missing_packages[*]}"
        return 1
    fi
    
    log_success "✅ All critical dependencies available"
}

# =============================================================================
# DATABASE CONNECTIVITY
# =============================================================================

wait_for_database() {
    log_info "🗄️  Waiting for PostgreSQL database..."
    
    local retries=0
    local max_retries=30
    local delay=5
    
    while [[ $retries -lt $max_retries ]]; do
        if python -c "
import psycopg2
import sys
try:
    conn = psycopg2.connect(
        host='$POSTGRES_HOST',
        database='$POSTGRES_DB',
        user='$POSTGRES_USER',
        password='${POSTGRES_PASSWORD:-}',
        connect_timeout=5
    )
    cursor = conn.cursor()
    cursor.execute('SELECT version();')
    version = cursor.fetchone()[0]
    print(f'PostgreSQL: {version[:50]}...')
    cursor.close()
    conn.close()
    sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null; then
            log_success "✅ PostgreSQL database is ready"
            return 0
        else
            ((retries++))
            log_warning "⏳ Database not ready (attempt $retries/$max_retries), waiting ${delay}s..."
            sleep $delay
        fi
    done
    
    log_error "❌ Database connection failed after $max_retries attempts"
    return 1
}

wait_for_redis() {
    log_info "🔴 Waiting for Redis cache..."
    
    local retries=0
    local max_retries=20
    local delay=3
    
    while [[ $retries -lt $max_retries ]]; do
        if python -c "
import redis
import sys
try:
    r = redis.Redis(host='$REDIS_HOST', port=$REDIS_PORT, db=0, socket_connect_timeout=3)
    r.ping()
    info = r.info()
    print(f'Redis {info.get(\"redis_version\")} - Memory: {info.get(\"used_memory_human\")}')
    sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null; then
            log_success "✅ Redis cache is ready"
            return 0
        else
            ((retries++))
            log_warning "⏳ Redis not ready (attempt $retries/$max_retries), waiting ${delay}s..."
            sleep $delay
        fi
    done
    
    log_error "❌ Redis connection failed after $max_retries attempts"
    return 1
}

# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

initialize_database() {
    log_info "🔧 Checking database schema..."
    
    # Check if database is already initialized
    if python -c "
import psycopg2
import sys
try:
    conn = psycopg2.connect(
        host='$POSTGRES_HOST',
        database='$POSTGRES_DB',
        user='$POSTGRES_USER',
        password='${POSTGRES_PASSWORD:-}'
    )
    cursor = conn.cursor()
    cursor.execute(\"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public' AND table_name='trading_signals';\")
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    if count > 0:
        sys.exit(0)
    else:
        sys.exit(1)
except Exception:
    sys.exit(2)
" 2>/dev/null; then
        log_success "✅ Database schema already exists"
        return 0
    fi
    
    # Initialize database schema
    if [[ -f "${APP_DIR}/scripts/create_trading_tables.sql" ]]; then
        log_info "📋 Initializing database schema..."
        
        if python -c "
import psycopg2
try:
    conn = psycopg2.connect(
        host='$POSTGRES_HOST',
        database='$POSTGRES_DB',
        user='$POSTGRES_USER',
        password='${POSTGRES_PASSWORD:-}'
    )
    cursor = conn.cursor()
    with open('${APP_DIR}/scripts/create_trading_tables.sql', 'r') as f:
        cursor.execute(f.read())
    conn.commit()
    cursor.close()
    conn.close()
except Exception as e:
    print(f'Schema creation failed: {e}')
    exit(1)
"; then
            log_success "✅ Database schema initialized successfully"
        else
            log_error "❌ Database schema initialization failed"
            return 1
        fi
    else
        log_warning "⚠️  Database schema file not found"
    fi
}

# =============================================================================
# HEALTH CHECKS
# =============================================================================

perform_health_checks() {
    log_info "🔍 Performing system health checks..."
    
    local health_issues=()
    
    # Check disk space
    local disk_usage
    disk_usage=$(df "${APP_DIR}" | tail -1 | awk '{print $5}' | sed 's/%//')
    if [[ $disk_usage -gt 90 ]]; then
        health_issues+=("High disk usage: ${disk_usage}%")
    fi
    
    # Check memory usage
    local memory_usage
    memory_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [[ $memory_usage -gt 90 ]]; then
        health_issues+=("High memory usage: ${memory_usage}%")
    fi
    
    # Check critical files
    local critical_files=(
        "${APP_DIR}/app/main.py"
        "${APP_DIR}/config/settings.py"
        "${APP_DIR}/requirements.txt"
    )
    
    for file in "${critical_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            health_issues+=("Missing file: $file")
        fi
    done
    
    if [[ ${#health_issues[@]} -eq 0 ]]; then
        log_success "✅ All health checks passed"
        return 0
    else
        log_error "❌ Health check issues:"
        for issue in "${health_issues[@]}"; do
            log_error "    - $issue"
        done
        return 1
    fi
}

# =============================================================================
# APPLICATION STARTUP
# =============================================================================

start_application() {
    log_info "🚀 Starting $APP_NAME..."
    
    # Change to app directory
    cd "${APP_DIR}" || {
        log_error "❌ Failed to change to app directory: ${APP_DIR}"
        return 1
    }
    
    # Set Python path
    export PYTHONPATH="${APP_DIR}:${PYTHONPATH:-}"
    
    # Create startup log
    python -c "
import json
from datetime import datetime
startup_info = {
    'timestamp': datetime.utcnow().isoformat(),
    'version': '$APP_VERSION',
    'features': {
        'pandas_ta_classic': True,
        'smart_money_concepts': True,
        'machine_learning': True,
        'real_time_trading': True
    }
}
with open('${LOGS_DIR}/startup.json', 'w') as f:
    json.dump(startup_info, f, indent=2)
" 2>/dev/null || true
    
    log_success "✅ Application environment ready"
    
    # Display startup summary
    echo
    log_info "📋 Startup Summary:"
    echo "    🐍 Python $(python --version 2>&1 | cut -d' ' -f2) + pandas-ta-classic"
    echo "    🗄️  Database: $POSTGRES_DB@$POSTGRES_HOST"
    echo "    🔴 Cache: Redis@$REDIS_HOST:$REDIS_PORT"
    echo "    📁 Data: $DATA_DIR"
    echo "    📊 Models: $MODELS_DIR"
    echo
    
    log_info "🌐 Web server starting..."
    echo "    📍 API Documentation: http://localhost:8000/docs"
    echo "    ❤️  Health Check: http://localhost:8000/health"
    echo "    📊 Metrics: http://localhost:8000/metrics"
    echo
    
    # Start the application
    exec python app/main.py "$@"
}

# =============================================================================
# ERROR HANDLING
# =============================================================================

handle_error() {
    local exit_code=$?
    local line_number=$1
    
    log_error "❌ Startup failed at line $line_number (exit code: $exit_code)"
    
    # Log error details
    {
        echo "Startup Error Details:"
        echo "  Timestamp: $(date)"
        echo "  Exit Code: $exit_code"
        echo "  Line: $line_number"
        echo "  Directory: $(pwd)"
        echo "  Environment:"
        env | grep -E "^(POSTGRES|REDIS|PYTHON|APP)" || true
    } >> "${LOGS_DIR}/startup_error.log" 2>/dev/null || true
    
    echo
    log_error "🔧 Troubleshooting:"
    echo "    1. Check services: docker-compose ps"
    echo "    2. View logs: docker-compose logs trading-bot"
    echo "    3. Database logs: docker-compose logs postgres"
    echo "    4. Redis logs: docker-compose logs redis"
    echo
    
    exit $exit_code
}

trap 'handle_error ${LINENO}' ERR

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    display_banner
    
    log_info "🚀 Initializing $APP_NAME..."
    log_info "📅 Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    log_info "🆔 Process ID: $$"
    echo
    
    # System information
    show_system_info
    
    # Phase 1: Environment Setup
    log_info "🔧 PHASE 1: Environment Setup"
    setup_directories
    check_python_version
    optimize_performance
    echo
    
    # Phase 2: Dependencies
    log_info "📦 PHASE 2: Dependency Verification"
    check_dependencies
    echo
    
    # Phase 3: Infrastructure
    log_info "🗄️  PHASE 3: Infrastructure Connectivity"
    wait_for_database
    wait_for_redis
    initialize_database
    echo
    
    # Phase 4: Health Checks
    log_info "🔍 PHASE 4: Health Verification"
    perform_health_checks
    echo
    
    # Phase 5: Application Launch
    log_info "🚀 PHASE 5: Application Launch"
    start_application "$@"
}

# Execute main if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi