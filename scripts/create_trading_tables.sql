-- AI/ML Trading Bot v2.0 Database Schema
-- Complete SQL schema for pandas-ta-classic with Smart Money Concepts
-- PostgreSQL 15+ compatible

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- ===================================================================
-- CORE TRADING TABLES
-- ===================================================================

-- Trading Instruments (Forex, Crypto, Stocks, Indices)
CREATE TABLE IF NOT EXISTS trading_instruments (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(100),
    instrument_type VARCHAR(20) NOT NULL, -- 'forex', 'crypto', 'stocks', 'indices'
    base_currency VARCHAR(10),
    quote_currency VARCHAR(10),
    pip_value DECIMAL(10,8),
    min_trade_size DECIMAL(15,8),
    max_trade_size DECIMAL(15,8),
    spread_avg DECIMAL(6,3),
    session_hours JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trading Accounts
CREATE TABLE IF NOT EXISTS trading_accounts (
    id SERIAL PRIMARY KEY,
    account_uuid UUID DEFAULT uuid_generate_v4(),
    broker VARCHAR(50) NOT NULL,
    account_number VARCHAR(50),
    account_type VARCHAR(20) DEFAULT 'demo', -- 'demo', 'live'
    base_currency VARCHAR(10) DEFAULT 'USD',
    initial_balance DECIMAL(15,2) DEFAULT 10000.00,
    current_balance DECIMAL(15,2) DEFAULT 10000.00,
    equity DECIMAL(15,2) DEFAULT 10000.00,
    free_margin DECIMAL(15,2) DEFAULT 10000.00,
    margin_level DECIMAL(8,2) DEFAULT 100.00,
    leverage INTEGER DEFAULT 100,
    max_drawdown DECIMAL(8,4) DEFAULT 0.0000,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trading Strategies
CREATE TABLE IF NOT EXISTS trading_strategies (
    id SERIAL PRIMARY KEY,
    strategy_uuid UUID DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    strategy_type VARCHAR(50) NOT NULL, -- 'pandas_ta_classic', 'smc', 'ml_ensemble'
    description TEXT,
    configuration JSONB NOT NULL,
    timeframes TEXT[] NOT NULL,
    supported_instruments TEXT[] NOT NULL,
    risk_parameters JSONB,
    performance_stats JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===================================================================
-- TRADING SIGNALS & ANALYSIS (pandas-ta-classic + SMC)
-- ===================================================================

-- Trading Signals with comprehensive analysis
CREATE TABLE IF NOT EXISTS trading_signals (
    id SERIAL PRIMARY KEY,
    signal_uuid UUID DEFAULT uuid_generate_v4(),
    strategy_id INTEGER REFERENCES trading_strategies(id),
    instrument_id INTEGER REFERENCES trading_instruments(id),
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL, -- 'M1', 'M5', 'M15', 'H1', 'H4', 'D1'
    signal_type VARCHAR(10) NOT NULL, -- 'BUY', 'SELL', 'HOLD'
    confidence DECIMAL(5,2) NOT NULL, -- 0.00 to 100.00
    
    -- Price levels
    entry_price DECIMAL(12,6) NOT NULL,
    stop_loss DECIMAL(12,6),
    take_profit DECIMAL(12,6),
    current_price DECIMAL(12,6) NOT NULL,
    
    -- Risk management
    risk_reward_ratio DECIMAL(6,3),
    position_size DECIMAL(15,8),
    risk_amount DECIMAL(12,2),
    potential_profit DECIMAL(12,2),
    
    -- Technical Analysis Data (pandas-ta-classic)
    technical_indicators JSONB,
    trend_strength DECIMAL(5,2), -- 0-100
    momentum_score DECIMAL(5,2), -- 0-100
    volume_confirmation DECIMAL(5,2), -- 0-100
    volatility_percentile DECIMAL(5,2), -- 0-100
    
    -- Smart Money Concepts
    smc_analysis JSONB,
    order_blocks_signal INTEGER DEFAULT 0, -- -1, 0, 1
    fair_value_gaps INTEGER DEFAULT 0, -- -1, 0, 1
    break_of_structure INTEGER DEFAULT 0, -- -1, 0, 1
    liquidity_sweeps INTEGER DEFAULT 0, -- -1, 0, 1
    smc_bias VARCHAR(10) DEFAULT 'NEUTRAL', -- 'BULLISH', 'BEARISH', 'NEUTRAL'
    smc_strength INTEGER DEFAULT 0,
    
    -- Multi-timeframe Analysis
    mtf_confluence JSONB,
    timeframes_analyzed INTEGER DEFAULT 0,
    bullish_confluence INTEGER DEFAULT 0,
    bearish_confluence INTEGER DEFAULT 0,
    confluence_ratio DECIMAL(4,3) DEFAULT 0.500,
    
    -- Machine Learning
    ml_predictions JSONB,
    lstm_prediction DECIMAL(4,3),
    xgboost_prediction DECIMAL(4,3),
    ensemble_prediction DECIMAL(4,3),
    model_confidence DECIMAL(5,2),
    
    -- Signal Status
    status VARCHAR(20) DEFAULT 'ACTIVE', -- 'ACTIVE', 'EXECUTED', 'CANCELLED', 'EXPIRED'
    execution_time TIMESTAMP,
    expiry_time TIMESTAMP,
    
    -- Metadata
    market_session VARCHAR(20),
    market_conditions JSONB,
    analysis_duration_ms INTEGER,
    data_quality_score DECIMAL(4,3),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trading History (Executed Trades)
CREATE TABLE IF NOT EXISTS trading_history (
    id SERIAL PRIMARY KEY,
    trade_uuid UUID DEFAULT uuid_generate_v4(),
    account_id INTEGER REFERENCES trading_accounts(id),
    signal_id INTEGER REFERENCES trading_signals(id),
    strategy_id INTEGER REFERENCES trading_strategies(id),
    instrument_id INTEGER REFERENCES trading_instruments(id),
    
    -- Trade Details
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    trade_type VARCHAR(10) NOT NULL, -- 'BUY', 'SELL'
    order_type VARCHAR(20) NOT NULL, -- 'market', 'limit', 'stop'
    
    -- Position Details
    quantity DECIMAL(15,8) NOT NULL,
    entry_price DECIMAL(12,6) NOT NULL,
    exit_price DECIMAL(12,6),
    stop_loss DECIMAL(12,6),
    take_profit DECIMAL(12,6),
    
    -- P&L Calculation
    gross_profit_loss DECIMAL(15,2),
    commission DECIMAL(10,2) DEFAULT 0.00,
    swap DECIMAL(10,2) DEFAULT 0.00,
    net_profit_loss DECIMAL(15,2),
    profit_loss_pips DECIMAL(8,1),
    
    -- Risk Metrics
    initial_risk DECIMAL(12,2),
    risk_reward_achieved DECIMAL(6,3),
    max_adverse_excursion DECIMAL(12,2), -- MAE
    max_favorable_excursion DECIMAL(12,2), -- MFE
    
    -- Trade Status & Timing
    status VARCHAR(20) DEFAULT 'OPEN', -- 'OPEN', 'CLOSED', 'CANCELLED'
    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    exit_time TIMESTAMP,
    duration_minutes INTEGER,
    
    -- Market Context
    market_session VARCHAR(20),
    volatility_at_entry DECIMAL(8,4),
    spread_at_entry DECIMAL(6,3),
    
    -- Exit Reason
    exit_reason VARCHAR(50), -- 'take_profit', 'stop_loss', 'trailing_stop', 'manual', 'time_exit'
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===================================================================
-- MARKET DATA & TECHNICAL ANALYSIS
-- ===================================================================

-- Market Data (OHLCV)
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    instrument_id INTEGER REFERENCES trading_instruments(id),
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- OHLCV Data
    open_price DECIMAL(12,6) NOT NULL,
    high_price DECIMAL(12,6) NOT NULL,
    low_price DECIMAL(12,6) NOT NULL,
    close_price DECIMAL(12,6) NOT NULL,
    volume DECIMAL(20,4) DEFAULT 0,
    tick_volume INTEGER DEFAULT 0,
    
    -- Additional Data
    spread DECIMAL(6,3),
    real_volume DECIMAL(20,4),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint to prevent duplicates
    UNIQUE(symbol, timeframe, timestamp)
);

-- Technical Indicators Cache (pandas-ta-classic)
CREATE TABLE IF NOT EXISTS technical_indicators (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Trend Indicators
    sma_20 DECIMAL(12,6),
    sma_50 DECIMAL(12,6),
    sma_200 DECIMAL(12,6),
    ema_12 DECIMAL(12,6),
    ema_26 DECIMAL(12,6),
    ema_50 DECIMAL(12,6),
    supertrend DECIMAL(12,6),
    supertrend_direction INTEGER,
    adx DECIMAL(6,3),
    aroon_up DECIMAL(6,3),
    aroon_down DECIMAL(6,3),
    
    -- Momentum Indicators  
    rsi_14 DECIMAL(6,3),
    macd DECIMAL(8,5),
    macd_signal DECIMAL(8,5),
    macd_histogram DECIMAL(8,5),
    stoch_k DECIMAL(6,3),
    stoch_d DECIMAL(6,3),
    williams_r DECIMAL(6,3),
    cci DECIMAL(8,3),
    
    -- Volume Indicators
    obv DECIMAL(20,2),
    ad DECIMAL(20,2),
    cmf DECIMAL(6,3),
    mfi DECIMAL(6,3),
    vwap DECIMAL(12,6),
    
    -- Volatility Indicators
    bb_upper DECIMAL(12,6),
    bb_middle DECIMAL(12,6),
    bb_lower DECIMAL(12,6),
    bb_width DECIMAL(8,5),
    bb_percent DECIMAL(6,3),
    atr DECIMAL(8,5),
    keltner_upper DECIMAL(12,6),
    keltner_lower DECIMAL(12,6),
    
    -- Composite Indicators
    trend_strength DECIMAL(5,2),
    momentum_composite DECIMAL(5,2),
    volume_confirmation DECIMAL(5,2),
    volatility_percentile DECIMAL(5,2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint
    UNIQUE(symbol, timeframe, timestamp)
);

-- Smart Money Concepts Data
CREATE TABLE IF NOT EXISTS smc_analysis (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Order Blocks
    order_block_signal INTEGER DEFAULT 0, -- -1 (bearish), 0 (none), 1 (bullish)
    order_block_price DECIMAL(12,6),
    order_block_strength DECIMAL(4,2),
    order_block_volume DECIMAL(20,4),
    
    -- Fair Value Gaps
    fvg_signal INTEGER DEFAULT 0,
    fvg_upper DECIMAL(12,6),
    fvg_lower DECIMAL(12,6),
    fvg_size DECIMAL(8,5),
    fvg_age INTEGER DEFAULT 0,
    
    -- Break of Structure
    bos_signal INTEGER DEFAULT 0,
    bos_level DECIMAL(12,6),
    bos_type VARCHAR(20), -- 'swing_high', 'swing_low', 'trend_line'
    bos_strength DECIMAL(4,2),
    
    -- Change of Character
    choch_signal INTEGER DEFAULT 0,
    choch_momentum_shift DECIMAL(6,3),
    choch_volume_confirmation BOOLEAN DEFAULT false,
    
    -- Liquidity Sweeps
    liquidity_sweep INTEGER DEFAULT 0,
    sweep_level DECIMAL(12,6),
    sweep_volume DECIMAL(20,4),
    sweep_reversal BOOLEAN DEFAULT false,
    
    -- Overall SMC Assessment
    smc_bias VARCHAR(10) DEFAULT 'NEUTRAL',
    smc_strength INTEGER DEFAULT 0,
    smc_confidence DECIMAL(5,2) DEFAULT 0.00,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(symbol, timeframe, timestamp)
);

-- ===================================================================
-- MACHINE LEARNING TABLES
-- ===================================================================

-- ML Model Registry
CREATE TABLE IF NOT EXISTS ml_models (
    id SERIAL PRIMARY KEY,
    model_uuid UUID DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'lstm', 'xgboost', 'ensemble'
    framework VARCHAR(30) NOT NULL, -- 'tensorflow', 'xgboost', 'sklearn'
    version VARCHAR(20) NOT NULL,
    
    -- Model Configuration
    hyperparameters JSONB,
    features JSONB,
    target_variable VARCHAR(50),
    
    -- Training Information
    training_data_start TIMESTAMP,
    training_data_end TIMESTAMP,
    training_samples INTEGER,
    validation_samples INTEGER,
    
    -- Performance Metrics
    accuracy DECIMAL(6,4),
    precision_score DECIMAL(6,4),
    recall_score DECIMAL(6,4),
    f1_score DECIMAL(6,4),
    auc_score DECIMAL(6,4),
    
    -- Model Files
    model_path TEXT,
    model_size_mb DECIMAL(8,2),
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    deployment_status VARCHAR(20) DEFAULT 'TRAINING', -- 'TRAINING', 'DEPLOYED', 'DEPRECATED'
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ML Predictions
CREATE TABLE IF NOT EXISTS ml_predictions (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_models(id),
    signal_id INTEGER REFERENCES trading_signals(id),
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    prediction_time TIMESTAMP NOT NULL,
    
    -- Input Features
    feature_values JSONB,
    
    -- Predictions
    predicted_direction VARCHAR(10), -- 'BUY', 'SELL', 'HOLD'
    prediction_probability DECIMAL(6,4),
    prediction_confidence DECIMAL(5,2),
    
    -- Model Specific Outputs
    raw_prediction DECIMAL(8,5),
    prediction_classes JSONB, -- For multi-class models
    feature_importance JSONB,
    
    -- Actual Outcome (for model evaluation)
    actual_direction VARCHAR(10),
    actual_outcome DECIMAL(8,4), -- Actual price movement
    prediction_accuracy DECIMAL(6,4),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===================================================================
-- PERFORMANCE & ANALYTICS
-- ===================================================================

-- Performance Metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES trading_accounts(id),
    strategy_id INTEGER REFERENCES trading_strategies(id),
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    
    -- Trading Performance
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(6,3) DEFAULT 0.000,
    
    -- Profitability
    gross_profit DECIMAL(15,2) DEFAULT 0.00,
    gross_loss DECIMAL(15,2) DEFAULT 0.00,
    net_profit DECIMAL(15,2) DEFAULT 0.00,
    profit_factor DECIMAL(8,4) DEFAULT 0.0000,
    
    -- Returns
    total_return DECIMAL(8,4) DEFAULT 0.0000,
    annualized_return DECIMAL(8,4) DEFAULT 0.0000,
    max_drawdown DECIMAL(8,4) DEFAULT 0.0000,
    current_drawdown DECIMAL(8,4) DEFAULT 0.0000,
    
    -- Risk Metrics
    sharpe_ratio DECIMAL(6,3) DEFAULT 0.000,
    sortino_ratio DECIMAL(6,3) DEFAULT 0.000,
    calmar_ratio DECIMAL(6,3) DEFAULT 0.000,
    var_95 DECIMAL(12,2) DEFAULT 0.00, -- Value at Risk 95%
    
    -- Trade Statistics
    avg_win DECIMAL(12,2) DEFAULT 0.00,
    avg_loss DECIMAL(12,2) DEFAULT 0.00,
    largest_win DECIMAL(12,2) DEFAULT 0.00,
    largest_loss DECIMAL(12,2) DEFAULT 0.00,
    avg_trade_duration_minutes INTEGER DEFAULT 0,
    
    -- System Metrics
    signals_generated INTEGER DEFAULT 0,
    signals_executed INTEGER DEFAULT 0,
    signal_execution_rate DECIMAL(6,3) DEFAULT 0.000,
    avg_signal_confidence DECIMAL(5,2) DEFAULT 0.00,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System Events & Logs
CREATE TABLE IF NOT EXISTS system_events (
    id SERIAL PRIMARY KEY,
    event_uuid UUID DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL, -- 'SIGNAL', 'TRADE', 'ERROR', 'SYSTEM', 'MODEL'
    event_category VARCHAR(30) NOT NULL, -- 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    
    -- Event Details
    title VARCHAR(200) NOT NULL,
    message TEXT,
    event_data JSONB,
    
    -- Context
    strategy_id INTEGER REFERENCES trading_strategies(id),
    account_id INTEGER REFERENCES trading_accounts(id),
    symbol VARCHAR(20),
    
    -- System Information
    server_id VARCHAR(50),
    process_id INTEGER,
    thread_id BIGINT,
    memory_usage_mb DECIMAL(8,2),
    cpu_usage_percent DECIMAL(5,2),
    
    -- Error Information (if applicable)
    error_code VARCHAR(20),
    stack_trace TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===================================================================
-- PERFORMANCE INDEXES
-- ===================================================================

-- Trading Signals indexes
CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_timeframe ON trading_signals(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_trading_signals_created_at ON trading_signals(created_at);
CREATE INDEX IF NOT EXISTS idx_trading_signals_status ON trading_signals(status);
CREATE INDEX IF NOT EXISTS idx_trading_signals_strategy_id ON trading_signals(strategy_id);
CREATE INDEX IF NOT EXISTS idx_trading_signals_confidence ON trading_signals(confidence);
CREATE INDEX IF NOT EXISTS idx_trading_signals_signal_type ON trading_signals(signal_type);
CREATE INDEX IF NOT EXISTS idx_trading_signals_smc_bias ON trading_signals(smc_bias);

-- Trading History indexes
CREATE INDEX IF NOT EXISTS idx_trading_history_symbol_timeframe ON trading_history(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_trading_history_entry_time ON trading_history(entry_time);
CREATE INDEX IF NOT EXISTS idx_trading_history_account_id ON trading_history(account_id);
CREATE INDEX IF NOT EXISTS idx_trading_history_strategy_id ON trading_history(strategy_id);
CREATE INDEX IF NOT EXISTS idx_trading_history_status ON trading_history(status);
CREATE INDEX IF NOT EXISTS idx_trading_history_trade_type ON trading_history(trade_type);

-- Market Data indexes
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timeframe_timestamp ON market_data(symbol, timeframe, timestamp);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol);

-- Technical Indicators indexes
CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_timeframe_timestamp ON technical_indicators(symbol, timeframe, timestamp);
CREATE INDEX IF NOT EXISTS idx_technical_indicators_timestamp ON technical_indicators(timestamp);

-- SMC Analysis indexes
CREATE INDEX IF NOT EXISTS idx_smc_analysis_symbol_timeframe_timestamp ON smc_analysis(symbol, timeframe, timestamp);
CREATE INDEX IF NOT EXISTS idx_smc_analysis_smc_bias ON smc_analysis(smc_bias);
CREATE INDEX IF NOT EXISTS idx_smc_analysis_timestamp ON smc_analysis(timestamp);

-- ML Predictions indexes
CREATE INDEX IF NOT EXISTS idx_ml_predictions_model_id ON ml_predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_prediction_time ON ml_predictions(prediction_time);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol ON ml_predictions(symbol);

-- Performance Metrics indexes
CREATE INDEX IF NOT EXISTS idx_performance_metrics_account_id ON performance_metrics(account_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_strategy_id ON performance_metrics(strategy_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_period_start ON performance_metrics(period_start);

-- System Events indexes
CREATE INDEX IF NOT EXISTS idx_system_events_event_type ON system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_system_events_created_at ON system_events(created_at);
CREATE INDEX IF NOT EXISTS idx_system_events_event_category ON system_events(event_category);
CREATE INDEX IF NOT EXISTS idx_system_events_symbol ON system_events(symbol);

-- ===================================================================
-- DEFAULT DATA
-- ===================================================================

-- Insert default trading instruments
INSERT INTO trading_instruments (symbol, name, instrument_type, base_currency, quote_currency, pip_value, min_trade_size, max_trade_size, spread_avg) VALUES
('EURUSD', 'Euro vs US Dollar', 'forex', 'EUR', 'USD', 0.0001, 0.01, 100.0, 1.2),
('GBPUSD', 'British Pound vs US Dollar', 'forex', 'GBP', 'USD', 0.0001, 0.01, 100.0, 1.8),
('USDJPY', 'US Dollar vs Japanese Yen', 'forex', 'USD', 'JPY', 0.01, 0.01, 100.0, 1.1),
('USDCHF', 'US Dollar vs Swiss Franc', 'forex', 'USD', 'CHF', 0.0001, 0.01, 100.0, 1.5),
('AUDUSD', 'Australian Dollar vs US Dollar', 'forex', 'AUD', 'USD', 0.0001, 0.01, 100.0, 1.4),
('USDCAD', 'US Dollar vs Canadian Dollar', 'forex', 'USD', 'CAD', 0.0001, 0.01, 100.0, 1.8),
('BTCUSDT', 'Bitcoin vs Tether', 'crypto', 'BTC', 'USDT', 0.01, 0.001, 10.0, 10.0),
('ETHUSDT', 'Ethereum vs Tether', 'crypto', 'ETH', 'USDT', 0.01, 0.01, 100.0, 5.0),
('BNBUSDT', 'Binance Coin vs Tether', 'crypto', 'BNB', 'USDT', 0.01, 0.01, 1000.0, 3.0),
('US30', 'Dow Jones Industrial Average', 'indices', 'USD', 'USD', 1.0, 0.1, 50.0, 3.0)
ON CONFLICT (symbol) DO NOTHING;

-- Insert default strategy
INSERT INTO trading_strategies (name, version, strategy_type, description, configuration, timeframes, supported_instruments) VALUES
('pandas-ta-classic Advanced Strategy', '2.0.0', 'pandas_ta_classic', 
 'Advanced strategy using pandas-ta-classic with 150+ indicators and Smart Money Concepts',
 '{"indicators": {"trend": ["sma", "ema", "supertrend"], "momentum": ["rsi", "macd", "stoch"], "volume": ["obv", "cmf"], "volatility": ["bbands", "atr"]}, "smc": {"order_blocks": true, "fair_value_gaps": true, "bos": true}, "risk": {"max_risk": 0.02, "rr_ratio": 2.0}}',
 ARRAY['M15', 'H1', 'H4', 'D1'], 
 ARRAY['forex', 'crypto', 'indices'])
ON CONFLICT DO NOTHING;

-- Insert default account
INSERT INTO trading_accounts (broker, account_number, account_type, base_currency, initial_balance, current_balance, equity, free_margin) VALUES
('demo_account', 'DEMO-001', 'demo', 'USD', 10000.00, 10000.00, 10000.00, 10000.00)
ON CONFLICT DO NOTHING;

-- ===================================================================
-- VIEWS FOR EASY DATA ACCESS
-- ===================================================================

-- View for latest signals with strategy info
CREATE OR REPLACE VIEW latest_trading_signals AS
SELECT 
    ts.id,
    ts.symbol,
    ts.timeframe,
    ts.signal_type,
    ts.confidence,
    ts.entry_price,
    ts.stop_loss,
    ts.take_profit,
    ts.risk_reward_ratio,
    ts.trend_strength,
    ts.momentum_score,
    ts.smc_bias,
    ts.smc_strength,
    ts.status,
    str.name as strategy_name,
    str.version as strategy_version,
    ts.created_at
FROM trading_signals ts
JOIN trading_strategies str ON ts.strategy_id = str.id
WHERE ts.created_at >= NOW() - INTERVAL '24 hours'
ORDER BY ts.created_at DESC;

-- View for performance summary
CREATE OR REPLACE VIEW strategy_performance_summary AS
SELECT 
    s.name as strategy_name,
    s.version,
    COUNT(th.id) as total_trades,
    SUM(CASE WHEN th.net_profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
    ROUND(AVG(CASE WHEN th.net_profit_loss > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
    ROUND(SUM(th.net_profit_loss), 2) as total_pnl,
    ROUND(AVG(th.risk_reward_achieved), 2) as avg_rr,
    COUNT(DISTINCT th.symbol) as instruments_traded
FROM trading_strategies s
LEFT JOIN trading_history th ON s.id = th.strategy_id
WHERE s.is_active = true
GROUP BY s.id, s.name, s.version
ORDER BY total_pnl DESC;

-- View for Smart Money Concepts analysis
CREATE OR REPLACE VIEW smc_signals_summary AS
SELECT 
    symbol,
    timeframe,
    COUNT(*) as total_signals,
    SUM(CASE WHEN order_block_signal != 0 THEN 1 ELSE 0 END) as order_blocks_count,
    SUM(CASE WHEN fvg_signal != 0 THEN 1 ELSE 0 END) as fvg_count,
    SUM(CASE WHEN bos_signal != 0 THEN 1 ELSE 0 END) as bos_count,
    SUM(CASE WHEN liquidity_sweep != 0 THEN 1 ELSE 0 END) as sweep_count,
    COUNT(CASE WHEN smc_bias = 'BULLISH' THEN 1 END) as bullish_bias,
    COUNT(CASE WHEN smc_bias = 'BEARISH' THEN 1 END) as bearish_bias,
    AVG(smc_confidence) as avg_confidence
FROM smc_analysis
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY symbol, timeframe
ORDER BY total_signals DESC;

-- ===================================================================
-- STORED PROCEDURES
-- ===================================================================

-- Function to calculate performance metrics
CREATE OR REPLACE FUNCTION calculate_strategy_performance(
    p_strategy_id INTEGER,
    p_start_date TIMESTAMP DEFAULT NOW() - INTERVAL '30 days',
    p_end_date TIMESTAMP DEFAULT NOW()
) 
RETURNS TABLE (
    total_trades BIGINT,
    winning_trades BIGINT,
    losing_trades BIGINT,
    win_rate NUMERIC,
    total_pnl NUMERIC,
    profit_factor NUMERIC,
    max_dd NUMERIC
) 
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_trades,
        SUM(CASE WHEN th.net_profit_loss > 0 THEN 1 ELSE 0 END)::BIGINT as winning_trades,
        SUM(CASE WHEN th.net_profit_loss <= 0 THEN 1 ELSE 0 END)::BIGINT as losing_trades,
        ROUND(AVG(CASE WHEN th.net_profit_loss > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
        ROUND(SUM(th.net_profit_loss), 2) as total_pnl,
        CASE 
            WHEN SUM(CASE WHEN th.net_profit_loss < 0 THEN ABS(th.net_profit_loss) ELSE 0 END) > 0 
            THEN ROUND(SUM(CASE WHEN th.net_profit_loss > 0 THEN th.net_profit_loss ELSE 0 END) / 
                      SUM(CASE WHEN th.net_profit_loss < 0 THEN ABS(th.net_profit_loss) ELSE 0 END), 2)
            ELSE 0
        END as profit_factor,
        COALESCE(MAX(ABS(th.max_adverse_excursion)), 0) as max_dd
    FROM trading_history th
    WHERE th.strategy_id = p_strategy_id
    AND th.entry_time BETWEEN p_start_date AND p_end_date
    AND th.status = 'CLOSED';
END;
$$ LANGUAGE plpgsql;

-- Function to get latest market data
CREATE OR REPLACE FUNCTION get_latest_market_data(
    p_symbol VARCHAR(20),
    p_timeframe VARCHAR(10),
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE (
    timestamp TIMESTAMP,
    open_price NUMERIC,
    high_price NUMERIC,
    low_price NUMERIC,
    close_price NUMERIC,
    volume NUMERIC
)
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        md.timestamp,
        md.open_price,
        md.high_price,
        md.low_price,
        md.close_price,
        md.volume
    FROM market_data md
    WHERE md.symbol = p_symbol
    AND md.timeframe = p_timeframe
    ORDER BY md.timestamp DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- ===================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- ===================================================================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update triggers to relevant tables
CREATE TRIGGER update_trading_instruments_updated_at BEFORE UPDATE ON trading_instruments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_trading_accounts_updated_at BEFORE UPDATE ON trading_accounts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_trading_strategies_updated_at BEFORE UPDATE ON trading_strategies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_trading_signals_updated_at BEFORE UPDATE ON trading_signals
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_trading_history_updated_at BEFORE UPDATE ON trading_history
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_ml_models_updated_at BEFORE UPDATE ON ml_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ===================================================================
-- DATABASE COMPLETION MESSAGE
-- ===================================================================

-- Log successful completion
DO $$
BEGIN
    RAISE NOTICE 'AI/ML Trading Bot v2.0 database schema created successfully!';
    RAISE NOTICE 'Features enabled: pandas-ta-classic, Smart Money Concepts, ML Pipeline';
    RAISE NOTICE 'Tables created: %, Indexes: %, Views: %, Functions: %', 
                 (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'),
                 (SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public'),
                 (SELECT COUNT(*) FROM information_schema.views WHERE table_schema = 'public'),
                 (SELECT COUNT(*) FROM information_schema.routines WHERE routine_schema = 'public');
END $$;