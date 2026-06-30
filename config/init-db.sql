-- ============================================
-- Binance Trading Bot Database Schema
-- Version: 0.2.0
-- PostgreSQL + TimescaleDB
-- ============================================

-- 创建 TimescaleDB 扩展
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================
-- 1. 市场数据表（时序表）
-- ============================================
CREATE TABLE IF NOT EXISTS market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    quote_asset_volume DECIMAL(20, 8),
    number_of_trades INTEGER,
    taker_buy_base_asset_volume DECIMAL(20, 8),
    taker_buy_quote_asset_volume DECIMAL(20, 8),
    interval VARCHAR(10) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (time, symbol, interval)
);

-- 转换为时序表
SELECT create_hypertable(
    'market_data',
    'time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- 建立索引
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time
ON market_data (symbol, time DESC);

CREATE INDEX IF NOT EXISTS idx_market_data_interval
ON market_data (interval, symbol, time DESC);

-- ============================================
-- 2. 订单表（时序表）
-- ============================================
CREATE TABLE IF NOT EXISTS orders (
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    order_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL, -- BUY or SELL
    order_type VARCHAR(20) NOT NULL, -- MARKET, LIMIT, etc
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8),
    status VARCHAR(20) NOT NULL, -- NEW, FILLED, CANCELED, REJECTED, PARTIAL
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    executed_price DECIMAL(20, 8),
    total_cost DECIMAL(20, 8),
    commission DECIMAL(20, 8) DEFAULT 0,
    commission_asset VARCHAR(10),
    strategy VARCHAR(100),
    risk_score DECIMAL(5, 4),
    error_message TEXT,
    PRIMARY KEY (created_at, order_id)
);

-- 转换为时序表
SELECT create_hypertable(
    'orders',
    'created_at',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 week'
);

-- 创建唯一索引来保证 order_id 唯一性
CREATE UNIQUE INDEX IF NOT EXISTS idx_orders_order_id
ON orders (order_id, created_at);

CREATE INDEX IF NOT EXISTS idx_orders_symbol_status
ON orders (symbol, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_orders_strategy
ON orders (strategy, created_at DESC);

-- ============================================
-- 3. 交易记录表（时序表）
-- ============================================
CREATE TABLE IF NOT EXISTS trades (
    entry_time TIMESTAMPTZ NOT NULL,
    trade_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exit_time TIMESTAMPTZ,
    entry_price DECIMAL(20, 8) NOT NULL,
    exit_price DECIMAL(20, 8),
    quantity DECIMAL(20, 8) NOT NULL,
    side VARCHAR(10) NOT NULL, -- LONG or SHORT
    pnl DECIMAL(20, 8),
    pnl_pct DECIMAL(10, 4),
    commission DECIMAL(20, 8) DEFAULT 0,
    status VARCHAR(20) NOT NULL, -- OPEN, CLOSED, CANCELED
    strategy VARCHAR(100),
    entry_order_id VARCHAR(50),
    exit_order_id VARCHAR(50),
    risk_score DECIMAL(5, 4),
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entry_time, trade_id)
);

SELECT create_hypertable(
    'trades',
    'entry_time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 month'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_trade_id
ON trades (trade_id, entry_time);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_status
ON trades (symbol, status, entry_time DESC);

CREATE INDEX IF NOT EXISTS idx_trades_strategy
ON trades (strategy, entry_time DESC);

-- ============================================
-- 4. 投资组合状态表（时序表）
-- ============================================
CREATE TABLE IF NOT EXISTS portfolio_state (
    timestamp TIMESTAMPTZ NOT NULL,
    total_balance DECIMAL(20, 8) NOT NULL,
    available_balance DECIMAL(20, 8) NOT NULL,
    locked_balance DECIMAL(20, 8),
    total_value DECIMAL(20, 8) NOT NULL,
    btc_value DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8),
    daily_return DECIMAL(10, 4),
    total_return DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable(
    'portfolio_state',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

CREATE INDEX IF NOT EXISTS idx_portfolio_state_timestamp
ON portfolio_state (timestamp DESC);

-- ============================================
-- 5. 审计日志表（时序表）- 不可变日志链
-- ============================================
CREATE TABLE IF NOT EXISTS audit_log (
    timestamp TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    user_id VARCHAR(100),
    action VARCHAR(50) NOT NULL,
    resource VARCHAR(200) NOT NULL,
    details JSONB NOT NULL,
    risk_score DECIMAL(5, 4),
    prev_hash VARCHAR(64) NOT NULL,
    current_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (timestamp, current_hash)
);

SELECT create_hypertable(
    'audit_log',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

CREATE INDEX IF NOT EXISTS idx_audit_log_event_type
ON audit_log (event_type, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_audit_log_resource
ON audit_log (resource, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_audit_log_hash
ON audit_log (current_hash);

-- ============================================
-- 6. 系统事件日志表（时序表）
-- ============================================
CREATE TABLE IF NOT EXISTS event_log (
    timestamp TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL, -- INFO, WARNING, ERROR, CRITICAL
    source VARCHAR(100),
    message TEXT NOT NULL,
    data JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (timestamp, event_type, severity)
);

SELECT create_hypertable(
    'event_log',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

CREATE INDEX IF NOT EXISTS idx_event_log_type_severity
ON event_log (event_type, severity, timestamp DESC);

-- ============================================
-- 7. 性能指标表（时序表）
-- ============================================
CREATE TABLE IF NOT EXISTS performance_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    total_return DECIMAL(10, 4),
    annual_return DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    sortino_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    win_rate DECIMAL(10, 4),
    profit_factor DECIMAL(10, 4),
    num_trades INTEGER,
    avg_trade_return DECIMAL(10, 4),
    avg_trade_duration INTERVAL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable(
    'performance_metrics',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- ============================================
-- 8. 模型版本表
-- ============================================
CREATE TABLE IF NOT EXISTS model_versions (
    id BIGSERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL UNIQUE,
    model_type VARCHAR(50) NOT NULL,
    model_path TEXT NOT NULL,
    parameters JSONB,
    performance_metrics JSONB,
    training_data_range JSONB,
    status VARCHAR(20) NOT NULL, -- TRAINING, TESTING, ACTIVE, INACTIVE, ARCHIVED
    deployed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_model_versions_status
ON model_versions (status, created_at DESC);

-- ============================================
-- 9. 风险拒绝记录表（时序表）
-- ============================================
CREATE TABLE IF NOT EXISTS risk_rejections (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    attempted_quantity DECIMAL(20, 8) NOT NULL,
    attempted_price DECIMAL(20, 8),
    rejection_reason TEXT NOT NULL,
    risk_layer VARCHAR(50) NOT NULL,
    risk_score DECIMAL(5, 4) NOT NULL,
    additional_data JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (timestamp, symbol, risk_layer)
);

SELECT create_hypertable(
    'risk_rejections',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 week'
);

CREATE INDEX IF NOT EXISTS idx_risk_rejections_symbol
ON risk_rejections (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_risk_rejections_layer
ON risk_rejections (risk_layer, timestamp DESC);

-- ============================================
-- 10. 系统状态快照表
-- ============================================
CREATE TABLE IF NOT EXISTS system_snapshots (
    id BIGSERIAL PRIMARY KEY,
    snapshot_id VARCHAR(50) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    portfolio_state JSONB NOT NULL,
    open_orders JSONB NOT NULL,
    positions JSONB NOT NULL,
    balance JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_system_snapshots_timestamp
ON system_snapshots (timestamp DESC);

-- ============================================
-- 数据保留策略
-- ============================================

-- 市场数据保留6个月
SELECT add_retention_policy(
    'market_data',
    INTERVAL '6 months',
    if_not_exists => TRUE
);

-- 事件日志保留3个月
SELECT add_retention_policy(
    'event_log',
    INTERVAL '3 months',
    if_not_exists => TRUE
);

-- 注意：system_snapshots 不是 hypertable，不需要 retention policy

-- ============================================
-- 自动压缩策略
-- ============================================

-- 启用压缩并设置压缩策略
-- 市场数据
ALTER TABLE market_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,interval'
);
SELECT add_compression_policy('market_data', INTERVAL '7 days', if_not_exists => TRUE);

-- 订单数据
ALTER TABLE orders SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('orders', INTERVAL '14 days', if_not_exists => TRUE);

-- 交易数据
ALTER TABLE trades SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('trades', INTERVAL '30 days', if_not_exists => TRUE);

-- 审计日志
ALTER TABLE audit_log SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'event_type'
);
SELECT add_compression_policy('audit_log', INTERVAL '7 days', if_not_exists => TRUE);

-- ============================================
-- 授予权限
-- ============================================

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trader;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trader;
GRANT USAGE ON SCHEMA public TO trader;

-- ============================================
-- 初始化完成
-- ============================================

-- 插入初始系统事件
INSERT INTO event_log (timestamp, event_type, severity, source, message, data)
VALUES (
    NOW(),
    'SYSTEM_INIT',
    'INFO',
    'init-db.sql',
    'Database schema initialized successfully',
    '{"version": "0.2.0"}'::jsonb
);

-- ============================================
-- 11. Arena 競技場表 (影子交易系統)
-- ============================================
CREATE TABLE IF NOT EXISTS arena_models (
    id BIGSERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL UNIQUE,
    model_type VARCHAR(100) NOT NULL,  -- e.g., "LSTM-Classic", "Transformer-RL"
    status VARCHAR(30) NOT NULL DEFAULT 'TRAINING',  -- TRAINING, SHADOW_RUNNING, LIVE, ARCHIVED
    win_rate DECIMAL(10, 6) DEFAULT 0,
    total_return DECIMAL(15, 6) DEFAULT 0,
    trade_count INTEGER DEFAULT 0,
    draw_down DECIMAL(10, 6) DEFAULT 0,
    sharpe_ratio DECIMAL(10, 6),
    profit_factor DECIMAL(10, 6),
    avg_hold_time VARCHAR(50),
    config JSONB DEFAULT '{}',
    shadow_start_at TIMESTAMPTZ,
    promoted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_arena_models_status
ON arena_models (status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_arena_models_version
ON arena_models (version);

-- ============================================
-- 12. Arena 競技場狀態表
-- ============================================
CREATE TABLE IF NOT EXISTS arena_state (
    id BIGSERIAL PRIMARY KEY,
    champion_version VARCHAR(50),
    challenger_version VARCHAR(50),
    comparison_days INTEGER DEFAULT 7,
    promotion_threshold DECIMAL(10, 6) DEFAULT 0.05,  -- 5% 優於冠軍才晉升
    is_comparing BOOLEAN DEFAULT FALSE,
    started_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (champion_version) REFERENCES arena_models(version) ON DELETE SET NULL,
    FOREIGN KEY (challenger_version) REFERENCES arena_models(version) ON DELETE SET NULL
);

-- 初始化 arena_state 為單行配置表
INSERT INTO arena_state (id, is_comparing, comparison_days, promotion_threshold)
VALUES (1, FALSE, 7, 0.05)
ON CONFLICT (id) DO NOTHING;

-- ============================================
-- 13. 影子交易記錄表 (時序表)
-- ============================================
CREATE TABLE IF NOT EXISTS shadow_trades (
    timestamp TIMESTAMPTZ NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- BUY, SELL
    signal_strength DECIMAL(10, 6),
    signal_confidence DECIMAL(10, 6),
    suggested_quantity DECIMAL(20, 8),
    suggested_price DECIMAL(20, 8),
    actual_price DECIMAL(20, 8),  -- 當時的實際市場價格
    virtual_pnl DECIMAL(20, 8),  -- 模擬盈虧
    would_have_triggered BOOLEAN DEFAULT FALSE,
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (timestamp, model_version, symbol)
);

SELECT create_hypertable(
    'shadow_trades',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

CREATE INDEX IF NOT EXISTS idx_shadow_trades_model
ON shadow_trades (model_version, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_shadow_trades_symbol
ON shadow_trades (symbol, timestamp DESC);

-- ============================================
-- 14. 模型晉升歷史表
-- ============================================
CREATE TABLE IF NOT EXISTS promotion_history (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    previous_champion VARCHAR(50),
    new_champion VARCHAR(50) NOT NULL,
    reason TEXT,
    champion_performance JSONB,  -- 舊冠軍在比較期間的表現
    challenger_performance JSONB,  -- 新冠軍在比較期間的表現
    promoted_by VARCHAR(100) DEFAULT 'system',  -- 'system' or 'manual'
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_promotion_history_timestamp
ON promotion_history (timestamp DESC);

-- ============================================
-- 15. 策略靈敏度配置表
-- ============================================
CREATE TABLE IF NOT EXISTS strategy_sensitivity (
    id BIGSERIAL PRIMARY KEY,
    sensitivity VARCHAR(20) NOT NULL DEFAULT 'moderate',  -- conservative, moderate, aggressive
    confidence_threshold DECIMAL(10, 6) DEFAULT 0.70,
    strength_threshold DECIMAL(10, 6) DEFAULT 0.60,
    position_ratio DECIMAL(10, 6) DEFAULT 0.6,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 初始化為單行配置表
INSERT INTO strategy_sensitivity (id, sensitivity, confidence_threshold, strength_threshold, position_ratio)
VALUES (1, 'moderate', 0.70, 0.60, 0.6)
ON CONFLICT (id) DO NOTHING;

-- ============================================
-- 16. Binance 原始交易記錄表 (時序表)
-- 存儲從 Binance API myTrades 同步的原始數據
-- ============================================
CREATE TABLE IF NOT EXISTS binance_trades (
    id BIGINT NOT NULL,                    -- Binance trade ID
    order_id BIGINT NOT NULL,              -- 對應的訂單 ID
    symbol VARCHAR(20) NOT NULL,           -- 交易對 (e.g., BTCUSDT)
    trade_time TIMESTAMPTZ NOT NULL,       -- 成交時間
    side VARCHAR(10) NOT NULL,             -- BUY or SELL
    price DECIMAL(20, 8) NOT NULL,         -- 成交價格
    qty DECIMAL(20, 8) NOT NULL,           -- 成交數量
    quote_qty DECIMAL(20, 8) NOT NULL,     -- 成交金額 (price * qty)
    commission DECIMAL(20, 8) NOT NULL,    -- 手續費
    commission_asset VARCHAR(20) NOT NULL, -- 手續費幣種
    is_buyer BOOLEAN NOT NULL,             -- 是否為買方
    is_maker BOOLEAN NOT NULL,             -- 是否為掛單方
    is_best_match BOOLEAN DEFAULT TRUE,
    synced_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trade_time, id, symbol)
);

SELECT create_hypertable(
    'binance_trades',
    'trade_time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 month'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_binance_trades_id
ON binance_trades (id, symbol);

CREATE INDEX IF NOT EXISTS idx_binance_trades_symbol
ON binance_trades (symbol, trade_time DESC);

CREATE INDEX IF NOT EXISTS idx_binance_trades_order
ON binance_trades (order_id, symbol);

-- ============================================
-- 17. 持倉成本追蹤表
-- 使用 FIFO 或平均成本法計算持倉成本
-- ============================================
CREATE TABLE IF NOT EXISTS position_costs (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,    -- 交易對
    base_asset VARCHAR(20) NOT NULL,       -- 基礎資產 (e.g., BTC)
    quote_asset VARCHAR(20) NOT NULL,      -- 報價資產 (e.g., USDT)
    quantity DECIMAL(20, 8) NOT NULL DEFAULT 0,      -- 當前持倉數量
    total_cost DECIMAL(20, 8) NOT NULL DEFAULT 0,    -- 總成本 (USDT)
    average_cost DECIMAL(20, 8) NOT NULL DEFAULT 0,  -- 平均成本價格
    realized_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,  -- 已實現盈虧
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,         -- 未實現盈虧 (需要實時更新)
    last_trade_id BIGINT,                  -- 最後一筆交易 ID
    last_trade_time TIMESTAMPTZ,           -- 最後交易時間
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_position_costs_symbol
ON position_costs (symbol);

CREATE INDEX IF NOT EXISTS idx_position_costs_base_asset
ON position_costs (base_asset);

-- ============================================
-- 18. 每日盈虧統計表 (時序表)
-- 記錄每日的盈虧摘要
-- ============================================
CREATE TABLE IF NOT EXISTS daily_pnl (
    date DATE NOT NULL,                    -- 日期
    symbol VARCHAR(20) NOT NULL DEFAULT 'ALL',  -- 交易對或 'ALL' 表示總計
    starting_balance DECIMAL(20, 8) NOT NULL DEFAULT 0,  -- 當日起始餘額
    ending_balance DECIMAL(20, 8) NOT NULL DEFAULT 0,    -- 當日結束餘額
    realized_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,      -- 當日已實現盈虧
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,             -- 當日未實現盈虧
    total_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,         -- 當日總盈虧
    pnl_percentage DECIMAL(10, 4) DEFAULT 0,             -- 盈虧百分比
    trade_count INTEGER DEFAULT 0,         -- 當日交易次數
    buy_volume DECIMAL(20, 8) DEFAULT 0,   -- 買入金額
    sell_volume DECIMAL(20, 8) DEFAULT 0,  -- 賣出金額
    commission_total DECIMAL(20, 8) DEFAULT 0,  -- 總手續費
    win_trades INTEGER DEFAULT 0,          -- 獲利交易次數
    loss_trades INTEGER DEFAULT 0,         -- 虧損交易次數
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_daily_pnl_date
ON daily_pnl (date DESC);

CREATE INDEX IF NOT EXISTS idx_daily_pnl_symbol
ON daily_pnl (symbol, date DESC);

-- ============================================
-- 19. 盈虧快照表 (時序表)
-- 定期記錄帳戶狀態快照，用於圖表繪製
-- ============================================
CREATE TABLE IF NOT EXISTS pnl_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    snapshot_type VARCHAR(20) NOT NULL DEFAULT 'hourly',  -- hourly, daily, manual
    total_balance_usdt DECIMAL(20, 8) NOT NULL,    -- 總餘額 (USDT)
    available_balance_usdt DECIMAL(20, 8),         -- 可用餘額
    in_position_usdt DECIMAL(20, 8),               -- 持倉價值
    total_realized_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,  -- 累計已實現盈虧
    total_unrealized_pnl DECIMAL(20, 8) DEFAULT 0,         -- 當前未實現盈虧
    total_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,           -- 總盈虧
    pnl_percentage DECIMAL(10, 4) DEFAULT 0,       -- 總盈虧百分比
    btc_price DECIMAL(20, 8),                      -- BTC 價格 (參考)
    position_count INTEGER DEFAULT 0,              -- 持倉數量
    metadata JSONB,                                -- 其他數據
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (timestamp, snapshot_type)
);

SELECT create_hypertable(
    'pnl_snapshots',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

CREATE INDEX IF NOT EXISTS idx_pnl_snapshots_type
ON pnl_snapshots (snapshot_type, timestamp DESC);

-- ============================================
-- 20. 交易成本明細表 (FIFO 用)
-- 追蹤每筆買入的成本批次
-- ============================================
CREATE TABLE IF NOT EXISTS cost_basis_lots (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    trade_id BIGINT NOT NULL,              -- 對應 binance_trades 的 id
    buy_time TIMESTAMPTZ NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,      -- 原始數量
    remaining_qty DECIMAL(20, 8) NOT NULL, -- 剩餘數量
    cost_per_unit DECIMAL(20, 8) NOT NULL, -- 單位成本
    total_cost DECIMAL(20, 8) NOT NULL,    -- 總成本
    is_closed BOOLEAN DEFAULT FALSE,       -- 是否已全部賣出
    closed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cost_basis_lots_symbol
ON cost_basis_lots (symbol, is_closed, buy_time);

CREATE INDEX IF NOT EXISTS idx_cost_basis_lots_remaining
ON cost_basis_lots (symbol, remaining_qty) WHERE remaining_qty > 0;

-- ============================================
-- 壓縮策略
-- ============================================

-- binance_trades 壓縮
ALTER TABLE binance_trades SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('binance_trades', INTERVAL '7 days', if_not_exists => TRUE);

-- pnl_snapshots 壓縮
ALTER TABLE pnl_snapshots SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'snapshot_type'
);
SELECT add_compression_policy('pnl_snapshots', INTERVAL '7 days', if_not_exists => TRUE);

-- ============================================
-- 授予權限
-- ============================================

GRANT ALL PRIVILEGES ON binance_trades TO trader;
GRANT ALL PRIVILEGES ON position_costs TO trader;
GRANT ALL PRIVILEGES ON daily_pnl TO trader;
GRANT ALL PRIVILEGES ON pnl_snapshots TO trader;
GRANT ALL PRIVILEGES ON cost_basis_lots TO trader;
GRANT USAGE, SELECT ON SEQUENCE position_costs_id_seq TO trader;
GRANT USAGE, SELECT ON SEQUENCE cost_basis_lots_id_seq TO trader;

-- 显示成功消息
DO $$
BEGIN
    RAISE NOTICE '✅ Database schema initialized successfully!';
    RAISE NOTICE '📊 Created 20 tables with TimescaleDB optimization';
    RAISE NOTICE '🔄 Configured automatic compression and retention policies';
    RAISE NOTICE '🏟️ Arena system tables ready for shadow trading';
    RAISE NOTICE '💰 PnL tracking tables ready (binance_trades, position_costs, daily_pnl, pnl_snapshots)';
    RAISE NOTICE '🎯 Ready for production deployment';
END $$;
