-- ============================================
-- Migration: 002_add_virtual_accounts.sql
-- 為競技場添加虛擬帳戶系統
-- 每個模型獲得 10000 USDT 虛擬資金
-- ============================================

-- ============================================
-- 1. 虛擬帳戶表
-- ============================================
CREATE TABLE IF NOT EXISTS virtual_accounts (
    id BIGSERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL UNIQUE,
    initial_balance DECIMAL(20, 8) NOT NULL DEFAULT 10000,  -- 初始資金
    balance DECIMAL(20, 8) NOT NULL DEFAULT 10000,          -- 當前餘額
    total_value DECIMAL(20, 8) NOT NULL DEFAULT 10000,      -- 總價值（餘額 + 持倉）
    realized_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,         -- 已實現盈虧
    unrealized_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,       -- 未實現盈虧
    total_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,            -- 總盈虧
    total_pnl_pct DECIMAL(10, 4) NOT NULL DEFAULT 0,        -- 總盈虧百分比
    max_drawdown DECIMAL(10, 4) NOT NULL DEFAULT 0,         -- 最大回撤
    peak_value DECIMAL(20, 8) NOT NULL DEFAULT 10000,       -- 歷史最高價值
    trade_count INTEGER NOT NULL DEFAULT 0,                  -- 交易次數
    win_count INTEGER NOT NULL DEFAULT 0,                    -- 獲勝次數
    loss_count INTEGER NOT NULL DEFAULT 0,                   -- 虧損次數
    win_rate DECIMAL(10, 4) NOT NULL DEFAULT 0,             -- 勝率
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_version) REFERENCES arena_models(version) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_virtual_accounts_model
ON virtual_accounts (model_version);

-- ============================================
-- 2. 虛擬持倉表
-- ============================================
CREATE TABLE IF NOT EXISTS virtual_positions (
    id BIGSERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL DEFAULT 'LONG',  -- LONG or SHORT
    quantity DECIMAL(20, 8) NOT NULL DEFAULT 0,
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    unrealized_pnl_pct DECIMAL(10, 4) DEFAULT 0,
    opened_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_version, symbol),
    FOREIGN KEY (model_version) REFERENCES arena_models(version) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_virtual_positions_model
ON virtual_positions (model_version);

CREATE INDEX IF NOT EXISTS idx_virtual_positions_symbol
ON virtual_positions (symbol);

-- ============================================
-- 3. 虛擬交易歷史表（時序表）
-- ============================================
CREATE TABLE IF NOT EXISTS virtual_trades (
    timestamp TIMESTAMPTZ NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- BUY or SELL
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    quote_value DECIMAL(20, 8) NOT NULL,  -- quantity * price
    commission DECIMAL(20, 8) DEFAULT 0,   -- 模擬手續費 0.1%
    realized_pnl DECIMAL(20, 8),           -- 賣出時的實現盈虧
    balance_after DECIMAL(20, 8),          -- 交易後餘額
    signal_strength DECIMAL(10, 6),
    signal_confidence DECIMAL(10, 6),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (timestamp, model_version, symbol, side)
);

SELECT create_hypertable(
    'virtual_trades',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

CREATE INDEX IF NOT EXISTS idx_virtual_trades_model
ON virtual_trades (model_version, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_virtual_trades_symbol
ON virtual_trades (symbol, timestamp DESC);

-- ============================================
-- 4. 為現有模型初始化虛擬帳戶
-- ============================================
INSERT INTO virtual_accounts (model_version, initial_balance, balance, total_value, peak_value)
SELECT version, 10000, 10000, 10000, 10000
FROM arena_models
WHERE version NOT IN (SELECT model_version FROM virtual_accounts)
ON CONFLICT (model_version) DO NOTHING;

-- ============================================
-- 壓縮策略
-- ============================================
ALTER TABLE virtual_trades SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'model_version,symbol'
);
SELECT add_compression_policy('virtual_trades', INTERVAL '7 days', if_not_exists => TRUE);

-- ============================================
-- 授予權限
-- ============================================
GRANT ALL PRIVILEGES ON virtual_accounts TO trader;
GRANT ALL PRIVILEGES ON virtual_positions TO trader;
GRANT ALL PRIVILEGES ON virtual_trades TO trader;
GRANT USAGE, SELECT ON SEQUENCE virtual_accounts_id_seq TO trader;
GRANT USAGE, SELECT ON SEQUENCE virtual_positions_id_seq TO trader;

-- 顯示成功消息
DO $$
BEGIN
    RAISE NOTICE '✅ Virtual accounts migration completed!';
    RAISE NOTICE '💰 Each model starts with 10,000 USDT';
END $$;
