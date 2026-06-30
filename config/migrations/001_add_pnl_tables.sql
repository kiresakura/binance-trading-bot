-- ============================================
-- Migration: Add PnL Tracking Tables
-- Version: 0.2.1
-- Date: 2024-12
-- Description: 添加盈虧追蹤相關表
-- ============================================

-- 檢查是否已存在 binance_trades 表
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'binance_trades') THEN
        -- 16. Binance 原始交易記錄表 (時序表)
        CREATE TABLE binance_trades (
            id BIGINT NOT NULL,
            order_id BIGINT NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            trade_time TIMESTAMPTZ NOT NULL,
            side VARCHAR(10) NOT NULL,
            price DECIMAL(20, 8) NOT NULL,
            qty DECIMAL(20, 8) NOT NULL,
            quote_qty DECIMAL(20, 8) NOT NULL,
            commission DECIMAL(20, 8) NOT NULL,
            commission_asset VARCHAR(20) NOT NULL,
            is_buyer BOOLEAN NOT NULL,
            is_maker BOOLEAN NOT NULL,
            is_best_match BOOLEAN DEFAULT TRUE,
            synced_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_time, id, symbol)
        );

        -- 轉換為時序表
        PERFORM create_hypertable(
            'binance_trades',
            'trade_time',
            if_not_exists => TRUE,
            chunk_time_interval => INTERVAL '1 month'
        );

        CREATE UNIQUE INDEX idx_binance_trades_id ON binance_trades (id, symbol);
        CREATE INDEX idx_binance_trades_symbol ON binance_trades (symbol, trade_time DESC);
        CREATE INDEX idx_binance_trades_order ON binance_trades (order_id, symbol);

        RAISE NOTICE '✅ Created binance_trades table';
    ELSE
        RAISE NOTICE 'ℹ️ binance_trades table already exists';
    END IF;
END $$;

-- 17. 持倉成本追蹤表
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'position_costs') THEN
        CREATE TABLE position_costs (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL UNIQUE,
            base_asset VARCHAR(20) NOT NULL,
            quote_asset VARCHAR(20) NOT NULL,
            quantity DECIMAL(20, 8) NOT NULL DEFAULT 0,
            total_cost DECIMAL(20, 8) NOT NULL DEFAULT 0,
            average_cost DECIMAL(20, 8) NOT NULL DEFAULT 0,
            realized_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
            unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
            last_trade_id BIGINT,
            last_trade_time TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX idx_position_costs_symbol ON position_costs (symbol);
        CREATE INDEX idx_position_costs_base_asset ON position_costs (base_asset);

        RAISE NOTICE '✅ Created position_costs table';
    ELSE
        RAISE NOTICE 'ℹ️ position_costs table already exists';
    END IF;
END $$;

-- 18. 每日盈虧統計表
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'daily_pnl') THEN
        CREATE TABLE daily_pnl (
            date DATE NOT NULL,
            symbol VARCHAR(20) NOT NULL DEFAULT 'ALL',
            starting_balance DECIMAL(20, 8) NOT NULL DEFAULT 0,
            ending_balance DECIMAL(20, 8) NOT NULL DEFAULT 0,
            realized_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
            unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
            total_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
            pnl_percentage DECIMAL(10, 4) DEFAULT 0,
            trade_count INTEGER DEFAULT 0,
            buy_volume DECIMAL(20, 8) DEFAULT 0,
            sell_volume DECIMAL(20, 8) DEFAULT 0,
            commission_total DECIMAL(20, 8) DEFAULT 0,
            win_trades INTEGER DEFAULT 0,
            loss_trades INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, symbol)
        );

        CREATE INDEX idx_daily_pnl_date ON daily_pnl (date DESC);
        CREATE INDEX idx_daily_pnl_symbol ON daily_pnl (symbol, date DESC);

        RAISE NOTICE '✅ Created daily_pnl table';
    ELSE
        RAISE NOTICE 'ℹ️ daily_pnl table already exists';
    END IF;
END $$;

-- 19. 盈虧快照表 (時序表)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'pnl_snapshots') THEN
        CREATE TABLE pnl_snapshots (
            timestamp TIMESTAMPTZ NOT NULL,
            snapshot_type VARCHAR(20) NOT NULL DEFAULT 'hourly',
            total_balance_usdt DECIMAL(20, 8) NOT NULL,
            available_balance_usdt DECIMAL(20, 8),
            in_position_usdt DECIMAL(20, 8),
            total_realized_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
            total_unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
            total_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
            pnl_percentage DECIMAL(10, 4) DEFAULT 0,
            btc_price DECIMAL(20, 8),
            position_count INTEGER DEFAULT 0,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (timestamp, snapshot_type)
        );

        PERFORM create_hypertable(
            'pnl_snapshots',
            'timestamp',
            if_not_exists => TRUE,
            chunk_time_interval => INTERVAL '1 day'
        );

        CREATE INDEX idx_pnl_snapshots_type ON pnl_snapshots (snapshot_type, timestamp DESC);

        RAISE NOTICE '✅ Created pnl_snapshots table';
    ELSE
        RAISE NOTICE 'ℹ️ pnl_snapshots table already exists';
    END IF;
END $$;

-- 20. 交易成本明細表 (FIFO 用)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'cost_basis_lots') THEN
        CREATE TABLE cost_basis_lots (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            trade_id BIGINT NOT NULL,
            buy_time TIMESTAMPTZ NOT NULL,
            quantity DECIMAL(20, 8) NOT NULL,
            remaining_qty DECIMAL(20, 8) NOT NULL,
            cost_per_unit DECIMAL(20, 8) NOT NULL,
            total_cost DECIMAL(20, 8) NOT NULL,
            is_closed BOOLEAN DEFAULT FALSE,
            closed_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX idx_cost_basis_lots_symbol ON cost_basis_lots (symbol, is_closed, buy_time);
        CREATE INDEX idx_cost_basis_lots_remaining ON cost_basis_lots (symbol, remaining_qty) WHERE remaining_qty > 0;

        RAISE NOTICE '✅ Created cost_basis_lots table';
    ELSE
        RAISE NOTICE 'ℹ️ cost_basis_lots table already exists';
    END IF;
END $$;

-- 設置壓縮（僅對 hypertables）
DO $$
BEGIN
    -- binance_trades 壓縮
    BEGIN
        ALTER TABLE binance_trades SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol'
        );
        PERFORM add_compression_policy('binance_trades', INTERVAL '7 days', if_not_exists => TRUE);
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'ℹ️ binance_trades compression already configured or not a hypertable';
    END;

    -- pnl_snapshots 壓縮
    BEGIN
        ALTER TABLE pnl_snapshots SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'snapshot_type'
        );
        PERFORM add_compression_policy('pnl_snapshots', INTERVAL '7 days', if_not_exists => TRUE);
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'ℹ️ pnl_snapshots compression already configured or not a hypertable';
    END;
END $$;

-- 授予權限
DO $$
BEGIN
    -- 檢查 trader 用戶是否存在
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'trader') THEN
        GRANT ALL PRIVILEGES ON binance_trades TO trader;
        GRANT ALL PRIVILEGES ON position_costs TO trader;
        GRANT ALL PRIVILEGES ON daily_pnl TO trader;
        GRANT ALL PRIVILEGES ON pnl_snapshots TO trader;
        GRANT ALL PRIVILEGES ON cost_basis_lots TO trader;
        GRANT USAGE, SELECT ON SEQUENCE position_costs_id_seq TO trader;
        GRANT USAGE, SELECT ON SEQUENCE cost_basis_lots_id_seq TO trader;
        RAISE NOTICE '✅ Granted permissions to trader';
    ELSE
        RAISE NOTICE 'ℹ️ trader user does not exist, skipping grants';
    END IF;
END $$;

-- 記錄遷移
INSERT INTO event_log (timestamp, event_type, severity, source, message, data)
VALUES (
    NOW(),
    'MIGRATION',
    'INFO',
    '001_add_pnl_tables.sql',
    'PnL tracking tables migration completed',
    '{"version": "0.2.1", "tables": ["binance_trades", "position_costs", "daily_pnl", "pnl_snapshots", "cost_basis_lots"]}'::jsonb
);

RAISE NOTICE '✅ Migration 001_add_pnl_tables completed successfully!';
