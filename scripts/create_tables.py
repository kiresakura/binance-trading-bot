#!/usr/bin/env python3
"""建立資料庫表結構腳本"""

import asyncio
import sys
import os

# 添加項目根目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from src.utils.database import async_db_manager
from src.models.base import Base
from src.models.trading import Position, Order, EmergencyEvent, StateSnapshot
from src.logger import logger


# SQL 建表語句 - 分開執行每個語句
CREATE_POSITIONS_SQL = """
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity NUMERIC(20, 8) NOT NULL,
    entry_price NUMERIC(20, 8) NOT NULL,
    current_price NUMERIC(20, 8),
    unrealized_pnl NUMERIC(20, 8) DEFAULT 0,
    realized_pnl NUMERIC(20, 8) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'open',
    stop_loss NUMERIC(20, 8),
    take_profit NUMERIC(20, 8),
    risk_score NUMERIC(5, 2),
    strategy VARCHAR(50),
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json JSONB
)
"""

CREATE_ORDERS_SQL = """
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(50) UNIQUE,
    client_order_id VARCHAR(50),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    quantity NUMERIC(20, 8) NOT NULL,
    price NUMERIC(20, 8),
    executed_qty NUMERIC(20, 8) DEFAULT 0,
    executed_price NUMERIC(20, 8),
    status VARCHAR(20) DEFAULT 'pending',
    commission NUMERIC(20, 8) DEFAULT 0,
    commission_asset VARCHAR(10),
    risk_score NUMERIC(5, 2),
    risk_reason TEXT,
    strategy VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    submitted_at TIMESTAMP,
    filled_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json JSONB
)
"""

CREATE_EMERGENCY_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS emergency_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(30) NOT NULL,
    reason TEXT NOT NULL,
    severity VARCHAR(20) DEFAULT 'critical',
    snapshot_id VARCHAR(100),
    open_positions_count INTEGER DEFAULT 0,
    open_orders_count INTEGER DEFAULT 0,
    total_exposure NUMERIC(20, 8),
    orders_cancelled INTEGER DEFAULT 0,
    positions_closed INTEGER DEFAULT 0,
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    resolution_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json JSONB
)
"""

CREATE_STATE_SNAPSHOTS_SQL = """
CREATE TABLE IF NOT EXISTS state_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_key VARCHAR(100) UNIQUE NOT NULL,
    reason VARCHAR(50) NOT NULL,
    total_value NUMERIC(20, 8),
    available_balance NUMERIC(20, 8),
    positions_count INTEGER DEFAULT 0,
    orders_count INTEGER DEFAULT 0,
    checksum VARCHAR(64),
    is_valid BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    metadata_json JSONB
)
"""

# 添加缺失欄位的 SQL（安全地處理已存在的情況）
ALTER_ORDERS_SQL = [
    "ALTER TABLE orders ADD COLUMN IF NOT EXISTS client_order_id VARCHAR(50)",
    "ALTER TABLE orders ADD COLUMN IF NOT EXISTS commission_asset VARCHAR(10)",
    "ALTER TABLE orders ADD COLUMN IF NOT EXISTS risk_score NUMERIC(5, 2)",
    "ALTER TABLE orders ADD COLUMN IF NOT EXISTS risk_reason TEXT",
    "ALTER TABLE orders ADD COLUMN IF NOT EXISTS strategy VARCHAR(50)",
    "ALTER TABLE orders ADD COLUMN IF NOT EXISTS submitted_at TIMESTAMP",
    "ALTER TABLE orders ADD COLUMN IF NOT EXISTS filled_at TIMESTAMP",
    "ALTER TABLE orders ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    "ALTER TABLE orders ADD COLUMN IF NOT EXISTS metadata_json JSONB",
]

# 索引
CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)",
    "CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id)",
    "CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
    "CREATE INDEX IF NOT EXISTS idx_orders_client_order_id ON orders(client_order_id)",
    "CREATE INDEX IF NOT EXISTS idx_emergency_events_type ON emergency_events(event_type)",
    "CREATE INDEX IF NOT EXISTS idx_emergency_events_resolved ON emergency_events(is_resolved)",
    "CREATE INDEX IF NOT EXISTS idx_state_snapshots_key ON state_snapshots(snapshot_key)",
    "CREATE INDEX IF NOT EXISTS idx_state_snapshots_created ON state_snapshots(created_at)",
]


async def create_tables():
    """建立資料庫表"""
    try:
        # 初始化資料庫連接
        await async_db_manager.init()
        logger.info("Database connection initialized")

        async with async_db_manager.pool.acquire() as conn:
            # 1. 建立表
            logger.info("Creating tables...")

            await conn.execute(CREATE_POSITIONS_SQL)
            logger.info("  ✓ positions table")

            await conn.execute(CREATE_ORDERS_SQL)
            logger.info("  ✓ orders table")

            await conn.execute(CREATE_EMERGENCY_EVENTS_SQL)
            logger.info("  ✓ emergency_events table")

            await conn.execute(CREATE_STATE_SNAPSHOTS_SQL)
            logger.info("  ✓ state_snapshots table")

            # 2. 添加可能缺失的欄位（針對已存在的表）
            logger.info("Adding missing columns to orders table...")
            for sql in ALTER_ORDERS_SQL:
                try:
                    await conn.execute(sql)
                except Exception as e:
                    # 忽略欄位已存在的錯誤
                    if "already exists" not in str(e).lower():
                        logger.warning(f"  Column alter warning: {e}")

            # 3. 建立索引
            logger.info("Creating indexes...")
            for sql in CREATE_INDEXES_SQL:
                try:
                    await conn.execute(sql)
                except Exception as e:
                    logger.warning(f"  Index warning: {e}")

            logger.info("✅ All tables created/updated successfully!")

        # 驗證表是否存在
        tables = ['positions', 'orders', 'emergency_events', 'state_snapshots']
        async with async_db_manager.pool.acquire() as conn:
            for table in tables:
                result = await conn.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
                    table
                )
                if result:
                    # 獲取欄位數量
                    col_count = await conn.fetchval(
                        "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = $1",
                        table
                    )
                    logger.info(f"  ✓ Table '{table}' exists ({col_count} columns)")
                else:
                    logger.error(f"  ✗ Table '{table}' not found!")

        logger.info("🎉 Database setup complete!")

    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise
    finally:
        await async_db_manager.close()


async def drop_tables():
    """刪除所有表（謹慎使用）"""
    try:
        await async_db_manager.init()

        drop_sql = """
        DROP TABLE IF EXISTS state_snapshots CASCADE;
        DROP TABLE IF EXISTS emergency_events CASCADE;
        DROP TABLE IF EXISTS orders CASCADE;
        DROP TABLE IF EXISTS positions CASCADE;
        """

        async with async_db_manager.pool.acquire() as conn:
            await conn.execute(drop_sql)
            logger.info("✅ All tables dropped!")

    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        raise
    finally:
        await async_db_manager.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Database table management')
    parser.add_argument('--drop', action='store_true', help='Drop all tables first')
    args = parser.parse_args()

    if args.drop:
        print("⚠️  WARNING: This will DROP all tables!")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            asyncio.run(drop_tables())
        else:
            print("Cancelled")
            sys.exit(0)

    asyncio.run(create_tables())
