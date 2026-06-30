#!/usr/bin/env python3
"""執行 Arena 資料表 Migration

連線字串必須透過環境變數 DATABASE_URL 提供，例如：

    export DATABASE_URL="postgresql://user:password@host:port/dbname"
    python scripts/run_migration.py

切勿將真實憑證寫死在原始碼中。
"""
import asyncio
import os
import sys

import asyncpg

DATABASE_URL = os.getenv("DATABASE_URL")

async def migrate():
    if not DATABASE_URL:
        print("❌ 未設定 DATABASE_URL 環境變數，請先 export DATABASE_URL=...")
        sys.exit(1)
    print("🚀 連接 Railway PostgreSQL...")
    conn = await asyncpg.connect(DATABASE_URL)
    print("✅ 連接成功！")
    
    # 1. shadow_trades
    print("📦 建立 shadow_trades 表...")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS shadow_trades (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_version TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            signal_strength REAL,
            signal_confidence REAL,
            suggested_quantity REAL,
            suggested_price REAL,
            actual_price REAL,
            virtual_pnl REAL,
            would_have_triggered BOOLEAN DEFAULT FALSE,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_shadow_trades_model_version ON shadow_trades(model_version)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_shadow_trades_timestamp ON shadow_trades(timestamp)")
    
    # 2. strategy_sensitivity
    print("📦 建立 strategy_sensitivity 表...")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS strategy_sensitivity (
            id INTEGER PRIMARY KEY,
            sensitivity TEXT NOT NULL DEFAULT 'moderate',
            confidence_threshold REAL NOT NULL DEFAULT 0.70,
            strength_threshold REAL NOT NULL DEFAULT 0.60,
            position_ratio REAL NOT NULL DEFAULT 0.6,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await conn.execute("""
        INSERT INTO strategy_sensitivity (id, sensitivity, confidence_threshold, strength_threshold, position_ratio)
        VALUES (1, 'moderate', 0.70, 0.60, 0.6)
        ON CONFLICT (id) DO NOTHING
    """)
    
    # 3. promotion_history
    print("📦 建立 promotion_history 表...")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS promotion_history (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            previous_champion TEXT,
            new_champion TEXT NOT NULL,
            reason TEXT,
            champion_performance TEXT,
            challenger_performance TEXT,
            promoted_by TEXT DEFAULT 'system'
        )
    """)
    
    # 4. virtual_accounts
    print("📦 建立 virtual_accounts 表...")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS virtual_accounts (
            model_version TEXT PRIMARY KEY,
            initial_balance REAL DEFAULT 10000,
            balance REAL DEFAULT 10000,
            total_value REAL DEFAULT 10000,
            realized_pnl REAL DEFAULT 0,
            unrealized_pnl REAL DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            total_pnl_pct REAL DEFAULT 0,
            max_drawdown REAL DEFAULT 0,
            peak_value REAL DEFAULT 10000,
            trade_count INTEGER DEFAULT 0,
            win_count INTEGER DEFAULT 0,
            loss_count INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 5. virtual_positions
    print("📦 建立 virtual_positions 表...")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS virtual_positions (
            id SERIAL PRIMARY KEY,
            model_version TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT DEFAULT 'LONG',
            quantity REAL DEFAULT 0,
            entry_price REAL NOT NULL,
            current_price REAL,
            unrealized_pnl REAL DEFAULT 0,
            unrealized_pnl_pct REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(model_version, symbol)
        )
    """)
    
    # 6. virtual_trades
    print("📦 建立 virtual_trades 表...")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS virtual_trades (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_version TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            quote_value REAL NOT NULL,
            commission REAL DEFAULT 0,
            realized_pnl REAL,
            balance_after REAL,
            signal_strength REAL,
            signal_confidence REAL
        )
    """)
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_virtual_trades_model ON virtual_trades(model_version)")
    
    await conn.close()
    print("")
    print("=" * 60)
    print("✅ Migration 完成！已建立所有 Arena 資料表！")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(migrate())
