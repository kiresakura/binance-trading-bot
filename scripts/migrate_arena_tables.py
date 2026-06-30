#!/usr/bin/env python3
"""
Arena 缺失資料表 Migration
建立 shadow_trades 和 strategy_sensitivity 表
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from src.config import settings


async def migrate():
    """建立缺失的 Arena 表"""
    
    print("=" * 60)
    print("🚀 開始建立缺失的 Arena 資料表...")
    print("=" * 60)
    
    from sqlalchemy.ext.asyncio import create_async_engine
    
    try:
        engine = create_async_engine(settings.ASYNC_DATABASE_URL, echo=True)
        
        async with engine.begin() as conn:
            
            # 1. shadow_trades 表 - 影子交易記錄
            print("\n📦 建立 shadow_trades 表...")
            await conn.execute(text("""
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
            """))
            
            # 建立索引
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_shadow_trades_model_version 
                ON shadow_trades(model_version)
            """))
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_shadow_trades_timestamp 
                ON shadow_trades(timestamp)
            """))
            
            # 2. strategy_sensitivity 表 - 策略靈敏度配置
            print("\n📦 建立 strategy_sensitivity 表...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS strategy_sensitivity (
                    id INTEGER PRIMARY KEY,
                    sensitivity TEXT NOT NULL DEFAULT 'moderate',
                    confidence_threshold REAL NOT NULL DEFAULT 0.70,
                    strength_threshold REAL NOT NULL DEFAULT 0.60,
                    position_ratio REAL NOT NULL DEFAULT 0.6,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # 初始化預設配置
            await conn.execute(text("""
                INSERT INTO strategy_sensitivity 
                (id, sensitivity, confidence_threshold, strength_threshold, position_ratio)
                VALUES (1, 'moderate', 0.70, 0.60, 0.6)
                ON CONFLICT (id) DO NOTHING
            """))
            
            # 3. promotion_history 表（如果不存在）
            print("\n📦 確保 promotion_history 表存在...")
            await conn.execute(text("""
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
            """))
            
            # 4. virtual_accounts 表（如果不存在）
            print("\n📦 確保 virtual_accounts 表存在...")
            await conn.execute(text("""
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
            """))
            
            # 5. virtual_positions 表
            print("\n📦 確保 virtual_positions 表存在...")
            await conn.execute(text("""
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
            """))
            
            # 6. virtual_trades 表
            print("\n📦 確保 virtual_trades 表存在...")
            await conn.execute(text("""
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
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_virtual_trades_model 
                ON virtual_trades(model_version)
            """))
        
        await engine.dispose()
        
        print("\n" + "=" * 60)
        print("✅ Migration 完成！已建立以下表：")
        print("   - shadow_trades")
        print("   - strategy_sensitivity")
        print("   - promotion_history")
        print("   - virtual_accounts")
        print("   - virtual_positions")
        print("   - virtual_trades")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Migration 失敗: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(migrate())
