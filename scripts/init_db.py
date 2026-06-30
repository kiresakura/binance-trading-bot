#!/usr/bin/env python3
"""
資料庫初始化腳本
在 Railway 或本地執行以建立所有需要的表
"""

import asyncio
import sys
import os

# 添加專案路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from src.config import settings
from src.logger import logger


async def init_database():
    """初始化資料庫表"""
    
    # 檢查是否使用 PostgreSQL
    if settings.should_use_sqlite:
        print("⚠️  當前配置使用 SQLite，不需要手動初始化")
        print("    SQLite 會自動建立表")
        return
    
    print("=" * 60)
    print("🚀 開始初始化 PostgreSQL 資料庫...")
    print(f"📍 Database URL: {settings.DATABASE_URL[:50]}...")
    print("=" * 60)
    
    # 導入必要的模組
    from sqlalchemy.ext.asyncio import create_async_engine
    from src.models.base import Base
    from src.models.trading import Position, Order, EmergencyEvent, StateSnapshot
    
    try:
        # 建立異步引擎
        engine = create_async_engine(
            settings.ASYNC_DATABASE_URL,
            echo=True  # 顯示 SQL 語句
        )
        
        # 建立所有表
        async with engine.begin() as conn:
            print("\n📦 建立 ORM 資料庫表...")
            await conn.run_sync(Base.metadata.create_all)
            
            # 建立 Arena 相關表（非 ORM）
            print("\n📦 建立 Arena 資料庫表...")
            
            # arena_models 表
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS arena_models (
                    id SERIAL PRIMARY KEY,
                    version TEXT NOT NULL UNIQUE,
                    model_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'TRAINING',
                    win_rate REAL DEFAULT 0,
                    total_return REAL DEFAULT 0,
                    trade_count INTEGER DEFAULT 0,
                    draw_down REAL DEFAULT 0,
                    sharpe_ratio REAL,
                    profit_factor REAL,
                    avg_hold_time TEXT,
                    config TEXT DEFAULT '{}',
                    shadow_start_at TIMESTAMP,
                    promoted_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # arena_state 表
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS arena_state (
                    id INTEGER PRIMARY KEY,
                    champion_version TEXT,
                    challenger_version TEXT,
                    comparison_days INTEGER DEFAULT 7,
                    promotion_threshold REAL DEFAULT 0.05,
                    is_comparing INTEGER DEFAULT 0,
                    started_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # 初始化 arena_state（如果不存在）
            await conn.execute(text("""
                INSERT INTO arena_state (id, is_comparing, comparison_days, promotion_threshold)
                VALUES (1, 0, 7, 0.05)
                ON CONFLICT (id) DO NOTHING
            """))
            
            # audit_log 表
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    details TEXT NOT NULL,
                    risk_score REAL,
                    prev_hash TEXT NOT NULL,
                    current_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # 建立索引
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON audit_log(event_type)"))
        
        print("\n✅ 資料庫表建立成功！")
        print("\n📋 已建立的表：")
        for table_name in Base.metadata.tables.keys():
            print(f"   - {table_name}")
        
        # 驗證表存在
        async with engine.connect() as conn:
            print("\n🔍 驗證表存在...")
            
            # 檢查每個表
            tables_to_check = ['positions', 'orders', 'emergency_events', 'state_snapshots']
            for table in tables_to_check:
                result = await conn.execute(
                    text(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table}')")
                )
                exists = result.scalar()
                status = "✅" if exists else "❌"
                print(f"   {status} {table}")
        
        await engine.dispose()
        print("\n🎉 資料庫初始化完成！")
        
    except Exception as e:
        print(f"\n❌ 初始化失敗: {e}")
        raise


async def drop_all_tables():
    """刪除所有表（危險操作！）"""
    
    if settings.should_use_sqlite:
        print("⚠️  SQLite 模式，跳過")
        return
    
    confirm = input("⚠️  這將刪除所有資料！確認? (輸入 'yes' 確認): ")
    if confirm.lower() != 'yes':
        print("取消操作")
        return
    
    from sqlalchemy.ext.asyncio import create_async_engine
    from src.models.base import Base
    
    engine = create_async_engine(settings.ASYNC_DATABASE_URL, echo=True)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()
    print("✅ 所有表已刪除")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="資料庫管理工具")
    parser.add_argument("--drop", action="store_true", help="刪除所有表（危險！）")
    args = parser.parse_args()
    
    if args.drop:
        asyncio.run(drop_all_tables())
    else:
        asyncio.run(init_database())
