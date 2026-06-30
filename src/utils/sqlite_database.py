"""SQLite 数据库连接管理 - 轻量级替代 PostgreSQL"""

import os
import sqlite3
import aiosqlite
from contextlib import contextmanager, asynccontextmanager
from typing import Generator, AsyncGenerator, Optional, List, Dict, Any
from pathlib import Path

from src.config import settings
from src.logger import logger


# 确保 data 目录存在
def ensure_data_directory():
    """确保数据目录存在"""
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


class SQLiteDatabaseManager:
    """同步 SQLite 数据库管理器"""

    def __init__(self):
        self.db_path: Optional[str] = None
        self.connection: Optional[sqlite3.Connection] = None
        self._initialized = False

    def init(self):
        """初始化数据库连接"""
        if self._initialized:
            logger.warning("SQLite database already initialized")
            return

        try:
            ensure_data_directory()
            self.db_path = settings.SQLITE_DB_PATH

            # 创建连接
            self.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self.connection.row_factory = sqlite3.Row

            # 启用 WAL 模式以提高并发性能
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self.connection.execute("PRAGMA cache_size=-64000")  # 64MB cache
            self.connection.execute("PRAGMA foreign_keys=ON")

            # 初始化表结构
            self._init_tables()

            self._initialized = True
            logger.info(f"SQLite database initialized: {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise

    def _init_tables(self):
        """初始化数据库表"""
        cursor = self.connection.cursor()

        # Arena 模型表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS arena_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                shadow_start_at TEXT,
                promoted_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Arena 状态表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS arena_state (
                id INTEGER PRIMARY KEY,
                champion_version TEXT,
                challenger_version TEXT,
                comparison_days INTEGER DEFAULT 7,
                promotion_threshold REAL DEFAULT 0.05,
                is_comparing INTEGER DEFAULT 0,
                started_at TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 初始化 arena_state
        cursor.execute("""
            INSERT OR IGNORE INTO arena_state (id, is_comparing, comparison_days, promotion_threshold)
            VALUES (1, 0, 7, 0.05)
        """)

        # 影子交易记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shadow_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_version TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                signal_strength REAL,
                signal_confidence REAL,
                suggested_quantity REAL,
                suggested_price REAL,
                actual_price REAL,
                virtual_pnl REAL,
                would_have_triggered INTEGER DEFAULT 0,
                reason TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 晋升历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS promotion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                previous_champion TEXT,
                new_champion TEXT NOT NULL,
                reason TEXT,
                champion_performance TEXT,
                challenger_performance TEXT,
                promoted_by TEXT DEFAULT 'system',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 策略灵敏度配置表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_sensitivity (
                id INTEGER PRIMARY KEY,
                sensitivity TEXT NOT NULL DEFAULT 'moderate',
                confidence_threshold REAL DEFAULT 0.70,
                strength_threshold REAL DEFAULT 0.60,
                position_ratio REAL DEFAULT 0.6,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 初始化策略灵敏度
        cursor.execute("""
            INSERT OR IGNORE INTO strategy_sensitivity (id, sensitivity, confidence_threshold, strength_threshold, position_ratio)
            VALUES (1, 'moderate', 0.70, 0.60, 0.6)
        """)

        # 审计日志表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user_id TEXT,
                action TEXT NOT NULL,
                resource TEXT NOT NULL,
                details TEXT NOT NULL,
                risk_score REAL,
                prev_hash TEXT NOT NULL,
                current_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_arena_models_status ON arena_models(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_arena_models_version ON arena_models(version)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_shadow_trades_model ON shadow_trades(model_version, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_shadow_trades_symbol ON shadow_trades(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON audit_log(event_type)")

        self.connection.commit()
        logger.info("SQLite tables initialized successfully")

    @contextmanager
    def get_session(self) -> Generator[sqlite3.Connection, None, None]:
        """获取数据库连接（上下文管理器）"""
        if not self._initialized:
            self.init()

        try:
            yield self.connection
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"SQLite session error: {e}")
            raise

    def execute_query(self, query: str, params: tuple = None) -> List[sqlite3.Row]:
        """执行查询"""
        with self.get_session() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()

    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            self._initialized = False
            logger.info("SQLite connection closed")

    def health_check(self) -> bool:
        """健康检查"""
        try:
            with self.get_session() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"SQLite health check failed: {e}")
            return False


class AsyncSQLiteDatabaseManager:
    """异步 SQLite 数据库管理器"""

    def __init__(self):
        self.db_path: Optional[str] = None
        self._initialized = False

    async def init(self):
        """初始化异步数据库连接"""
        if self._initialized:
            logger.warning("Async SQLite database already initialized")
            return

        try:
            ensure_data_directory()
            self.db_path = settings.SQLITE_DB_PATH

            # 测试连接并初始化表
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                # 启用 WAL 模式
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("PRAGMA synchronous=NORMAL")
                await db.execute("PRAGMA cache_size=-64000")
                await db.execute("PRAGMA foreign_keys=ON")

                # 初始化表结构
                await self._init_tables(db)
                await db.commit()

            self._initialized = True
            logger.info(f"Async SQLite database initialized: {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize async SQLite database: {e}")
            raise

    async def _init_tables(self, db: aiosqlite.Connection):
        """初始化数据库表"""
        # Arena 模型表
        await db.execute("""
            CREATE TABLE IF NOT EXISTS arena_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                shadow_start_at TEXT,
                promoted_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Arena 状态表
        await db.execute("""
            CREATE TABLE IF NOT EXISTS arena_state (
                id INTEGER PRIMARY KEY,
                champion_version TEXT,
                challenger_version TEXT,
                comparison_days INTEGER DEFAULT 7,
                promotion_threshold REAL DEFAULT 0.05,
                is_comparing INTEGER DEFAULT 0,
                started_at TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 初始化 arena_state
        await db.execute("""
            INSERT OR IGNORE INTO arena_state (id, is_comparing, comparison_days, promotion_threshold)
            VALUES (1, 0, 7, 0.05)
        """)

        # 影子交易记录表
        await db.execute("""
            CREATE TABLE IF NOT EXISTS shadow_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_version TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                signal_strength REAL,
                signal_confidence REAL,
                suggested_quantity REAL,
                suggested_price REAL,
                actual_price REAL,
                virtual_pnl REAL,
                would_have_triggered INTEGER DEFAULT 0,
                reason TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 晋升历史表
        await db.execute("""
            CREATE TABLE IF NOT EXISTS promotion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                previous_champion TEXT,
                new_champion TEXT NOT NULL,
                reason TEXT,
                champion_performance TEXT,
                challenger_performance TEXT,
                promoted_by TEXT DEFAULT 'system',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 策略灵敏度配置表
        await db.execute("""
            CREATE TABLE IF NOT EXISTS strategy_sensitivity (
                id INTEGER PRIMARY KEY,
                sensitivity TEXT NOT NULL DEFAULT 'moderate',
                confidence_threshold REAL DEFAULT 0.70,
                strength_threshold REAL DEFAULT 0.60,
                position_ratio REAL DEFAULT 0.6,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 初始化策略灵敏度
        await db.execute("""
            INSERT OR IGNORE INTO strategy_sensitivity (id, sensitivity, confidence_threshold, strength_threshold, position_ratio)
            VALUES (1, 'moderate', 0.70, 0.60, 0.6)
        """)

        # 审计日志表
        await db.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user_id TEXT,
                action TEXT NOT NULL,
                resource TEXT NOT NULL,
                details TEXT NOT NULL,
                risk_score REAL,
                prev_hash TEXT NOT NULL,
                current_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建索引
        await db.execute("CREATE INDEX IF NOT EXISTS idx_arena_models_status ON arena_models(status)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_arena_models_version ON arena_models(version)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_shadow_trades_model ON shadow_trades(model_version, timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_shadow_trades_symbol ON shadow_trades(symbol, timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON audit_log(event_type)")

        logger.info("Async SQLite tables initialized successfully")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """获取异步数据库连接"""
        if not self._initialized:
            await self.init()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            try:
                yield db
                await db.commit()
            except Exception as e:
                await db.rollback()
                logger.error(f"Async SQLite session error: {e}")
                raise

    async def execute(self, query: str, *args) -> List[Dict[str, Any]]:
        """执行查询（返回多行）- 兼容 asyncpg 风格"""
        if not self._initialized:
            await self.init()

        # 转换 PostgreSQL 风格的参数占位符 $1, $2 为 SQLite 的 ?
        converted_query = self._convert_placeholders(query)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            try:
                cursor = await db.execute(converted_query, args)
                rows = await cursor.fetchall()
                # 对于 INSERT/UPDATE/DELETE 需要 commit
                await db.commit()
                # 转换为字典列表，兼容 asyncpg 风格
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Query execution failed: {e}, query: {query}")
                raise

    async def execute_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """执行查询（返回单行）- 兼容 asyncpg 风格"""
        if not self._initialized:
            await self.init()

        # 转换 PostgreSQL 风格的参数占位符
        converted_query = self._convert_placeholders(query)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            try:
                cursor = await db.execute(converted_query, args)
                row = await cursor.fetchone()
                # 对于 INSERT/UPDATE/DELETE 需要 commit
                await db.commit()
                return dict(row) if row else None
            except Exception as e:
                logger.error(f"Query execution failed: {e}, query: {query}")
                raise

    async def execute_many(self, query: str, args_list: list):
        """批量执行查询"""
        if not self._initialized:
            await self.init()

        converted_query = self._convert_placeholders(query)

        async with aiosqlite.connect(self.db_path) as db:
            try:
                await db.executemany(converted_query, args_list)
                await db.commit()
            except Exception as e:
                logger.error(f"Batch query execution failed: {e}")
                raise

    def _convert_placeholders(self, query: str) -> str:
        """转换 PostgreSQL 参数占位符 $1, $2 为 SQLite 的 ?"""
        import re
        # 替换 $1, $2, ... 为 ?
        return re.sub(r'\$\d+', '?', query)

    async def close(self):
        """关闭数据库连接"""
        self._initialized = False
        logger.info("Async SQLite connection closed")

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            result = await self.execute_one("SELECT 1 as check_value")
            return result is not None
        except Exception as e:
            logger.error(f"Async SQLite health check failed: {e}")
            return False


# 全局数据库管理器实例
sqlite_db_manager = SQLiteDatabaseManager()
async_sqlite_db_manager = AsyncSQLiteDatabaseManager()
