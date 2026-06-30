"""数据库连接管理

支持两种模式：
- SQLite 模式（USE_SQLITE=True）：轻量级，无需外部服务
- PostgreSQL 模式（USE_SQLITE=False）：重量级，需要 PostgreSQL 服务
"""

from contextlib import contextmanager, asynccontextmanager
from typing import Generator, AsyncGenerator, Optional, List, Dict, Any
from src.config import settings
from src.logger import logger


# 根据配置决定使用哪种数据库
# 使用 should_use_sqlite 自動判斷（沒有 DATABASE_URL 時用 SQLite）
if settings.should_use_sqlite:
    # ==================== SQLite 模式 ====================
    from src.utils.sqlite_database import (
        SQLiteDatabaseManager,
        AsyncSQLiteDatabaseManager,
        sqlite_db_manager,
        async_sqlite_db_manager
    )

    # 使用 SQLite 数据库管理器
    db_manager = sqlite_db_manager
    async_db_manager = async_sqlite_db_manager
    logger.info("Using SQLite database mode")

else:
    # ==================== PostgreSQL 模式 ====================
    from sqlalchemy import create_engine, event, pool
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    import asyncpg

    class DatabaseManager:
        """同步数据库管理器"""

        def __init__(self):
            self.engine = None
            self.session_factory = None
            self._initialized = False

        def init(self):
            """初始化数据库连接"""
            if self._initialized:
                logger.warning("Database already initialized")
                return

            try:
                # 创建同步引擎
                self.engine = create_engine(
                    settings.DATABASE_URL,
                    echo=settings.DEBUG,
                    pool_pre_ping=True,  # 连接前ping测试
                    pool_size=10,  # 连接池大小
                    max_overflow=20,  # 最大溢出连接数
                    pool_recycle=3600,  # 连接回收时间（秒）
                    connect_args={
                        "connect_timeout": 10,
                    }
                )

                # 设置连接池事件监听
                @event.listens_for(self.engine, "connect")
                def receive_connect(dbapi_conn, connection_record):
                    """连接建立时的回调"""
                    logger.debug("Database connection established")

                @event.listens_for(self.engine, "close")
                def receive_close(dbapi_conn, connection_record):
                    """连接关闭时的回调"""
                    logger.debug("Database connection closed")

                # 建立 Session 工厂
                self.session_factory = sessionmaker(
                    bind=self.engine,
                    autocommit=False,
                    autoflush=False
                )

                self._initialized = True
                logger.info("Database connection initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize database connection: {e}")
                raise

        @contextmanager
        def get_session(self) -> Generator[Session, None, None]:
            """获取数据库会话（上下文管理器）

            Usage:
                with db_manager.get_session() as session:
                    result = session.query(Model).all()
            """
            if not self._initialized:
                self.init()

            session = self.session_factory()
            try:
                yield session
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                session.close()

        def execute_query(self, query: str, params: dict = None):
            """执行原始 SQL 查询

            Args:
                query: SQL 查询语句
                params: 查询参数

            Returns:
                查询结果
            """
            with self.get_session() as session:
                result = session.execute(query, params or {})
                return result.fetchall()

        def close(self):
            """关闭数据库连接"""
            if self.engine:
                self.engine.dispose()
                self._initialized = False
                logger.info("Database connection closed")

        def health_check(self) -> bool:
            """健康检查"""
            try:
                with self.get_session() as session:
                    session.execute("SELECT 1")
                return True
            except Exception as e:
                logger.error(f"Database health check failed: {e}")
                return False


    class AsyncDatabaseManager:
        """异步数据库管理器"""

        def __init__(self):
            self.engine = None
            self.async_session_factory = None
            self.pool = None
            self._initialized = False

        async def init(self):
            """初始化异步数据库连接"""
            if self._initialized:
                logger.warning("Async database already initialized")
                return

            try:
                # 创建异步引擎（用于 SQLAlchemy ORM）
                self.engine = create_async_engine(
                    settings.ASYNC_DATABASE_URL,
                    echo=settings.DEBUG,
                    pool_pre_ping=True,
                    pool_size=10,
                    max_overflow=20,
                    pool_recycle=3600,
                )

                # 建立异步 Session 工厂
                self.async_session_factory = async_sessionmaker(
                    self.engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )

                # 创建 asyncpg 连接池（用于原始 SQL）
                # 優先使用 DATABASE_URL（Railway 環境）
                if settings.DATABASE_URL:
                    # 直接使用 DATABASE_URL 建立連線池
                    self.pool = await asyncpg.create_pool(
                        dsn=settings.DATABASE_URL,
                        min_size=5,
                        max_size=20,
                        timeout=30,
                        command_timeout=60,
                    )
                else:
                    self.pool = await asyncpg.create_pool(
                        host=settings.DB_HOST,
                        port=settings.DB_PORT,
                        database=settings.DB_NAME,
                        user=settings.DB_USER,
                        password=settings.DB_PASSWORD,
                        min_size=5,
                        max_size=20,
                        timeout=30,
                        command_timeout=60,
                )

                self._initialized = True
                logger.info("Async database connection initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize async database connection: {e}")
                raise

        @asynccontextmanager
        async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
            """获取异步数据库会话（上下文管理器）

            Usage:
                async with async_db_manager.get_session() as session:
                    result = await session.execute(query)
            """
            if not self._initialized:
                await self.init()

            async with self.async_session_factory() as session:
                try:
                    yield session
                    await session.commit()
                except Exception as e:
                    await session.rollback()
                    logger.error(f"Async database session error: {e}")
                    raise

        async def execute(self, query: str, *args):
            """执行异步查询（返回多行）

            Args:
                query: SQL 查询语句
                *args: 查询参数

            Returns:
                查询结果列表
            """
            if not self._initialized:
                await self.init()

            async with self.pool.acquire() as connection:
                try:
                    result = await connection.fetch(query, *args)
                    return result
                except Exception as e:
                    logger.error(f"Query execution failed: {e}")
                    raise

        async def execute_one(self, query: str, *args):
            """执行异步查询（返回单行）

            Args:
                query: SQL 查询语句
                *args: 查询参数

            Returns:
                单行查询结果
            """
            if not self._initialized:
                await self.init()

            async with self.pool.acquire() as connection:
                try:
                    result = await connection.fetchrow(query, *args)
                    return result
                except Exception as e:
                    logger.error(f"Query execution failed: {e}")
                    raise

        async def execute_many(self, query: str, args_list: list):
            """批量执行异步查询

            Args:
                query: SQL 查询语句
                args_list: 参数列表

            Returns:
                执行结果
            """
            if not self._initialized:
                await self.init()

            async with self.pool.acquire() as connection:
                try:
                    result = await connection.executemany(query, args_list)
                    return result
                except Exception as e:
                    logger.error(f"Batch query execution failed: {e}")
                    raise

        async def close(self):
            """关闭异步数据库连接"""
            if self.pool:
                await self.pool.close()

            if self.engine:
                await self.engine.dispose()

            self._initialized = False
            logger.info("Async database connection closed")

        async def health_check(self) -> bool:
            """健康检查"""
            try:
                result = await self.execute_one("SELECT 1")
                return result is not None
            except Exception as e:
                logger.error(f"Async database health check failed: {e}")
                return False

        # 別名方法 - 為了兼容性
        async def fetch_all(self, query: str, *args):
            """fetch_all 是 execute 的別名"""
            return await self.execute(query, *args)

        async def fetch_one(self, query: str, *args):
            """fetch_one 是 execute_one 的別名"""
            return await self.execute_one(query, *args)

    # 使用 PostgreSQL 数据库管理器
    db_manager = DatabaseManager()
    async_db_manager = AsyncDatabaseManager()
    logger.info("Using PostgreSQL database mode")


# 便捷装饰器
def with_db_session(func):
    """数据库会话装饰器（同步）"""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with db_manager.get_session() as session:
            return func(session, *args, **kwargs)

    return wrapper


def with_async_db_session(func):
    """数据库会话装饰器（异步）"""
    import functools

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        async with async_db_manager.get_session() as session:
            return await func(session, *args, **kwargs)

    return wrapper
