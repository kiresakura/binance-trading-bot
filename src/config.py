"""应用配置管理"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List
import os


class Settings(BaseSettings):
    """应用配置"""

    # ==================== 应用基本信息 ====================
    APP_NAME: str = "Binance Trading Bot - Enterprise Edition"
    VERSION: str = "0.2.0"
    DEBUG: bool = False
    # 安全預設：預設為 production（最嚴格）。本地開發請顯式設定 ENVIRONMENT=development。
    ENVIRONMENT: str = Field(default="production", description="development, staging, production")

    # ==================== 数据库配置 ====================
    # SQLite 配置（轻量级模式）
    # 如果有 DATABASE_URL 就用 PostgreSQL，沒有就 fallback 到 SQLite
    USE_SQLITE: bool = False  # 會根據 DATABASE_URL 自動判斷
    SQLITE_DB_PATH: str = "data/trading_bot.db"

    # PostgreSQL 配置（重量级模式，可选）
    # Railway 會自動設定 DATABASE_URL，優先使用
    DATABASE_URL: Optional[str] = None  # Railway 自動注入
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "trading_bot"
    DB_USER: str = Field(default="", description="Database user - must be set via DB_USER env var")
    DB_PASSWORD: str = Field(default="", description="Database password - must be set via DB_PASSWORD env var")

    @property
    def POSTGRES_URL(self) -> str:
        """PostgreSQL 连接 URL（優先使用 DATABASE_URL）"""
        if self.DATABASE_URL:
            # Railway 格式: postgresql://user:pass@host:port/db
            return self.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://")
        return f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        """异步 PostgreSQL 连接 URL"""
        if self.DATABASE_URL:
            return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # ==================== 缓存配置 ====================
    # 内存缓存配置（轻量级模式）
    # 如果有 REDIS_URL 就用 Redis，沒有就 fallback 到 MemoryCache
    USE_MEMORY_CACHE: bool = False  # 會根據 REDIS_URL 自動判斷
    CACHE_MAX_SIZE: int = 10000  # 内存缓存最大条目数
    CACHE_DEFAULT_TTL: int = 3600  # 默认过期时间（秒）

    # Redis 配置（重量级模式，可选）
    # Railway 會自動設定 REDIS_URL，優先使用
    REDIS_URL: Optional[str] = None  # Railway 自動注入
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = Field(default="", description="Redis password - must be set via REDIS_PASSWORD env var")
    REDIS_DB: int = 0

    @property
    def REDIS_CONNECTION_URL(self) -> str:
        """Redis 连接 URL（優先使用 REDIS_URL）"""
        if self.REDIS_URL:
            return self.REDIS_URL
        return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def should_use_sqlite(self) -> bool:
        """自動判斷是否應該使用 SQLite（沒有 DATABASE_URL 時）"""
        return self.USE_SQLITE or not self.DATABASE_URL

    @property
    def should_use_memory_cache(self) -> bool:
        """自動判斷是否應該使用 MemoryCache（沒有 REDIS_URL 時）"""
        return self.USE_MEMORY_CACHE or not self.REDIS_URL

    # ==================== RabbitMQ 配置 ====================
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "guest"
    RABBITMQ_PASSWORD: str = "guest"

    @property
    def RABBITMQ_URL(self) -> str:
        """RabbitMQ 连接 URL"""
        return f"amqp://{self.RABBITMQ_USER}:{self.RABBITMQ_PASSWORD}@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}/"

    # ==================== Binance API 配置 ====================
    BINANCE_API_KEY: Optional[str] = None
    BINANCE_API_SECRET: Optional[str] = None
    BINANCE_TESTNET: bool = True

    @property
    def BINANCE_BASE_URL(self) -> str:
        """Binance API 基础 URL"""
        if self.BINANCE_TESTNET:
            return "https://testnet.binance.vision"
        return "https://api.binance.com"

    @property
    def BINANCE_WS_URL(self) -> str:
        """Binance WebSocket URL"""
        if self.BINANCE_TESTNET:
            return "wss://testnet.binance.vision/ws"
        return "wss://stream.binance.com:9443/ws"

    # ==================== Telegram 通知配置 ====================
    TELEGRAM_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    TELEGRAM_ENABLED: bool = False

    # ==================== AI 助手配置 (Claude API) ====================
    ANTHROPIC_API_KEY: Optional[str] = None
    AI_MODEL: str = "claude-sonnet-4-20250514"  # 使用 Sonnet 平衡成本和品質
    AI_MAX_TOKENS: int = 2000  # 單次回應最大 token
    AI_DAILY_REPORT_HOUR: int = 8  # 每日報告時間（台灣時間，UTC+8）
    AI_ENABLED: bool = False  # 需明確啟用
    AI_DAILY_TOKEN_BUDGET: int = 50000  # 每日 token 預算（約 $0.15）

    # ==================== 日志配置 ====================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or text
    LOG_PATH: str = "logs"
    LOG_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 10

    # ==================== 监控配置 ====================
    PROMETHEUS_PORT: int = 9000
    PROMETHEUS_ENABLED: bool = True
    METRICS_EXPORT_INTERVAL: int = 15  # 秒

    # ==================== FastAPI 配置 ====================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    API_RELOAD: bool = False

    # ==================== 交易配置 ====================
    TRADING_ENABLED: bool = True  # 是否啟用交易功能
    DRY_RUN: bool = True  # 安全預設：True=只記錄不下單；要真實下單須顯式設定 DRY_RUN=false
    AUTO_START_TRADING: bool = True  # 啟動時自動開始交易循環
    TRADING_PAIRS: List[str] = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOGEUSDT"]
    DEFAULT_TRADING_PAIR: str = "BTCUSDT"

    # ==================== 訓練配置 ====================
    # TRAINING_ENABLED=false 時禁用自動 ML 訓練（Railway 線上環境）
    # TRAINING_ENABLED=true 時啟用訓練（本地訓練環境）
    TRAINING_ENABLED: bool = False  # Railway 預設關閉訓練
    MODEL_DIR: str = "models"  # 模型存儲目錄
    AUTO_LOAD_MODELS: bool = True  # 啟動時自動載入模型

    # ==================== 风险管理配置 ====================
    # 第1层：仓位控制
    MAX_POSITION_SIZE: float = 0.1  # 单笔最大10%
    MAX_TOTAL_EXPOSURE: float = 0.5  # 总暴露最大50%
    MIN_ORDER_VALUE: float = 6.0  # 最低訂單金額 $6（Binance 要求 $5）
    MAX_CONCURRENT_POSITIONS: int = 2  # 最多同時持有 2 個倉位

    # 第2层：回撤保护
    MAX_DAILY_LOSS: float = 0.02  # 日内最大亏损2%
    MAX_WEEKLY_LOSS: float = 0.05  # 周内最大亏损5%
    MAX_MONTHLY_LOSS: float = 0.10  # 月内最大亏损10%
    MAX_DRAWDOWN: float = 0.10  # 最大回撤10%

    # 第3层：相关性分析
    MAX_CORRELATION: float = 0.8  # 最大相关系数0.8

    # 第4层：黑天鹅保护
    VOLATILITY_THRESHOLD: float = 3.0  # 3倍标准差
    VOLUME_SPIKE_THRESHOLD: float = 5.0  # 5倍平均成交量

    # 第5层：熔断机制
    MAX_TRADES_PER_MINUTE: int = 10
    MAX_TRADES_PER_HOUR: int = 100
    CIRCUIT_BREAKER_COOLDOWN: int = 300  # 5分钟冷却期

    # ==================== 订单执行配置 ====================
    ORDER_EXECUTION_TIMEOUT: float = 10.0  # 订单执行超时（秒）
    ORDER_MAX_RETRIES: int = 3  # 最大重试次数
    COMMISSION_RATE: float = 0.001  # 手续费率0.1%

    # ==================== 策略配置 ====================
    DEFAULT_STRATEGY: str = "adaptive"  # adaptive, twap, vwap, immediate
    BACKTEST_LOOKBACK_PERIOD: int = 50  # 回测lookback周期

    # ==================== 安全配置 ====================
    # 重要：必須透過 SECRET_KEY 環境變數設定（不提供任何不安全的固定預設值）。
    # 未設定時：production 於啟動時拒絕；非 production 會自動產生一次性隨機金鑰。
    SECRET_KEY: str = Field(
        default="",
        description="JWT signing key - MUST be set via SECRET_KEY env var (no insecure default)"
    )
    API_KEY_ROTATION_DAYS: int = 30  # API密钥轮换周期（天）

    # ==================== 审计日志配置 ====================
    AUDIT_LOG_ENABLED: bool = True
    AUDIT_LOG_RETENTION_DAYS: int = 90  # 审计日志保留90天

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# 全局设置实例
settings = Settings()


# 环境检查函数
def validate_settings():
    """验证关键配置"""
    errors = []
    warnings = []

    # 生產環境安全檢查
    if settings.ENVIRONMENT == "production":
        # 必須設定 SECRET_KEY（不得為空）
        if not settings.SECRET_KEY:
            errors.append("SECRET_KEY must be set via the SECRET_KEY environment variable in production")

        # 如果使用 PostgreSQL，必須設定密碼
        if not settings.USE_SQLITE:
            if not settings.DB_USER:
                errors.append("DB_USER must be set when using PostgreSQL")
            if not settings.DB_PASSWORD:
                errors.append("DB_PASSWORD must be set when using PostgreSQL")

        # 如果使用 Redis，必須設定密碼
        if not settings.USE_MEMORY_CACHE:
            if not settings.REDIS_PASSWORD:
                warnings.append("REDIS_PASSWORD is not set - Redis may be insecure")

    # 检查 Binance API 配置
    if settings.TRADING_ENABLED:
        if not settings.BINANCE_API_KEY:
            errors.append("BINANCE_API_KEY is required when TRADING_ENABLED=True")
        if not settings.BINANCE_API_SECRET:
            errors.append("BINANCE_API_SECRET is required when TRADING_ENABLED=True")

    # 检查 Telegram 配置
    if settings.TELEGRAM_ENABLED:
        if not settings.TELEGRAM_TOKEN:
            errors.append("TELEGRAM_TOKEN is required when TELEGRAM_ENABLED=True")
        if not settings.TELEGRAM_CHAT_ID:
            errors.append("TELEGRAM_CHAT_ID is required when TELEGRAM_ENABLED=True")

    # 检查风险管理参数
    if settings.MAX_POSITION_SIZE > 1.0 or settings.MAX_POSITION_SIZE < 0:
        errors.append("MAX_POSITION_SIZE must be between 0 and 1")

    if settings.MAX_TOTAL_EXPOSURE > 1.0 or settings.MAX_TOTAL_EXPOSURE < 0:
        errors.append("MAX_TOTAL_EXPOSURE must be between 0 and 1")

    # 輸出警告
    for warning in warnings:
        import logging
        logging.warning(f"Configuration warning: {warning}")

    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {e}" for e in errors))


# 非 production 環境若未設定 SECRET_KEY，產生一次性隨機金鑰（避免可預測的固定金鑰）。
# production 的強制驗證改於應用「啟動時」執行（見 src/api/app.py 的 startup 事件），
# 以免在「匯入設定」階段（測試、CLI 工具）即崩潰。
if not settings.SECRET_KEY and settings.ENVIRONMENT != "production":
    import secrets as _secrets
    import logging as _logging
    settings.SECRET_KEY = _secrets.token_urlsafe(48)
    _logging.warning(
        "SECRET_KEY 未設定，已為非 production 環境產生一次性隨機金鑰（重啟後既有 token 失效）"
    )
