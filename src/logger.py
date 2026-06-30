"""日志系统配置"""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional


class JsonFormatter(logging.Formatter):
    """JSON 格式日志"""

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # 添加额外字段
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        return json.dumps(log_data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """彩色控制台格式化器"""

    # ANSI 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
    }
    RESET = '\033[0m'

    def format(self, record):
        """格式化日志记录"""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "trading_bot",
    log_level: str = "INFO",
    log_format: str = "json",
    log_path: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 10
) -> logging.Logger:
    """设置日志记录器

    Args:
        name: 日志记录器名称
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: 日志格式 (json, text)
        log_path: 日志文件路径
        max_bytes: 单个日志文件最大大小
        backup_count: 日志文件备份数量

    Returns:
        配置好的 Logger 实例
    """
    # 创建日志目录
    log_dir = Path(log_path)
    log_dir.mkdir(exist_ok=True, parents=True)

    # 获取或创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # 清除现有的处理器
    logger.handlers.clear()

    # 选择格式化器
    if log_format == "json":
        formatter = JsonFormatter()
        console_formatter = ColoredFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_formatter = ColoredFormatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )

    # === 控制台处理器 ===
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)  # 控制台只显示 INFO 及以上
    logger.addHandler(console_handler)

    # === 主日志文件处理器（所有级别）===
    main_log_file = log_dir / f"{name}.log"
    file_handler = RotatingFileHandler(
        main_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
    logger.addHandler(file_handler)

    # === 错误日志文件处理器（仅 ERROR 和 CRITICAL）===
    error_log_file = log_dir / f"{name}.error.log"
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=max_bytes,
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)

    # === 调试日志文件处理器（仅在 DEBUG 模式）===
    if log_level.upper() == "DEBUG":
        debug_log_file = log_dir / f"{name}.debug.log"
        debug_handler = RotatingFileHandler(
            debug_log_file,
            maxBytes=max_bytes,
            backupCount=3,
            encoding='utf-8'
        )
        debug_handler.setFormatter(formatter)
        debug_handler.setLevel(logging.DEBUG)
        logger.addHandler(debug_handler)

    # 防止日志传播到根 logger
    logger.propagate = False

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取 logger 实例

    Args:
        name: logger 名称，如果为 None 则返回默认 logger

    Returns:
        Logger 实例
    """
    if name is None:
        return logging.getLogger("trading_bot")
    return logging.getLogger(f"trading_bot.{name}")


# 初始化全局 logger
try:
    from src.config import settings

    logger = setup_logger(
        name="trading_bot",
        log_level=settings.LOG_LEVEL,
        log_format=settings.LOG_FORMAT,
        log_path=settings.LOG_PATH,
        max_bytes=settings.LOG_MAX_SIZE,
        backup_count=settings.LOG_BACKUP_COUNT
    )
except ImportError:
    # 如果配置文件不可用，使用默认配置
    logger = setup_logger()


# 便捷函数
def log_debug(message: str, **kwargs):
    """记录 DEBUG 日志"""
    logger.debug(message, extra={'extra_data': kwargs} if kwargs else None)


def log_info(message: str, **kwargs):
    """记录 INFO 日志"""
    logger.info(message, extra={'extra_data': kwargs} if kwargs else None)


def log_warning(message: str, **kwargs):
    """记录 WARNING 日志"""
    logger.warning(message, extra={'extra_data': kwargs} if kwargs else None)


def log_error(message: str, **kwargs):
    """记录 ERROR 日志"""
    logger.error(message, extra={'extra_data': kwargs} if kwargs else None)


def log_critical(message: str, **kwargs):
    """记录 CRITICAL 日志"""
    logger.critical(message, extra={'extra_data': kwargs} if kwargs else None)


# 日志装饰器
def log_execution(func):
    """日志装饰器：记录函数执行"""
    import functools
    import time

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = f"{func.__module__}.{func.__name__}"

        logger.debug(f"Executing {func_name}")

        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"{func_name} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"{func_name} failed after {elapsed:.3f}s: {str(e)}",
                exc_info=True
            )
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = f"{func.__module__}.{func.__name__}"

        logger.debug(f"Executing {func_name}")

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"{func_name} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"{func_name} failed after {elapsed:.3f}s: {str(e)}",
                exc_info=True
            )
            raise

    # 判断是否为异步函数
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
