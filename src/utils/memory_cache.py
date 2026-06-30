"""内存缓存系统 - 轻量级替代 Redis"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import OrderedDict
from threading import Lock
import numpy as np
import pandas as pd

from src.logger import logger


class MemoryCache:
    """
    线程安全的内存缓存

    功能：
    - 支持 TTL（过期时间）
    - 支持 LRU 淘汰策略
    - 支持 List 操作（lpush, lrange, ltrim）
    - 兼容 Redis API 风格
    """

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        """
        初始化缓存

        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认过期时间（秒）
        """
        self._cache: OrderedDict = OrderedDict()
        self._ttls: Dict[str, datetime] = {}
        self._lists: Dict[str, List[Any]] = {}
        self._lock = Lock()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._initialized = True

        logger.info(f"MemoryCache initialized (max_size={max_size}, default_ttl={default_ttl}s)")

    def _is_expired(self, key: str) -> bool:
        """检查键是否过期"""
        if key in self._ttls:
            return datetime.utcnow() > self._ttls[key]
        return False

    def _evict_expired(self):
        """清理过期的键"""
        now = datetime.utcnow()
        expired_keys = [k for k, v in self._ttls.items() if now > v]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._ttls.pop(key, None)
            self._lists.pop(key, None)

    def _evict_lru(self):
        """LRU 淘汰"""
        while len(self._cache) > self.max_size:
            oldest = next(iter(self._cache))
            self._cache.pop(oldest)
            self._ttls.pop(oldest, None)

    # ==================== 基本操作 ====================

    def _set_sync(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """
        同步设置值（内部方法）

        Args:
            key: 键
            value: 值
            ex: 过期时间（秒）
        """
        with self._lock:
            self._evict_expired()

            # 移动到末尾（最近访问）
            if key in self._cache:
                self._cache.move_to_end(key)

            self._cache[key] = value

            # 设置 TTL
            ttl_seconds = ex or self.default_ttl
            self._ttls[key] = datetime.utcnow() + timedelta(seconds=ttl_seconds)

            self._evict_lru()
            return True

    def set(self, key: str, value: Any, ex: Optional[int] = None, ttl: Optional[int] = None) -> bool:
        """
        设置值（同步接口，兼容旧代码）

        Args:
            key: 键
            value: 值
            ex: 过期时间（秒）
            ttl: 过期时间（秒），与 ex 同义
        """
        return self._set_sync(key, value, ex or ttl)

    async def set_async(self, key: str, value: Any, ex: Optional[int] = None, ttl: Optional[int] = None) -> bool:
        """
        异步设置值

        Args:
            key: 键
            value: 值
            ex: 过期时间（秒）
            ttl: 过期时间（秒），与 ex 同义
        """
        return self._set_sync(key, value, ex or ttl)

    def get(self, key: str) -> Optional[Any]:
        """获取值"""
        with self._lock:
            if key not in self._cache:
                return None

            if self._is_expired(key):
                self._cache.pop(key, None)
                self._ttls.pop(key, None)
                return None

            # 移动到末尾（最近访问）
            self._cache.move_to_end(key)
            return self._cache[key]

    def delete(self, key: str) -> bool:
        """删除键"""
        with self._lock:
            result = key in self._cache or key in self._lists
            self._cache.pop(key, None)
            self._ttls.pop(key, None)
            self._lists.pop(key, None)
            return result

    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            if key in self._cache:
                if self._is_expired(key):
                    self._cache.pop(key, None)
                    self._ttls.pop(key, None)
                    return False
                return True
            return key in self._lists

    def expire(self, key: str, seconds: int) -> bool:
        """设置过期时间"""
        with self._lock:
            if key in self._cache or key in self._lists:
                self._ttls[key] = datetime.utcnow() + timedelta(seconds=seconds)
                return True
            return False

    # ==================== List 操作 ====================

    def lpush(self, key: str, value: Any) -> int:
        """左侧插入"""
        with self._lock:
            if key not in self._lists:
                self._lists[key] = []
            self._lists[key].insert(0, value)
            return len(self._lists[key])

    def rpush(self, key: str, value: Any) -> int:
        """右侧插入"""
        with self._lock:
            if key not in self._lists:
                self._lists[key] = []
            self._lists[key].append(value)
            return len(self._lists[key])

    def lrange(self, key: str, start: int, end: int) -> List[Any]:
        """获取列表范围"""
        with self._lock:
            if key not in self._lists:
                return []
            if end == -1:
                return self._lists[key][start:]
            return self._lists[key][start:end + 1]

    def ltrim(self, key: str, start: int, end: int) -> bool:
        """裁剪列表"""
        with self._lock:
            if key not in self._lists:
                return True
            if end == -1:
                self._lists[key] = self._lists[key][start:]
            else:
                self._lists[key] = self._lists[key][start:end + 1]
            return True

    def llen(self, key: str) -> int:
        """获取列表长度"""
        with self._lock:
            return len(self._lists.get(key, []))

    # ==================== 工具方法 ====================

    def clear(self):
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
            self._ttls.clear()
            self._lists.clear()

    def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配的键（简单的 * 通配符支持）"""
        with self._lock:
            self._evict_expired()
            all_keys = list(self._cache.keys()) + list(self._lists.keys())
            if pattern == "*":
                return all_keys
            # 简单的前缀匹配
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                return [k for k in all_keys if k.startswith(prefix)]
            return [k for k in all_keys if k == pattern]

    def info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        with self._lock:
            return {
                "cache_size": len(self._cache),
                "list_count": len(self._lists),
                "max_size": self.max_size,
                "default_ttl": self.default_ttl
            }

    async def init(self):
        """初始化（兼容异步接口）"""
        self._initialized = True
        logger.debug("MemoryCache async init called")

    async def ping(self) -> bool:
        """健康检查"""
        return self._initialized

    async def close(self):
        """关闭（兼容接口）"""
        pass


class AsyncMemoryCache:
    """
    异步内存缓存

    包装 MemoryCache 提供异步接口，兼容 aioredis API
    """

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self._cache = MemoryCache(max_size=max_size, default_ttl=default_ttl)
        self._initialized = True

    async def init(self):
        """初始化（兼容接口）"""
        self._initialized = True
        logger.debug("AsyncMemoryCache init called")

    async def ping(self) -> bool:
        """健康检查"""
        return True

    async def set(self, key: str, value: Any, ex: Optional[int] = None, ttl: Optional[int] = None) -> bool:
        """异步设置值"""
        return self._cache.set(key, value, ex=ex, ttl=ttl)

    async def get(self, key: str) -> Optional[Any]:
        """异步获取值"""
        return self._cache.get(key)

    async def delete(self, key: str) -> bool:
        """异步删除"""
        return self._cache.delete(key)

    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        return self._cache.exists(key)

    async def expire(self, key: str, seconds: int) -> bool:
        """设置过期时间"""
        return self._cache.expire(key, seconds)

    async def lpush(self, key: str, value: Any) -> int:
        """左侧插入"""
        return self._cache.lpush(key, value)

    async def rpush(self, key: str, value: Any) -> int:
        """右侧插入"""
        return self._cache.rpush(key, value)

    async def lrange(self, key: str, start: int, end: int) -> List[Any]:
        """获取列表范围"""
        return self._cache.lrange(key, start, end)

    async def ltrim(self, key: str, start: int, end: int) -> bool:
        """裁剪列表"""
        return self._cache.ltrim(key, start, end)

    async def llen(self, key: str) -> int:
        """获取列表长度"""
        return self._cache.llen(key)

    async def close(self):
        """关闭连接"""
        pass

    def info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return self._cache.info()


# 全局内存缓存实例
memory_cache = MemoryCache()
async_memory_cache = AsyncMemoryCache()


class MemoryFeatureStore:
    """
    特征存储管理器 - 使用内存缓存替代 Redis

    功能：
    - 缓存最近的特征数据
    - 支持多交易对
    - 支持多时间框架
    - 特征版本管理
    """

    def __init__(self):
        self.cache = AsyncMemoryCache(max_size=50000, default_ttl=86400)
        self.feature_prefix = "features:"
        self.kline_prefix = "klines:"
        self.signal_prefix = "signals:"
        self._initialized = True

    async def init(self):
        """初始化（兼容接口）"""
        self._initialized = True
        logger.info("MemoryFeatureStore initialized")

    async def close(self):
        """关闭连接"""
        await self.cache.close()
        self._initialized = False

    # ==================== K线数据 ====================

    async def store_klines(
        self,
        symbol: str,
        interval: str = "1h",
        klines: pd.DataFrame = None,
        max_size: int = 1000
    ):
        """存储 K 线数据"""
        if not self._initialized or klines is None:
            return

        key = f"{self.kline_prefix}{symbol}:{interval}"

        try:
            # 转换为可序列化格式
            data = klines.tail(max_size).to_json(orient='split', date_format='iso')
            await self.cache.set(key, data, ex=86400)  # 24 小时过期
            logger.debug(f"Stored {len(klines)} klines for {symbol}:{interval}")
        except Exception as e:
            logger.error(f"Failed to store klines: {e}")

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """获取 K 线数据"""
        if not self._initialized:
            return None

        key = f"{self.kline_prefix}{symbol}:{interval}"

        try:
            data = await self.cache.get(key)
            if data:
                df = pd.read_json(data, orient='split')
                if limit:
                    df = df.tail(limit)
                return df
        except Exception as e:
            logger.error(f"Failed to get klines: {e}")

        return None

    # ==================== 特征数据 ====================

    async def store_features(
        self,
        symbol: str,
        interval: str = "1h",
        features: pd.DataFrame = None,
        version: str = "v1"
    ):
        """存储计算好的特征"""
        if not self._initialized or features is None:
            return

        key = f"{self.feature_prefix}{symbol}:{interval}:{version}"

        try:
            data = features.tail(500).to_json(orient='split', date_format='iso')
            await self.cache.set(key, data, ex=3600)  # 1 小时过期
            logger.debug(f"Stored features for {symbol}:{interval}")
        except Exception as e:
            logger.error(f"Failed to store features: {e}")

    async def get_features(
        self,
        symbol: str,
        interval: str = "1h",
        version: str = "v1",
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """获取特征数据"""
        if not self._initialized:
            return None

        key = f"{self.feature_prefix}{symbol}:{interval}:{version}"

        try:
            data = await self.cache.get(key)
            if data:
                df = pd.read_json(data, orient='split')
                if limit:
                    df = df.tail(limit)
                return df
        except Exception as e:
            logger.error(f"Failed to get features: {e}")

        return None

    async def get_latest_features(
        self,
        symbol: str,
        interval: str = "1h",
        version: str = "v1"
    ) -> Optional[Dict[str, float]]:
        """获取最新一行特征作为字典"""
        df = await self.get_features(symbol, interval, version, limit=1)
        if df is not None and len(df) > 0:
            return df.iloc[-1].to_dict()
        return None

    # ==================== 信号数据 ====================

    async def store_signal(
        self,
        symbol: str,
        signal: Dict[str, Any]
    ):
        """存储交易信号"""
        if not self._initialized:
            return

        key = f"{self.signal_prefix}{symbol}:latest"
        history_key = f"{self.signal_prefix}{symbol}:history"

        try:
            signal['timestamp'] = datetime.utcnow().isoformat()
            signal_json = json.dumps(signal)

            # 存储最新信号
            await self.cache.set(key, signal_json)

            # 添加到历史（保留最近 100 个）
            await self.cache.lpush(history_key, signal_json)
            await self.cache.ltrim(history_key, 0, 99)

            logger.debug(f"Stored signal for {symbol}: {signal.get('direction')}")
        except Exception as e:
            logger.error(f"Failed to store signal: {e}")

    async def get_latest_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取最新信号"""
        if not self._initialized:
            return None

        key = f"{self.signal_prefix}{symbol}:latest"

        try:
            data = await self.cache.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Failed to get signal: {e}")

        return None

    async def get_signal_history(
        self,
        symbol: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """获取信号历史"""
        if not self._initialized:
            return []

        history_key = f"{self.signal_prefix}{symbol}:history"

        try:
            data_list = await self.cache.lrange(history_key, 0, limit - 1)
            return [json.loads(d) for d in data_list]
        except Exception as e:
            logger.error(f"Failed to get signal history: {e}")

        return []

    # ==================== 模型状态 ====================

    async def store_model_state(
        self,
        model_name: str,
        state: Dict[str, Any]
    ):
        """存储模型状态"""
        if not self._initialized:
            return

        key = f"model_state:{model_name}"

        try:
            state['updated_at'] = datetime.utcnow().isoformat()
            await self.cache.set(key, json.dumps(state))
            logger.debug(f"Stored state for model {model_name}")
        except Exception as e:
            logger.error(f"Failed to store model state: {e}")

    async def get_model_state(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型状态"""
        if not self._initialized:
            return None

        key = f"model_state:{model_name}"

        try:
            data = await self.cache.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Failed to get model state: {e}")

        return None

    # ==================== 绩效追踪 ====================

    async def record_trade_result(
        self,
        symbol: str,
        strategy: str,
        result: Dict[str, Any]
    ):
        """记录交易结果用于学习"""
        if not self._initialized:
            return

        key = f"trade_results:{symbol}:{strategy}"

        try:
            result['timestamp'] = datetime.utcnow().isoformat()
            await self.cache.lpush(key, json.dumps(result))
            await self.cache.ltrim(key, 0, 999)  # 保留最近 1000 笔

            logger.info(f"Recorded trade result for {symbol}:{strategy}, PnL: {result.get('pnl_percent', 0):.2f}%")
        except Exception as e:
            logger.error(f"Failed to record trade result: {e}")

    async def get_strategy_performance(
        self,
        symbol: str,
        strategy: str,
        days: int = 7
    ) -> Dict[str, float]:
        """获取策略绩效统计"""
        if not self._initialized:
            return {}

        key = f"trade_results:{symbol}:{strategy}"

        try:
            data_list = await self.cache.lrange(key, 0, -1)

            if not data_list:
                return {}

            trades = [json.loads(d) for d in data_list]

            # 过滤时间范围
            cutoff = datetime.utcnow() - timedelta(days=days)
            trades = [
                t for t in trades
                if datetime.fromisoformat(t['timestamp']) > cutoff
            ]

            if not trades:
                return {}

            # 计算统计
            pnls = [t.get('pnl_percent', 0) for t in trades]

            return {
                'total_trades': len(trades),
                'win_rate': sum(1 for p in pnls if p > 0) / len(pnls),
                'avg_pnl': np.mean(pnls),
                'total_pnl': sum(pnls),
                'max_drawdown': min(pnls),
                'sharpe_ratio': np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(len(pnls)),
                'profit_factor': sum(p for p in pnls if p > 0) / (abs(sum(p for p in pnls if p < 0)) + 1e-10)
            }

        except Exception as e:
            logger.error(f"Failed to get strategy performance: {e}")
            return {}


# 全局特征存储实例
memory_feature_store = MemoryFeatureStore()
