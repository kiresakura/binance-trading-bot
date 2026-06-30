"""
特徵存儲管理模組

管理特徵的緩存、存儲和檢索

支持两种模式：
- 内存缓存模式（USE_MEMORY_CACHE=True）：轻量级，无需外部服务
- Redis 模式（USE_MEMORY_CACHE=False）：重量级，需要 Redis 服务
"""

import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from src.config import settings
from src.logger import logger

# 根据配置决定使用哪种缓存
if settings.USE_MEMORY_CACHE:
    from src.utils.memory_cache import memory_feature_store
else:
    import redis.asyncio as aioredis


class FeatureStore:
    """
    特徵存儲管理器

    功能：
    - Redis 緩存最近的特徵數據
    - 支持多交易對
    - 支持多時間框架
    - 特徵版本管理
    """

    def __init__(self):
        self.redis_client: Optional[aioredis.Redis] = None
        self.feature_prefix = "features:"
        self.kline_prefix = "klines:"
        self.signal_prefix = "signals:"
        self._initialized = False

    async def init(self):
        """初始化 Redis 連接"""
        if self._initialized:
            return

        try:
            self.redis_client = aioredis.Redis.from_url(
                settings.REDIS_CONNECTION_URL,
                encoding='utf-8',
                decode_responses=True
            )
            await self.redis_client.ping()
            self._initialized = True
            logger.info("FeatureStore initialized with Redis")
        except Exception as e:
            logger.warning(f"FeatureStore Redis not available: {e}")
            self._initialized = False

    async def close(self):
        """關閉連接"""
        if self.redis_client:
            await self.redis_client.close()
            self._initialized = False

    # ==================== K線數據 ====================

    async def store_klines(
        self,
        symbol: str,
        interval: str,
        klines: pd.DataFrame,
        max_size: int = 1000
    ):
        """
        存儲 K 線數據

        Args:
            symbol: 交易對（如 BTCUSDT）
            interval: 時間間隔（如 1m, 5m, 1h）
            klines: OHLCV DataFrame
            max_size: 最大保留數量
        """
        if not self._initialized:
            return

        key = f"{self.kline_prefix}{symbol}:{interval}"

        try:
            # 轉換為可序列化格式
            data = klines.tail(max_size).to_json(orient='split', date_format='iso')

            await self.redis_client.set(key, data)
            await self.redis_client.expire(key, 86400)  # 24 小時過期

            logger.debug(f"Stored {len(klines)} klines for {symbol}:{interval}")

        except Exception as e:
            logger.error(f"Failed to store klines: {e}")

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        獲取 K 線數據

        Args:
            symbol: 交易對
            interval: 時間間隔
            limit: 返回數量限制

        Returns:
            OHLCV DataFrame 或 None
        """
        if not self._initialized:
            return None

        key = f"{self.kline_prefix}{symbol}:{interval}"

        try:
            data = await self.redis_client.get(key)
            if data:
                df = pd.read_json(data, orient='split')
                if limit:
                    df = df.tail(limit)
                return df
        except Exception as e:
            logger.error(f"Failed to get klines: {e}")

        return None

    # ==================== 特徵數據 ====================

    async def store_features(
        self,
        symbol: str,
        interval: str,
        features: pd.DataFrame,
        version: str = "v1"
    ):
        """
        存儲計算好的特徵

        Args:
            symbol: 交易對
            interval: 時間間隔
            features: 特徵 DataFrame
            version: 特徵版本
        """
        if not self._initialized:
            return

        key = f"{self.feature_prefix}{symbol}:{interval}:{version}"

        try:
            # 只保留最近的數據
            data = features.tail(500).to_json(orient='split', date_format='iso')

            await self.redis_client.set(key, data)
            await self.redis_client.expire(key, 3600)  # 1 小時過期

            logger.debug(f"Stored features for {symbol}:{interval}")

        except Exception as e:
            logger.error(f"Failed to store features: {e}")

    async def get_features(
        self,
        symbol: str,
        interval: str,
        version: str = "v1",
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """獲取特徵數據"""
        if not self._initialized:
            return None

        key = f"{self.feature_prefix}{symbol}:{interval}:{version}"

        try:
            data = await self.redis_client.get(key)
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
        interval: str,
        version: str = "v1"
    ) -> Optional[Dict[str, float]]:
        """獲取最新一行特徵作為字典"""
        df = await self.get_features(symbol, interval, version, limit=1)
        if df is not None and len(df) > 0:
            return df.iloc[-1].to_dict()
        return None

    # ==================== 信號數據 ====================

    async def store_signal(
        self,
        symbol: str,
        signal: Dict[str, Any]
    ):
        """
        存儲交易信號

        Args:
            symbol: 交易對
            signal: 信號字典，包含：
                - timestamp: 時間戳
                - direction: 方向 (1=做多, -1=做空, 0=觀望)
                - strength: 信號強度 (0-1)
                - strategy: 策略名稱
                - features: 關鍵特徵
        """
        if not self._initialized:
            return

        key = f"{self.signal_prefix}{symbol}:latest"
        history_key = f"{self.signal_prefix}{symbol}:history"

        try:
            signal['timestamp'] = datetime.utcnow().isoformat()
            signal_json = json.dumps(signal)

            # 存儲最新信號
            await self.redis_client.set(key, signal_json)

            # 添加到歷史（保留最近 100 個）
            await self.redis_client.lpush(history_key, signal_json)
            await self.redis_client.ltrim(history_key, 0, 99)

            logger.debug(f"Stored signal for {symbol}: {signal['direction']}")

        except Exception as e:
            logger.error(f"Failed to store signal: {e}")

    async def get_latest_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """獲取最新信號"""
        if not self._initialized:
            return None

        key = f"{self.signal_prefix}{symbol}:latest"

        try:
            data = await self.redis_client.get(key)
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
        """獲取信號歷史"""
        if not self._initialized:
            return []

        history_key = f"{self.signal_prefix}{symbol}:history"

        try:
            data_list = await self.redis_client.lrange(history_key, 0, limit - 1)
            return [json.loads(d) for d in data_list]
        except Exception as e:
            logger.error(f"Failed to get signal history: {e}")

        return []

    # ==================== 模型狀態 ====================

    async def store_model_state(
        self,
        model_name: str,
        state: Dict[str, Any]
    ):
        """存儲模型狀態"""
        if not self._initialized:
            return

        key = f"model_state:{model_name}"

        try:
            state['updated_at'] = datetime.utcnow().isoformat()
            await self.redis_client.set(key, json.dumps(state))
            logger.debug(f"Stored state for model {model_name}")
        except Exception as e:
            logger.error(f"Failed to store model state: {e}")

    async def get_model_state(self, model_name: str) -> Optional[Dict[str, Any]]:
        """獲取模型狀態"""
        if not self._initialized:
            return None

        key = f"model_state:{model_name}"

        try:
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Failed to get model state: {e}")

        return None

    # ==================== 績效追蹤 ====================

    async def record_trade_result(
        self,
        symbol: str,
        strategy: str,
        result: Dict[str, Any]
    ):
        """
        記錄交易結果用於學習

        Args:
            symbol: 交易對
            strategy: 策略名稱
            result: 交易結果，包含：
                - entry_price, exit_price
                - pnl, pnl_percent
                - duration_minutes
                - signal_strength
        """
        if not self._initialized:
            return

        key = f"trade_results:{symbol}:{strategy}"

        try:
            result['timestamp'] = datetime.utcnow().isoformat()
            await self.redis_client.lpush(key, json.dumps(result))
            await self.redis_client.ltrim(key, 0, 999)  # 保留最近 1000 筆

            logger.info(f"Recorded trade result for {symbol}:{strategy}, PnL: {result.get('pnl_percent', 0):.2f}%")

        except Exception as e:
            logger.error(f"Failed to record trade result: {e}")

    async def get_strategy_performance(
        self,
        symbol: str,
        strategy: str,
        days: int = 7
    ) -> Dict[str, float]:
        """
        獲取策略績效統計

        Returns:
            績效指標字典
        """
        if not self._initialized:
            return {}

        key = f"trade_results:{symbol}:{strategy}"

        try:
            data_list = await self.redis_client.lrange(key, 0, -1)

            if not data_list:
                return {}

            trades = [json.loads(d) for d in data_list]

            # 過濾時間範圍
            cutoff = datetime.utcnow() - timedelta(days=days)
            trades = [
                t for t in trades
                if datetime.fromisoformat(t['timestamp']) > cutoff
            ]

            if not trades:
                return {}

            # 計算統計
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


# 全局實例
# 根据配置选择使用内存缓存或 Redis
if settings.USE_MEMORY_CACHE:
    feature_store = memory_feature_store
    logger.info("Using memory cache for feature store")
else:
    feature_store = FeatureStore()
    logger.info("Using Redis for feature store")
