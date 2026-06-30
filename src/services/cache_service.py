"""
Redis 緩存服務
負責統一管理 Redis 連線與緩存操作
使用 redis-py 5.x (內建 async 支援，不需要 aioredis)
"""

import json
import asyncio
from typing import Optional, Any, Union
from datetime import timedelta

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

from src.config import settings
from src.logger import logger

class CacheService:
    def __init__(self):
        self.redis = None
        self._initialized = False

    async def init(self):
        """初始化 Redis 連線"""
        if self._initialized:
            return

        if aioredis is None:
            logger.warning("⚠️ redis library not available, cache service disabled")
            return

        try:
            redis_url = getattr(settings, 'REDIS_URL', None)
            if redis_url:
                self.redis = await aioredis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                # 測試連線
                await self.redis.ping()
                self._initialized = True
                logger.info("✅ Redis cache service initialized")
            else:
                logger.warning("⚠️ REDIS_URL not set, cache service disabled")
        except Exception as e:
            logger.error(f"❌ Redis initialization failed: {e}")
            self.redis = None

    async def close(self):
        """關閉 Redis 連線"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")

    async def get(self, key: str) -> Optional[Any]:
        """獲取緩存值 (自動 JSON decode)"""
        if not self.redis:
            return None
        
        try:
            value = await self.redis.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.error(f"Redis get error for {key}: {e}")
            return None

    async def set(
        self, 
        key: str, 
        value: Any, 
        expire: Union[int, timedelta] = 300
    ) -> bool:
        """設置緩存值 (自動 JSON encode)"""
        if not self.redis:
            return False

        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            await self.redis.set(key, value, ex=expire)
            return True
        except Exception as e:
            logger.error(f"Redis set error for {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """刪除緩存"""
        if not self.redis:
            return False
            
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error for {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """檢查 Key 是否存在"""
        if not self.redis:
            return False
            
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis exists error for {key}: {e}")
            return False

# 全局實例
cache_service = CacheService()
