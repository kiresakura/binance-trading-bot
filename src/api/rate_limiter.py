"""
Rate Limiter - API 限流中間件

實現多層級限流：
1. IP 層級限流 - 防止單一 IP 濫用
2. 用戶層級限流 - 限制認證用戶的請求頻率
3. 端點層級限流 - 敏感端點有更嚴格的限制
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from functools import wraps

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.logger import logger
from src.config import settings


@dataclass
class RateLimitConfig:
    """限流配置"""
    # 預設限制（每分鐘請求數）
    default_rpm: int = 120  # 每分鐘 120 次
    default_burst: int = 20  # 突發容量

    # IP 層級限制
    ip_rpm: int = 300  # 每 IP 每分鐘 300 次
    ip_burst: int = 50

    # 認證用戶限制（更寬鬆）
    user_rpm: int = 600  # 認證用戶每分鐘 600 次
    user_burst: int = 100

    # 敏感端點限制
    sensitive_rpm: int = 10  # 敏感操作每分鐘 10 次
    sensitive_burst: int = 3

    # 訂單端點限制
    order_rpm: int = 30  # 下單每分鐘 30 次
    order_burst: int = 5

    # 窗口大小（秒）
    window_size: int = 60


class TokenBucket:
    """令牌桶算法實現"""

    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: 每秒添加的令牌數
            capacity: 桶的最大容量
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> Tuple[bool, float]:
        """
        嘗試消費令牌

        Returns:
            (是否成功, 需要等待的時間)
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update

            # 添加新令牌
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, 0.0
            else:
                # 計算需要等待的時間
                wait_time = (tokens - self.tokens) / self.rate
                return False, wait_time


class SlidingWindowCounter:
    """滑動窗口計數器"""

    def __init__(self, window_size: int = 60, max_requests: int = 60):
        """
        Args:
            window_size: 窗口大小（秒）
            max_requests: 窗口內最大請求數
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests: Dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> Tuple[bool, int, int]:
        """
        檢查請求是否允許

        Returns:
            (是否允許, 剩餘請求數, 重置時間)
        """
        async with self._lock:
            now = time.time()
            window_start = now - self.window_size

            # 清理過期請求
            self.requests[key] = [
                t for t in self.requests[key] if t > window_start
            ]

            current_count = len(self.requests[key])
            remaining = max(0, self.max_requests - current_count)
            reset_time = int(self.window_size - (now - window_start))

            if current_count < self.max_requests:
                self.requests[key].append(now)
                return True, remaining - 1, reset_time
            else:
                return False, 0, reset_time

    async def cleanup(self):
        """清理過期數據"""
        async with self._lock:
            now = time.time()
            window_start = now - self.window_size

            keys_to_delete = []
            for key, timestamps in self.requests.items():
                self.requests[key] = [t for t in timestamps if t > window_start]
                if not self.requests[key]:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del self.requests[key]


class RateLimiter:
    """多層級限流器"""

    # 敏感端點列表
    SENSITIVE_ENDPOINTS = {
        "/api/v1/system/emergency-stop",
        "/api/v1/system/resume",
        "/api/v1/learning/train",
        "/api/v1/arena/promote",
    }

    # 訂單相關端點
    ORDER_ENDPOINTS = {
        "/api/v1/orders",
    }

    # 不限流的端點
    EXCLUDED_ENDPOINTS = {
        "/health",
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/metrics",
        "/ws",  # WebSocket 連線
    }

    # 高頻端點（較寬鬆的限制）
    HIGH_FREQUENCY_ENDPOINTS = {
        "/api/v1/arena/status",
        "/api/v1/market/regime",
        "/api/v1/trading/status",
        "/api/v1/data/prices",
        "/api/v1/strategies/sensitivity",
    }

    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()

        # IP 層級限流器
        self.ip_limiters: Dict[str, TokenBucket] = {}

        # 用戶層級限流器
        self.user_limiters: Dict[str, TokenBucket] = {}

        # 端點層級滑動窗口
        self.endpoint_windows: Dict[str, SlidingWindowCounter] = {}

        # 全局鎖
        self._lock = asyncio.Lock()

        # 清理任務
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """啟動清理任務"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Rate limiter cleanup task started")

    async def stop(self):
        """停止清理任務"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_loop(self):
        """定期清理過期數據"""
        while True:
            try:
                await asyncio.sleep(300)  # 每 5 分鐘清理
                await self._cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rate limiter cleanup error: {e}")

    async def _cleanup(self):
        """清理過期的限流器"""
        async with self._lock:
            now = time.time()

            # 清理 5 分鐘未使用的 IP 限流器
            stale_ips = [
                ip for ip, bucket in self.ip_limiters.items()
                if now - bucket.last_update > 300
            ]
            for ip in stale_ips:
                del self.ip_limiters[ip]

            # 清理用戶限流器
            stale_users = [
                user for user, bucket in self.user_limiters.items()
                if now - bucket.last_update > 300
            ]
            for user in stale_users:
                del self.user_limiters[user]

            # 清理滑動窗口
            for window in self.endpoint_windows.values():
                await window.cleanup()

            if stale_ips or stale_users:
                logger.debug(f"Cleaned up {len(stale_ips)} IP limiters, {len(stale_users)} user limiters")

    def _get_client_ip(self, request: Request) -> str:
        """獲取客戶端 IP"""
        # 檢查代理頭
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _get_user_id(self, request: Request) -> Optional[str]:
        """從請求中獲取用戶 ID"""
        # 從請求狀態獲取（由 auth middleware 設置）
        if hasattr(request.state, 'user') and request.state.user:
            return request.state.user.sub
        return None

    async def _get_ip_bucket(self, ip: str) -> TokenBucket:
        """獲取或創建 IP 限流桶"""
        async with self._lock:
            if ip not in self.ip_limiters:
                rate = self.config.ip_rpm / 60  # 轉換為每秒
                self.ip_limiters[ip] = TokenBucket(rate, self.config.ip_burst)
            return self.ip_limiters[ip]

    async def _get_user_bucket(self, user_id: str) -> TokenBucket:
        """獲取或創建用戶限流桶"""
        async with self._lock:
            if user_id not in self.user_limiters:
                rate = self.config.user_rpm / 60
                self.user_limiters[user_id] = TokenBucket(rate, self.config.user_burst)
            return self.user_limiters[user_id]

    async def _get_endpoint_window(self, endpoint: str, is_sensitive: bool, is_order: bool) -> SlidingWindowCounter:
        """獲取或創建端點滑動窗口"""
        async with self._lock:
            if endpoint not in self.endpoint_windows:
                if is_sensitive:
                    max_requests = self.config.sensitive_rpm
                elif is_order:
                    max_requests = self.config.order_rpm
                else:
                    max_requests = self.config.default_rpm

                self.endpoint_windows[endpoint] = SlidingWindowCounter(
                    self.config.window_size,
                    max_requests
                )
            return self.endpoint_windows[endpoint]

    async def check_rate_limit(self, request: Request) -> Tuple[bool, Dict]:
        """
        檢查請求是否超過限制

        Returns:
            (是否允許, 響應頭信息)
        """
        path = request.url.path

        # 排除不需要限流的端點
        if path in self.EXCLUDED_ENDPOINTS or path.startswith("/ws"):
            return True, {}

        client_ip = self._get_client_ip(request)
        user_id = self._get_user_id(request)
        is_sensitive = path in self.SENSITIVE_ENDPOINTS
        is_order = any(path.startswith(ep) for ep in self.ORDER_ENDPOINTS) and request.method == "POST"
        is_high_frequency = any(path.startswith(ep) for ep in self.HIGH_FREQUENCY_ENDPOINTS)

        headers = {}

        # 1. IP 層級檢查
        ip_bucket = await self._get_ip_bucket(client_ip)
        ip_allowed, ip_wait = await ip_bucket.consume()

        if not ip_allowed:
            logger.warning(f"IP rate limit exceeded: {client_ip}")
            headers["X-RateLimit-Type"] = "ip"
            headers["Retry-After"] = str(int(ip_wait) + 1)
            return False, headers

        # 高頻端點只做 IP 層級檢查，跳過更嚴格的限制
        if is_high_frequency:
            return True, headers

        # 2. 用戶層級檢查（如果已認證）
        if user_id:
            user_bucket = await self._get_user_bucket(user_id)
            user_allowed, user_wait = await user_bucket.consume()

            if not user_allowed:
                logger.warning(f"User rate limit exceeded: {user_id}")
                headers["X-RateLimit-Type"] = "user"
                headers["Retry-After"] = str(int(user_wait) + 1)
                return False, headers

        # 3. 端點層級檢查（敏感端點和訂單端點）
        if is_sensitive or is_order:
            key = f"{client_ip}:{path}" if not user_id else f"{user_id}:{path}"
            window = await self._get_endpoint_window(path, is_sensitive, is_order)
            allowed, remaining, reset = await window.is_allowed(key)

            headers["X-RateLimit-Limit"] = str(window.max_requests)
            headers["X-RateLimit-Remaining"] = str(remaining)
            headers["X-RateLimit-Reset"] = str(reset)

            if not allowed:
                endpoint_type = "sensitive" if is_sensitive else "order"
                logger.warning(f"{endpoint_type.title()} endpoint rate limit exceeded: {key}")
                headers["X-RateLimit-Type"] = endpoint_type
                headers["Retry-After"] = str(reset)
                return False, headers

        return True, headers


class RateLimitMiddleware(BaseHTTPMiddleware):
    """限流中間件"""

    def __init__(self, app, rate_limiter: RateLimiter = None):
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter()

    async def dispatch(self, request: Request, call_next):
        """處理請求"""
        allowed, headers = await self.rate_limiter.check_rate_limit(request)

        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "type": headers.get("X-RateLimit-Type", "unknown"),
                    "retry_after": int(headers.get("Retry-After", 60))
                },
                headers=headers
            )

        response = await call_next(request)

        # 添加限流響應頭
        for key, value in headers.items():
            response.headers[key] = value

        return response


# 裝飾器版本的限流（用於特定端點）
def rate_limit(rpm: int = 60, burst: int = 10):
    """
    端點級別的限流裝飾器

    Usage:
        @app.get("/api/expensive")
        @rate_limit(rpm=10, burst=2)
        async def expensive_endpoint():
            ...
    """
    limiter = TokenBucket(rpm / 60, burst)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            allowed, wait_time = await limiter.consume()
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Rate limit exceeded",
                        "retry_after": int(wait_time) + 1
                    },
                    headers={"Retry-After": str(int(wait_time) + 1)}
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# 全局實例
rate_limiter = RateLimiter()
