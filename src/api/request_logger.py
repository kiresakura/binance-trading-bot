"""
Request Logger - API 請求日誌中間件

功能：
1. 記錄所有 API 請求和響應
2. 計算請求處理時間
3. 記錄錯誤詳情
4. 支持結構化日誌輸出
5. 敏感數據脫敏處理
"""

import time
import uuid
import json
from typing import Optional, Set, Dict, Any
from datetime import datetime

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.logger import logger


class RequestLogger:
    """請求日誌記錄器"""

    # 需要脫敏的請求頭
    SENSITIVE_HEADERS: Set[str] = {
        "authorization",
        "x-api-key",
        "cookie",
        "x-auth-token",
    }

    # 需要脫敏的請求體字段
    SENSITIVE_FIELDS: Set[str] = {
        "password",
        "secret",
        "token",
        "api_key",
        "api_secret",
        "credit_card",
        "card_number",
    }

    # 不記錄請求體的端點（可能包含大量數據）
    SKIP_BODY_ENDPOINTS: Set[str] = {
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    # 不記錄的端點
    SKIP_LOGGING_ENDPOINTS: Set[str] = {
        "/health",
        "/",
        "/metrics",
    }

    def __init__(
        self,
        log_request_body: bool = True,
        log_response_body: bool = False,
        max_body_length: int = 10000,
    ):
        """
        Args:
            log_request_body: 是否記錄請求體
            log_response_body: 是否記錄響應體
            max_body_length: 請求體最大記錄長度
        """
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_length = max_body_length

    def _mask_sensitive_value(self, value: str) -> str:
        """脫敏處理敏感值"""
        if not value:
            return value
        if len(value) <= 8:
            return "***"
        return f"{value[:4]}...{value[-4:]}"

    def _mask_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """脫敏處理請求頭"""
        masked = {}
        for key, value in headers.items():
            if key.lower() in self.SENSITIVE_HEADERS:
                masked[key] = self._mask_sensitive_value(value)
            else:
                masked[key] = value
        return masked

    def _mask_body(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """脫敏處理請求體"""
        if not isinstance(body, dict):
            return body

        masked = {}
        for key, value in body.items():
            if key.lower() in self.SENSITIVE_FIELDS:
                if isinstance(value, str):
                    masked[key] = self._mask_sensitive_value(value)
                else:
                    masked[key] = "***"
            elif isinstance(value, dict):
                masked[key] = self._mask_body(value)
            elif isinstance(value, list):
                masked[key] = [
                    self._mask_body(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                masked[key] = value
        return masked

    def _get_client_ip(self, request: Request) -> str:
        """獲取客戶端 IP"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _truncate(self, text: str, max_length: int = None) -> str:
        """截斷過長的文本"""
        max_len = max_length or self.max_body_length
        if len(text) > max_len:
            return text[:max_len] + f"... (truncated, total: {len(text)})"
        return text

    async def log_request(
        self,
        request: Request,
        request_id: str,
        body: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """記錄請求信息"""
        path = request.url.path

        log_data = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "path": path,
            "query_params": dict(request.query_params),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("User-Agent", ""),
        }

        # 記錄請求頭（脫敏）
        log_data["headers"] = self._mask_headers(dict(request.headers))

        # 記錄請求體（如果啟用且不在跳過列表中）
        if self.log_request_body and path not in self.SKIP_BODY_ENDPOINTS:
            if body:
                try:
                    body_str = body.decode("utf-8")
                    body_json = json.loads(body_str)
                    log_data["body"] = self._mask_body(body_json)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    log_data["body"] = self._truncate(body.decode("utf-8", errors="replace"))
                except Exception:
                    log_data["body"] = "<unable to parse>"

        return log_data

    def log_response(
        self,
        request_id: str,
        status_code: int,
        duration_ms: float,
        response_body: Optional[bytes] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """記錄響應信息"""
        log_data = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
        }

        if error:
            log_data["error"] = error

        # 記錄響應體（如果啟用）
        if self.log_response_body and response_body:
            try:
                body_str = response_body.decode("utf-8")
                if len(body_str) <= self.max_body_length:
                    body_json = json.loads(body_str)
                    log_data["response_body"] = body_json
                else:
                    log_data["response_body"] = f"<truncated, length: {len(body_str)}>"
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        return log_data


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """請求日誌中間件"""

    def __init__(self, app, request_logger: RequestLogger = None):
        super().__init__(app)
        self.request_logger = request_logger or RequestLogger()

    async def dispatch(self, request: Request, call_next) -> Response:
        """處理請求並記錄日誌"""
        path = request.url.path

        # 跳過不需要記錄的端點
        if path in RequestLogger.SKIP_LOGGING_ENDPOINTS:
            return await call_next(request)

        # 生成請求 ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]

        # 設置請求 ID 到 state（供其他中間件使用）
        request.state.request_id = request_id

        # 讀取請求體
        body = None
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.body()
            except Exception:
                pass

        # 記錄請求
        request_data = await self.request_logger.log_request(request, request_id, body)

        # 計算處理時間
        start_time = time.time()
        error_msg = None
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000

            # 記錄響應
            response_data = self.request_logger.log_response(
                request_id=request_id,
                status_code=status_code,
                duration_ms=duration_ms,
                error=error_msg
            )

            # 合併日誌數據
            log_entry = {**request_data, **response_data}

            # 根據狀態碼選擇日誌級別
            if status_code >= 500:
                logger.error(f"[{request_id}] {request.method} {path} -> {status_code} ({duration_ms:.2f}ms)",
                            extra={"request_log": log_entry})
            elif status_code >= 400:
                logger.warning(f"[{request_id}] {request.method} {path} -> {status_code} ({duration_ms:.2f}ms)",
                              extra={"request_log": log_entry})
            else:
                # 成功請求只記錄簡要信息
                logger.info(f"[{request_id}] {request.method} {path} -> {status_code} ({duration_ms:.2f}ms)")

        # 添加請求 ID 到響應頭
        response.headers["X-Request-ID"] = request_id

        return response


# 全局實例
request_logger = RequestLogger()
