"""
API 依賴項模組

從 auth.py 重新導出常用的依賴項，方便其他模組 import。
"""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.config import settings
from src.api.auth import (
    get_current_user,
    get_optional_user,
    verify_api_key as _verify_api_key,
    TokenData,
    RequireTrading,
    RequireAdmin,
    RequireAuth,
    OptionalAuth,
)

# API Key Header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str = Security(api_key_header)
) -> str:
    """
    驗證 API Key 的 FastAPI 依賴項
    
    用法:
        @router.get("/protected")
        async def protected_route(_: str = Depends(verify_api_key)):
            ...
    """
    # 開發環境允許無認證
    if settings.ENVIRONMENT == "development":
        return "development"
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    key_info = _verify_api_key(api_key)
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return key_info["name"]


# 重新導出，方便其他模組使用
__all__ = [
    "verify_api_key",
    "get_current_user",
    "get_optional_user",
    "TokenData",
    "RequireTrading",
    "RequireAdmin",
    "RequireAuth",
    "OptionalAuth",
]
