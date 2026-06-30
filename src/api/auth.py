"""JWT 認證模組"""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Security, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
import jwt
from pydantic import BaseModel

from src.config import settings
from src.logger import logger


# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class TokenData(BaseModel):
    """Token 數據"""
    sub: str  # subject (user id or api key name)
    exp: datetime
    iat: datetime
    scope: str = "trading"  # trading, read_only, admin


class AuthConfig:
    """認證配置"""
    # JWT 設定
    SECRET_KEY: str = settings.SECRET_KEY
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 小時

    # API Keys (可以從環境變數或資料庫載入)
    # 格式: {api_key: {"name": "...", "scope": "..."}}
    API_KEYS: dict = {}

    @classmethod
    def load_api_keys(cls):
        """從環境變數載入 API Keys"""
        import os
        # 格式: API_KEY_NAME=key:scope
        # 例如: API_KEY_IOS_APP=abc123:trading
        for key, value in os.environ.items():
            if key.startswith("API_KEY_"):
                name = key.replace("API_KEY_", "").lower()
                parts = value.split(":")
                if len(parts) >= 1:
                    api_key = parts[0]
                    scope = parts[1] if len(parts) > 1 else "read_only"
                    cls.API_KEYS[api_key] = {"name": name, "scope": scope}

        # 不提供任何內建/預設 API Key。所有金鑰一律經由 API_KEY_* 環境變數注入，
        # 以高強度隨機字串設定，避免將可預測的金鑰寫進原始碼。


# 載入 API Keys
AuthConfig.load_api_keys()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """創建 JWT Access Token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow()
    })

    encoded_jwt = jwt.encode(
        to_encode,
        AuthConfig.SECRET_KEY,
        algorithm=AuthConfig.ALGORITHM
    )

    return encoded_jwt


def decode_token(token: str) -> Optional[TokenData]:
    """解碼並驗證 JWT Token"""
    try:
        payload = jwt.decode(
            token,
            AuthConfig.SECRET_KEY,
            algorithms=[AuthConfig.ALGORITHM]
        )

        return TokenData(
            sub=payload.get("sub", ""),
            exp=datetime.fromtimestamp(payload.get("exp", 0)),
            iat=datetime.fromtimestamp(payload.get("iat", 0)),
            scope=payload.get("scope", "read_only")
        )

    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None


def verify_api_key(api_key: str) -> Optional[dict]:
    """驗證 API Key"""
    if api_key in AuthConfig.API_KEYS:
        return AuthConfig.API_KEYS[api_key]
    return None


async def get_current_user(
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header)
) -> TokenData:
    """
    獲取當前用戶（支持 JWT Token 或 API Key）

    認證方式：
    1. Bearer Token (JWT): Authorization: Bearer <token>
    2. API Key: X-API-Key: <key>
    """
    # 嘗試 Bearer Token
    if bearer and bearer.credentials:
        token_data = decode_token(bearer.credentials)
        if token_data:
            return token_data

    # 嘗試 API Key
    if api_key:
        key_info = verify_api_key(api_key)
        if key_info:
            return TokenData(
                sub=key_info["name"],
                exp=datetime.utcnow() + timedelta(days=365),  # API Key 不過期
                iat=datetime.utcnow(),
                scope=key_info["scope"]
            )

    # 不再提供「開發環境免認證」後門：任何環境下，缺少有效憑證一律拒絕。
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_optional_user(
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header)
) -> Optional[TokenData]:
    """
    可選認證 - 不強制要求認證
    用於公開端點但需要知道用戶身份的情況
    """
    try:
        return await get_current_user(bearer, api_key)
    except HTTPException:
        return None


def require_scope(required_scope: str):
    """
    權限檢查裝飾器

    使用方式:
    @app.get("/admin/users")
    async def get_users(user: TokenData = Depends(require_scope("admin"))):
        ...
    """
    async def check_scope(
        user: TokenData = Depends(get_current_user)
    ) -> TokenData:
        scope_hierarchy = {
            "read_only": 0,
            "trading": 1,
            "admin": 2
        }

        user_level = scope_hierarchy.get(user.scope, 0)
        required_level = scope_hierarchy.get(required_scope, 0)

        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {required_scope}, Got: {user.scope}"
            )

        return user

    return check_scope


# 便捷依賴項
RequireTrading = Depends(require_scope("trading"))
RequireAdmin = Depends(require_scope("admin"))
RequireAuth = Depends(get_current_user)
OptionalAuth = Depends(get_optional_user)
