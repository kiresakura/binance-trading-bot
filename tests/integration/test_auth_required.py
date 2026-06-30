"""認證強制測試 - 驗證碰錢/管理端點在缺少憑證時被拒絕。

對應 COMMISSION.md §3 驗收情境：未帶認證呼叫受保護端點應回 401/403。
這些測試刻意設計為最小依賴（不需資料庫/Redis/Binance），可在輕量 venv 執行。
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.auth import create_access_token, decode_token


client = TestClient(app)

# 受保護的「寫入/碰錢」端點：未帶憑證必須被拒絕
PROTECTED_WRITE = [
    ("post", "/api/v1/orders", {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.01}),
    ("post", "/api/v1/trading/sync", None),
    ("post", "/api/v1/system/resume", None),
    ("post", "/api/v1/learning/train", None),
    ("delete", "/api/v1/trading/positions/BTCUSDT", None),
]

# 受保護的「帳戶資料讀取」端點：未帶憑證必須被拒絕
PROTECTED_READ = [
    "/api/v1/portfolio/summary",
    "/api/v1/orders",
    "/api/v1/orders/open",
    "/api/v1/trading/positions",
]


@pytest.mark.parametrize("method,path,body", PROTECTED_WRITE)
def test_protected_write_endpoints_reject_anonymous(method, path, body):
    """未認證呼叫寫入端點 → 401/403（絕不可為 200）。"""
    resp = getattr(client, method)(path, json=body) if body is not None else getattr(client, method)(path)
    assert resp.status_code in (401, 403), (
        f"{method.upper()} {path} 在未認證下回 {resp.status_code}，預期 401/403"
    )


@pytest.mark.parametrize("path", PROTECTED_READ)
def test_protected_read_endpoints_reject_anonymous(path):
    """未認證呼叫帳戶資料讀取端點 → 401/403。"""
    resp = client.get(path)
    assert resp.status_code in (401, 403), (
        f"GET {path} 在未認證下回 {resp.status_code}，預期 401/403"
    )


def test_debug_endpoint_not_anonymous():
    """/debug/db 不可匿名存取（401/403；正式環境亦可能 404）。"""
    resp = client.get("/debug/db")
    assert resp.status_code in (401, 403, 404)


def test_health_remains_public():
    """健康檢查維持公開（部署探針需要）。"""
    resp = client.get("/health")
    assert resp.status_code == 200


def test_invalid_token_rejected():
    """偽造/無效的 Bearer token 應被拒絕。"""
    resp = client.get(
        "/api/v1/portfolio/summary",
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert resp.status_code in (401, 403)


def test_jwt_roundtrip_with_configured_secret():
    """以設定的 SECRET_KEY 簽發的 JWT 可被解出且帶正確 scope（認證層正常運作）。"""
    token = create_access_token({"sub": "tester", "scope": "trading"})
    data = decode_token(token)
    assert data is not None
    assert data.sub == "tester"
    assert data.scope == "trading"


def test_no_builtin_backdoor_api_key():
    """確認沒有任何「程式碼內建」的後門金鑰（過去曾自動註冊一把 admin scope 的開發金鑰）。

    以特徵斷言（name=development 且 scope=admin 的自動註冊金鑰）而非寫死字串，
    避免測試本身被祕密掃描誤判。
    """
    from src.api.auth import AuthConfig

    assert not any(
        info.get("name") == "development" and info.get("scope") == "admin"
        for info in AuthConfig.API_KEYS.values()
    )
