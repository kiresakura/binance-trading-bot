"""Integration tests for the FastAPI application（對照現行 API surface 重寫）。

舊版測試針對多個現已不存在的端點（/api/v1/system/status、/portfolio/positions、
/audit?limit=、order-by-id 等）與舊的回應結構，已全面失效。本檔改為對照
src/api/app.py 的真實端點，並對受保護端點帶上有效 JWT。
未認證「拒絕」情境集中在 test_auth_required.py，本檔不重複。
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from src.api.app import app
from src.api.auth import create_access_token


client = TestClient(app)


def _auth(scope: str = "trading") -> dict:
    """產生帶有效 JWT 的 Authorization 標頭。"""
    token = create_access_token({"sub": "test", "scope": scope})
    return {"Authorization": f"Bearer {token}"}


class TestPublicEndpoints:
    def test_root(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert data["components"]["api"] == "healthy"

    def test_metrics_prometheus_format(self):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]

    def test_trading_status_returns_payload(self):
        # 端點以 try/except 回退，缺引擎時仍回 200 + 預設結構
        resp = client.get("/api/v1/trading/status")
        assert resp.status_code == 200
        assert "is_running" in resp.json()

    def test_docs_available_in_non_production(self):
        # conftest 設 ENVIRONMENT=development → /docs 應開啟
        assert client.get("/docs").status_code == 200


class TestOrderFlow:
    def test_create_order_success_with_auth(self):
        with patch("src.api.app.risk_management_system") as mock_risk, \
             patch("src.api.app.smart_order_executor") as mock_exec, \
             patch("src.api.app.audit_logger") as mock_audit, \
             patch("src.api.app.metrics_collector"):
            mock_risk.validate_order = AsyncMock(
                return_value=MagicMock(is_valid=True, risk_score=0.3, reason=None, warnings=[])
            )
            mock_exec.execute_order = AsyncMock(return_value=[MagicMock(
                success=True, order_id="t1",
                executed_price=Decimal("42000"), executed_quantity=Decimal("0.5"),
                total_cost=Decimal("21000"), commission=Decimal("21"),
            )])
            mock_audit.log_trade = AsyncMock()

            resp = client.post(
                "/api/v1/orders",
                json={"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.5, "price": 42000},
                headers=_auth("trading"),
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["order_id"] == "t1"

    def test_create_order_risk_rejection(self):
        with patch("src.api.app.risk_management_system") as mock_risk, \
             patch("src.api.app.audit_logger") as mock_audit, \
             patch("src.api.app.metrics_collector"):
            mock_risk.validate_order = AsyncMock(
                return_value=MagicMock(is_valid=False, risk_score=0.95, reason="position too large", warnings=[])
            )
            mock_audit.log_risk_rejection = AsyncMock()

            resp = client.post(
                "/api/v1/orders",
                json={"symbol": "BTCUSDT", "side": "BUY", "quantity": 50, "price": 42000},
                headers=_auth("trading"),
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is False
            assert "position too large" in data["message"]

    def test_create_order_invalid_body_422(self):
        # 帶有效認證，但數量為負（gt=0 失敗）→ 422
        resp = client.post(
            "/api/v1/orders",
            json={"symbol": "BTCUSDT", "side": "BUY", "quantity": -1},
            headers=_auth("trading"),
        )
        assert resp.status_code == 422

    def test_validate_order_is_public(self):
        # /orders/validate 不需認證（僅驗證不下單）
        with patch("src.api.app.risk_management_system") as mock_risk:
            mock_risk.validate_order = AsyncMock(
                return_value=MagicMock(is_valid=True, risk_score=0.2, reason=None, warnings=[])
            )
            resp = client.post(
                "/api/v1/orders/validate",
                json={"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.5, "price": 42000},
            )
            assert resp.status_code == 200
            assert resp.json()["is_valid"] is True


class TestAuditEndpoints:
    def test_audit_verify_with_auth(self):
        with patch("src.api.app.audit_logger") as mock_audit:
            mock_audit.verify_chain_integrity = AsyncMock(return_value=True)
            resp = client.get("/api/v1/audit/verify", headers=_auth("read_only"))
            assert resp.status_code == 200
            assert resp.json()["is_valid"] is True

    def test_audit_trail_with_auth(self):
        with patch("src.api.app.audit_logger") as mock_audit:
            mock_audit.get_audit_trail = AsyncMock(return_value=[{"event_type": "TRADE", "data": {}}])
            resp = client.get("/api/v1/audit/trail?limit=10", headers=_auth("read_only"))
            assert resp.status_code == 200
            assert resp.json()["count"] == 1


class TestErrorHandling:
    def test_not_found(self):
        assert client.get("/api/v1/nonexistent").status_code == 404

    def test_method_not_allowed(self):
        assert client.delete("/health").status_code == 405
