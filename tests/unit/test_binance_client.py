"""Unit tests for ``src.services.binance_client.BinanceClient``.

All HTTP traffic is mocked: the tests replace the client's request helpers
with ``AsyncMock`` objects, so no network calls or API keys are needed.
"""

from unittest.mock import AsyncMock

from src.services.binance_client import BinanceClient


class _FakeResponse:
    status = 200

    async def json(self):
        return [[1700000000000, "42000.0", "42100.0", "41900.0", "42050.0", "12.5"]]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Records the headers of each GET instead of opening a connection."""

    def __init__(self):
        self.sent_headers = []

    def get(self, url, params=None, headers=None):
        self.sent_headers.append(headers)
        return _FakeResponse()


class TestRequestHeaders:
    """Public market-data endpoints must work without an API key."""

    async def test_no_api_key_header_when_key_is_not_configured(self):
        client = BinanceClient()
        client.api_key = None
        session = _FakeSession()
        client._get_session = AsyncMock(return_value=session)

        klines = await client.get_klines("BTCUSDT", limit=1)

        assert len(klines) == 1
        assert "X-MBX-APIKEY" not in session.sent_headers[0]

    async def test_api_key_header_sent_when_key_is_configured(self):
        client = BinanceClient()
        client.api_key = "test-key"
        session = _FakeSession()
        client._get_session = AsyncMock(return_value=session)

        await client.get_klines("BTCUSDT", limit=1)

        assert session.sent_headers[0]["X-MBX-APIKEY"] == "test-key"


class TestCancelAllOrders:
    """``cancel_all_orders`` backs the Telegram emergency-stop command."""

    async def test_cancels_every_open_order(self):
        client = BinanceClient()
        client.get_open_orders = AsyncMock(
            return_value=[{"orderId": 1}, {"orderId": 2}]
        )
        client.cancel_order = AsyncMock(
            side_effect=lambda symbol, order_id: {"orderId": order_id, "status": "CANCELED"}
        )

        cancelled = await client.cancel_all_orders("BTCUSDT")

        assert [o["orderId"] for o in cancelled] == [1, 2]
        assert client.cancel_order.await_count == 2

    async def test_failed_cancellations_are_left_out(self):
        client = BinanceClient()
        client.get_open_orders = AsyncMock(
            return_value=[{"orderId": 1}, {"orderId": 2}]
        )

        async def cancel(symbol, order_id):
            if order_id == 2:
                raise RuntimeError("order already filled")
            return {"orderId": order_id, "status": "CANCELED"}

        client.cancel_order = AsyncMock(side_effect=cancel)

        cancelled = await client.cancel_all_orders("BTCUSDT")

        assert [o["orderId"] for o in cancelled] == [1]

    async def test_no_open_orders_cancels_nothing(self):
        client = BinanceClient()
        client.get_open_orders = AsyncMock(return_value=[])
        client.cancel_order = AsyncMock()

        assert await client.cancel_all_orders("BTCUSDT") == []
        client.cancel_order.assert_not_awaited()
