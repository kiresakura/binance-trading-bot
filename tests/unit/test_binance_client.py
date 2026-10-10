"""Unit tests for ``src.services.binance_client.BinanceClient``.

All HTTP traffic is mocked: the tests replace the client's request helpers
with ``AsyncMock`` objects, so no network calls or API keys are needed.
"""

from unittest.mock import AsyncMock

from src.services.binance_client import BinanceClient


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
