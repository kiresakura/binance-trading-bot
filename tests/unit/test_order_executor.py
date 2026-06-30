"""Unit tests for the smart order execution system.

Every exchange / network collaborator is mocked at the
``src.services.order_executor.<name>`` boundary so the suite runs fully
offline and finishes in well under a second.

The autouse ``_isolate_network`` fixture is the safety net that turns the
otherwise-hanging suite into a fast, deterministic one:

* replaces ``binance_client`` so ``OrderBookAnalyzer.analyze_depth``
  (``get_order_book``) and ``SlippagePredictor._get_volatility_factor``
  (``get_klines``, reached via ``predict``) never touch the network;
* neutralises ``asyncio.sleep`` so the per-chunk pacing delays in
  ``execute_large_order`` (``i * uniform(5, 15)`` seconds) don't stall the run;
* clears the class-level ``OrderBookAnalyzer._depth_cache`` so the 5s depth
  cache can't leak between tests.

Tests that need to steer the execution strategy mock ``analyze_depth``'s
``liquidity_score`` directly (the real lever the executor branches on),
while ``fault_tolerant_engine`` / ``risk_management_system`` are mocked per
test to drive real success / rejection / failure code paths.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services import order_executor as oe
from src.services.order_executor import (
    OrderChunk,
    ExecutionResult,
    OrderBookAnalyzer,
    SlippagePredictor,
    MarketImpactModel,
)
from src.services.risk_manager import Order, ValidationResult


# --------------------------------------------------------------------------- #
# Deterministic offline market data
# --------------------------------------------------------------------------- #
def _fake_order_book():
    """Binance-style order book (``[price_str, qty_str]`` pairs), balanced book."""
    return {
        "bids": [["42000.00", "1.0"], ["41999.00", "2.0"], ["41998.00", "3.0"]],
        "asks": [["42001.00", "1.0"], ["42002.00", "2.0"], ["42003.00", "3.0"]],
    }


def _fake_klines():
    """20 Binance-style klines whose closes swing hard.

    The +/-~4.7% close-to-close swing pushes the realised-volatility estimate
    above the 0.01 normalisation ceiling, so ``_get_volatility_factor`` returns
    its capped value of 1.0 deterministically (close price is index 4).
    """
    closes = [42000.0 if i % 2 == 0 else 44000.0 for i in range(20)]
    # [openTime, open, high, low, close, volume, closeTime, qav, n, tbb, tbq, ign]
    return [[0, c, c, c, c, 1.0, 0, 0.0, 0, 0.0, 0.0, 0] for c in closes]


@pytest.fixture(autouse=True)
def _isolate_network(monkeypatch):
    """Make every test in this module fully offline and fast."""
    OrderBookAnalyzer._depth_cache.clear()

    fake_client = MagicMock(name="binance_client")
    fake_client.get_order_book = AsyncMock(return_value=_fake_order_book())
    fake_client.get_klines = AsyncMock(return_value=_fake_klines())
    monkeypatch.setattr(oe, "binance_client", fake_client)

    # execute_large_order awaits asyncio.sleep(chunk.delay) per chunk; the real
    # delays are 5-60s, so neutralise sleep to keep the suite instant.
    monkeypatch.setattr(asyncio, "sleep", AsyncMock(name="sleep"))

    yield

    OrderBookAnalyzer._depth_cache.clear()


def _valid_risk():
    risk = MagicMock(name="risk_management_system")
    risk.validate_order = AsyncMock(return_value=ValidationResult(is_valid=True))
    return risk


def _engine(return_value):
    engine = MagicMock(name="fault_tolerant_engine")
    engine.execute_with_failover = AsyncMock(return_value=return_value)
    return engine


# --------------------------------------------------------------------------- #
# OrderBookAnalyzer
# --------------------------------------------------------------------------- #
class TestOrderBookAnalyzer:
    """Order book analysis."""

    @pytest.mark.asyncio
    async def test_analyze_depth(self):
        """analyze_depth parses the order book and derives liquidity metrics."""
        analyzer = OrderBookAnalyzer()

        depth = await analyzer.analyze_depth("BTCUSDT", "BUY")

        # Real call signature against the exchange client.
        oe.binance_client.get_order_book.assert_awaited_once_with("BTCUSDT", limit=20)

        # Shape.
        for key in ("bids", "asks", "liquidity_score", "spread",
                    "imbalance", "bid_volume", "ask_volume"):
            assert key in depth

        # Raw string pairs are parsed into (Decimal, Decimal) tuples.
        assert depth["bids"][0] == (Decimal("42000.00"), Decimal("1.0"))
        assert depth["asks"][0] == (Decimal("42001.00"), Decimal("1.0"))

        # Volumes = sum of first-10 level quantities (1 + 2 + 3).
        assert depth["bid_volume"] == 6.0
        assert depth["ask_volume"] == 6.0

        # Balanced book -> zero imbalance.
        assert depth["imbalance"] == pytest.approx(0.0)

        # spread = (best_ask - best_bid) / best_bid = 1 / 42000.
        assert depth["spread"] == pytest.approx(1.0 / 42000.0)

        # liquidity_score is normalised into [0, 1] and floored at 0.1.
        assert 0.1 <= depth["liquidity_score"] <= 1.0

        # A second call inside the 5s TTL is served from the class cache:
        # same object, no extra network round-trip.
        depth2 = await analyzer.analyze_depth("BTCUSDT", "BUY")
        assert depth2 is depth
        oe.binance_client.get_order_book.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_analyze_depth_fallback_on_empty_book(self):
        """An empty/None order book yields the neutral fallback depth."""
        oe.binance_client.get_order_book = AsyncMock(return_value=None)
        analyzer = OrderBookAnalyzer()

        depth = await analyzer.analyze_depth("ETHUSDT", "SELL")

        assert depth["bids"] == []
        assert depth["asks"] == []
        assert depth["liquidity_score"] == 0.5
        assert depth["spread"] == 0.001

    @pytest.mark.asyncio
    async def test_calculate_sufficient_liquidity(self, mock_order_book):
        """Liquidity calc reports sufficiency once cumulative qty covers target."""
        analyzer = OrderBookAnalyzer()
        target_quantity = Decimal("2.0")

        result = analyzer.calculate_available_liquidity(
            mock_order_book["bids"], target_quantity
        )

        assert result["sufficient_liquidity"] is True
        # First two levels (1.5 + 2.0 = 3.5) already cover 2.0.
        assert result["levels_needed"] == 2
        assert result["total_available"] == Decimal("3.5")
        # Volume-weighted avg price sits between the two consumed price levels.
        assert Decimal("41980") <= result["avg_price"] <= Decimal("41990")

    @pytest.mark.asyncio
    async def test_calculate_insufficient_liquidity(self, mock_order_book):
        """Liquidity calc reports insufficiency when depth can't cover target."""
        analyzer = OrderBookAnalyzer()
        target_quantity = Decimal("100.0")  # Larger than the whole book.

        result = analyzer.calculate_available_liquidity(
            mock_order_book["bids"], target_quantity
        )

        assert result["sufficient_liquidity"] is False
        # All three levels consumed: 1.5 + 2.0 + 1.0 = 4.5.
        assert result["levels_needed"] == 3
        assert result["total_available"] == Decimal("4.5")
        assert result["total_available"] < target_quantity


# --------------------------------------------------------------------------- #
# SlippagePredictor
# --------------------------------------------------------------------------- #
class TestSlippagePredictor:
    """Slippage prediction."""

    @pytest.mark.asyncio
    async def test_predict_small_order(self):
        """predict returns the heuristic fields and the real volatility factor."""
        predictor = SlippagePredictor()
        chunk = OrderChunk(
            quantity=Decimal("0.1"), price=Decimal("42000"), delay=0.0, chunk_id=0
        )

        prediction = await predictor.predict(chunk)

        assert {"predicted_slippage", "confidence", "volatility_factor"} <= set(prediction)
        assert prediction["predicted_slippage"] >= 0
        # 0.75 confidence is the success path (the error fallback returns 0.5).
        assert prediction["confidence"] == 0.75
        # Klines fixture swings hard -> volatility normalised to its 1.0 ceiling.
        assert prediction["volatility_factor"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_predict_large_order(self):
        """Larger chunks predict strictly larger slippage than small chunks."""
        predictor = SlippagePredictor()
        large = OrderChunk(
            quantity=Decimal("10.0"), price=Decimal("42000"), delay=0.0, chunk_id=0
        )
        small = OrderChunk(
            quantity=Decimal("0.1"), price=Decimal("42000"), delay=0.0, chunk_id=0
        )

        large_pred = await predictor.predict(large)
        small_pred = await predictor.predict(small)

        # base = qty * 1e-4 = 1e-3, doubled by vol=1.0, then * U(0.8, 1.2):
        # always strictly above 1e-3.
        assert large_pred["predicted_slippage"] > 0.001
        assert large_pred["predicted_slippage"] > small_pred["predicted_slippage"]

    @pytest.mark.asyncio
    async def test_model_update(self):
        """update_model accumulates history and caps it at 1000 (FIFO)."""
        predictor = SlippagePredictor()

        for i in range(10):
            predictor.update_model(0.002 + i * 0.0001)

        assert len(predictor.historical_slippage) == 10
        assert predictor.historical_slippage[0] == pytest.approx(0.002)

        # Overflow the cap; oldest entries are evicted from the front.
        for _ in range(1500):
            predictor.update_model(0.002)

        assert len(predictor.historical_slippage) == 1000
        assert predictor.historical_slippage[-1] == pytest.approx(0.002)


# --------------------------------------------------------------------------- #
# MarketImpactModel
# --------------------------------------------------------------------------- #
class TestMarketImpactModel:
    """Market impact estimation (impact = 0.01 * sqrt(order/volume), capped 5%)."""

    def test_estimate_small_impact(self):
        """A small participation ratio yields sub-1% impact."""
        model = MarketImpactModel()

        impact = model.estimate_impact(
            order_size=Decimal("1.0"), average_volume=Decimal("1000.0")
        )

        # 0.01 * sqrt(1/1000) ~= 3.16e-4.
        assert impact == pytest.approx(0.01 * (0.001 ** 0.5))
        assert impact < 0.01

    def test_estimate_large_impact(self):
        """Larger orders impact more, but the sqrt model stays under the 5% cap.

        NOTE: the previous assertion ``impact > 0.01`` was stale -- it did not
        match the real sqrt model. For order 100 / volume 1000 the participation
        ratio is 0.1, giving 0.01 * sqrt(0.1) ~= 3.16e-3, which is *below* 0.01.
        The real, monotonic behaviour ("larger order -> higher impact") plus the
        cap is asserted instead.
        """
        model = MarketImpactModel()

        small = model.estimate_impact(Decimal("1.0"), Decimal("1000.0"))
        large = model.estimate_impact(Decimal("100.0"), Decimal("1000.0"))

        assert large == pytest.approx(0.01 * (0.1 ** 0.5))  # ~3.16e-3
        assert large > small                                # higher impact
        assert large <= 0.05                                # never exceeds cap

        # The cap only engages when the order dwarfs the volume:
        # ratio 25 -> 0.01 * sqrt(25) = 0.05 exactly; larger -> still capped.
        assert model.estimate_impact(Decimal("25000.0"), Decimal("1000.0")) == 0.05
        assert model.estimate_impact(Decimal("1000000.0"), Decimal("1000.0")) == 0.05

    def test_zero_volume_handling(self):
        """Zero average volume short-circuits to the 1% default impact."""
        model = MarketImpactModel()

        impact = model.estimate_impact(
            order_size=Decimal("10.0"), average_volume=Decimal("0")
        )

        assert impact == 0.01


# --------------------------------------------------------------------------- #
# SmartOrderExecutor
# --------------------------------------------------------------------------- #
class TestSmartOrderExecutor:
    """Smart order execution."""

    @pytest.mark.asyncio
    async def test_execute_order_immediate_high_liquidity(self, sample_order, order_executor):
        """High liquidity -> single immediate fill with the exchange's price."""
        engine = _engine(
            {"status": "success", "order_id": "test_123", "price": 42000.0}
        )

        with patch.object(oe, "fault_tolerant_engine", engine), \
             patch.object(oe, "risk_management_system", _valid_risk()) as risk, \
             patch.object(
                 order_executor.order_book_analyzer, "analyze_depth",
                 AsyncMock(return_value={"liquidity_score": 0.9}),
             ):
            results = await order_executor.execute_order(sample_order, strategy="adaptive")

        assert len(results) == 1
        res = results[0]
        assert isinstance(res, ExecutionResult)
        assert res.success is True
        assert res.order_id == "test_123"
        assert res.executed_price == Decimal("42000.0")
        assert res.executed_quantity == sample_order.quantity  # 0.5
        assert res.commission == Decimal("42000.0") * sample_order.quantity * Decimal("0.001")

        # Risk validation ran and a single MARKET order hit the failover engine.
        risk.validate_order.assert_awaited_once()
        engine.execute_with_failover.assert_awaited_once()
        order_data = engine.execute_with_failover.await_args.args[0]
        assert order_data["symbol"] == "BTCUSDT"
        assert order_data["side"] == "BUY"
        assert order_data["type"] == "MARKET"
        assert order_data["quantity"] == float(sample_order.quantity)

    @pytest.mark.asyncio
    async def test_reject_invalid_order(self, risky_order, order_executor):
        """A risk rejection returns a single failure result and skips execution."""
        risk = MagicMock()
        risk.validate_order = AsyncMock(
            return_value=ValidationResult(is_valid=False, reason="Position size too large")
        )
        engine = _engine({"status": "success"})

        with patch.object(oe, "risk_management_system", risk), \
             patch.object(oe, "fault_tolerant_engine", engine):
            results = await order_executor.execute_order(risky_order)

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error_message == "Position size too large"
        assert results[0].executed_quantity == Decimal("0")
        # Rejected before reaching the exchange.
        engine.execute_with_failover.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_large_order_splitting(self, large_order, order_executor):
        """Low liquidity splits a large order into multiple chunks summing to total."""
        engine = _engine(
            {"status": "success", "order_id": "test_chunk", "price": 42000.0}
        )

        with patch.object(oe, "fault_tolerant_engine", engine), \
             patch.object(
                 order_executor.order_book_analyzer, "analyze_depth",
                 AsyncMock(return_value={"liquidity_score": 0.3}),
             ):
            results = await order_executor.execute_large_order(large_order)

        # liquidity 0.3 -> (< 0.6) -> 5 chunks.
        assert len(results) == 5
        assert all(isinstance(r, ExecutionResult) for r in results)
        assert all(r.success for r in results)
        # Chunks reconstruct the parent quantity exactly.
        assert sum(r.executed_quantity for r in results) == large_order.quantity  # 10.0
        # One failover call per chunk.
        assert engine.execute_with_failover.await_count == 5

    @pytest.mark.asyncio
    async def test_calculate_optimal_chunks(self, large_order, order_executor, mock_order_book):
        """Optimal chunking: quantities conserve, delays rise, ends are smaller."""
        # Pin the random pacing so the strictly-increasing-delay check is stable
        # (real code: delay = i * uniform(5, 15), which can be non-monotonic).
        with patch.object(oe.np.random, "uniform", return_value=10.0):
            chunks = order_executor._calculate_optimal_chunks(large_order, mock_order_book)

        # mock_order_book liquidity 0.75 -> default 3 chunks.
        assert len(chunks) == 3
        assert all(isinstance(c, OrderChunk) for c in chunks)

        # Quantities conserve the parent order exactly.
        assert sum(c.quantity for c in chunks) == large_order.quantity

        # Deterministic, strictly increasing delays: i * 10.0.
        assert [c.delay for c in chunks] == [0.0, 10.0, 20.0]
        for i in range(1, len(chunks)):
            assert chunks[i].delay > chunks[i - 1].delay

        # Non-uniform split: first and last chunks are smaller than the middle.
        assert chunks[0].quantity < chunks[1].quantity
        assert chunks[-1].quantity <= chunks[1].quantity
        assert [c.chunk_id for c in chunks] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_adaptive_strategy_high_liquidity(self, sample_order, order_executor):
        """Adaptive + high liquidity -> immediate single execution (no chunking)."""
        engine = _engine(
            {"status": "success", "order_id": "test_123", "price": 42000.0}
        )

        with patch.object(oe, "fault_tolerant_engine", engine):
            results = await order_executor._execute_adaptive(
                sample_order, {"liquidity_score": 0.9}
            )

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].order_id == "test_123"
        engine.execute_with_failover.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_adaptive_strategy_low_liquidity(self, large_order, order_executor):
        """Adaptive + low liquidity -> routed to the large-order splitter."""
        engine = _engine(
            {"status": "success", "order_id": "test_chunk", "price": 42000.0}
        )

        with patch.object(oe, "fault_tolerant_engine", engine), \
             patch.object(
                 order_executor.order_book_analyzer, "analyze_depth",
                 AsyncMock(return_value={"liquidity_score": 0.3}),
             ):
            results = await order_executor._execute_adaptive(
                large_order, {"liquidity_score": 0.3}
            )

        assert len(results) > 1
        assert all(r.success for r in results)
        assert sum(r.executed_quantity for r in results) == large_order.quantity

    @pytest.mark.asyncio
    async def test_execution_error_handling(self, sample_order, order_executor):
        """A failed exchange status surfaces as a failure result with the message."""
        engine = _engine({"status": "failed", "msg": "Connection timeout"})

        with patch.object(oe, "fault_tolerant_engine", engine), \
             patch.object(oe, "risk_management_system", _valid_risk()), \
             patch.object(
                 order_executor.order_book_analyzer, "analyze_depth",
                 AsyncMock(return_value={"liquidity_score": 0.9}),
             ):
            results = await order_executor.execute_order(sample_order)

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].executed_quantity == Decimal("0")
        assert results[0].error_message is not None
        assert "Connection timeout" in results[0].error_message

    @pytest.mark.asyncio
    async def test_slippage_tracking(self, sample_order, order_executor):
        """A successful fill above order price records the realised slippage."""
        # Exchange fills 100 above the order's 42000 limit.
        engine = _engine(
            {"status": "success", "order_id": "test_123", "price": 42100.0}
        )

        assert order_executor.slippage_predictor.historical_slippage == []

        with patch.object(oe, "fault_tolerant_engine", engine), \
             patch.object(oe, "risk_management_system", _valid_risk()), \
             patch.object(
                 order_executor.order_book_analyzer, "analyze_depth",
                 AsyncMock(return_value={"liquidity_score": 0.9}),
             ):
            results = await order_executor.execute_order(sample_order)

        assert results[0].success is True
        assert results[0].executed_price == Decimal("42100.0")

        # One fill -> one recorded slippage ~= |42100 - 42000| / 42000.
        hist = order_executor.slippage_predictor.historical_slippage
        assert len(hist) == 1
        assert hist[-1] == pytest.approx(100.0 / 42000.0)


# --------------------------------------------------------------------------- #
# MARKET-order minimum-value guard (regression: get_symbol_price did not exist)
# --------------------------------------------------------------------------- #
class TestMinOrderValueGuard:
    """execute_order 對 MARKET 單的最低金額守門。

    迴歸測試：舊版呼叫不存在的 ``binance_client.get_symbol_price``，被 try/except
    吞掉 → ``order_value=inf`` → 守門形同虛設。修正後改用 ``get_ticker_price``。
    """

    @pytest.mark.asyncio
    async def test_market_order_below_min_value_is_rejected(self, order_executor):
        # 0.0001 BTC * 42000 = $4.2 < MIN_ORDER_VALUE($6) → 應被拒
        order = Order(symbol="BTCUSDT", side="BUY", order_type="MARKET",
                      quantity=Decimal("0.0001"), price=None, timestamp=datetime.utcnow())
        with patch("src.services.binance_client.binance_client") as bc:
            bc.get_ticker_price = AsyncMock(return_value={"symbol": "BTCUSDT", "price": "42000.00"})
            results = await order_executor.execute_order(order)
            # 證明用了「正確的方法」抓價（舊的 get_symbol_price 根本不存在）
            bc.get_ticker_price.assert_awaited_once_with("BTCUSDT")
        assert len(results) == 1
        assert results[0].success is False
        assert "below minimum" in (results[0].error_message or "")

    @pytest.mark.asyncio
    async def test_market_order_above_min_value_passes_guard(self, order_executor):
        # 1 BTC * 42000 = $42000 > $6 → 通過守門並真的執行
        order = Order(symbol="BTCUSDT", side="BUY", order_type="MARKET",
                      quantity=Decimal("1.0"), price=None, timestamp=datetime.utcnow())
        with patch("src.services.binance_client.binance_client") as bc, \
             patch.object(oe, "risk_management_system", _valid_risk()), \
             patch.object(oe, "fault_tolerant_engine",
                          _engine({"status": "FILLED", "orderId": "x1", "price": "42000.0"})):
            bc.get_ticker_price = AsyncMock(return_value={"symbol": "BTCUSDT", "price": "42000.00"})
            results = await order_executor.execute_order(order)
        # 通過守門（沒有 below-minimum 拒絕）
        assert all("below minimum" not in (r.error_message or "") for r in results)
