"""Unit tests for the multi-layer risk management system.

These tests exercise the *current* public API of ``src.services.risk_manager``:
each risk layer is an async ``RiskLayer`` whose ``validate(order)`` reads market
/ account state through small async I/O helpers (``_get_portfolio_value``,
``_calculate_drawdown``, ``_get_correlation``, ``_get_current_volatility`` ...)
and returns a ``ValidationResult(is_valid, reason, risk_score, warnings)``.

The I/O helpers talk to Binance / Redis, so the tests stub *only* those helpers
with controlled values and then assert the real decision logic (threshold
comparisons, risk-score maths, warning generation, short-circuiting) produced by
each ``validate`` implementation.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

from src.services.risk_manager import (
    Order,
    PositionSizeController,
    DrawdownProtection,
    CorrelationAnalyzer,
    BlackSwanProtection,
    CircuitBreaker,
    RiskManagementSystem,
    ValidationResult,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_order(symbol="BTCUSDT", side="BUY", quantity="0.5", price="42000.0"):
    return Order(
        symbol=symbol,
        side=side,
        order_type="LIMIT",
        quantity=Decimal(quantity),
        price=Decimal(price),
        timestamp=datetime.utcnow(),
    )


def _drawdown_stub(day=0.0, week=0.0, month=0.0):
    """Return an async stand-in for ``DrawdownProtection._calculate_drawdown``.

    ``validate`` calls it once per period (``day``/``week``/``month``); the stub
    returns the (negative) drawdown configured for the requested period.
    """
    mapping = {"day": day, "week": week, "month": month}

    async def _calc(period="day"):
        return mapping[period]

    return _calc


def _patch_system_layers(
    manager,
    *,
    portfolio=1_000_000.0,
    exposure=0.0,
    drawdowns=(0.0, 0.0, 0.0),
    positions=None,
    correlation=0.0,
    cur_vol=0.02,
    hist_vol=0.02,
    cur_volume=1000.0,
    avg_volume=1000.0,
    trade_count=0,
):
    """Stub every layer's I/O helpers on a ``RiskManagementSystem`` instance so
    the full ``validate_order`` pipeline runs deterministically without network.
    """
    pos, draw, corr, swan, breaker = manager.risk_layers

    pos._get_portfolio_value = AsyncMock(return_value=portfolio)
    pos._get_total_exposure = AsyncMock(return_value=exposure)

    draw._calculate_drawdown = _drawdown_stub(*drawdowns)

    corr._get_current_positions = AsyncMock(return_value=positions or [])
    corr._get_correlation = AsyncMock(return_value=correlation)

    swan._get_current_volatility = AsyncMock(return_value=cur_vol)
    swan._get_historical_volatility = AsyncMock(return_value=hist_vol)
    swan._get_current_volume = AsyncMock(return_value=cur_volume)
    swan._get_average_volume = AsyncMock(return_value=avg_volume)

    breaker._get_trade_count = AsyncMock(return_value=trade_count)
    breaker.is_circuit_broken = False
    breaker.circuit_break_time = None


# --------------------------------------------------------------------------- #
# Layer 1: PositionSizeController
# --------------------------------------------------------------------------- #
class TestPositionSizeController:
    """Position-size / total-exposure limits (max_position_size, max_total_exposure)."""

    @pytest.mark.asyncio
    async def test_validate_normal_position(self, sample_order):
        """A position well under the per-trade cap is accepted with a low score."""
        controller = PositionSizeController()  # defaults: 10% per trade, 50% total
        controller._get_portfolio_value = AsyncMock(return_value=1_000_000.0)
        controller._get_total_exposure = AsyncMock(return_value=0.0)

        result = await controller.validate(sample_order)

        # 0.5 * 42000 = 21000 -> 2.1% of a $1M portfolio.
        assert result.is_valid is True
        assert result.risk_score == pytest.approx(0.21)  # 0.021 / 0.10
        assert result.risk_score < 0.5
        assert result.warnings == []

    @pytest.mark.asyncio
    async def test_reject_oversized_position(self, risky_order):
        """A single position exceeding max_position_size is rejected (score 1.0)."""
        controller = PositionSizeController()
        controller._get_portfolio_value = AsyncMock(return_value=100_000.0)
        controller._get_total_exposure = AsyncMock(return_value=0.0)

        # 50 * 42000 = 2_100_000 -> 2100% of a $100k portfolio.
        result = await controller.validate(risky_order)

        assert result.is_valid is False
        assert "exceeds max" in result.reason.lower()
        assert result.risk_score == 1.0

    @pytest.mark.asyncio
    async def test_reject_when_total_exposure_exceeded(self):
        """A position that passes the per-trade cap but pushes aggregate exposure
        over max_total_exposure is rejected (score 0.9)."""
        controller = PositionSizeController()
        controller._get_portfolio_value = AsyncMock(return_value=100_000.0)
        controller._get_total_exposure = AsyncMock(return_value=0.48)  # already 48%

        order = _make_order(quantity="0.1", price="50000.0")  # 5000 -> 5% of 100k

        result = await controller.validate(order)

        # 5% per-trade is fine, but 48% + 5% = 53% > 50% total cap.
        assert result.is_valid is False
        assert "exposure" in result.reason.lower()
        assert result.risk_score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_warns_when_approaching_position_limit(self):
        """A position above 70% of the cap is still valid but emits a warning."""
        controller = PositionSizeController()
        controller._get_portfolio_value = AsyncMock(return_value=100_000.0)
        controller._get_total_exposure = AsyncMock(return_value=0.0)

        order = _make_order(quantity="0.16", price="50000.0")  # 8000 -> 8% (>7%, <10%)

        result = await controller.validate(order)

        assert result.is_valid is True
        assert result.risk_score == pytest.approx(0.8)  # 0.08 / 0.10
        assert len(result.warnings) >= 1
        assert "approaching" in result.warnings[0].lower()


# --------------------------------------------------------------------------- #
# Layer 2: DrawdownProtection
# --------------------------------------------------------------------------- #
class TestDrawdownProtection:
    """Daily / weekly / monthly drawdown limits."""

    @pytest.mark.asyncio
    async def test_no_drawdown(self, sample_order):
        """No drawdown across all windows -> valid with zero risk score."""
        protection = DrawdownProtection()
        protection._calculate_drawdown = _drawdown_stub(day=0.0, week=0.0, month=0.0)

        result = await protection.validate(sample_order)

        assert result.is_valid is True
        assert result.risk_score == 0.0

    @pytest.mark.asyncio
    async def test_moderate_drawdown(self, sample_order):
        """Drawdowns at roughly half of each limit -> valid, mid-range score."""
        protection = DrawdownProtection()  # limits: day 2%, week 5%, month 10%
        protection._calculate_drawdown = _drawdown_stub(
            day=-0.01, week=-0.02, month=-0.03
        )

        result = await protection.validate(sample_order)

        assert result.is_valid is True
        # max(0.01/0.02, 0.02/0.05, 0.03/0.10) = max(0.5, 0.4, 0.3) = 0.5
        assert result.risk_score == pytest.approx(0.5)
        assert 0.3 < result.risk_score < 0.7

    @pytest.mark.asyncio
    async def test_daily_drawdown_rejection(self, sample_order):
        """A daily drawdown beyond the 2% limit is rejected (score 1.0)."""
        protection = DrawdownProtection()
        protection._calculate_drawdown = _drawdown_stub(day=-0.025)  # 2.5% > 2%

        result = await protection.validate(sample_order)

        assert result.is_valid is False
        assert "daily drawdown" in result.reason.lower()
        assert result.risk_score == 1.0

    @pytest.mark.asyncio
    async def test_weekly_drawdown_rejection(self, sample_order):
        """Daily within limit but weekly beyond 5% -> rejected (score 0.95)."""
        protection = DrawdownProtection()
        protection._calculate_drawdown = _drawdown_stub(day=-0.01, week=-0.06)

        result = await protection.validate(sample_order)

        assert result.is_valid is False
        assert "weekly drawdown" in result.reason.lower()
        assert result.risk_score == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_monthly_drawdown_rejection(self, sample_order):
        """Daily/weekly within limit but monthly beyond 10% -> rejected (score 0.9)."""
        protection = DrawdownProtection()
        protection._calculate_drawdown = _drawdown_stub(
            day=-0.01, week=-0.02, month=-0.12
        )

        result = await protection.validate(sample_order)

        assert result.is_valid is False
        assert "monthly drawdown" in result.reason.lower()
        assert result.risk_score == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# Layer 3: CorrelationAnalyzer
# --------------------------------------------------------------------------- #
class TestCorrelationAnalyzer:
    """Correlation-with-existing-positions limit (max_correlation)."""

    @pytest.mark.asyncio
    async def test_no_open_positions(self, sample_order):
        """With no existing positions there is nothing to correlate -> valid, 0 risk."""
        analyzer = CorrelationAnalyzer()
        analyzer._get_current_positions = AsyncMock(return_value=[])

        result = await analyzer.validate(sample_order)

        assert result.is_valid is True
        assert result.risk_score == 0.0

    @pytest.mark.asyncio
    async def test_low_correlation(self, sample_order):
        """Correlation below max_correlation -> valid, scaled risk score."""
        analyzer = CorrelationAnalyzer()  # default max_correlation = 0.8
        analyzer._get_current_positions = AsyncMock(
            return_value=[{"symbol": "ETHUSDT", "quantity": 5.0, "value": 5000.0}]
        )
        analyzer._get_correlation = AsyncMock(return_value=0.3)

        result = await analyzer.validate(sample_order)

        assert result.is_valid is True
        assert result.risk_score == pytest.approx(0.375)  # 0.3 / 0.8
        assert result.risk_score < 0.5

    @pytest.mark.asyncio
    async def test_high_correlation_rejection(self, sample_order):
        """Correlation above max_correlation -> rejected, score equals correlation."""
        analyzer = CorrelationAnalyzer()
        analyzer._get_current_positions = AsyncMock(
            return_value=[{"symbol": "ETHUSDT", "quantity": 5.0, "value": 50000.0}]
        )
        analyzer._get_correlation = AsyncMock(return_value=0.95)  # > 0.8

        result = await analyzer.validate(sample_order)

        assert result.is_valid is False
        assert "correlation" in result.reason.lower()
        assert result.risk_score == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_correlation_warning(self, sample_order):
        """Correlation between 70% and 100% of the limit -> valid but warned."""
        analyzer = CorrelationAnalyzer()
        analyzer._get_current_positions = AsyncMock(
            return_value=[{"symbol": "ETHUSDT", "quantity": 5.0, "value": 5000.0}]
        )
        analyzer._get_correlation = AsyncMock(return_value=0.6)  # >0.56, <0.8

        result = await analyzer.validate(sample_order)

        assert result.is_valid is True
        assert result.risk_score == pytest.approx(0.75)  # 0.6 / 0.8
        assert len(result.warnings) >= 1


# --------------------------------------------------------------------------- #
# Layer 4: BlackSwanProtection
# --------------------------------------------------------------------------- #
class TestBlackSwanProtection:
    """Extreme-volatility / volume-spike detection."""

    @pytest.mark.asyncio
    async def test_normal_market_conditions(self, sample_order):
        """Current ≈ historical volatility and volume -> valid, low score."""
        protection = BlackSwanProtection()  # vol x3, volume x5 thresholds
        protection._get_current_volatility = AsyncMock(return_value=0.02)
        protection._get_historical_volatility = AsyncMock(return_value=0.02)
        protection._get_current_volume = AsyncMock(return_value=1000.0)
        protection._get_average_volume = AsyncMock(return_value=1000.0)

        result = await protection.validate(sample_order)

        assert result.is_valid is True
        # max(1.0/3.0, 1.0/5.0) = 0.333...
        assert result.risk_score == pytest.approx(1.0 / 3.0)
        assert result.risk_score < 0.5

    @pytest.mark.asyncio
    async def test_extreme_volatility_rejection(self, sample_order):
        """Volatility above 3x normal -> rejected (score 1.0)."""
        protection = BlackSwanProtection()
        protection._get_current_volatility = AsyncMock(return_value=0.10)  # 5x of 0.02
        protection._get_historical_volatility = AsyncMock(return_value=0.02)
        protection._get_current_volume = AsyncMock(return_value=1000.0)
        protection._get_average_volume = AsyncMock(return_value=1000.0)

        result = await protection.validate(sample_order)

        assert result.is_valid is False
        assert "volatility" in result.reason.lower()
        assert result.risk_score == 1.0

    @pytest.mark.asyncio
    async def test_volume_spike_rejection(self, sample_order):
        """Normal volatility but volume above 5x average -> rejected (score 0.9)."""
        protection = BlackSwanProtection()
        protection._get_current_volatility = AsyncMock(return_value=0.02)
        protection._get_historical_volatility = AsyncMock(return_value=0.02)
        protection._get_current_volume = AsyncMock(return_value=10000.0)  # 10x of 1000
        protection._get_average_volume = AsyncMock(return_value=1000.0)

        result = await protection.validate(sample_order)

        assert result.is_valid is False
        assert "volume" in result.reason.lower()
        assert result.risk_score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_elevated_volatility_warning(self, sample_order):
        """Volatility above 70% of the threshold -> valid but warned."""
        protection = BlackSwanProtection()
        protection._get_current_volatility = AsyncMock(return_value=0.05)  # 2.5x of 0.02
        protection._get_historical_volatility = AsyncMock(return_value=0.02)
        protection._get_current_volume = AsyncMock(return_value=1000.0)
        protection._get_average_volume = AsyncMock(return_value=1000.0)

        result = await protection.validate(sample_order)

        assert result.is_valid is True
        # 2.5x > 3.0*0.7 = 2.1 -> warning; score = max(2.5/3, 1/5) = 0.833...
        assert result.risk_score == pytest.approx(2.5 / 3.0)
        assert len(result.warnings) >= 1


# --------------------------------------------------------------------------- #
# Layer 5: CircuitBreaker
# --------------------------------------------------------------------------- #
class TestCircuitBreaker:
    """Trade-frequency circuit breaker with cooldown."""

    @pytest.mark.asyncio
    async def test_normal_operation(self, sample_order):
        """No recent trades and breaker not tripped -> valid, zero score."""
        breaker = CircuitBreaker()
        breaker._trade_timestamps = []  # isolate from shared class-level state

        result = await breaker.validate(sample_order)

        assert result.is_valid is True
        assert result.risk_score == 0.0

    @pytest.mark.asyncio
    async def test_active_circuit_breaker_blocks(self, sample_order):
        """While tripped and inside the cooldown window every order is rejected."""
        breaker = CircuitBreaker()
        breaker._trade_timestamps = []
        breaker.is_circuit_broken = True
        breaker.circuit_break_time = datetime.utcnow()  # just tripped, cooldown 300s

        result = await breaker.validate(sample_order)

        assert result.is_valid is False
        assert "circuit breaker" in result.reason.lower()
        assert result.risk_score == 1.0

    @pytest.mark.asyncio
    async def test_trips_on_trades_per_minute(self, sample_order):
        """Reaching max_trades_per_minute trips the breaker and rejects the order."""
        breaker = CircuitBreaker()  # default max_trades_per_minute = 10
        breaker._trade_timestamps = []
        for _ in range(breaker.max_trades_per_minute):
            breaker.record_trade()

        result = await breaker.validate(sample_order)

        assert result.is_valid is False
        assert "minute" in result.reason.lower()
        assert result.risk_score == 1.0
        # Side effect: the breaker is now latched open.
        assert breaker.is_circuit_broken is True

    @pytest.mark.asyncio
    async def test_resets_after_cooldown(self, sample_order):
        """Once the cooldown elapses the breaker auto-resets and orders pass again."""
        breaker = CircuitBreaker()
        breaker._trade_timestamps = []
        breaker.is_circuit_broken = True
        breaker.circuit_break_time = datetime.utcnow() - timedelta(seconds=400)  # >300

        result = await breaker.validate(sample_order)

        assert breaker.is_circuit_broken is False
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_record_trade_and_count_window(self):
        """record_trade / _get_trade_count honour the time window they are asked for."""
        breaker = CircuitBreaker()
        breaker._trade_timestamps = []

        for _ in range(3):
            breaker.record_trade()
        # A trade outside the 1-minute window but inside the 60-minute window.
        breaker._trade_timestamps.append(datetime.utcnow() - timedelta(minutes=5))

        count_hour = await breaker._get_trade_count(minutes=60)
        count_minute = await breaker._get_trade_count(minutes=1)

        assert count_hour == 4
        assert count_minute == 3


# --------------------------------------------------------------------------- #
# Integration: RiskManagementSystem.validate_order
# --------------------------------------------------------------------------- #
class TestRiskManagementSystem:
    """The orchestrator that runs all five layers in sequence."""

    @pytest.mark.asyncio
    async def test_multi_layer_validation_pass(self, sample_order, risk_manager):
        """An order that satisfies every layer is accepted."""
        _patch_system_layers(risk_manager)  # all-safe defaults

        result = await risk_manager.validate_order(sample_order)

        assert result.is_valid is True
        assert result.reason is None
        assert isinstance(result.warnings, list)

    @pytest.mark.asyncio
    async def test_multi_layer_validation_fail(self, risky_order, risk_manager):
        """An order rejected by a layer (position size) fails the whole pipeline."""
        _patch_system_layers(risk_manager)

        result = await risk_manager.validate_order(risky_order)

        assert result.is_valid is False
        assert result.reason and len(result.reason) > 0
        assert result.risk_score > 0.5

    @pytest.mark.asyncio
    async def test_risk_score_calculation(self, sample_order, risk_manager):
        """The aggregate risk score stays within the [0, 1] range."""
        _patch_system_layers(risk_manager)

        result = await risk_manager.validate_order(sample_order)

        assert 0.0 <= result.risk_score <= 1.0

    @pytest.mark.asyncio
    async def test_alert_sent_on_rejection(self, risky_order, risk_manager):
        """A rejection triggers exactly one risk alert for the offending order."""
        _patch_system_layers(risk_manager)

        alerts = []

        async def _capture_alert(order, result):
            alerts.append((order, result))

        risk_manager._send_alert = _capture_alert

        result = await risk_manager.validate_order(risky_order)

        assert result.is_valid is False
        assert len(alerts) == 1
        assert alerts[0][0] is risky_order
        assert alerts[0][1].is_valid is False

    @pytest.mark.asyncio
    async def test_warnings_aggregated_across_layers(self, sample_order, risk_manager):
        """Warnings from multiple passing layers are collected on the final result."""
        # Portfolio sized so the position lands at 8.4% (>70% of the 10% cap -> warn)
        # and volatility at 2.5x (>70% of the 3x cap -> warn).
        _patch_system_layers(
            risk_manager,
            portfolio=250_000.0,
            cur_vol=0.05,
            hist_vol=0.02,
        )

        result = await risk_manager.validate_order(sample_order)

        assert result.is_valid is True
        assert len(result.warnings) >= 2


# --------------------------------------------------------------------------- #
# Regression: risk-layer mutable state must be per-instance (not class-level)
# --------------------------------------------------------------------------- #
class TestRiskLayerStateIsolation:
    """迴歸測試：風控層可變狀態應為實例級別，不得跨實例共享。

    舊版 ``_trade_timestamps`` / ``_peak_values`` / ``_pnl_history`` 宣告在 class
    層級 → 所有實例（與測試）共用同一個 list/dict，造成狀態洩漏。
    """

    def test_circuit_breaker_timestamps_not_shared(self):
        cb1 = CircuitBreaker()
        cb2 = CircuitBreaker()
        cb1.record_trade()
        cb1.record_trade()
        assert len(cb1._trade_timestamps) == 2
        assert cb2._trade_timestamps == []                      # 未被 cb1 污染
        assert cb1._trade_timestamps is not cb2._trade_timestamps

    def test_drawdown_state_not_shared(self):
        d1 = DrawdownProtection()
        d2 = DrawdownProtection()
        d1._peak_values["day"] = 12345.0
        d1._pnl_history["day"].append(1.0)
        assert d2._peak_values["day"] == 0.0                   # d2 有獨立初始狀態
        assert d2._pnl_history["day"] == []
        assert d1._peak_values is not d2._peak_values
        assert d1._pnl_history is not d2._pnl_history
