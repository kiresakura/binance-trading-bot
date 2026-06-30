"""Unit tests for the monitoring / metrics system.

These tests exercise the REAL ``MetricsCollector`` API in
``src/services/monitoring.py``.  A few notes about how they assert:

* ``MetricsCollector`` builds its OWN ``CollectorRegistry`` per instance
  (``self.registry``); it never touches the process-global
  ``prometheus_client.REGISTRY``.  Every assertion therefore reads values
  through ``collector.registry.get_sample_value(...)`` rather than the global
  registry, and the fixture handing out a fresh collector per test keeps the
  tests fully isolated (no duplicate-registration problems).
* Counters created with names already ending in ``_total`` keep that exact
  sample name (``trading_bot_trades_total``).  A counter whose name does NOT
  end in ``_total`` gets the suffix appended on export.  (Note: realized PnL is
  a Gauge, not a Counter, so it stays ``trading_bot_realized_pnl_usdt`` with no
  ``_total`` suffix and may be negative.)
* Histograms expose ``<name>_count`` / ``<name>_sum`` / ``<name>_bucket`` samples.
"""

import asyncio

import pytest
from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
)

from src.services.monitoring import MetricsCollector


def metric_value(collector, name, **labels):
    """Read a single Prometheus sample value from the collector's registry."""
    return collector.registry.get_sample_value(name, labels or None)


class TestMetricsCollector:
    """Test Prometheus metrics collection against the current API."""

    def test_metrics_collector_initialization(self, metrics_collector):
        """A fresh collector owns a private registry and registers its metrics."""
        # Private, per-instance registry (not the global one).
        assert isinstance(metrics_collector.registry, CollectorRegistry)

        # Core metric objects exist with the expected Prometheus types.
        assert isinstance(metrics_collector.trades_total, Counter)
        assert isinstance(metrics_collector.trade_latency, Histogram)
        assert isinstance(metrics_collector.current_balance, Gauge)
        assert isinstance(metrics_collector.sharpe_ratio, Gauge)
        assert isinstance(metrics_collector.risk_score, Gauge)

        # Unlabeled gauges + app info are exported immediately after init.
        exposition = metrics_collector.get_metrics().decode()
        assert "trading_bot_balance_usdt" in exposition
        assert "trading_bot_sharpe_ratio" in exposition
        assert "trading_bot_risk_score" in exposition
        assert "trading_bot_app_info" in exposition

    def test_record_trade(self, metrics_collector):
        """record_trade bumps the trade counter, volume and latency histogram."""
        labels = dict(symbol="BTCUSDT", side="BUY", status="filled")

        # Label child does not exist until the first record.
        assert metric_value(metrics_collector, "trading_bot_trades_total", **labels) is None

        metrics_collector.record_trade(
            symbol="BTCUSDT",
            side="BUY",
            status="filled",
            volume=21000.0,
            latency=0.5,
            slippage=0.02,
        )

        assert metric_value(metrics_collector, "trading_bot_trades_total", **labels) == 1.0
        assert metric_value(
            metrics_collector, "trading_bot_trade_volume_total", symbol="BTCUSDT"
        ) == 21000.0
        assert metric_value(
            metrics_collector,
            "trading_bot_trade_latency_seconds_count",
            symbol="BTCUSDT",
            order_type="MARKET",
        ) == 1.0
        assert metric_value(
            metrics_collector,
            "trading_bot_trade_latency_seconds_sum",
            symbol="BTCUSDT",
            order_type="MARKET",
        ) == 0.5

        # Counters accumulate across calls.
        metrics_collector.record_trade(
            symbol="BTCUSDT",
            side="BUY",
            status="filled",
            volume=1000.0,
            latency=0.2,
            slippage=0.0,
        )
        assert metric_value(metrics_collector, "trading_bot_trades_total", **labels) == 2.0
        assert metric_value(
            metrics_collector, "trading_bot_trade_volume_total", symbol="BTCUSDT"
        ) == 22000.0

    def test_record_trade_latency(self, metrics_collector):
        """Trade execution latency is observed into the latency histogram."""
        latencies = [0.1, 0.3, 0.7]
        for lat in latencies:
            metrics_collector.record_trade(
                symbol="BTCUSDT",
                side="BUY",
                status="filled",
                volume=100.0,
                latency=lat,
                slippage=0.0,
            )

        assert metric_value(
            metrics_collector,
            "trading_bot_trade_latency_seconds_count",
            symbol="BTCUSDT",
            order_type="MARKET",
        ) == len(latencies)
        assert metric_value(
            metrics_collector,
            "trading_bot_trade_latency_seconds_sum",
            symbol="BTCUSDT",
            order_type="MARKET",
        ) == pytest.approx(sum(latencies))

    def test_record_slippage(self, metrics_collector):
        """Slippage is only observed when strictly positive (the module's branch)."""
        # slippage == 0 -> histogram child is never created.
        metrics_collector.record_trade(
            symbol="BTCUSDT",
            side="BUY",
            status="filled",
            volume=100.0,
            latency=0.1,
            slippage=0.0,
        )
        assert metric_value(
            metrics_collector, "trading_bot_slippage_percent_count", symbol="BTCUSDT"
        ) is None

        # slippage > 0 -> observed.
        metrics_collector.record_trade(
            symbol="BTCUSDT",
            side="BUY",
            status="filled",
            volume=100.0,
            latency=0.1,
            slippage=0.02,
        )
        assert metric_value(
            metrics_collector, "trading_bot_slippage_percent_count", symbol="BTCUSDT"
        ) == 1.0
        assert metric_value(
            metrics_collector, "trading_bot_slippage_percent_sum", symbol="BTCUSDT"
        ) == pytest.approx(0.02)

    def test_update_balance(self, metrics_collector):
        """Balance + unrealized PnL gauges are driven by update_portfolio_metrics."""
        metrics_collector.update_portfolio_metrics(
            balance=105000.0,
            unrealized_pnl=5000.0,
            position_count=3,
            drawdown=0.05,
            sharpe_ratio=2.5,
        )
        assert metric_value(metrics_collector, "trading_bot_balance_usdt") == 105000.0
        assert metric_value(metrics_collector, "trading_bot_unrealized_pnl_usdt") == 5000.0

    def test_update_portfolio_metrics(self, metrics_collector):
        """All five portfolio gauges reflect the supplied values."""
        metrics_collector.update_portfolio_metrics(
            balance=110000.0,
            unrealized_pnl=-250.0,
            position_count=7,
            drawdown=0.12,
            sharpe_ratio=1.8,
        )
        assert metric_value(metrics_collector, "trading_bot_balance_usdt") == 110000.0
        assert metric_value(metrics_collector, "trading_bot_unrealized_pnl_usdt") == -250.0
        assert metric_value(metrics_collector, "trading_bot_open_positions") == 7.0
        assert metric_value(metrics_collector, "trading_bot_drawdown_percent") == 0.12
        assert metric_value(metrics_collector, "trading_bot_sharpe_ratio") == 1.8

    def test_record_realized_pnl_accumulates_and_allows_losses(self, metrics_collector):
        """Realized PnL is a signed cumulative Gauge: wins add, losses subtract.

        迴歸測試：先前用 Counter，虧損（負值）會丟 ValueError；改 Gauge 後負值正常。
        """
        metrics_collector.record_realized_pnl(100.0)
        metrics_collector.record_realized_pnl(50.0)
        # 一筆虧損（負值）——舊版 Counter 會在此丟 ValueError
        metrics_collector.record_realized_pnl(-200.0)
        assert metric_value(
            metrics_collector, "trading_bot_realized_pnl_usdt"
        ) == pytest.approx(-50.0)

    def test_update_position_value(self, metrics_collector):
        """Position value is a per-symbol gauge that holds the latest value."""
        metrics_collector.update_position_value("BTCUSDT", 42000.0)
        assert metric_value(
            metrics_collector, "trading_bot_position_value_usdt", symbol="BTCUSDT"
        ) == 42000.0

        metrics_collector.update_position_value("BTCUSDT", 50000.0)
        assert metric_value(
            metrics_collector, "trading_bot_position_value_usdt", symbol="BTCUSDT"
        ) == 50000.0

    def test_update_risk_score(self, metrics_collector):
        """Risk score gauge holds the most recently set value."""
        metrics_collector.update_risk_score(0.35)
        assert metric_value(metrics_collector, "trading_bot_risk_score") == 0.35

        metrics_collector.update_risk_score(0.7)
        assert metric_value(metrics_collector, "trading_bot_risk_score") == 0.7

    def test_record_risk_rejection(self, metrics_collector):
        """Risk rejections are counted per reason label."""
        metrics_collector.record_risk_rejection("position_size_exceeded")
        metrics_collector.record_risk_rejection("position_size_exceeded")
        metrics_collector.record_risk_rejection("max_drawdown")

        assert metric_value(
            metrics_collector,
            "trading_bot_risk_rejections_total",
            reason="position_size_exceeded",
        ) == 2.0
        assert metric_value(
            metrics_collector, "trading_bot_risk_rejections_total", reason="max_drawdown"
        ) == 1.0

    def test_record_error(self, metrics_collector):
        """Errors are counted per (component, error_type) label pair."""
        metrics_collector.record_error("order_executor", "timeout")
        metrics_collector.record_error("order_executor", "validation_error")

        assert metric_value(
            metrics_collector,
            "trading_bot_errors_total",
            component="order_executor",
            error_type="timeout",
        ) == 1.0
        assert metric_value(
            metrics_collector,
            "trading_bot_errors_total",
            component="order_executor",
            error_type="validation_error",
        ) == 1.0

    def test_record_api_request(self, metrics_collector):
        """API requests increment a counter and observe a latency histogram."""
        metrics_collector.record_api_request(
            endpoint="/api/v1/orders", method="POST", status_code=200, latency=0.3
        )

        assert metric_value(
            metrics_collector,
            "trading_bot_api_requests_total",
            endpoint="/api/v1/orders",
            method="POST",
            status_code="200",
        ) == 1.0
        assert metric_value(
            metrics_collector,
            "trading_bot_api_latency_seconds_count",
            endpoint="/api/v1/orders",
            method="POST",
        ) == 1.0
        assert metric_value(
            metrics_collector,
            "trading_bot_api_latency_seconds_sum",
            endpoint="/api/v1/orders",
            method="POST",
        ) == pytest.approx(0.3)

    def test_update_system_metrics(self, metrics_collector):
        """update_system_metrics reads psutil and updates the resource gauges."""
        # Defaults before the call.
        assert metric_value(metrics_collector, "trading_bot_memory_usage_bytes") == 0.0

        metrics_collector.update_system_metrics()

        mem_bytes = metric_value(metrics_collector, "trading_bot_memory_usage_bytes")
        assert mem_bytes is not None and mem_bytes > 0  # real, non-default reading

        mem_pct = metric_value(metrics_collector, "trading_bot_memory_usage_percent")
        assert mem_pct is not None and 0 < mem_pct <= 100

        cpu_pct = metric_value(metrics_collector, "trading_bot_cpu_usage_percent")
        # CPU can legitimately read 0.0 over a short sampling interval.
        assert cpu_pct is not None and cpu_pct >= 0

    def test_record_websocket_status(self, metrics_collector):
        """WebSocket status gauge encodes connected=1 / disconnected=0."""
        endpoint = "wss://stream.binance.com/ws"

        metrics_collector.record_websocket_status(endpoint, True)
        assert metric_value(
            metrics_collector, "trading_bot_websocket_connected", endpoint=endpoint
        ) == 1.0

        metrics_collector.record_websocket_status(endpoint, False)
        assert metric_value(
            metrics_collector, "trading_bot_websocket_connected", endpoint=endpoint
        ) == 0.0

    def test_record_websocket_latency(self, metrics_collector):
        """WebSocket latency is observed into a per-endpoint histogram (ms)."""
        endpoint = "wss://stream.binance.com/ws"
        metrics_collector.record_websocket_latency(endpoint, 12.5)

        assert metric_value(
            metrics_collector, "trading_bot_websocket_latency_ms_count", endpoint=endpoint
        ) == 1.0
        assert metric_value(
            metrics_collector, "trading_bot_websocket_latency_ms_sum", endpoint=endpoint
        ) == pytest.approx(12.5)

    def test_open_positions_gauge_up_and_down(self, metrics_collector):
        """The open-positions gauge can move up and down via portfolio updates."""
        metrics_collector.update_portfolio_metrics(
            balance=100000.0, unrealized_pnl=0.0, position_count=5,
            drawdown=0.0, sharpe_ratio=1.0,
        )
        assert metric_value(metrics_collector, "trading_bot_open_positions") == 5.0

        metrics_collector.update_portfolio_metrics(
            balance=100000.0, unrealized_pnl=0.0, position_count=2,
            drawdown=0.0, sharpe_ratio=1.0,
        )
        assert metric_value(metrics_collector, "trading_bot_open_positions") == 2.0

    def test_record_strategy_signal(self, metrics_collector):
        """Strategy signals are counted per (strategy, signal_type)."""
        metrics_collector.record_strategy_signal("momentum_v1", "BUY")
        metrics_collector.record_strategy_signal("momentum_v1", "BUY")

        assert metric_value(
            metrics_collector,
            "trading_bot_strategy_signals_total",
            strategy="momentum_v1",
            signal_type="BUY",
        ) == 2.0

    def test_update_strategy_performance(self, metrics_collector):
        """Strategy return gauge holds the latest per-strategy value."""
        metrics_collector.update_strategy_performance("momentum_v1", 2.5)
        assert metric_value(
            metrics_collector, "trading_bot_strategy_return_percent", strategy="momentum_v1"
        ) == 2.5

        metrics_collector.update_strategy_performance("mean_reversion", -1.0)
        assert metric_value(
            metrics_collector,
            "trading_bot_strategy_return_percent",
            strategy="mean_reversion",
        ) == -1.0

    def test_multiple_symbols(self, metrics_collector):
        """Trades for distinct symbols are tracked independently."""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        for symbol in symbols:
            metrics_collector.record_trade(
                symbol=symbol,
                side="BUY",
                status="filled",
                volume=1000.0,
                latency=0.1,
                slippage=0.0,
            )

        for symbol in symbols:
            assert metric_value(
                metrics_collector,
                "trading_bot_trades_total",
                symbol=symbol,
                side="BUY",
                status="filled",
            ) == 1.0
            assert metric_value(
                metrics_collector, "trading_bot_trade_volume_total", symbol=symbol
            ) == 1000.0

    def test_concurrent_metric_updates(self, metrics_collector):
        """Many sequential async updates accumulate without losing data."""
        async def update_metrics():
            for i in range(10):
                metrics_collector.record_trade(
                    symbol="BTCUSDT",
                    side="BUY",
                    status="filled",
                    volume=100.0,
                    latency=0.1,
                    slippage=0.0,
                )
                metrics_collector.update_portfolio_metrics(
                    balance=100000.0 + i * 100,
                    unrealized_pnl=0.0,
                    position_count=1,
                    drawdown=0.0,
                    sharpe_ratio=1.0,
                )

        asyncio.run(update_metrics())

        assert metric_value(
            metrics_collector,
            "trading_bot_trades_total",
            symbol="BTCUSDT",
            side="BUY",
            status="filled",
        ) == 10.0
        # Last balance written wins (i == 9 -> 100900).
        assert metric_value(metrics_collector, "trading_bot_balance_usdt") == 100900.0

    def test_trade_volume_tracking(self, metrics_collector):
        """Trade volume counter accumulates per symbol."""
        metrics_collector.record_trade(
            symbol="BTCUSDT", side="BUY", status="filled",
            volume=21000.0, latency=0.1, slippage=0.0,
        )
        metrics_collector.record_trade(
            symbol="BTCUSDT", side="SELL", status="filled",
            volume=9000.0, latency=0.1, slippage=0.0,
        )
        assert metric_value(
            metrics_collector, "trading_bot_trade_volume_total", symbol="BTCUSDT"
        ) == 30000.0

    def test_metrics_labels(self, metrics_collector):
        """Distinct label combinations are tracked as independent series."""
        for status in ["filled", "partial", "cancelled"]:
            metrics_collector.record_trade(
                symbol="BTCUSDT", side="BUY", status=status,
                volume=500.0, latency=0.1, slippage=0.0,
            )
        for status in ["filled", "partial", "cancelled"]:
            assert metric_value(
                metrics_collector,
                "trading_bot_trades_total",
                symbol="BTCUSDT",
                side="BUY",
                status=status,
            ) == 1.0

        for error_type in ["timeout", "invalid_request", "rate_limit"]:
            metrics_collector.record_error("/api/v1/orders", error_type)
        for error_type in ["timeout", "invalid_request", "rate_limit"]:
            assert metric_value(
                metrics_collector,
                "trading_bot_errors_total",
                component="/api/v1/orders",
                error_type=error_type,
            ) == 1.0

    def test_histogram_buckets(self, metrics_collector):
        """Latency observations fall into the expected cumulative buckets."""
        latencies = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
        for lat in latencies:
            metrics_collector.record_trade(
                symbol="BTCUSDT", side="BUY", status="filled",
                volume=100.0, latency=lat, slippage=0.0,
            )

        base = dict(symbol="BTCUSDT", order_type="MARKET")
        assert metric_value(
            metrics_collector, "trading_bot_trade_latency_seconds_count", **base
        ) == len(latencies)

        # Cumulative bucket counts (<= le).
        assert metric_value(
            metrics_collector, "trading_bot_trade_latency_seconds_bucket", le="0.01", **base
        ) == 2  # 0.001, 0.01
        assert metric_value(
            metrics_collector, "trading_bot_trade_latency_seconds_bucket", le="0.1", **base
        ) == 3  # + 0.1
        assert metric_value(
            metrics_collector, "trading_bot_trade_latency_seconds_bucket", le="+Inf", **base
        ) == 6  # all

    def test_gauge_increase_decrease(self, metrics_collector):
        """A gauge reflects the latest value and can both rise and fall."""
        def set_balance(value):
            metrics_collector.update_portfolio_metrics(
                balance=value, unrealized_pnl=0.0, position_count=1,
                drawdown=0.0, sharpe_ratio=1.0,
            )

        set_balance(100000.0)
        assert metric_value(metrics_collector, "trading_bot_balance_usdt") == 100000.0

        set_balance(95000.0)
        assert metric_value(metrics_collector, "trading_bot_balance_usdt") == 95000.0

        set_balance(105000.0)
        assert metric_value(metrics_collector, "trading_bot_balance_usdt") == 105000.0

    def test_prometheus_registry(self, metrics_collector):
        """Metrics live in the collector's private registry, not the global one."""
        # Populate a labeled metric so it is exported.
        metrics_collector.record_trade(
            symbol="BTCUSDT", side="BUY", status="filled",
            volume=1000.0, latency=0.1, slippage=0.0,
        )
        metrics_collector.update_portfolio_metrics(
            balance=100000.0, unrealized_pnl=0.0, position_count=1,
            drawdown=0.0, sharpe_ratio=2.5,
        )

        exposition = metrics_collector.get_metrics().decode()
        expected = [
            "trading_bot_trades_total",
            "trading_bot_balance_usdt",
            "trading_bot_sharpe_ratio",
            "trading_bot_risk_score",
        ]
        for name in expected:
            assert name in exposition

        # Isolation: these metrics are NOT registered on the process-global registry.
        assert REGISTRY.get_sample_value("trading_bot_balance_usdt") is None
        assert metrics_collector.registry.get_sample_value(
            "trading_bot_balance_usdt"
        ) == 100000.0
