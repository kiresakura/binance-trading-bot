"""Unit tests for the fault tolerance and recovery system.

These tests exercise the REAL current API of ``src.services.fault_tolerance``:

* The state-snapshot type is the ``StateSnapshotData`` dataclass (the name
  ``StateSnapshot`` in the module refers to a SQLAlchemy DB model, imported
  internally as ``StateSnapshotModel`` -- it is not the snapshot value object).
* ``verify_execution`` only treats the Binance order statuses
  ``FILLED`` / ``NEW`` / ``PARTIALLY_FILLED`` as valid (NOT ``"success"``).
* ``execute_with_failover`` returns ``None`` *only* while the engine is
  emergency-stopped; on exhausted retries it returns the last error dict.
  It tracks consecutive total failures and triggers ``emergency_shutdown``
  once ``max_consecutive_failures`` (default 3) is reached; any success resets
  the counter.

External I/O (the shared ``binance_client`` singleton, the cache and the
database) is mocked so no real network/DB calls happen.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.fault_tolerance import (
    ConnectionStatus,
    StateSnapshotData,
    StateManager,
    BinanceConnection,
    BinanceBackupConnection,
    FaultTolerantTradingEngine,
)

# The connections look ``binance_client`` up in the fault_tolerance module
# globals at call time, so this is the patch target that neutralises I/O.
BINANCE_PATH = "src.services.fault_tolerance.binance_client"


def _fake_binance(**overrides):
    """Build a fully-async stand-in for the global ``binance_client``."""
    bc = MagicMock(name="binance_client")
    bc.get_ticker_price = AsyncMock(
        return_value={"symbol": "BTCUSDT", "price": "42000.00"}
    )
    bc.create_order = AsyncMock(
        return_value={
            "orderId": 999,
            "status": "FILLED",
            "symbol": "BTCUSDT",
            "price": "42000.00",
        }
    )
    bc.get_open_orders = AsyncMock(return_value=[])
    bc.cancel_order = AsyncMock(return_value={"orderId": 1, "status": "CANCELED"})
    bc.get_balances = AsyncMock(
        return_value=[{"asset": "USDT", "free": "100000", "locked": "0", "total": "100000"}]
    )
    for key, value in overrides.items():
        setattr(bc, key, value)
    return bc


def _make_snapshot(**overrides) -> StateSnapshotData:
    """Construct a real ``StateSnapshotData`` value object for tests."""
    data = dict(
        timestamp=datetime.utcnow(),
        portfolio_state={"total_value": 100000.0},
        open_orders=[],
        positions=[],
        balance={"USDT": 100000.0},
        metadata={"version": "1.0"},
    )
    data.update(overrides)
    return StateSnapshotData(**data)


class TestStateSnapshotData:
    """Tests for the ``StateSnapshotData`` value object (the real snapshot type)."""

    def test_snapshot_to_dict(self):
        """``to_dict`` serialises the timestamp to an ISO string and keeps fields."""
        snapshot = _make_snapshot()

        snapshot_dict = snapshot.to_dict()

        assert set(snapshot_dict) >= {
            "timestamp",
            "portfolio_state",
            "open_orders",
            "positions",
            "balance",
            "metadata",
        }
        assert isinstance(snapshot_dict["timestamp"], str)
        assert snapshot_dict["balance"] == {"USDT": 100000.0}

    def test_snapshot_from_dict(self, mock_state_snapshot):
        """``from_dict`` rebuilds a snapshot, parsing the ISO timestamp back."""
        snapshot = StateSnapshotData.from_dict(mock_state_snapshot)

        assert isinstance(snapshot, StateSnapshotData)
        assert isinstance(snapshot.timestamp, datetime)
        assert snapshot.balance["USDT"] == 100000.0
        assert snapshot.positions[0]["symbol"] == "BTCUSDT"

    def test_snapshot_roundtrip(self):
        """A snapshot survives a to_dict/from_dict round-trip unchanged."""
        snapshot = _make_snapshot(metadata={"version": "1.0", "reason": "rt"})

        restored = StateSnapshotData.from_dict(snapshot.to_dict())

        assert restored.timestamp == snapshot.timestamp
        assert restored.portfolio_state == snapshot.portfolio_state
        assert restored.balance == snapshot.balance
        assert restored.metadata == snapshot.metadata

    def test_snapshot_checksum(self):
        """``compute_checksum`` is a stable sha256 hex digest sensitive to content."""
        snapshot = _make_snapshot()

        checksum = snapshot.compute_checksum()

        assert isinstance(checksum, str)
        assert len(checksum) == 64  # sha256 hex digest
        # Deterministic for identical content...
        assert snapshot.compute_checksum() == checksum
        # ...and different when the content changes.
        changed = _make_snapshot(
            timestamp=snapshot.timestamp, balance={"USDT": 999.0}
        )
        assert changed.compute_checksum() != checksum


class TestStateManager:
    """Tests for snapshot creation and restoration in ``StateManager``."""

    async def test_create_snapshot(self):
        """``create_snapshot`` gathers state, caches it, and records it as current."""
        manager = StateManager()

        # _db_initialized is False by default, so metadata persistence is skipped.
        manager._save_snapshot_to_cache = AsyncMock(return_value="state_snapshot:key")
        manager._get_portfolio_state = AsyncMock(return_value={"total_value": 100000.0})
        manager._get_open_orders = AsyncMock(return_value=[])
        manager._get_positions = AsyncMock(return_value=[])
        manager._get_balance = AsyncMock(return_value={"USDT": 100000.0})

        snapshot = await manager.create_snapshot(reason="unit_test")

        assert isinstance(snapshot, StateSnapshotData)
        assert snapshot.portfolio_state["total_value"] == 100000.0
        assert snapshot.metadata["reason"] == "unit_test"
        assert manager.current_snapshot is snapshot
        manager._save_snapshot_to_cache.assert_awaited_once()

    async def test_restore_snapshot(self):
        """``restore_snapshot`` cancels orders then restores portfolio and positions."""
        manager = StateManager()
        manager._cancel_all_orders = AsyncMock()
        manager._restore_portfolio_state = AsyncMock()
        manager._restore_positions = AsyncMock()

        snapshot = _make_snapshot(
            positions=[{"symbol": "BTCUSDT", "quantity": 1.0}],
        )

        await manager.restore_snapshot(snapshot)

        manager._cancel_all_orders.assert_awaited_once()
        manager._restore_portfolio_state.assert_awaited_once_with(
            snapshot.portfolio_state
        )
        manager._restore_positions.assert_awaited_once_with(snapshot.positions)


class TestBinanceConnection:
    """Tests for the primary ``BinanceConnection``."""

    async def test_connect_success(self):
        """A successful ticker probe marks the connection CONNECTED."""
        fake = _fake_binance()
        with patch(BINANCE_PATH, fake):
            connection = BinanceConnection()
            await connection.connect()

        assert connection.status == ConnectionStatus.CONNECTED
        fake.get_ticker_price.assert_awaited_once_with("BTCUSDT")

    async def test_connect_failure(self):
        """A failing probe marks the connection FAILED and re-raises."""
        fake = _fake_binance(
            get_ticker_price=AsyncMock(side_effect=RuntimeError("network down"))
        )
        connection = BinanceConnection()
        with patch(BINANCE_PATH, fake):
            with pytest.raises(RuntimeError):
                await connection.connect()

        assert connection.status == ConnectionStatus.FAILED

    async def test_execute_order(self):
        """Executing while connected delegates to ``binance_client.create_order``."""
        fake = _fake_binance()
        with patch(BINANCE_PATH, fake):
            connection = BinanceConnection()
            await connection.connect()

            order = {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.5}
            result = await connection.execute(order)

        # Returns whatever the exchange client returned, unchanged.
        assert result["status"] == "FILLED"
        assert result["orderId"] == 999
        fake.create_order.assert_awaited_once_with(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=Decimal("0.5"),
            price=None,
        )

    async def test_execute_without_connection(self):
        """Executing while not connected raises ``ConnectionError``."""
        connection = BinanceConnection()
        order = {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.5}

        with pytest.raises(ConnectionError):
            await connection.execute(order)

    async def test_disconnect(self):
        """``disconnect`` returns the connection to DISCONNECTED."""
        fake = _fake_binance()
        with patch(BINANCE_PATH, fake):
            connection = BinanceConnection()
            await connection.connect()
            assert connection.status == ConnectionStatus.CONNECTED

            await connection.disconnect()

        assert connection.status == ConnectionStatus.DISCONNECTED


class TestBinanceBackupConnection:
    """Tests for the backup ``BinanceBackupConnection``."""

    async def test_backup_connect(self):
        """Backup connect marks the connection CONNECTED."""
        backup = BinanceBackupConnection()

        await backup.connect()

        assert backup.status == ConnectionStatus.CONNECTED

    async def test_backup_execute(self):
        """Backup execution delegates to ``binance_client.create_order``."""
        fake = _fake_binance(
            create_order=AsyncMock(
                return_value={"orderId": "backup_1", "status": "NEW"}
            )
        )
        with patch(BINANCE_PATH, fake):
            backup = BinanceBackupConnection()
            await backup.connect()

            order = {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.5}
            result = await backup.execute(order)

        assert result["orderId"] == "backup_1"
        assert result["status"] == "NEW"
        fake.create_order.assert_awaited_once()


class TestFaultTolerantTradingEngine:
    """Tests for the orchestrating ``FaultTolerantTradingEngine``."""

    async def test_successful_execution(self, fault_tolerant_engine):
        """A verified primary execution returns immediately without failover."""
        engine = fault_tolerant_engine
        # Pretend we are already connected so the loop is reached directly.
        engine.primary_connection.status = ConnectionStatus.CONNECTED
        engine.primary_connection.execute = AsyncMock(
            return_value={"orderId": "test_123", "status": "FILLED", "price": "42000.0"}
        )
        engine.backup_connection.connect = AsyncMock()
        engine.backup_connection.execute = AsyncMock()
        engine.state_manager.create_snapshot = AsyncMock(return_value=_make_snapshot())

        order = {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.5}
        result = await engine.execute_with_failover(order)

        assert result is not None
        assert result["status"] == "FILLED"
        assert result["orderId"] == "test_123"
        engine.primary_connection.execute.assert_awaited_once()
        # Verified on the first try -> backup never touched.
        engine.backup_connection.execute.assert_not_awaited()

    async def test_failover_to_backup(self):
        """A primary ConnectionError fails over to the backup connection."""
        engine = FaultTolerantTradingEngine()
        engine.primary_connection.status = ConnectionStatus.CONNECTED
        engine.primary_connection.execute = AsyncMock(
            side_effect=ConnectionError("Primary failed")
        )
        engine.backup_connection.connect = AsyncMock()
        engine.backup_connection.execute = AsyncMock(
            return_value={"orderId": "backup_123", "status": "NEW"}
        )
        engine.state_manager.create_snapshot = AsyncMock(return_value=_make_snapshot())

        order = {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.5}
        result = await engine.execute_with_failover(order, max_retries=1)

        assert result is not None
        assert result["orderId"] == "backup_123"
        engine.backup_connection.connect.assert_awaited()
        engine.backup_connection.execute.assert_awaited()

    async def test_execution_timeout_fails_over(self):
        """A timed-out primary execution is routed to the backup connection."""
        engine = FaultTolerantTradingEngine()
        engine.primary_connection.status = ConnectionStatus.CONNECTED
        # Simulate the timeout that _execute_with_timeout would raise.
        engine.primary_connection.execute = AsyncMock(side_effect=asyncio.TimeoutError())
        engine.backup_connection.connect = AsyncMock()
        engine.backup_connection.execute = AsyncMock(
            return_value={"orderId": "backup_123", "status": "FILLED"}
        )
        engine.state_manager.create_snapshot = AsyncMock(return_value=_make_snapshot())

        order = {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.5}
        result = await engine.execute_with_failover(order, max_retries=1)

        assert result is not None
        assert result["orderId"] == "backup_123"
        engine.backup_connection.connect.assert_awaited()

    async def test_execute_with_timeout_helper(self):
        """``_execute_with_timeout`` raises ``TimeoutError`` when the coro is too slow."""
        engine = FaultTolerantTradingEngine()

        async def slow():
            await asyncio.sleep(1)
            return {"status": "FILLED"}

        with pytest.raises(asyncio.TimeoutError):
            await engine._execute_with_timeout(slow(), timeout=0.05)

    async def test_all_connections_fail(self):
        """A SINGLE total failure returns the last error dict and restores the
        snapshot, but does NOT yet trigger emergency stop (below the consecutive
        threshold). Repeated failures are covered by the next test.
        """
        engine = FaultTolerantTradingEngine()
        engine.primary_connection.status = ConnectionStatus.CONNECTED
        engine.primary_connection.execute = AsyncMock(
            side_effect=ConnectionError("Primary failed")
        )
        engine.backup_connection.connect = AsyncMock()
        engine.backup_connection.execute = AsyncMock(
            side_effect=ConnectionError("Backup failed")
        )
        engine.state_manager.create_snapshot = AsyncMock(return_value=_make_snapshot())
        engine.state_manager.restore_snapshot = AsyncMock()
        engine.emergency_shutdown = AsyncMock()

        order = {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.5}
        result = await engine.execute_with_failover(order, max_retries=2)

        assert result is not None
        assert result["status"] == "error"
        engine.state_manager.restore_snapshot.assert_awaited()
        # 單筆失敗（連續=1）不應停機
        engine.emergency_shutdown.assert_not_awaited()
        assert engine._consecutive_failures == 1

    async def test_consecutive_failures_trigger_emergency_stop(self):
        """連續達到 max_consecutive_failures 次完全失敗 → 自動緊急停止。

        迴歸測試：舊版重試耗盡只回 error dict，永不觸發 emergency_shutdown。
        """
        engine = FaultTolerantTradingEngine()
        engine.primary_connection.status = ConnectionStatus.CONNECTED
        engine.primary_connection.execute = AsyncMock(
            side_effect=ConnectionError("Primary failed")
        )
        engine.backup_connection.connect = AsyncMock()
        engine.backup_connection.execute = AsyncMock(
            side_effect=ConnectionError("Backup failed")
        )
        engine.state_manager.create_snapshot = AsyncMock(return_value=_make_snapshot())
        engine.state_manager.restore_snapshot = AsyncMock()
        engine.emergency_shutdown = AsyncMock()
        engine.max_consecutive_failures = 3

        order = {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.5}
        # 前兩次完全失敗：不停機
        for _ in range(2):
            await engine.execute_with_failover(order, max_retries=1)
        engine.emergency_shutdown.assert_not_awaited()
        # 第三次完全失敗：觸發緊急停止
        await engine.execute_with_failover(order, max_retries=1)
        engine.emergency_shutdown.assert_awaited_once()

    async def test_consecutive_failures_reset_on_success(self):
        """任一次成功即把連續失敗計數歸零。"""
        engine = FaultTolerantTradingEngine()
        engine.primary_connection.status = ConnectionStatus.CONNECTED
        engine.state_manager.create_snapshot = AsyncMock(return_value=_make_snapshot())
        engine._record_order_to_db = AsyncMock(return_value=None)
        engine._update_order_status = AsyncMock()
        engine.emergency_shutdown = AsyncMock()

        order = {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.5}

        # 先一次失敗（連續=1）
        engine.primary_connection.execute = AsyncMock(side_effect=ConnectionError("x"))
        engine.backup_connection.connect = AsyncMock()
        engine.backup_connection.execute = AsyncMock(side_effect=ConnectionError("y"))
        await engine.execute_with_failover(order, max_retries=1)
        assert engine._consecutive_failures == 1

        # 再一次成功 → 計數歸零
        engine.primary_connection.execute = AsyncMock(
            return_value={"orderId": "ok", "status": "FILLED"}
        )
        result = await engine.execute_with_failover(order, max_retries=1)
        assert result["status"] == "FILLED"
        assert engine._consecutive_failures == 0

    async def test_state_restoration_on_failure(self):
        """The pre-execution snapshot is restored when the backup also fails."""
        engine = FaultTolerantTradingEngine()
        engine.primary_connection.status = ConnectionStatus.CONNECTED
        engine.primary_connection.execute = AsyncMock(
            side_effect=ConnectionError("Primary down")
        )
        engine.backup_connection.connect = AsyncMock()
        engine.backup_connection.execute = AsyncMock(
            side_effect=Exception("Backup down")
        )
        snapshot = _make_snapshot()
        engine.state_manager.create_snapshot = AsyncMock(return_value=snapshot)
        engine.state_manager.restore_snapshot = AsyncMock()

        order = {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.5}
        await engine.execute_with_failover(order, max_retries=1)

        engine.state_manager.restore_snapshot.assert_awaited_once_with(snapshot)

    async def test_verify_execution_success(self, fault_tolerant_engine):
        """Valid Binance order statuses verify as successful."""
        for status in ("FILLED", "NEW", "PARTIALLY_FILLED"):
            result = {"status": status, "orderId": "123", "price": 42000}
            assert await fault_tolerant_engine.verify_execution(result) is True

    async def test_verify_execution_failure(self, fault_tolerant_engine):
        """Empty, error-coded and non-trading statuses verify as failures."""
        engine = fault_tolerant_engine
        assert await engine.verify_execution({}) is False
        assert await engine.verify_execution({"status": "failed"}) is False
        assert await engine.verify_execution({"status": "error", "msg": "x"}) is False
        assert await engine.verify_execution({"code": -2010, "msg": "insufficient"}) is False
        # 'success' is NOT a Binance order status, so it is treated as invalid.
        assert await engine.verify_execution({"status": "success"}) is False

    async def test_emergency_shutdown(self):
        """Emergency shutdown stops the engine, cancels orders and disconnects."""
        engine = FaultTolerantTradingEngine()
        engine.state_manager.create_snapshot = AsyncMock(
            return_value=_make_snapshot(positions=[{"symbol": "BTCUSDT"}])
        )
        engine.primary_connection.disconnect = AsyncMock()
        engine._send_emergency_notification = AsyncMock()

        open_order = {"symbol": "BTCUSDT", "orderId": 555}
        fake = _fake_binance(
            get_open_orders=AsyncMock(return_value=[open_order]),
            cancel_order=AsyncMock(),
        )

        with patch(BINANCE_PATH, fake):
            await engine.emergency_shutdown("Risk breach")

        assert engine.is_emergency_stopped is True
        engine.state_manager.create_snapshot.assert_awaited_once_with(reason="emergency")
        fake.get_open_orders.assert_awaited()
        fake.cancel_order.assert_awaited_once_with(symbol="BTCUSDT", order_id=555)
        engine.primary_connection.disconnect.assert_awaited_once()
        engine._send_emergency_notification.assert_awaited_once_with("Risk breach")

    async def test_no_trading_during_emergency(self):
        """No order executes while the engine is emergency-stopped (returns None)."""
        engine = FaultTolerantTradingEngine()
        engine.is_emergency_stopped = True
        # Guard: even if reached, execution would be observable.
        engine.primary_connection.execute = AsyncMock()

        order = {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.5}
        result = await engine.execute_with_failover(order)

        assert result is None
        engine.primary_connection.execute.assert_not_awaited()

    async def test_resume_trading(self):
        """Resuming clears the emergency flag and reconnects the primary."""
        engine = FaultTolerantTradingEngine()
        engine.is_emergency_stopped = True
        engine.primary_connection.connect = AsyncMock()

        await engine.resume_trading()

        assert engine.is_emergency_stopped is False
        engine.primary_connection.connect.assert_awaited_once()

    async def test_resume_trading_noop_when_active(self):
        """Resuming while already active is a no-op (no reconnect)."""
        engine = FaultTolerantTradingEngine()
        engine.is_emergency_stopped = False
        engine.primary_connection.connect = AsyncMock()

        await engine.resume_trading()

        assert engine.is_emergency_stopped is False
        engine.primary_connection.connect.assert_not_awaited()

    async def test_retry_mechanism(self):
        """The retry loop keeps trying the primary until it returns a verified fill."""
        engine = FaultTolerantTradingEngine()
        engine.primary_connection.status = ConnectionStatus.CONNECTED

        call_count = 0

        async def intermittent_execute(order):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return {"orderId": "123", "status": "FILLED"}

        engine.primary_connection.execute = intermittent_execute
        # Backup also fails so the loop falls through to the next primary retry.
        engine.backup_connection.connect = AsyncMock()
        engine.backup_connection.execute = AsyncMock(
            side_effect=ConnectionError("Backup down")
        )
        engine.state_manager.create_snapshot = AsyncMock(return_value=_make_snapshot())
        engine.state_manager.restore_snapshot = AsyncMock()

        order = {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.5}
        result = await engine.execute_with_failover(order, max_retries=3)

        assert result is not None
        assert result["status"] == "FILLED"
        assert call_count == 3
