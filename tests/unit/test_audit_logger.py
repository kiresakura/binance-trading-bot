"""Unit tests for the immutable audit logging system.

Target module: src/services/audit_logger.py

The module exposes:
  * ``AuditEntry`` dataclass with fields
    (timestamp, event_type, user_id, action, resource, details,
     risk_score, prev_hash, current_hash) and a ``to_dict()`` serialiser.
  * ``AuditLogger`` with async ``log_trade`` / ``log_order`` /
    ``log_risk_rejection`` / ``log_system_event`` (all return ``None`` and
    push an ``AuditEntry`` into ``_write_to_immutable_log``), a SHA-256
    hash-chain (``_calculate_hash`` / ``_get_genesis_hash`` / ``last_hash``),
    plus ``verify_chain_integrity`` and ``get_audit_trail`` which read from
    ``async_db_manager``.

There is no ``AuditEventType`` enum: ``event_type`` is a plain string
(``"TRADE"``, ``"ORDER"``, ``"RISK_REJECTION"`` and arbitrary strings for
system events).
"""

import asyncio
import hashlib
import json
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.audit_logger import AuditEntry, AuditLogger


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _trade_kwargs(**overrides):
    """Default keyword arguments for ``AuditLogger.log_trade``."""
    kwargs = dict(
        symbol="BTCUSDT",
        side="BUY",
        quantity=Decimal("0.5"),
        price=Decimal("42000.0"),
        order_id="order_1",
        strategy="momentum",
        risk_score=0.3,
        commission=Decimal("0.001"),
    )
    kwargs.update(overrides)
    return kwargs


def _make_chain_row(
    logger,
    *,
    prev_hash,
    details,
    timestamp,
    event_type="TRADE",
    user_id="system",
    action="BUY",
    resource="BTCUSDT",
    risk_score=0.1,
):
    """Build a DB-shaped ``audit_log`` row consistent with the module.

    ``verify_chain_integrity`` reads ``details`` as a JSON string and
    recomputes ``current_hash`` over the *other* fields (details parsed back
    into a dict).  We therefore store ``details`` as JSON and seed
    ``current_hash`` using the module's own ``_calculate_hash`` over exactly
    the dict the verifier will reconstruct, producing a genuinely valid link.
    """
    details_json = json.dumps(details)
    hashed_view = {
        "timestamp": timestamp,
        "event_type": event_type,
        "user_id": user_id,
        "action": action,
        "resource": resource,
        "details": json.loads(details_json),
        "risk_score": risk_score,
        "prev_hash": prev_hash,
    }
    return {
        "timestamp": timestamp,
        "event_type": event_type,
        "user_id": user_id,
        "action": action,
        "resource": resource,
        "details": details_json,
        "risk_score": risk_score,
        "prev_hash": prev_hash,
        "current_hash": logger._calculate_hash(hashed_view),
    }


# --------------------------------------------------------------------------- #
# AuditEntry dataclass
# --------------------------------------------------------------------------- #
class TestAuditEntry:
    """Tests for the AuditEntry dataclass."""

    def test_entry_creation(self):
        ts = datetime(2024, 1, 1, 12, 0, 0)
        entry = AuditEntry(
            timestamp=ts,
            event_type="TRADE",
            user_id="system",
            action="BUY",
            resource="BTCUSDT",
            details={"symbol": "BTCUSDT", "quantity": 0.5},
            risk_score=0.1,
            prev_hash="def456",
            current_hash="abc123",
        )

        assert entry.timestamp == ts
        assert entry.event_type == "TRADE"
        assert entry.user_id == "system"
        assert entry.action == "BUY"
        assert entry.resource == "BTCUSDT"
        assert entry.details == {"symbol": "BTCUSDT", "quantity": 0.5}
        assert entry.prev_hash == "def456"
        assert entry.current_hash == "abc123"

    def test_to_dict_serialises_timestamp_and_decimals(self):
        """to_dict() must ISO-format the timestamp and recursively convert
        Decimal values inside ``details`` to float (incl. nested dict/list)."""
        entry = AuditEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            event_type="TRADE",
            user_id="system",
            action="BUY",
            resource="BTCUSDT",
            details={
                "symbol": "BTCUSDT",
                "price": Decimal("42000.50"),
                "nested": {"fee": Decimal("0.10")},
                "legs": [Decimal("1.5"), Decimal("2.0")],
            },
            risk_score=0.2,
            prev_hash="def",
            current_hash="abc",
        )

        d = entry.to_dict()

        # Scalar fields preserved
        assert d["event_type"] == "TRADE"
        assert d["resource"] == "BTCUSDT"
        assert d["current_hash"] == "abc"
        assert d["prev_hash"] == "def"

        # Timestamp rendered as an ISO-8601 string
        assert isinstance(d["timestamp"], str)
        assert d["timestamp"] == "2024-01-01T12:00:00"

        # Decimal -> float conversion, including nested dict and list members
        assert isinstance(d["details"]["price"], float)
        assert d["details"]["price"] == 42000.50
        assert isinstance(d["details"]["nested"]["fee"], float)
        assert d["details"]["nested"]["fee"] == 0.10
        assert isinstance(d["details"]["legs"][0], float)
        assert d["details"]["legs"] == [1.5, 2.0]


# --------------------------------------------------------------------------- #
# Logging public methods
# --------------------------------------------------------------------------- #
class TestAuditLogging:
    """Tests for the public log_* methods.

    Each method builds an AuditEntry internally and writes it via
    ``_write_to_immutable_log``; we stub that sink with an AsyncMock and
    inspect the real entry it receives, so all hash/chain logic still runs.
    """

    async def test_log_trade_records_entry(self, audit_logger):
        audit_logger._write_to_immutable_log = AsyncMock()

        await audit_logger.log_trade(
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("0.5"),
            price=Decimal("42000.0"),
            order_id="test_123",
            strategy="momentum",
            risk_score=0.3,
            commission=Decimal("0.001"),
        )

        audit_logger._write_to_immutable_log.assert_awaited_once()
        entry = audit_logger._write_to_immutable_log.call_args.args[0]

        assert isinstance(entry, AuditEntry)
        assert entry.event_type == "TRADE"
        assert entry.action == "BUY"            # action == side
        assert entry.resource == "BTCUSDT"      # resource == symbol
        assert entry.risk_score == 0.3
        assert entry.details["symbol"] == "BTCUSDT"
        assert entry.details["order_id"] == "test_123"
        assert entry.details["strategy"] == "momentum"
        assert entry.details["quantity"] == 0.5
        assert entry.details["price"] == 42000.0
        assert entry.details["commission"] == 0.001
        assert entry.details["total_cost"] == 0.5 * 42000.0
        assert len(entry.current_hash) == 64    # SHA-256 hex digest
        # The logger advances its chain head to this entry's hash.
        assert audit_logger.last_hash == entry.current_hash

    async def test_log_order_records_entry(self, audit_logger):
        audit_logger._write_to_immutable_log = AsyncMock()

        await audit_logger.log_order(
            order_id="ord_1",
            symbol="BTCUSDT",
            order_type="LIMIT",
            side="SELL",
            quantity=Decimal("0.5"),
            price=Decimal("42000.0"),
            status="FILLED",
        )

        entry = audit_logger._write_to_immutable_log.call_args.args[0]
        assert entry.event_type == "ORDER"
        assert entry.action == "FILLED"         # action == status
        assert entry.resource == "ord_1"        # resource == order_id
        assert entry.details["symbol"] == "BTCUSDT"
        assert entry.details["type"] == "LIMIT"
        assert entry.details["side"] == "SELL"
        assert entry.details["price"] == 42000.0
        assert entry.details["status"] == "FILLED"
        assert entry.risk_score == 0.0
        assert len(entry.current_hash) == 64

    async def test_log_order_allows_market_order_without_price(self, audit_logger):
        audit_logger._write_to_immutable_log = AsyncMock()

        await audit_logger.log_order(
            order_id="ord_2",
            symbol="ETHUSDT",
            order_type="MARKET",
            side="BUY",
            quantity=Decimal("1.0"),
            price=None,
            status="NEW",
        )

        entry = audit_logger._write_to_immutable_log.call_args.args[0]
        assert entry.details["price"] is None
        assert entry.details["type"] == "MARKET"

    async def test_log_risk_rejection_records_entry(self, audit_logger):
        audit_logger._write_to_immutable_log = AsyncMock()

        await audit_logger.log_risk_rejection(
            symbol="BTCUSDT",
            reason="Position size too large",
            risk_score=0.95,
            attempted_quantity=Decimal("50.0"),
            attempted_side="BUY",
        )

        entry = audit_logger._write_to_immutable_log.call_args.args[0]
        assert entry.event_type == "RISK_REJECTION"
        assert entry.action == "REJECTED"
        assert entry.resource == "BTCUSDT"
        assert entry.risk_score == 0.95
        assert entry.details["reason"] == "Position size too large"
        assert entry.details["attempted_quantity"] == 50.0
        assert entry.details["attempted_side"] == "BUY"
        assert len(entry.current_hash) == 64

    async def test_log_system_event_records_entry(self, audit_logger):
        audit_logger._write_to_immutable_log = AsyncMock()

        await audit_logger.log_system_event(
            event_type="STARTUP",
            action="START",
            details={"event": "system_startup", "version": "0.2.0"},
        )

        entry = audit_logger._write_to_immutable_log.call_args.args[0]
        assert entry.event_type == "STARTUP"
        assert entry.action == "START"
        assert entry.resource == "SYSTEM"
        assert entry.details["event"] == "system_startup"
        assert entry.details["version"] == "0.2.0"
        assert entry.risk_score == 0.0

    async def test_log_trade_handles_decimal_values(self, audit_logger):
        """Decimal inputs (common in trading) must be normalised to float in
        the stored details and must not break hash computation."""
        audit_logger._write_to_immutable_log = AsyncMock()

        await audit_logger.log_trade(
            **_trade_kwargs(
                quantity=Decimal("0.5"),
                price=Decimal("42000.50"),
                commission=Decimal("0.001"),
            )
        )

        entry = audit_logger._write_to_immutable_log.call_args.args[0]
        assert isinstance(entry.details["quantity"], float)
        assert entry.details["quantity"] == 0.5
        assert entry.details["price"] == 42000.50
        assert entry.details["commission"] == 0.001
        assert entry.details["total_cost"] == float(Decimal("0.5") * Decimal("42000.50"))
        assert len(entry.current_hash) == 64

    async def test_concurrent_logging_produces_unique_hashes(self, audit_logger):
        """Logging many distinct trades concurrently must yield one valid
        SHA-256 entry per call with no hash collisions."""
        audit_logger._write_to_immutable_log = AsyncMock()

        await asyncio.gather(
            *[
                audit_logger.log_trade(
                    **_trade_kwargs(
                        symbol=f"SYM{i}USDT",
                        order_id=f"o{i}",
                        quantity=Decimal(i),
                    )
                )
                for i in range(10)
            ]
        )

        entries = [
            c.args[0] for c in audit_logger._write_to_immutable_log.call_args_list
        ]
        assert len(entries) == 10
        hashes = [e.current_hash for e in entries]
        assert all(len(h) == 64 for h in hashes)
        assert len(set(hashes)) == 10


# --------------------------------------------------------------------------- #
# Hash chain mechanics
# --------------------------------------------------------------------------- #
class TestHashChain:
    """Tests for the SHA-256 hash-chain construction."""

    async def test_genesis_hash_value(self, audit_logger):
        expected = hashlib.sha256(b"genesis_block").hexdigest()
        assert audit_logger._get_genesis_hash() == expected
        # A fresh logger is seeded with the genesis hash as its chain head.
        assert audit_logger.last_hash == expected

    async def test_first_entry_links_to_genesis(self, audit_logger):
        audit_logger._write_to_immutable_log = AsyncMock()

        await audit_logger.log_trade(**_trade_kwargs())

        entry = audit_logger._write_to_immutable_log.call_args.args[0]
        assert entry.prev_hash == audit_logger._get_genesis_hash()
        assert len(entry.current_hash) == 64

    async def test_entries_form_linked_chain(self, audit_logger):
        audit_logger._write_to_immutable_log = AsyncMock()

        await audit_logger.log_trade(**_trade_kwargs(symbol="BTCUSDT", order_id="o1"))
        await audit_logger.log_trade(**_trade_kwargs(symbol="ETHUSDT", order_id="o2"))
        await audit_logger.log_trade(**_trade_kwargs(symbol="BNBUSDT", order_id="o3"))

        e1, e2, e3 = (
            c.args[0] for c in audit_logger._write_to_immutable_log.call_args_list
        )

        # Each entry's prev_hash points at the previous entry's current_hash.
        assert e2.prev_hash == e1.current_hash
        assert e3.prev_hash == e2.current_hash
        # The chain head tracks the most recent entry.
        assert audit_logger.last_hash == e3.current_hash
        # Distinct entries hash to distinct values.
        assert len({e1.current_hash, e2.current_hash, e3.current_hash}) == 3


class TestCalculateHash:
    """Tests for the low-level _calculate_hash primitive."""

    async def test_deterministic_and_is_sha256_over_canonical_json(self, audit_logger):
        entry = {
            "timestamp": "2024-01-01T00:00:00",
            "event_type": "TRADE",
            "user_id": "system",
            "action": "BUY",
            "resource": "BTCUSDT",
            "details": {"symbol": "BTCUSDT", "quantity": 0.5, "price": 42000.0},
            "risk_score": 0.1,
            "prev_hash": "test_prev_hash",
        }

        h1 = audit_logger._calculate_hash(entry)
        h2 = audit_logger._calculate_hash(dict(entry))

        # Same input -> same hash, 64 hex chars.
        assert h1 == h2
        assert len(h1) == 64
        # And it is exactly SHA-256 over the canonical (sorted-key) JSON.
        expected = hashlib.sha256(
            json.dumps(entry, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        assert h1 == expected

    async def test_excludes_current_hash_field(self, audit_logger):
        """The current_hash field itself is excluded from the digest input."""
        base = {"event_type": "TRADE", "details": {"x": 1}, "prev_hash": "p"}
        with_hash = dict(base, current_hash="SHOULD_BE_IGNORED")
        assert audit_logger._calculate_hash(base) == audit_logger._calculate_hash(with_hash)

    async def test_sensitive_to_details_change(self, audit_logger):
        e1 = {"event_type": "TRADE", "details": {"symbol": "BTCUSDT"}, "prev_hash": "p"}
        e2 = {"event_type": "TRADE", "details": {"symbol": "ETHUSDT"}, "prev_hash": "p"}
        assert audit_logger._calculate_hash(e1) != audit_logger._calculate_hash(e2)

    async def test_sensitive_to_prev_hash_change(self, audit_logger):
        """Changing prev_hash changes the digest -- this is what links the
        chain: tampering with any entry breaks every descendant."""
        e1 = {"event_type": "TRADE", "details": {"symbol": "BTCUSDT"}, "prev_hash": "a"}
        e2 = {"event_type": "TRADE", "details": {"symbol": "BTCUSDT"}, "prev_hash": "b"}
        assert audit_logger._calculate_hash(e1) != audit_logger._calculate_hash(e2)


# --------------------------------------------------------------------------- #
# Chain verification (tamper detection)
# --------------------------------------------------------------------------- #
class TestVerifyChainIntegrity:
    """Tests for verify_chain_integrity against the (mocked) audit_log table.

    ``async_db_manager`` is imported into the module namespace, so it is
    patched at ``src.services.audit_logger.async_db_manager``.
    """

    async def test_valid_chain_passes(self, audit_logger):
        genesis = audit_logger._get_genesis_hash()
        row1 = _make_chain_row(
            audit_logger,
            prev_hash=genesis,
            details={"symbol": "BTCUSDT", "qty": 0.5},
            timestamp="2024-01-01T00:00:00",
            resource="BTCUSDT",
        )
        row2 = _make_chain_row(
            audit_logger,
            prev_hash=row1["current_hash"],
            details={"symbol": "ETHUSDT", "qty": 1.0},
            timestamp="2024-01-01T00:01:00",
            resource="ETHUSDT",
        )

        mock_db = MagicMock()
        mock_db.execute = AsyncMock(return_value=[row1, row2])
        with patch("src.services.audit_logger.async_db_manager", mock_db):
            result = await audit_logger.verify_chain_integrity()

        assert result is True

    async def test_tampered_details_detected(self, audit_logger):
        """Mutating a stored entry's data without re-signing it must fail."""
        row1 = _make_chain_row(
            audit_logger,
            prev_hash=audit_logger._get_genesis_hash(),
            details={"symbol": "BTCUSDT", "qty": 0.5},
            timestamp="2024-01-01T00:00:00",
            resource="BTCUSDT",
        )
        row2 = _make_chain_row(
            audit_logger,
            prev_hash=row1["current_hash"],
            details={"symbol": "ETHUSDT", "qty": 1.0},
            timestamp="2024-01-01T00:01:00",
            resource="ETHUSDT",
        )

        # Tamper row2's persisted details but keep its original current_hash.
        tampered = dict(row2)
        tampered["details"] = json.dumps({"symbol": "ETHUSDT", "qty": 999.0})

        mock_db = MagicMock()
        mock_db.execute = AsyncMock(return_value=[row1, tampered])
        with patch("src.services.audit_logger.async_db_manager", mock_db):
            result = await audit_logger.verify_chain_integrity()

        assert result is False

    async def test_broken_link_detected(self, audit_logger):
        """An internally-consistent entry whose prev_hash does not point at
        its predecessor must be rejected (broken chain link)."""
        row1 = _make_chain_row(
            audit_logger,
            prev_hash=audit_logger._get_genesis_hash(),
            details={"symbol": "BTCUSDT", "qty": 0.5},
            timestamp="2024-01-01T00:00:00",
            resource="BTCUSDT",
        )
        # row2 is self-consistent (its own hash matches its contents) but its
        # prev_hash deliberately does NOT match row1.current_hash.
        row2 = _make_chain_row(
            audit_logger,
            prev_hash="0" * 64,
            details={"symbol": "ETHUSDT", "qty": 1.0},
            timestamp="2024-01-01T00:01:00",
            resource="ETHUSDT",
        )

        mock_db = MagicMock()
        mock_db.execute = AsyncMock(return_value=[row1, row2])
        with patch("src.services.audit_logger.async_db_manager", mock_db):
            result = await audit_logger.verify_chain_integrity()

        assert result is False

    async def test_empty_range_is_considered_valid(self, audit_logger):
        mock_db = MagicMock()
        mock_db.execute = AsyncMock(return_value=[])
        with patch("src.services.audit_logger.async_db_manager", mock_db):
            result = await audit_logger.verify_chain_integrity()

        assert result is True

    async def test_db_error_returns_false(self, audit_logger):
        mock_db = MagicMock()
        mock_db.execute = AsyncMock(side_effect=RuntimeError("db down"))
        with patch("src.services.audit_logger.async_db_manager", mock_db):
            result = await audit_logger.verify_chain_integrity()

        assert result is False


# --------------------------------------------------------------------------- #
# Audit trail retrieval
# --------------------------------------------------------------------------- #
class TestGetAuditTrail:
    """Tests for get_audit_trail query building and result handling."""

    async def test_returns_rows_and_passes_filters(self, audit_logger):
        rows = [
            {
                "timestamp": "2024-01-01T00:00:00",
                "event_type": "TRADE",
                "user_id": "system",
                "action": "BUY",
                "resource": "BTCUSDT",
                "details": "{}",
                "risk_score": 0.1,
                "current_hash": "h1",
            }
        ]
        mock_db = MagicMock()
        mock_db.execute = AsyncMock(return_value=rows)

        with patch("src.services.audit_logger.async_db_manager", mock_db):
            trail = await audit_logger.get_audit_trail(
                resource="BTCUSDT", event_type="TRADE", limit=25
            )

        assert trail == [dict(r) for r in rows]

        passed = mock_db.execute.call_args.args
        sql = passed[0]
        params = passed[1:]
        assert "audit_log" in sql
        # The resource & event_type filters and the limit are bound as params.
        assert "BTCUSDT" in params
        assert "TRADE" in params
        assert 25 in params

    async def test_default_limit_is_applied(self, audit_logger):
        mock_db = MagicMock()
        mock_db.execute = AsyncMock(return_value=[])

        with patch("src.services.audit_logger.async_db_manager", mock_db):
            trail = await audit_logger.get_audit_trail()

        assert trail == []
        # Default limit of 100 is bound even when no filters are supplied.
        assert 100 in mock_db.execute.call_args.args[1:]

    async def test_returns_empty_list_on_db_error(self, audit_logger):
        mock_db = MagicMock()
        mock_db.execute = AsyncMock(side_effect=RuntimeError("db down"))

        with patch("src.services.audit_logger.async_db_manager", mock_db):
            trail = await audit_logger.get_audit_trail(resource="BTCUSDT")

        assert trail == []


# --------------------------------------------------------------------------- #
# Regression: real write -> verify roundtrip (the chain used to always fail)
# --------------------------------------------------------------------------- #
def _rows_from_written(written, *, tamper_index=None, tamper_details=None):
    """把實際寫入的 AuditEntry 轉成「DB 讀回」的 rows。

    刻意把 timestamp 以 ``str(datetime)``（空格分隔）讀回——這正是舊版會讓
    雜湊對不上的情境（寫入用 isoformat 的 T 分隔）。修正後仍必須驗證通過。
    """
    rows = []
    for i, e in enumerate(written):
        details = tamper_details if (i == tamper_index) else e.details
        rows.append({
            "timestamp": str(e.timestamp),            # space-separated, NOT isoformat
            "event_type": e.event_type,
            "user_id": e.user_id,
            "action": e.action,
            "resource": e.resource,
            "details": json.dumps(details),
            "risk_score": e.risk_score,
            "prev_hash": e.prev_hash,
            "current_hash": e.current_hash,
        })
    return rows


class TestChainRoundTripVerification:
    """真實「寫入→驗證」往返。舊版 verify_chain_integrity 對任何真實資料一律
    回 False（寫入雜湊 isoformat 字串、驗證雜湊 DB datetime/空格格式且為另一次
    utcnow）。修正：寫入只取一次時間戳 + 雜湊時正規化 timestamp。
    """

    @pytest.mark.asyncio
    async def test_real_logged_chain_verifies_true(self):
        logger = AuditLogger()
        written = []
        logger._write_to_immutable_log = AsyncMock(side_effect=lambda e: written.append(e))

        await logger.log_system_event(event_type="STARTUP", action="boot", details={"v": 1})
        await logger.log_trade(**_trade_kwargs())
        assert len(written) == 2

        rows = _rows_from_written(written)
        with patch("src.services.audit_logger.async_db_manager") as db:
            db.execute = AsyncMock(return_value=rows)
            ok = await logger.verify_chain_integrity()
        assert ok is True

    @pytest.mark.asyncio
    async def test_tampered_details_break_verification(self):
        logger = AuditLogger()
        written = []
        logger._write_to_immutable_log = AsyncMock(side_effect=lambda e: written.append(e))

        await logger.log_system_event(event_type="STARTUP", action="boot", details={"v": 1})
        await logger.log_trade(**_trade_kwargs())

        rows = _rows_from_written(written, tamper_index=1, tamper_details={"hacked": True})
        with patch("src.services.audit_logger.async_db_manager") as db:
            db.execute = AsyncMock(return_value=rows)
            ok = await logger.verify_chain_integrity()
        assert ok is False
