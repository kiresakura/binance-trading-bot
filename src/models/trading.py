"""交易相關資料庫模型"""

from datetime import datetime
from decimal import Decimal
from sqlalchemy import Column, Integer, String, DateTime, Numeric, Boolean, Text, JSON, Enum as SQLEnum
from sqlalchemy.sql import func
from enum import Enum
from src.models.base import Base


class PositionStatus(str, Enum):
    """持倉狀態"""
    OPEN = "open"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"


class OrderStatus(str, Enum):
    """訂單狀態"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class EmergencyEventType(str, Enum):
    """緊急事件類型"""
    CONNECTION_FAILURE = "connection_failure"
    EXECUTION_FAILURE = "execution_failure"
    RISK_BREACH = "risk_breach"
    MANUAL_STOP = "manual_stop"
    SYSTEM_ERROR = "system_error"


class Position(Base):
    """持倉記錄表"""
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # LONG / SHORT
    quantity = Column(Numeric(20, 8), nullable=False)
    entry_price = Column(Numeric(20, 8), nullable=False)
    current_price = Column(Numeric(20, 8), nullable=True)
    unrealized_pnl = Column(Numeric(20, 8), default=0)
    realized_pnl = Column(Numeric(20, 8), default=0)
    status = Column(String(20), default=PositionStatus.OPEN.value, index=True)

    # 風控相關
    stop_loss = Column(Numeric(20, 8), nullable=True)
    take_profit = Column(Numeric(20, 8), nullable=True)
    risk_score = Column(Numeric(5, 2), nullable=True)

    # 策略相關
    strategy = Column(String(50), nullable=True)

    # 時間戳
    opened_at = Column(DateTime, default=func.now(), nullable=False)
    closed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # 元數據
    metadata_json = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<Position {self.symbol} {self.side} {self.quantity}>"

    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": str(self.quantity),
            "entry_price": str(self.entry_price),
            "current_price": str(self.current_price) if self.current_price else None,
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "status": self.status,
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "take_profit": str(self.take_profit) if self.take_profit else None,
            "strategy": self.strategy,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
        }


class Order(Base):
    """訂單記錄表"""
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(50), unique=True, index=True)  # 交易所訂單 ID
    client_order_id = Column(String(50), nullable=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY / SELL
    order_type = Column(String(20), nullable=False)  # MARKET / LIMIT
    quantity = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8), nullable=True)  # 限價單價格
    executed_qty = Column(Numeric(20, 8), default=0)
    executed_price = Column(Numeric(20, 8), nullable=True)  # 平均成交價
    status = Column(String(20), default=OrderStatus.PENDING.value, index=True)

    # 費用
    commission = Column(Numeric(20, 8), default=0)
    commission_asset = Column(String(10), nullable=True)

    # 風控相關
    risk_score = Column(Numeric(5, 2), nullable=True)
    risk_reason = Column(Text, nullable=True)

    # 策略相關
    strategy = Column(String(50), nullable=True)

    # 時間戳
    created_at = Column(DateTime, default=func.now(), nullable=False)
    submitted_at = Column(DateTime, nullable=True)
    filled_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # 元數據
    metadata_json = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<Order {self.order_id} {self.symbol} {self.side} {self.status}>"

    def to_dict(self):
        return {
            "id": self.id,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": str(self.quantity),
            "price": str(self.price) if self.price else None,
            "executed_qty": str(self.executed_qty),
            "executed_price": str(self.executed_price) if self.executed_price else None,
            "status": self.status,
            "commission": str(self.commission),
            "strategy": self.strategy,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
        }


class EmergencyEvent(Base):
    """緊急事件記錄表"""
    __tablename__ = "emergency_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(30), nullable=False, index=True)
    reason = Column(Text, nullable=False)
    severity = Column(String(20), default="critical")  # warning / critical / fatal

    # 事件發生時的狀態
    snapshot_id = Column(String(100), nullable=True)  # Redis 快照 key
    open_positions_count = Column(Integer, default=0)
    open_orders_count = Column(Integer, default=0)
    total_exposure = Column(Numeric(20, 8), nullable=True)

    # 處理結果
    orders_cancelled = Column(Integer, default=0)
    positions_closed = Column(Integer, default=0)
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    resolution_notes = Column(Text, nullable=True)

    # 時間戳
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # 元數據
    metadata_json = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<EmergencyEvent {self.event_type} at {self.created_at}>"

    def to_dict(self):
        return {
            "id": self.id,
            "event_type": self.event_type,
            "reason": self.reason,
            "severity": self.severity,
            "snapshot_id": self.snapshot_id,
            "open_positions_count": self.open_positions_count,
            "open_orders_count": self.open_orders_count,
            "is_resolved": self.is_resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class StateSnapshot(Base):
    """狀態快照記錄表（備份 Redis 快照的元數據）"""
    __tablename__ = "state_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_key = Column(String(100), unique=True, nullable=False, index=True)  # Redis key
    reason = Column(String(50), nullable=False)  # manual / auto / emergency

    # 快照內容摘要
    total_value = Column(Numeric(20, 8), nullable=True)
    available_balance = Column(Numeric(20, 8), nullable=True)
    positions_count = Column(Integer, default=0)
    orders_count = Column(Integer, default=0)

    # 驗證
    checksum = Column(String(64), nullable=True)  # SHA256 of snapshot data
    is_valid = Column(Boolean, default=True)

    # 時間戳
    created_at = Column(DateTime, default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=True)

    # 元數據
    metadata_json = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<StateSnapshot {self.snapshot_key} at {self.created_at}>"

    def to_dict(self):
        return {
            "id": self.id,
            "snapshot_key": self.snapshot_key,
            "reason": self.reason,
            "total_value": str(self.total_value) if self.total_value else None,
            "positions_count": self.positions_count,
            "orders_count": self.orders_count,
            "is_valid": self.is_valid,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
