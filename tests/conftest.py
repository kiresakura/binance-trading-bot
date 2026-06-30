"""Pytest configuration and shared fixtures"""

# 測試環境設定必須在匯入 src.* 之前完成（src.config 於匯入時即讀取環境變數）。
import os

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production-use")
os.environ.setdefault("TRADING_ENABLED", "false")
os.environ.setdefault("BINANCE_TESTNET", "true")
os.environ.setdefault("DRY_RUN", "true")

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any
from unittest.mock import MagicMock, AsyncMock

from src.services.risk_manager import Order, RiskManagementSystem
from src.services.order_executor import SmartOrderExecutor
from src.services.fault_tolerance import FaultTolerantTradingEngine, StateManager
from src.services.audit_logger import AuditLogger
from src.services.monitoring import MetricsCollector


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_order() -> Order:
    """Sample order for testing"""
    return Order(
        symbol="BTCUSDT",
        side="BUY",
        order_type="LIMIT",
        quantity=Decimal("0.5"),
        price=Decimal("42000.0"),
        timestamp=datetime.utcnow()
    )


@pytest.fixture
def large_order() -> Order:
    """Large order for testing order splitting"""
    return Order(
        symbol="BTCUSDT",
        side="BUY",
        order_type="LIMIT",
        quantity=Decimal("10.0"),
        price=Decimal("42000.0"),
        timestamp=datetime.utcnow()
    )


@pytest.fixture
def risky_order() -> Order:
    """Risky order that should trigger risk rejection"""
    return Order(
        symbol="BTCUSDT",
        side="BUY",
        order_type="LIMIT",
        quantity=Decimal("50.0"),  # Very large position
        price=Decimal("42000.0"),
        timestamp=datetime.utcnow()
    )


@pytest.fixture
def mock_order_book() -> Dict[str, Any]:
    """Mock order book data"""
    return {
        'bids': [(Decimal('41990'), Decimal('1.5')),
                 (Decimal('41980'), Decimal('2.0')),
                 (Decimal('41970'), Decimal('1.0'))],
        'asks': [(Decimal('42010'), Decimal('1.5')),
                 (Decimal('42020'), Decimal('2.0')),
                 (Decimal('42030'), Decimal('1.0'))],
        'liquidity_score': 0.75,
        'spread': 0.0005,
        'imbalance': 0.1
    }


@pytest.fixture
async def risk_manager():
    """Risk management system instance"""
    manager = RiskManagementSystem()
    # Initialize with test portfolio
    manager.current_portfolio_value = Decimal('100000.0')
    manager.available_balance = Decimal('50000.0')
    return manager


@pytest.fixture
async def order_executor():
    """Smart order executor instance"""
    executor = SmartOrderExecutor()
    return executor


@pytest.fixture
async def fault_tolerant_engine():
    """Fault tolerant engine instance (with mocked connections)"""
    engine = FaultTolerantTradingEngine()
    # Mock the connections to avoid actual API calls
    engine.primary_connection.execute = AsyncMock(
        return_value={'status': 'success', 'order_id': 'test_123', 'price': 42000.0}
    )
    engine.backup_connection.execute = AsyncMock(
        return_value={'status': 'success', 'order_id': 'test_124', 'price': 42000.0}
    )
    return engine


@pytest.fixture
async def audit_logger():
    """Audit logger instance (with mocked database)"""
    logger = AuditLogger()
    # Mock database operations
    logger._save_to_db = AsyncMock()
    return logger


@pytest.fixture
def metrics_collector():
    """Metrics collector instance"""
    collector = MetricsCollector()
    return collector


@pytest.fixture
def mock_execution_result() -> Dict[str, Any]:
    """Mock execution result from exchange"""
    return {
        'status': 'success',
        'order_id': 'test_order_123',
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'type': 'LIMIT',
        'price': 42000.0,
        'quantity': 0.5,
        'filled_quantity': 0.5,
        'commission': 0.001,
        'timestamp': datetime.utcnow().isoformat()
    }


@pytest.fixture
def mock_state_snapshot() -> Dict[str, Any]:
    """Mock state snapshot"""
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'portfolio_state': {
            'total_value': 100000.0,
            'available_balance': 50000.0,
            'locked_balance': 50000.0
        },
        'open_orders': [],
        'positions': [
            {'symbol': 'BTCUSDT', 'quantity': 1.0, 'entry_price': 41000.0}
        ],
        'balance': {'USDT': 100000.0},
        'metadata': {'version': '1.0', 'reason': 'test_snapshot'}
    }
