"""故障容错和恢复机制 - 連接資料庫版本"""

import asyncio
import json
import hashlib
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from decimal import Decimal
from enum import Enum
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.logger import logger
from src.config import settings
from src.utils.database import async_db_manager
from src.services.binance_client import binance_client
from src.models.trading import (
    Position, Order, EmergencyEvent, StateSnapshot as StateSnapshotModel,
    PositionStatus, OrderStatus, EmergencyEventType
)

# 根據配置決定使用 Redis 還是內存緩存
if settings.USE_MEMORY_CACHE:
    from src.utils.memory_cache import async_memory_cache as memory_cache
else:
    import redis.asyncio as aioredis


class ConnectionStatus(Enum):
    """连接状态"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class StateSnapshotData:
    """状态快照數據"""
    timestamp: datetime
    portfolio_state: Dict[str, Any]
    open_orders: List[Dict]
    positions: List[Dict]
    balance: Dict[str, float]
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict):
        """从字典创建"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

    def compute_checksum(self) -> str:
        """計算快照 checksum"""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


class StateManager:
    """状态管理器 - 連接資料庫和緩存（Redis 或內存）"""

    def __init__(self):
        self.redis_client = None  # 僅在非內存緩存模式使用
        self.snapshot_key_prefix = "state_snapshot:"
        self.current_snapshot: Optional[StateSnapshotData] = None
        self._db_initialized = False
        self._cache_initialized = False

    async def init(self):
        """初始化緩存和資料庫連接"""
        # 初始化緩存
        if settings.USE_MEMORY_CACHE:
            # 使用內存緩存
            await memory_cache.init()
            self._cache_initialized = True
            logger.info("✅ State manager using memory cache")
        else:
            # 使用 Redis
            try:
                self.redis_client = aioredis.Redis.from_url(
                    settings.REDIS_URL,
                    encoding='utf-8',
                    decode_responses=True
                )
                await self.redis_client.ping()
                self._cache_initialized = True
                logger.info("✅ State manager Redis connected")
            except Exception as e:
                logger.warning(f"⚠️ Redis not available: {e}")
                self.redis_client = None
                self._cache_initialized = False

        # 初始化資料庫
        try:
            await async_db_manager.init()
            self._db_initialized = True
            logger.info("✅ State manager database connected")
        except Exception as e:
            logger.warning(f"⚠️ Database not available: {e}")
            self._db_initialized = False

    async def create_snapshot(self, reason: str = "manual_snapshot") -> StateSnapshotData:
        """创建状态快照"""
        try:
            snapshot = StateSnapshotData(
                timestamp=datetime.utcnow(),
                portfolio_state=await self._get_portfolio_state(),
                open_orders=await self._get_open_orders(),
                positions=await self._get_positions(),
                balance=await self._get_balance(),
                metadata={
                    'version': '1.0',
                    'reason': reason
                }
            )

            # 保存到緩存
            snapshot_key = await self._save_snapshot_to_cache(snapshot)

            # 保存元數據到資料庫
            if self._db_initialized:
                await self._save_snapshot_metadata(snapshot, snapshot_key)

            self.current_snapshot = snapshot
            logger.info(f"State snapshot created: {snapshot.timestamp}, reason: {reason}")

            return snapshot

        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            raise

    async def restore_snapshot(self, snapshot: StateSnapshotData):
        """恢复状态快照"""
        try:
            logger.info(f"Restoring snapshot from {snapshot.timestamp}")

            # 取消所有开放订单
            await self._cancel_all_orders()

            # 恢复投资组合状态
            await self._restore_portfolio_state(snapshot.portfolio_state)

            # 恢复头寸
            await self._restore_positions(snapshot.positions)

            logger.info("State snapshot restored successfully")

        except Exception as e:
            logger.error(f"Failed to restore snapshot: {e}")
            raise

    async def _save_snapshot_to_cache(self, snapshot: StateSnapshotData) -> str:
        """保存快照到緩存（Redis 或內存）"""
        key = f"{self.snapshot_key_prefix}{snapshot.timestamp.isoformat()}"
        data = json.dumps(snapshot.to_dict())

        if settings.USE_MEMORY_CACHE:
            # 使用內存緩存
            await memory_cache.set(key, data, ttl=86400)
        elif self.redis_client:
            # 使用 Redis
            await self.redis_client.setex(
                key,
                86400,  # 保存24小时
                data
            )

        return key

    async def _save_snapshot_metadata(self, snapshot: StateSnapshotData, snapshot_key: str):
        """保存快照元數據到資料庫"""
        try:
            async with async_db_manager.get_session() as session:
                db_snapshot = StateSnapshotModel(
                    snapshot_key=snapshot_key,
                    reason=snapshot.metadata.get('reason', 'unknown'),
                    total_value=Decimal(str(snapshot.portfolio_state.get('total_value', 0))),
                    available_balance=Decimal(str(snapshot.portfolio_state.get('available_balance', 0))),
                    positions_count=len(snapshot.positions),
                    orders_count=len(snapshot.open_orders),
                    checksum=snapshot.compute_checksum(),
                    is_valid=True,
                    expires_at=datetime.utcnow() + timedelta(hours=24),
                    metadata_json=snapshot.metadata
                )
                session.add(db_snapshot)
                await session.commit()
                logger.debug(f"Snapshot metadata saved to database: {snapshot_key}")
        except Exception as e:
            logger.error(f"Failed to save snapshot metadata: {e}")

    async def _get_portfolio_state(self) -> Dict[str, Any]:
        """獲取投資組合狀態 - 從 Binance API"""
        try:
            balances = await binance_client.get_balances()

            total_value = Decimal('0')
            available_balance = Decimal('0')
            locked_balance = Decimal('0')

            for balance in balances:
                asset = balance['asset']
                free = Decimal(balance['free'])
                locked = Decimal(balance['locked'])

                # 簡單計算：只考慮 USDT
                if asset == 'USDT':
                    total_value += free + locked
                    available_balance += free
                    locked_balance += locked

            return {
                'total_value': float(total_value),
                'available_balance': float(available_balance),
                'locked_balance': float(locked_balance),
                'balances': balances
            }
        except Exception as e:
            logger.error(f"Failed to get portfolio state: {e}")
            return {
                'total_value': 0,
                'available_balance': 0,
                'locked_balance': 0,
                'balances': [],
                'error': str(e)
            }

    async def _get_open_orders(self) -> List[Dict]:
        """獲取開放訂單 - 從 Binance API + 資料庫"""
        orders = []

        # 1. 從 Binance API 獲取真實掛單
        try:
            binance_orders = await binance_client.get_open_orders()
            orders.extend(binance_orders)
            logger.debug(f"Got {len(binance_orders)} open orders from Binance")
        except Exception as e:
            logger.error(f"Failed to get orders from Binance: {e}")

        # 2. 從資料庫獲取本地訂單記錄（補充）
        if self._db_initialized:
            try:
                async with async_db_manager.get_session() as session:
                    result = await session.execute(
                        select(Order).where(
                            Order.status.in_([
                                OrderStatus.PENDING.value,
                                OrderStatus.SUBMITTED.value,
                                OrderStatus.PARTIALLY_FILLED.value
                            ])
                        )
                    )
                    db_orders = result.scalars().all()

                    # 合併，避免重複
                    binance_order_ids = {o.get('orderId') for o in orders}
                    for db_order in db_orders:
                        if db_order.order_id not in binance_order_ids:
                            orders.append(db_order.to_dict())

                    logger.debug(f"Got {len(db_orders)} orders from database")
            except Exception as e:
                logger.error(f"Failed to get orders from database: {e}")

        return orders

    async def _get_positions(self) -> List[Dict]:
        """獲取當前持倉 - 從資料庫"""
        positions = []

        if self._db_initialized:
            try:
                async with async_db_manager.get_session() as session:
                    result = await session.execute(
                        select(Position).where(
                            Position.status == PositionStatus.OPEN.value
                        )
                    )
                    db_positions = result.scalars().all()
                    positions = [p.to_dict() for p in db_positions]
                    logger.debug(f"Got {len(positions)} positions from database")
            except Exception as e:
                logger.error(f"Failed to get positions from database: {e}")

        return positions

    async def _get_balance(self) -> Dict[str, float]:
        """獲取賬戶餘額 - 從 Binance API"""
        try:
            balances = await binance_client.get_balances()
            return {
                b['asset']: float(Decimal(b['free']) + Decimal(b['locked']))
                for b in balances
            }
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return {}

    async def _cancel_all_orders(self):
        """取消所有訂單"""
        logger.info("Cancelling all open orders...")

        try:
            open_orders = await binance_client.get_open_orders()
            cancelled_count = 0

            for order in open_orders:
                try:
                    await binance_client.cancel_order(
                        symbol=order['symbol'],
                        order_id=order['orderId']
                    )
                    cancelled_count += 1
                except Exception as e:
                    logger.error(f"Failed to cancel order {order['orderId']}: {e}")

            logger.info(f"Cancelled {cancelled_count}/{len(open_orders)} orders")

            # 更新資料庫訂單狀態
            if self._db_initialized:
                async with async_db_manager.get_session() as session:
                    await session.execute(
                        update(Order)
                        .where(Order.status.in_([
                            OrderStatus.PENDING.value,
                            OrderStatus.SUBMITTED.value
                        ]))
                        .values(status=OrderStatus.CANCELLED.value)
                    )
                    await session.commit()

        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")

    async def _restore_portfolio_state(self, state: Dict[str, Any]):
        """恢复投资组合状态"""
        logger.info(f"Restoring portfolio state: total_value={state.get('total_value')}")
        # 投資組合狀態主要是參考用，實際狀態由交易所決定

    async def _restore_positions(self, positions: List[Dict]):
        """恢复头寸"""
        logger.info(f"Restoring {len(positions)} positions")
        # 持倉恢復主要是更新資料庫記錄，實際持倉由交易所決定

    async def close(self):
        """关闭連接"""
        if settings.USE_MEMORY_CACHE:
            await memory_cache.close()
        elif self.redis_client:
            await self.redis_client.close()
        logger.info("State manager closed")


class BinanceConnection:
    """主 Binance 连接"""

    def __init__(self):
        self.status = ConnectionStatus.DISCONNECTED
        self.retry_count = 0
        self.max_retries = 3

    async def connect(self):
        """连接到 Binance"""
        try:
            # 測試連接
            await binance_client.get_ticker_price("BTCUSDT")
            self.status = ConnectionStatus.CONNECTED
            logger.info("Connected to Binance (primary)")
        except Exception as e:
            self.status = ConnectionStatus.FAILED
            logger.error(f"Failed to connect to Binance: {e}")
            raise

    async def execute(self, order: Dict) -> Dict:
        """执行订单"""
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Not connected to Binance")

        result = await binance_client.create_order(
            symbol=order['symbol'],
            side=order['side'],
            order_type=order.get('order_type', 'MARKET'),
            quantity=Decimal(str(order['quantity'])),
            price=Decimal(str(order['price'])) if order.get('price') else None
        )

        return result

    async def disconnect(self):
        """断开连接"""
        self.status = ConnectionStatus.DISCONNECTED
        logger.info("Disconnected from Binance (primary)")


class BinanceBackupConnection:
    """备用 Binance 连接（使用不同 endpoint）"""

    def __init__(self):
        self.status = ConnectionStatus.DISCONNECTED

    async def connect(self):
        """连接到备用服务器"""
        try:
            # 備用連接使用相同客戶端，但可以配置不同 endpoint
            self.status = ConnectionStatus.CONNECTED
            logger.info("Connected to Binance (backup)")
        except Exception as e:
            self.status = ConnectionStatus.FAILED
            logger.error(f"Failed to connect to Binance backup: {e}")
            raise

    async def execute(self, order: Dict) -> Dict:
        """执行订单"""
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Not connected to Binance backup")

        # 使用相同的客戶端執行
        result = await binance_client.create_order(
            symbol=order['symbol'],
            side=order['side'],
            order_type=order.get('order_type', 'MARKET'),
            quantity=Decimal(str(order['quantity'])),
            price=Decimal(str(order['price'])) if order.get('price') else None
        )

        return result


class FaultTolerantTradingEngine:
    """容错交易引擎 - 連接資料庫版本"""

    def __init__(self):
        self.primary_connection = BinanceConnection()
        self.backup_connection = BinanceBackupConnection()
        self._state_manager = StateManager()
        self.is_emergency_stopped = False
        self._db_initialized = False
        # 連續「完全失敗」計數：達到門檻才觸發緊急停止，
        # 避免單筆暫時性失敗就停機；任一次成功即歸零。
        self._consecutive_failures = 0
        self.max_consecutive_failures = 3

    @property
    def state_manager(self) -> StateManager:
        return self._state_manager

    async def init(self):
        """初始化引擎"""
        # 初始化狀態管理器（包含 Redis 和資料庫）
        await self._state_manager.init()
        self._db_initialized = self._state_manager._db_initialized

        # 連接 Binance
        try:
            await self.primary_connection.connect()
        except Exception as e:
            logger.warning(f"Primary connection failed: {e}")

        logger.info("✅ Fault-tolerant trading engine initialized")

    async def execute_with_failover(
        self,
        order: Dict,
        max_retries: int = 3
    ) -> Optional[Dict]:
        """带故障转移的订单执行"""

        if self.is_emergency_stopped:
            logger.error("Trading engine is in emergency stop mode")
            return None

        # 確保連接已建立
        if self.primary_connection.status != ConnectionStatus.CONNECTED:
            logger.info("Primary connection not connected, attempting to connect...")
            try:
                await self.primary_connection.connect()
            except Exception as e:
                logger.error(f"Failed to connect to Binance: {e}")
                return {'status': 'error', 'msg': f'Connection failed: {e}'}

        snapshot = None
        last_result = None

        for attempt in range(max_retries):
            try:
                logger.info(f"Order execution attempt {attempt + 1}/{max_retries}")

                # 创建状态快照
                snapshot = await self._state_manager.create_snapshot(reason="pre_execution")

                # 記錄訂單到資料庫
                db_order = await self._record_order_to_db(order)

                # 尝试主连接执行
                result = await self._execute_with_timeout(
                    self.primary_connection.execute(order),
                    timeout=10.0
                )

                # 验证执行结果
                if await self.verify_execution(result):
                    # 更新資料庫訂單狀態
                    await self._update_order_status(
                        db_order.order_id if db_order else None,
                        result
                    )
                    logger.info(f"Order executed successfully: {result}")
                    self._consecutive_failures = 0
                    return result

                # 驗證失敗但有結果，記錄並在最後一次嘗試後返回
                logger.warning(f"Execution verification failed: {result}")
                last_result = result

            except (ConnectionError, asyncio.TimeoutError) as e:
                logger.error(f"Primary connection failed: {e}")
                last_result = {'status': 'error', 'msg': f'Connection error: {e}'}

                # 尝试备用连接
                try:
                    await self.backup_connection.connect()
                    result = await self._execute_with_timeout(
                        self.backup_connection.execute(order),
                        timeout=10.0
                    )

                    if await self.verify_execution(result):
                        logger.info(f"Order executed via backup: {result}")
                        self._consecutive_failures = 0
                        return result
                    last_result = result

                except Exception as backup_error:
                    logger.critical(f"Backup connection also failed: {backup_error}")
                    last_result = {'status': 'error', 'msg': f'Backup failed: {backup_error}'}

                    # 恢复到快照状态
                    if snapshot:
                        await self._state_manager.restore_snapshot(snapshot)

            except Exception as e:
                logger.error(f"Unexpected error during execution: {e}")
                last_result = {'status': 'error', 'msg': f'Execution error: {e}'}

        # 所有重試都失敗（這是一次「完全失敗」）
        self._consecutive_failures += 1
        if last_result is None:
            last_result = {'status': 'error', 'msg': 'All execution attempts failed'}
        logger.error(
            f"All {max_retries} execution attempts failed "
            f"(consecutive failures: {self._consecutive_failures}). Last result: {last_result}"
        )

        # 連續多次完全失敗 → 自動觸發緊急停止（安全機制）
        if self._consecutive_failures >= self.max_consecutive_failures:
            logger.critical(
                f"{self._consecutive_failures} consecutive total failures — triggering emergency stop"
            )
            await self.emergency_shutdown(
                f"{self._consecutive_failures} consecutive execution failures"
            )

        return last_result

    async def _record_order_to_db(self, order: Dict) -> Optional[Order]:
        """記錄訂單到資料庫"""
        if not self._db_initialized:
            return None

        try:
            async with async_db_manager.get_session() as session:
                db_order = Order(
                    order_id=f"local_{datetime.utcnow().timestamp()}",
                    symbol=order['symbol'],
                    side=order['side'],
                    order_type=order.get('order_type', 'MARKET'),
                    quantity=Decimal(str(order['quantity'])),
                    price=Decimal(str(order['price'])) if order.get('price') else None,
                    status=OrderStatus.PENDING.value,
                    strategy=order.get('strategy'),
                    metadata_json=order
                )
                session.add(db_order)
                await session.commit()
                await session.refresh(db_order)
                return db_order
        except Exception as e:
            logger.error(f"Failed to record order to database: {e}")
            return None

    async def _update_order_status(self, local_order_id: str, result: Dict):
        """更新訂單狀態"""
        if not self._db_initialized or not local_order_id:
            return

        try:
            async with async_db_manager.get_session() as session:
                await session.execute(
                    update(Order)
                    .where(Order.order_id == local_order_id)
                    .values(
                        order_id=str(result.get('orderId', local_order_id)),
                        status=OrderStatus.FILLED.value if result.get('status') == 'FILLED' else OrderStatus.SUBMITTED.value,
                        executed_qty=Decimal(str(result.get('executedQty', 0))),
                        executed_price=Decimal(str(result.get('price', 0))) if result.get('price') else None,
                        filled_at=datetime.utcnow() if result.get('status') == 'FILLED' else None
                    )
                )
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to update order status: {e}")

    async def _execute_with_timeout(
        self,
        coro,
        timeout: float
    ):
        """带超时的执行"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"Execution timeout after {timeout}s")
            raise

    async def verify_execution(self, result: Dict) -> bool:
        """验证执行结果"""
        if not result:
            logger.warning("Execution result is None or empty")
            return False

        # 檢查是否有 Binance API 錯誤
        if 'code' in result and result.get('code') != 0:
            error_msg = result.get('msg', 'Unknown error')
            logger.error(f"Binance API error: code={result['code']}, msg={error_msg}")
            return False

        status = result.get('status')
        if status in ['FILLED', 'NEW', 'PARTIALLY_FILLED']:
            return True

        # 檢查是否是自定義錯誤狀態
        if status == 'error':
            logger.warning(f"Custom error status: {result.get('msg', 'Unknown')}")
            return False

        logger.warning(f"Execution verification failed with status '{status}': {result}")
        return False

    async def emergency_shutdown(self, reason: str):
        """紧急停止 - 記錄到資料庫"""
        logger.critical(f"🚨 EMERGENCY SHUTDOWN: {reason}")

        self.is_emergency_stopped = True
        snapshot = None
        orders_cancelled = 0
        positions_count = 0

        try:
            # 1. 創建緊急快照
            snapshot = await self._state_manager.create_snapshot(reason="emergency")
            positions_count = len(snapshot.positions)

            # 2. 取消所有开放订单
            logger.info("Cancelling all open orders...")
            open_orders = await binance_client.get_open_orders()
            for order in open_orders:
                try:
                    await binance_client.cancel_order(
                        symbol=order['symbol'],
                        order_id=order['orderId']
                    )
                    orders_cancelled += 1
                except Exception as e:
                    logger.error(f"Failed to cancel order: {e}")

            # 3. 記錄緊急事件到資料庫
            await self._record_emergency_event(
                event_type=EmergencyEventType.SYSTEM_ERROR.value,
                reason=reason,
                snapshot_id=f"{self._state_manager.snapshot_key_prefix}{snapshot.timestamp.isoformat()}" if snapshot else None,
                open_positions_count=positions_count,
                open_orders_count=len(open_orders),
                orders_cancelled=orders_cancelled
            )

            # 4. 发送紧急通知
            await self._send_emergency_notification(reason)

            # 5. 关闭连接
            await self.primary_connection.disconnect()

            logger.critical("Emergency shutdown completed")

        except Exception as e:
            logger.critical(f"Error during emergency shutdown: {e}")

    async def _record_emergency_event(
        self,
        event_type: str,
        reason: str,
        snapshot_id: str = None,
        open_positions_count: int = 0,
        open_orders_count: int = 0,
        orders_cancelled: int = 0
    ):
        """記錄緊急事件到資料庫"""
        if not self._db_initialized:
            return

        try:
            async with async_db_manager.get_session() as session:
                event = EmergencyEvent(
                    event_type=event_type,
                    reason=reason,
                    severity="critical",
                    snapshot_id=snapshot_id,
                    open_positions_count=open_positions_count,
                    open_orders_count=open_orders_count,
                    orders_cancelled=orders_cancelled,
                    is_resolved=False,
                    metadata_json={
                        "triggered_at": datetime.utcnow().isoformat(),
                        "engine_state": "stopped"
                    }
                )
                session.add(event)
                await session.commit()
                logger.info(f"Emergency event recorded to database: {event.id}")
        except Exception as e:
            logger.error(f"Failed to record emergency event: {e}")

    async def _send_emergency_notification(self, reason: str):
        """发送紧急通知"""
        message = f"""
        🚨 紧急停止警报 🚨

        时间: {datetime.utcnow().isoformat()}
        原因: {reason}

        系统已自动停止所有交易活动。
        请立即检查系统状态！
        """

        logger.critical(message)
        # TODO: 發送 Telegram、Email 通知

    async def resume_trading(self):
        """恢复交易"""
        if not self.is_emergency_stopped:
            logger.info("Trading is already active")
            return

        logger.info("Resuming trading operations...")
        # 恢復交易時重置連續失敗計數
        self._consecutive_failures = 0

        try:
            # 重新连接
            await self.primary_connection.connect()

            # 更新資料庫中的緊急事件狀態
            if self._db_initialized:
                async with async_db_manager.get_session() as session:
                    await session.execute(
                        update(EmergencyEvent)
                        .where(EmergencyEvent.is_resolved == False)
                        .values(
                            is_resolved=True,
                            resolved_at=datetime.utcnow(),
                            resolution_notes="Trading resumed via API"
                        )
                    )
                    await session.commit()

            self.is_emergency_stopped = False

            logger.info("✅ Trading resumed successfully")

        except Exception as e:
            logger.error(f"Failed to resume trading: {e}")
            raise

    async def close(self):
        """关闭引擎"""
        await self._state_manager.close()
        await self.primary_connection.disconnect()
        logger.info("Fault-tolerant trading engine closed")


# 全局容错交易引擎实例
fault_tolerant_engine = FaultTolerantTradingEngine()
