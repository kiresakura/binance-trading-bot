"""
Arena Service - 影子交易競技場服務

實現 Champion/Challenger 模型比較系統：
- 模型註冊與管理
- 影子交易追蹤
- 虛擬帳戶交易（每個模型 10000 USDT）
- 模型晉升邏輯
- 策略靈敏度配置
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, asdict
import json

from src.logger import logger
from src.utils.database import async_db_manager
from src.api.websocket_manager import ws_manager
from src.services.risk_management_service import risk_service, RiskAction


# 虛擬交易配置
INITIAL_VIRTUAL_BALANCE = Decimal("10000")  # 初始虛擬資金
VIRTUAL_COMMISSION_RATE = Decimal("0.001")  # 0.1% 手續費
MAX_POSITION_SIZE_PCT = Decimal("0.3")      # 單筆最大倉位 30%


def _parse_timestamp(ts: Union[str, datetime, None]) -> Optional[str]:
    """解析时间戳为 ISO 格式字符串"""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.isoformat()
    if isinstance(ts, str):
        return ts
    return str(ts)


@dataclass
class ModelStats:
    """模型統計數據"""
    version: str
    type: str
    win_rate: float
    total_return: float
    trade_count: int
    draw_down: float
    status: str
    sharpe_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_hold_time: Optional[str] = None
    last_update: Optional[str] = None
    # 虛擬帳戶數據
    virtual_balance: Optional[float] = None
    virtual_total_value: Optional[float] = None
    virtual_pnl: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "type": self.type,
            "win_rate": self.win_rate,
            "total_return": self.total_return,
            "trade_count": self.trade_count,
            "draw_down": self.draw_down,
            "status": self.status,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "avg_hold_time": self.avg_hold_time,
            "last_update": self.last_update,
            "virtual_balance": self.virtual_balance,
            "virtual_total_value": self.virtual_total_value,
            "virtual_pnl": self.virtual_pnl,
        }


@dataclass 
class VirtualPosition:
    """虛擬持倉"""
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": float(self.quantity),
            "entry_price": float(self.entry_price),
            "current_price": float(self.current_price),
            "unrealized_pnl": float(self.unrealized_pnl),
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
        }


@dataclass
class ArenaStatus:
    """競技場狀態"""
    champion: Optional[ModelStats]
    challenger: Optional[ModelStats]
    comparison_days: int
    promotion_threshold: float
    is_comparing: bool
    started_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "champion": self.champion.to_dict() if self.champion else None,
            "challenger": self.challenger.to_dict() if self.challenger else None,
            "comparison_days": self.comparison_days,
            "promotion_threshold": self.promotion_threshold,
            "is_comparing": self.is_comparing,
            "started_at": self.started_at,
        }


@dataclass
class SensitivityConfig:
    """策略靈敏度配置"""
    sensitivity: str
    confidence_threshold: float
    strength_threshold: float
    position_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ArenaService:
    """影子交易競技場服務"""

    def __init__(self):
        self._initialized = False

    async def initialize(self):
        """初始化服務"""
        if self._initialized:
            return

        try:
            await self._ensure_tables()
            self._initialized = True
            logger.info("ArenaService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ArenaService: {e}")
            raise

    async def _ensure_tables(self):
        """確保必要的表存在，如不存在則創建"""
        try:
            # 檢查並創建 shadow_trades 表
            try:
                await async_db_manager.execute_one("SELECT 1 FROM shadow_trades LIMIT 1")
            except Exception:
                logger.info("Creating shadow_trades table...")
                await async_db_manager.execute("""
                    CREATE TABLE IF NOT EXISTS shadow_trades (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        model_version VARCHAR(100) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        signal_strength DECIMAL(10, 6),
                        signal_confidence DECIMAL(10, 6),
                        suggested_quantity DECIMAL(20, 8),
                        suggested_price DECIMAL(20, 8),
                        actual_price DECIMAL(20, 8),
                        virtual_pnl DECIMAL(20, 8),
                        would_have_triggered BOOLEAN DEFAULT FALSE,
                        reason TEXT
                    )
                """)
                await async_db_manager.execute(
                    "CREATE INDEX IF NOT EXISTS idx_shadow_trades_model ON shadow_trades (model_version, timestamp DESC)"
                )
                await async_db_manager.execute(
                    "CREATE INDEX IF NOT EXISTS idx_shadow_trades_symbol ON shadow_trades (symbol, timestamp DESC)"
                )
                logger.info("shadow_trades table created successfully")

            # 檢查並創建 strategy_sensitivity 表
            try:
                await async_db_manager.execute_one("SELECT 1 FROM strategy_sensitivity LIMIT 1")
            except Exception:
                logger.info("Creating strategy_sensitivity table...")
                await async_db_manager.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_sensitivity (
                        id BIGSERIAL PRIMARY KEY,
                        sensitivity VARCHAR(20) NOT NULL DEFAULT 'moderate',
                        confidence_threshold DECIMAL(10, 6) DEFAULT 0.70,
                        strength_threshold DECIMAL(10, 6) DEFAULT 0.60,
                        position_ratio DECIMAL(10, 6) DEFAULT 0.6,
                        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # 插入預設值
                await async_db_manager.execute("""
                    INSERT INTO strategy_sensitivity (id, sensitivity, confidence_threshold, strength_threshold, position_ratio)
                    VALUES (1, 'moderate', 0.70, 0.60, 0.6)
                    ON CONFLICT (id) DO NOTHING
                """)
                logger.info("strategy_sensitivity table created successfully")

            # 檢查其他表
            try:
                await async_db_manager.execute_one("SELECT 1 FROM arena_models LIMIT 1")
            except Exception:
                logger.warning("arena_models table does not exist")

            try:
                await async_db_manager.execute_one("SELECT 1 FROM virtual_accounts LIMIT 1")
            except Exception:
                logger.warning("virtual_accounts table does not exist")

            # 檢查並創建 risk_alerts 表
            try:
                await async_db_manager.execute_one("SELECT 1 FROM risk_alerts LIMIT 1")
            except Exception:
                logger.info("Creating risk_alerts table...")
                await async_db_manager.execute("""
                    CREATE TABLE IF NOT EXISTS risk_alerts (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        alert_type VARCHAR(50) NOT NULL,
                        model_version VARCHAR(100) NOT NULL,
                        symbol VARCHAR(20),
                        message TEXT,
                        severity VARCHAR(20) NOT NULL,
                        action_taken VARCHAR(50),
                        details TEXT
                    )
                """)
                await async_db_manager.execute(
                    "CREATE INDEX IF NOT EXISTS idx_risk_alerts_model ON risk_alerts (model_version, timestamp DESC)"
                )
                await async_db_manager.execute(
                    "CREATE INDEX IF NOT EXISTS idx_risk_alerts_severity ON risk_alerts (severity, timestamp DESC)"
                )
                logger.info("risk_alerts table created successfully")

            # 確保 virtual_positions 表有 peak_price 欄位
            try:
                await async_db_manager.execute_one("SELECT peak_price FROM virtual_positions LIMIT 1")
            except Exception:
                logger.info("Adding peak_price column to virtual_positions...")
                try:
                    await async_db_manager.execute(
                        "ALTER TABLE virtual_positions ADD COLUMN IF NOT EXISTS peak_price DECIMAL(20, 8)"
                    )
                    logger.info("peak_price column added to virtual_positions")
                except Exception as alter_err:
                    logger.warning(f"Could not add peak_price column: {alter_err}")

        except Exception as e:
            logger.error(f"Error ensuring Arena tables: {e}")

    # ==================== 虛擬帳戶管理 ====================

    async def init_virtual_account(self, model_version: str) -> bool:
        """為模型初始化虛擬帳戶（10000 USDT）"""
        try:
            await async_db_manager.execute(
                """
                INSERT INTO virtual_accounts 
                (model_version, initial_balance, balance, total_value, peak_value)
                VALUES ($1, $2, $2, $2, $2)
                ON CONFLICT (model_version) DO NOTHING
                """,
                model_version, float(INITIAL_VIRTUAL_BALANCE)
            )
            logger.info(f"Virtual account initialized for {model_version} with {INITIAL_VIRTUAL_BALANCE} USDT")
            return True
        except Exception as e:
            logger.error(f"Failed to init virtual account for {model_version}: {e}")
            return False

    async def get_virtual_account(self, model_version: str) -> Optional[Dict[str, Any]]:
        """獲取虛擬帳戶狀態"""
        try:
            row = await async_db_manager.execute_one(
                """
                SELECT balance, total_value, realized_pnl, unrealized_pnl, 
                       total_pnl, total_pnl_pct, max_drawdown, peak_value,
                       trade_count, win_count, loss_count, win_rate
                FROM virtual_accounts
                WHERE model_version = $1
                """,
                model_version
            )
            if not row:
                return None
            return {
                "balance": float(row['balance']),
                "total_value": float(row['total_value']),
                "realized_pnl": float(row['realized_pnl']),
                "unrealized_pnl": float(row['unrealized_pnl']),
                "total_pnl": float(row['total_pnl']),
                "total_pnl_pct": float(row['total_pnl_pct']),
                "max_drawdown": float(row['max_drawdown']),
                "peak_value": float(row['peak_value']),
                "trade_count": int(row['trade_count']),
                "win_count": int(row['win_count']),
                "loss_count": int(row['loss_count']),
                "win_rate": float(row['win_rate']),
            }
        except Exception as e:
            logger.error(f"Failed to get virtual account for {model_version}: {e}")
            return None


    async def execute_virtual_trade(
        self,
        model_version: str,
        symbol: str,
        side: str,  # BUY or SELL
        price: Decimal,
        quantity: Optional[Decimal] = None,
        signal_strength: Optional[float] = None,
        signal_confidence: Optional[float] = None,
        skip_risk_check: bool = False  # 用於風險管理自動平倉時跳過檢查
    ) -> Dict[str, Any]:
        """
        執行虛擬交易

        BUY: 用 USDT 買入幣種
        SELL: 賣出持有的幣種換回 USDT
        """
        try:
            # 風險管理：檢查交易頻率限制（僅對買入有效）
            if side.upper() == "BUY" and not skip_risk_check:
                can_trade, reason = await risk_service.can_trade(model_version, symbol)
                if not can_trade:
                    logger.warning(f"Trade blocked by risk management: {reason}")
                    return {"success": False, "error": reason, "blocked_by_risk": True}

            # 獲取虛擬帳戶
            account = await self.get_virtual_account(model_version)
            if not account:
                # 自動初始化帳戶
                await self.init_virtual_account(model_version)
                account = await self.get_virtual_account(model_version)

            balance = Decimal(str(account['balance']))
            
            if side.upper() == "BUY":
                return await self._execute_virtual_buy(
                    model_version, symbol, price, quantity, balance,
                    signal_strength, signal_confidence
                )
            elif side.upper() == "SELL":
                return await self._execute_virtual_sell(
                    model_version, symbol, price, quantity,
                    signal_strength, signal_confidence
                )
            else:
                return {"success": False, "error": f"Invalid side: {side}"}
                
        except Exception as e:
            logger.error(f"Virtual trade error for {model_version}: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_virtual_buy(
        self,
        model_version: str,
        symbol: str,
        price: Decimal,
        quantity: Optional[Decimal],
        balance: Decimal,
        signal_strength: Optional[float],
        signal_confidence: Optional[float]
    ) -> Dict[str, Any]:
        """執行虛擬買入"""
        try:
            # 計算買入數量（最多用 30% 資金）
            max_spend = balance * MAX_POSITION_SIZE_PCT
            if quantity is None:
                spend_amount = max_spend
            else:
                spend_amount = min(quantity * price, max_spend)
            
            if spend_amount < Decimal("1"):
                return {"success": False, "error": "Insufficient balance"}
            
            # 計算手續費和實際買入數量
            commission = spend_amount * VIRTUAL_COMMISSION_RATE
            actual_spend = spend_amount - commission
            buy_quantity = (actual_spend / price).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
            
            if buy_quantity <= 0:
                return {"success": False, "error": "Quantity too small"}
            
            # 檢查是否已有持倉（加倉）
            existing = await async_db_manager.execute_one(
                "SELECT quantity, entry_price FROM virtual_positions WHERE model_version = $1 AND symbol = $2",
                model_version, symbol
            )
            
            if existing:
                # 加倉：計算新的平均成本
                old_qty = Decimal(str(existing['quantity']))
                old_price = Decimal(str(existing['entry_price']))
                new_qty = old_qty + buy_quantity
                new_avg_price = ((old_qty * old_price) + (buy_quantity * price)) / new_qty

                await async_db_manager.execute(
                    """
                    UPDATE virtual_positions
                    SET quantity = $1, entry_price = $2, current_price = $3,
                        peak_price = GREATEST(COALESCE(peak_price, $3), $3),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE model_version = $4 AND symbol = $5
                    """,
                    float(new_qty), float(new_avg_price), float(price), model_version, symbol
                )
            else:
                # 新建持倉
                await async_db_manager.execute(
                    """
                    INSERT INTO virtual_positions (model_version, symbol, quantity, entry_price, current_price, peak_price)
                    VALUES ($1, $2, $3, $4, $4, $4)
                    """,
                    model_version, symbol, float(buy_quantity), float(price)
                )
            
            # 更新餘額（注意：trade_count 只在 SELL 時增加，因為 trade_count 應該等於 win_count + loss_count）
            new_balance = balance - spend_amount
            await async_db_manager.execute(
                """
                UPDATE virtual_accounts
                SET balance = $1, updated_at = CURRENT_TIMESTAMP
                WHERE model_version = $2
                """,
                float(new_balance), model_version
            )
            
            # 記錄交易
            await async_db_manager.execute(
                """
                INSERT INTO virtual_trades 
                (timestamp, model_version, symbol, side, quantity, price, quote_value, commission, balance_after, signal_strength, signal_confidence)
                VALUES (CURRENT_TIMESTAMP, $1, $2, 'BUY', $3, $4, $5, $6, $7, $8, $9)
                """,
                model_version, symbol, float(buy_quantity), float(price), 
                float(spend_amount), float(commission), float(new_balance),
                signal_strength, signal_confidence
            )
            
            # 更新總價值
            await self._update_account_stats(model_version)

            # 風險管理：記錄交易（用於頻率限制）
            risk_service.record_trade(model_version)

            logger.info(f"Virtual BUY: {model_version} bought {buy_quantity} {symbol} @ {price}")

            # WebSocket 廣播交易事件
            try:
                await ws_manager.broadcast_arena_trade(
                    model_version=model_version,
                    symbol=symbol,
                    side="BUY",
                    quantity=float(buy_quantity),
                    price=float(price),
                    is_winner=None,  # 買入時尚未知道結果
                    pnl_pct=None
                )
            except Exception as ws_err:
                logger.warning(f"Failed to broadcast arena trade: {ws_err}")

            return {
                "success": True,
                "side": "BUY",
                "symbol": symbol,
                "quantity": float(buy_quantity),
                "price": float(price),
                "commission": float(commission),
                "balance_after": float(new_balance)
            }

        except Exception as e:
            logger.error(f"Virtual buy error: {e}")
            return {"success": False, "error": str(e)}


    async def _execute_virtual_sell(
        self,
        model_version: str,
        symbol: str,
        price: Decimal,
        quantity: Optional[Decimal],
        signal_strength: Optional[float],
        signal_confidence: Optional[float]
    ) -> Dict[str, Any]:
        """執行虛擬賣出"""
        try:
            # 獲取持倉
            position = await async_db_manager.execute_one(
                "SELECT quantity, entry_price FROM virtual_positions WHERE model_version = $1 AND symbol = $2",
                model_version, symbol
            )
            
            if not position or Decimal(str(position['quantity'])) <= 0:
                return {"success": False, "error": f"No position for {symbol}"}
            
            pos_qty = Decimal(str(position['quantity']))
            entry_price = Decimal(str(position['entry_price']))
            
            # 決定賣出數量（默認全部賣出）
            sell_qty = quantity if quantity and quantity < pos_qty else pos_qty
            
            # 計算收益
            gross_value = sell_qty * price
            commission = gross_value * VIRTUAL_COMMISSION_RATE
            net_value = gross_value - commission
            
            # 計算實現盈虧
            cost_basis = sell_qty * entry_price
            realized_pnl = net_value - cost_basis
            is_win = realized_pnl > 0
            
            # 獲取當前帳戶
            account = await self.get_virtual_account(model_version)
            balance = Decimal(str(account['balance']))
            new_balance = balance + net_value
            
            # 更新或刪除持倉
            remaining_qty = pos_qty - sell_qty
            if remaining_qty <= Decimal("0.00000001"):
                # 清空持倉
                await async_db_manager.execute(
                    "DELETE FROM virtual_positions WHERE model_version = $1 AND symbol = $2",
                    model_version, symbol
                )
            else:
                # 更新持倉
                await async_db_manager.execute(
                    """
                    UPDATE virtual_positions 
                    SET quantity = $1, current_price = $2, updated_at = CURRENT_TIMESTAMP
                    WHERE model_version = $3 AND symbol = $4
                    """,
                    float(remaining_qty), float(price), model_version, symbol
                )
            
            # 更新帳戶
            win_inc = 1 if is_win else 0
            loss_inc = 0 if is_win else 1
            await async_db_manager.execute(
                """
                UPDATE virtual_accounts 
                SET balance = $1, 
                    realized_pnl = realized_pnl + $2,
                    trade_count = trade_count + 1,
                    win_count = win_count + $3,
                    loss_count = loss_count + $4,
                    updated_at = CURRENT_TIMESTAMP
                WHERE model_version = $5
                """,
                float(new_balance), float(realized_pnl), win_inc, loss_inc, model_version
            )
            
            # 記錄交易
            await async_db_manager.execute(
                """
                INSERT INTO virtual_trades 
                (timestamp, model_version, symbol, side, quantity, price, quote_value, commission, realized_pnl, balance_after, signal_strength, signal_confidence)
                VALUES (CURRENT_TIMESTAMP, $1, $2, 'SELL', $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                model_version, symbol, float(sell_qty), float(price),
                float(gross_value), float(commission), float(realized_pnl), float(new_balance),
                signal_strength, signal_confidence
            )
            
            # 更新統計
            await self._update_account_stats(model_version)

            # 風險管理：記錄交易（用於頻率限制）
            risk_service.record_trade(model_version)

            # 計算盈虧百分比
            pnl_pct = (float(realized_pnl) / float(cost_basis) * 100) if cost_basis > 0 else 0

            pnl_emoji = "🟢" if is_win else "🔴"
            logger.info(f"Virtual SELL: {model_version} sold {sell_qty} {symbol} @ {price}, PnL: {pnl_emoji} {realized_pnl:.4f} ({pnl_pct:.2f}%)")

            # WebSocket 廣播交易事件
            try:
                await ws_manager.broadcast_arena_trade(
                    model_version=model_version,
                    symbol=symbol,
                    side="SELL",
                    quantity=float(sell_qty),
                    price=float(price),
                    is_winner=is_win,
                    pnl_pct=pnl_pct
                )
            except Exception as ws_err:
                logger.warning(f"Failed to broadcast arena trade: {ws_err}")

            return {
                "success": True,
                "side": "SELL",
                "symbol": symbol,
                "quantity": float(sell_qty),
                "price": float(price),
                "commission": float(commission),
                "realized_pnl": float(realized_pnl),
                "is_win": is_win,
                "balance_after": float(new_balance)
            }

        except Exception as e:
            logger.error(f"Virtual sell error: {e}")
            return {"success": False, "error": str(e)}


    async def _update_account_stats(self, model_version: str):
        """更新帳戶統計數據"""
        try:
            # 計算持倉價值
            positions = await async_db_manager.execute(
                """
                SELECT symbol, quantity, entry_price, current_price 
                FROM virtual_positions 
                WHERE model_version = $1 AND quantity > 0
                """,
                model_version
            )
            
            total_position_value = Decimal("0")
            total_unrealized_pnl = Decimal("0")
            
            for pos in positions:
                qty = Decimal(str(pos['quantity']))
                entry = Decimal(str(pos['entry_price']))
                current = Decimal(str(pos['current_price'])) if pos['current_price'] else entry
                
                position_value = qty * current
                unrealized = position_value - (qty * entry)
                
                total_position_value += position_value
                total_unrealized_pnl += unrealized
            
            # 獲取帳戶餘額
            account = await async_db_manager.execute_one(
                "SELECT balance, realized_pnl, peak_value, trade_count, win_count FROM virtual_accounts WHERE model_version = $1",
                model_version
            )
            
            if not account:
                return
                
            balance = Decimal(str(account['balance']))
            realized_pnl = Decimal(str(account['realized_pnl']))
            peak_value = Decimal(str(account['peak_value']))
            trade_count = int(account['trade_count'])
            win_count = int(account['win_count'])
            
            # 計算總價值和盈虧
            total_value = balance + total_position_value
            total_pnl = realized_pnl + total_unrealized_pnl
            total_pnl_pct = (total_pnl / INITIAL_VIRTUAL_BALANCE * 100) if INITIAL_VIRTUAL_BALANCE > 0 else 0
            
            # 更新峰值和回撤
            if total_value > peak_value:
                peak_value = total_value
            
            drawdown = ((peak_value - total_value) / peak_value * 100) if peak_value > 0 else 0
            
            # 計算勝率（只計算已完成的交易：win + loss）
            # 獲取 loss_count
            loss_count_row = await async_db_manager.execute_one(
                "SELECT loss_count FROM virtual_accounts WHERE model_version = $1",
                model_version
            )
            loss_count = int(loss_count_row['loss_count']) if loss_count_row else 0
            completed_trades = win_count + loss_count
            win_rate = (win_count / completed_trades * 100) if completed_trades > 0 else 0
            
            # 更新帳戶
            await async_db_manager.execute(
                """
                UPDATE virtual_accounts 
                SET total_value = $1,
                    unrealized_pnl = $2,
                    total_pnl = $3,
                    total_pnl_pct = $4,
                    peak_value = $5,
                    max_drawdown = $6,
                    win_rate = $7,
                    updated_at = CURRENT_TIMESTAMP
                WHERE model_version = $8
                """,
                float(total_value), float(total_unrealized_pnl), float(total_pnl),
                float(total_pnl_pct), float(peak_value), float(drawdown), float(win_rate),
                model_version
            )
            
            # 同步到 arena_models 表
            await async_db_manager.execute(
                """
                UPDATE arena_models 
                SET win_rate = $1,
                    total_return = $2,
                    trade_count = $3,
                    draw_down = $4,
                    updated_at = CURRENT_TIMESTAMP
                WHERE version = $5
                """,
                float(win_rate), float(total_pnl_pct), trade_count, float(drawdown), model_version
            )
            
        except Exception as e:
            logger.error(f"Failed to update account stats for {model_version}: {e}")

    async def update_prices(self, prices: Dict[str, Decimal]):
        """
        批量更新所有持倉的當前價格
        prices: {"BTCUSDT": Decimal("100000"), "ETHUSDT": Decimal("3500"), ...}
        """
        try:
            for symbol, price in prices.items():
                # 更新所有模型的該幣種持倉價格，並追蹤 peak_price
                await async_db_manager.execute(
                    """
                    UPDATE virtual_positions
                    SET current_price = $1,
                        peak_price = GREATEST(COALESCE(peak_price, $1), $1),
                        unrealized_pnl = (quantity * $1) - (quantity * entry_price),
                        unrealized_pnl_pct = CASE
                            WHEN entry_price > 0 THEN (($1 - entry_price) / entry_price * 100)
                            ELSE 0
                        END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = $2
                    """,
                    float(price), symbol
                )
            
            # 更新所有有持倉的帳戶統計
            models = await async_db_manager.execute(
                "SELECT DISTINCT model_version FROM virtual_positions WHERE quantity > 0"
            )
            
            for row in models:
                await self._update_account_stats(row['model_version'])
                
        except Exception as e:
            logger.error(f"Failed to update prices: {e}")

    async def get_virtual_positions(self, model_version: str) -> List[VirtualPosition]:
        """獲取模型的虛擬持倉"""
        try:
            rows = await async_db_manager.execute(
                """
                SELECT symbol, quantity, entry_price, current_price, unrealized_pnl, unrealized_pnl_pct
                FROM virtual_positions
                WHERE model_version = $1 AND quantity > 0
                """,
                model_version
            )
            return [
                VirtualPosition(
                    symbol=row['symbol'],
                    quantity=Decimal(str(row['quantity'])),
                    entry_price=Decimal(str(row['entry_price'])),
                    current_price=Decimal(str(row['current_price'] or row['entry_price'])),
                    unrealized_pnl=Decimal(str(row['unrealized_pnl'] or 0)),
                    unrealized_pnl_pct=float(row['unrealized_pnl_pct'] or 0)
                )
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get virtual positions for {model_version}: {e}")
            return []

    async def get_virtual_trades(self, model_version: str, limit: int = 50) -> List[Dict[str, Any]]:
        """獲取模型的虛擬交易歷史"""
        try:
            rows = await async_db_manager.execute(
                """
                SELECT timestamp, symbol, side, quantity, price, quote_value, 
                       commission, realized_pnl, balance_after
                FROM virtual_trades
                WHERE model_version = $1
                ORDER BY timestamp DESC
                LIMIT $2
                """,
                model_version, limit
            )
            return [
                {
                    "timestamp": _parse_timestamp(row['timestamp']),
                    "symbol": row['symbol'],
                    "side": row['side'],
                    "quantity": float(row['quantity']),
                    "price": float(row['price']),
                    "quote_value": float(row['quote_value']),
                    "commission": float(row['commission'] or 0),
                    "realized_pnl": float(row['realized_pnl']) if row['realized_pnl'] else None,
                    "balance_after": float(row['balance_after']),
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get virtual trades for {model_version}: {e}")
            return []


    # ==================== 模型管理 ====================

    async def get_model(self, version: str) -> Optional[ModelStats]:
        """獲取模型詳情"""
        try:
            row = await async_db_manager.execute_one(
                """
                SELECT version, model_type, status, win_rate, total_return,
                       trade_count, draw_down, sharpe_ratio, profit_factor,
                       avg_hold_time, updated_at
                FROM arena_models
                WHERE version = $1
                """,
                version
            )

            if not row:
                return None

            # 獲取虛擬帳戶數據
            account = await self.get_virtual_account(version)

            return ModelStats(
                version=row['version'],
                type=row['model_type'],
                status=row['status'],
                win_rate=float(row['win_rate'] or 0),
                total_return=float(row['total_return'] or 0),
                trade_count=int(row['trade_count'] or 0),
                draw_down=float(row['draw_down'] or 0),
                sharpe_ratio=float(row['sharpe_ratio']) if row['sharpe_ratio'] else None,
                profit_factor=float(row['profit_factor']) if row['profit_factor'] else None,
                avg_hold_time=row['avg_hold_time'],
                last_update=_parse_timestamp(row['updated_at']),
                virtual_balance=account['balance'] if account else None,
                virtual_total_value=account['total_value'] if account else None,
                virtual_pnl=account['total_pnl'] if account else None,
            )
        except Exception as e:
            logger.error(f"Failed to get model {version}: {e}")
            return None

    async def list_models(self, status: Optional[str] = None) -> List[ModelStats]:
        """列出所有模型"""
        try:
            if status:
                rows = await async_db_manager.execute(
                    """
                    SELECT version, model_type, status, win_rate, total_return,
                           trade_count, draw_down, sharpe_ratio, profit_factor,
                           avg_hold_time, updated_at
                    FROM arena_models
                    WHERE status = $1
                    ORDER BY created_at DESC
                    """,
                    status
                )
            else:
                rows = await async_db_manager.execute(
                    """
                    SELECT version, model_type, status, win_rate, total_return,
                           trade_count, draw_down, sharpe_ratio, profit_factor,
                           avg_hold_time, updated_at
                    FROM arena_models
                    ORDER BY created_at DESC
                    """
                )

            results = []
            for row in rows:
                account = await self.get_virtual_account(row['version'])
                results.append(ModelStats(
                    version=row['version'],
                    type=row['model_type'],
                    status=row['status'],
                    win_rate=float(row['win_rate'] or 0),
                    total_return=float(row['total_return'] or 0),
                    trade_count=int(row['trade_count'] or 0),
                    draw_down=float(row['draw_down'] or 0),
                    sharpe_ratio=float(row['sharpe_ratio']) if row['sharpe_ratio'] else None,
                    profit_factor=float(row['profit_factor']) if row['profit_factor'] else None,
                    avg_hold_time=row['avg_hold_time'],
                    last_update=_parse_timestamp(row['updated_at']),
                    virtual_balance=account['balance'] if account else None,
                    virtual_total_value=account['total_value'] if account else None,
                    virtual_pnl=account['total_pnl'] if account else None,
                ))
            return results
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    async def register_model(
        self,
        version: str,
        model_type: str,
        config: Optional[Dict] = None
    ) -> bool:
        """註冊新模型並初始化虛擬帳戶"""
        try:
            await async_db_manager.execute(
                """
                INSERT INTO arena_models (version, model_type, status, config, updated_at)
                VALUES ($1, $2, 'TRAINING', $3, CURRENT_TIMESTAMP)
                ON CONFLICT(version) DO UPDATE SET
                    model_type = excluded.model_type,
                    config = excluded.config,
                    updated_at = CURRENT_TIMESTAMP
                """,
                version,
                model_type,
                json.dumps(config or {})
            )
            
            # 初始化虛擬帳戶
            await self.init_virtual_account(version)

            logger.info(f"Model {version} registered with 10000 USDT virtual account")

            # WebSocket 廣播新模型註冊
            try:
                await ws_manager.broadcast_arena_model_registered(
                    model_id=version,
                    model_version=version,
                    model_type=model_type,
                    accuracy=0.0,  # 初始準確率為 0
                    symbol=config.get("symbol") if config else None,
                    source="arena_registration"
                )
            except Exception as ws_err:
                logger.warning(f"Failed to broadcast model registration: {ws_err}")

            return True
        except Exception as e:
            logger.error(f"Failed to register model {version}: {e}")
            return False


    # ==================== 競技場管理 ====================

    async def get_arena_status(self) -> ArenaStatus:
        """獲取競技場狀態"""
        try:
            row = await async_db_manager.execute_one(
                """
                SELECT champion_version, challenger_version, comparison_days,
                       promotion_threshold, is_comparing, started_at
                FROM arena_state
                WHERE id = 1
                """
            )

            if not row:
                return ArenaStatus(
                    champion=None, challenger=None,
                    comparison_days=7, promotion_threshold=0.05,
                    is_comparing=False, started_at=None
                )

            champion = None
            challenger = None

            if row['champion_version']:
                champion = await self.get_model(row['champion_version'])
            if row['challenger_version']:
                challenger = await self.get_model(row['challenger_version'])

            return ArenaStatus(
                champion=champion,
                challenger=challenger,
                comparison_days=row['comparison_days'],
                promotion_threshold=float(row['promotion_threshold']),
                is_comparing=bool(row['is_comparing']),
                started_at=_parse_timestamp(row['started_at'])
            )
        except Exception as e:
            logger.error(f"Failed to get arena status: {e}")
            return ArenaStatus(
                champion=None, challenger=None,
                comparison_days=7, promotion_threshold=0.05,
                is_comparing=False, started_at=None
            )

    async def start_shadow_trading(self, challenger_version: str) -> bool:
        """啟動影子交易模式"""
        try:
            model = await self.get_model(challenger_version)
            if not model:
                logger.warning(f"Model {challenger_version} not found")
                return False

            # 確保有虛擬帳戶
            await self.init_virtual_account(challenger_version)

            await async_db_manager.execute(
                """
                UPDATE arena_models
                SET status = 'SHADOW_RUNNING', shadow_start_at = CURRENT_TIMESTAMP
                WHERE version = $1
                """,
                challenger_version
            )

            await async_db_manager.execute(
                """
                INSERT INTO arena_state (id, challenger_version, is_comparing, started_at, comparison_days, promotion_threshold)
                VALUES (1, $1, 1, CURRENT_TIMESTAMP, 7, 0.05)
                ON CONFLICT(id) DO UPDATE SET
                    challenger_version = excluded.challenger_version,
                    is_comparing = 1,
                    started_at = CURRENT_TIMESTAMP
                """,
                challenger_version
            )

            logger.info(f"Shadow trading started for model {challenger_version}")
            return True
        except Exception as e:
            logger.error(f"Failed to start shadow trading: {e}")
            return False

    async def stop_shadow_trading(self) -> bool:
        """停止影子交易模式"""
        try:
            row = await async_db_manager.execute_one(
                "SELECT challenger_version FROM arena_state WHERE id = 1"
            )
            challenger_version = row['challenger_version'] if row else None

            if challenger_version:
                await async_db_manager.execute(
                    """
                    UPDATE arena_models
                    SET status = 'TRAINING'
                    WHERE version = $1 AND status = 'SHADOW_RUNNING'
                    """,
                    challenger_version
                )

            await async_db_manager.execute(
                "UPDATE arena_state SET challenger_version = NULL, is_comparing = 0 WHERE id = 1"
            )

            logger.info("Shadow trading stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop shadow trading: {e}")
            return False

    async def promote_model(self, challenger_version: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """晉升挑戰者為冠軍"""
        try:
            arena_status = await self.get_arena_status()
            challenger = await self.get_model(challenger_version)
            
            if not challenger:
                return {"success": False, "message": f"Challenger model {challenger_version} not found"}

            previous_champion = arena_status.champion

            if previous_champion:
                await async_db_manager.execute(
                    "UPDATE arena_models SET status = 'ARCHIVED' WHERE version = $1 AND status = 'LIVE'",
                    previous_champion.version
                )

            await async_db_manager.execute(
                "UPDATE arena_models SET status = 'LIVE', promoted_at = CURRENT_TIMESTAMP WHERE version = $1",
                challenger_version
            )

            await async_db_manager.execute(
                "UPDATE arena_state SET champion_version = $1, challenger_version = NULL, is_comparing = 0 WHERE id = 1",
                challenger_version
            )

            await async_db_manager.execute(
                """
                INSERT INTO promotion_history
                (previous_champion, new_champion, reason, champion_performance, challenger_performance, promoted_by)
                VALUES ($1, $2, $3, $4, $5, 'manual')
                """,
                previous_champion.version if previous_champion else None,
                challenger_version,
                reason or "Manual promotion",
                json.dumps(previous_champion.to_dict()) if previous_champion else "{}",
                json.dumps(challenger.to_dict())
            )

            logger.info(f"Model {challenger_version} promoted to champion")

            # WebSocket 廣播冠軍變更
            try:
                await ws_manager.broadcast_arena_champion_changed(
                    old_champion=previous_champion.version if previous_champion else None,
                    new_champion=challenger_version,
                    new_champion_stats=challenger.to_dict()
                )
            except Exception as ws_err:
                logger.warning(f"Failed to broadcast champion change: {ws_err}")

            return {
                "success": True,
                "message": f"Model {challenger_version} promoted successfully",
                "new_champion": challenger.to_dict(),
                "previous_champion": previous_champion.to_dict() if previous_champion else None
            }
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return {"success": False, "message": str(e)}


    # ==================== 影子交易記錄（舊版相容）====================

    async def record_shadow_trade(
        self,
        model_version: str,
        symbol: str,
        side: str,
        signal_strength: float,
        signal_confidence: float,
        suggested_quantity: Decimal,
        suggested_price: Decimal,
        actual_price: Decimal,
        would_have_triggered: bool = False,
        reason: Optional[str] = None
    ) -> bool:
        """記錄影子交易（舊版相容，現在會執行虛擬交易）"""
        try:
            # 執行虛擬交易
            result = await self.execute_virtual_trade(
                model_version=model_version,
                symbol=symbol,
                side=side,
                price=actual_price,
                quantity=suggested_quantity,
                signal_strength=signal_strength,
                signal_confidence=signal_confidence
            )
            
            # 也記錄到舊的 shadow_trades 表
            await async_db_manager.execute(
                """
                INSERT INTO shadow_trades
                (timestamp, model_version, symbol, side, signal_strength, signal_confidence,
                 suggested_quantity, suggested_price, actual_price, would_have_triggered, reason)
                VALUES (CURRENT_TIMESTAMP, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                model_version, symbol, side, signal_strength, signal_confidence,
                float(suggested_quantity), float(suggested_price), float(actual_price), 
                would_have_triggered, reason
            )
            
            return result.get("success", False)
        except Exception as e:
            logger.error(f"Failed to record shadow trade: {e}")
            return False

    async def get_shadow_trades(self, model_version: str, limit: int = 100) -> List[Dict[str, Any]]:
        """獲取影子交易記錄"""
        try:
            rows = await async_db_manager.execute(
                """
                SELECT timestamp, symbol, side, signal_strength, signal_confidence,
                       suggested_quantity, suggested_price, actual_price,
                       virtual_pnl, would_have_triggered, reason
                FROM shadow_trades
                WHERE model_version = $1
                ORDER BY timestamp DESC
                LIMIT $2
                """,
                model_version, limit
            )
            return [
                {
                    "timestamp": _parse_timestamp(row['timestamp']),
                    "symbol": row['symbol'],
                    "side": row['side'],
                    "signal_strength": float(row['signal_strength']) if row['signal_strength'] else None,
                    "signal_confidence": float(row['signal_confidence']) if row['signal_confidence'] else None,
                    "suggested_quantity": float(row['suggested_quantity']) if row['suggested_quantity'] else None,
                    "suggested_price": float(row['suggested_price']) if row['suggested_price'] else None,
                    "actual_price": float(row['actual_price']) if row['actual_price'] else None,
                    "virtual_pnl": float(row['virtual_pnl']) if row['virtual_pnl'] else None,
                    "would_have_triggered": bool(row['would_have_triggered']),
                    "reason": row['reason'],
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get shadow trades: {e}")
            return []

    # ==================== 策略靈敏度 ====================

    async def get_sensitivity_config(self) -> SensitivityConfig:
        """獲取策略靈敏度配置"""
        try:
            row = await async_db_manager.execute_one(
                "SELECT sensitivity, confidence_threshold, strength_threshold, position_ratio FROM strategy_sensitivity WHERE id = 1"
            )
            if not row:
                return SensitivityConfig("moderate", 0.70, 0.60, 0.6)
            return SensitivityConfig(
                sensitivity=row['sensitivity'],
                confidence_threshold=float(row['confidence_threshold']),
                strength_threshold=float(row['strength_threshold']),
                position_ratio=float(row['position_ratio'])
            )
        except Exception as e:
            logger.error(f"Failed to get sensitivity config: {e}")
            return SensitivityConfig("moderate", 0.70, 0.60, 0.6)

    async def update_sensitivity_config(
        self, sensitivity: str, confidence_threshold: float,
        strength_threshold: float, position_ratio: float
    ) -> bool:
        """更新策略靈敏度配置"""
        try:
            result = await async_db_manager.execute(
                """
                UPDATE strategy_sensitivity
                SET sensitivity = $1, confidence_threshold = $2, strength_threshold = $3, 
                    position_ratio = $4, updated_at = CURRENT_TIMESTAMP
                WHERE id = 1
                RETURNING id
                """,
                sensitivity, confidence_threshold, strength_threshold, position_ratio
            )
            if not result:
                await async_db_manager.execute(
                    """
                    INSERT INTO strategy_sensitivity (id, sensitivity, confidence_threshold, strength_threshold, position_ratio)
                    VALUES (1, $1, $2, $3, $4)
                    """,
                    sensitivity, confidence_threshold, strength_threshold, position_ratio
                )
            logger.info(f"Sensitivity config updated to: {sensitivity}")
            return True
        except Exception as e:
            logger.error(f"Failed to update sensitivity config: {e}")
            return False

    # ==================== 晉升歷史 ====================

    async def get_promotion_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """獲取晉升歷史"""
        try:
            rows = await async_db_manager.execute(
                """
                SELECT timestamp, previous_champion, new_champion, reason,
                       champion_performance, challenger_performance, promoted_by
                FROM promotion_history
                ORDER BY timestamp DESC
                LIMIT $1
                """,
                limit
            )
            return [
                {
                    "timestamp": _parse_timestamp(row['timestamp']),
                    "previous_champion": row['previous_champion'],
                    "new_champion": row['new_champion'],
                    "reason": row['reason'],
                    "champion_performance": row['champion_performance'],
                    "challenger_performance": row['challenger_performance'],
                    "promoted_by": row['promoted_by'],
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get promotion history: {e}")
            return []

    # ==================== 虛擬交易代理方法 ====================

    async def init_virtual_account(self, model_version: str) -> bool:
        """初始化模型的虛擬帳戶"""
        from src.services.virtual_trading_service import virtual_trading_service
        return await virtual_trading_service.ensure_account(model_version)

    async def get_virtual_account(self, model_version: str) -> Optional[Dict[str, Any]]:
        """獲取虛擬帳戶"""
        from src.services.virtual_trading_service import virtual_trading_service
        account = await virtual_trading_service.get_account(model_version)
        return account.to_dict() if account else None

    async def get_virtual_positions(self, model_version: str) -> List:
        """獲取虛擬持倉"""
        from src.services.virtual_trading_service import virtual_trading_service
        return await virtual_trading_service.get_all_positions(model_version)

    async def get_virtual_trades(self, model_version: str, limit: int = 50) -> List[Dict[str, Any]]:
        """獲取虛擬交易歷史"""
        from src.services.virtual_trading_service import virtual_trading_service
        return await virtual_trading_service.get_trade_history(model_version, limit)

    async def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """獲取排行榜"""
        from src.services.virtual_trading_service import virtual_trading_service
        return await virtual_trading_service.get_leaderboard(limit)

    # ==================== 排行榜 ====================

    async def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """獲取模型排行榜（按總回報排序）"""
        try:
            rows = await async_db_manager.execute(
                """
                SELECT va.model_version, va.total_value, va.total_pnl, va.total_pnl_pct,
                       va.win_rate, va.max_drawdown, va.trade_count,
                       am.model_type, am.status
                FROM virtual_accounts va
                JOIN arena_models am ON va.model_version = am.version
                ORDER BY va.total_pnl DESC
                LIMIT $1
                """,
                limit
            )
            return [
                {
                    "rank": idx + 1,
                    "model_version": row['model_version'],
                    "model_type": row['model_type'],
                    "status": row['status'],
                    "total_value": float(row['total_value']),
                    "total_pnl": float(row['total_pnl']),
                    "total_pnl_pct": float(row['total_pnl_pct']),
                    "win_rate": float(row['win_rate']),
                    "max_drawdown": float(row['max_drawdown']),
                    "trade_count": int(row['trade_count']),
                }
                for idx, row in enumerate(rows)
            ]
        except Exception as e:
            logger.error(f"Failed to get leaderboard: {e}")
            return []


    # ==================== 相容性方法 ====================

    async def update_model_stats(self, model_version: str = None, version: str = None, **kwargs) -> bool:
        """更新模型統計數據（相容舊 API，支援 model_version 或 version 參數，忽略額外參數）"""
        # 支援兩種參數命名
        ver = model_version or version
        if not ver:
            logger.error("update_model_stats: no model_version or version provided")
            return False
        try:
            # 注意：_update_account_stats 會從資料庫重新計算統計，不需要外部傳入的 trade_count 等
            await self._update_account_stats(ver)
            return True
        except Exception as e:
            logger.error(f"Failed to update model stats for {ver}: {e}")
            return False


# 單例實例
arena_service = ArenaService()
