"""
Virtual Trading Service - 虛擬交易服務

為競技場模型提供 10000 USDT 虛擬資金進行模擬交易
追蹤虛擬持倉、盈虧、勝率等統計數據
"""

from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from src.logger import logger
from src.utils.database import async_db_manager


INITIAL_BALANCE = Decimal("10000")  # 初始虛擬資金
COMMISSION_RATE = Decimal("0.001")  # 0.1% 手續費


@dataclass
class VirtualAccount:
    """虛擬帳戶"""
    model_version: str
    balance: Decimal
    total_value: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal
    total_pnl_pct: float
    max_drawdown: float
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "balance": float(self.balance),
            "total_value": float(self.total_value),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "total_pnl": float(self.total_pnl),
            "total_pnl_pct": self.total_pnl_pct,
            "max_drawdown": self.max_drawdown,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_rate,
        }


@dataclass
class VirtualPosition:
    """虛擬持倉"""
    model_version: str
    symbol: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": float(self.quantity),
            "entry_price": float(self.entry_price),
            "current_price": float(self.current_price),
            "unrealized_pnl": float(self.unrealized_pnl),
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
        }


class VirtualTradingService:
    """虛擬交易服務"""

    def __init__(self):
        self._initialized = False

    async def initialize(self):
        """初始化服務"""
        if self._initialized:
            return
        self._initialized = True
        logger.info("VirtualTradingService initialized")

    # ==================== 帳戶管理 ====================

    async def ensure_account(self, model_version: str) -> bool:
        """確保模型有虛擬帳戶，沒有則創建"""
        try:
            existing = await async_db_manager.execute_one(
                "SELECT 1 FROM virtual_accounts WHERE model_version = $1",
                model_version
            )
            
            if not existing:
                await async_db_manager.execute(
                    """
                    INSERT INTO virtual_accounts 
                    (model_version, initial_balance, balance, total_value, peak_value)
                    VALUES ($1, $2, $2, $2, $2)
                    ON CONFLICT (model_version) DO NOTHING
                    """,
                    model_version, INITIAL_BALANCE
                )
                logger.info(f"Created virtual account for {model_version} with {INITIAL_BALANCE} USDT")
            
            return True
        except Exception as e:
            logger.error(f"Failed to ensure account for {model_version}: {e}")
            return False

    async def get_account(self, model_version: str) -> Optional[VirtualAccount]:
        """獲取虛擬帳戶"""
        try:
            await self.ensure_account(model_version)
            
            row = await async_db_manager.execute_one(
                """
                SELECT model_version, balance, total_value, realized_pnl, 
                       unrealized_pnl, total_pnl, total_pnl_pct, max_drawdown,
                       trade_count, win_count, loss_count, win_rate
                FROM virtual_accounts
                WHERE model_version = $1
                """,
                model_version
            )
            
            if not row:
                return None
            
            return VirtualAccount(
                model_version=row['model_version'],
                balance=Decimal(str(row['balance'])),
                total_value=Decimal(str(row['total_value'])),
                realized_pnl=Decimal(str(row['realized_pnl'])),
                unrealized_pnl=Decimal(str(row['unrealized_pnl'])),
                total_pnl=Decimal(str(row['total_pnl'])),
                total_pnl_pct=float(row['total_pnl_pct']),
                max_drawdown=float(row['max_drawdown']),
                trade_count=int(row['trade_count']),
                win_count=int(row['win_count']),
                loss_count=int(row['loss_count']),
                win_rate=float(row['win_rate']),
            )
        except Exception as e:
            logger.error(f"Failed to get account for {model_version}: {e}")
            return None

    async def reset_account(self, model_version: str) -> bool:
        """重置虛擬帳戶"""
        try:
            # 清除持倉
            await async_db_manager.execute(
                "DELETE FROM virtual_positions WHERE model_version = $1",
                model_version
            )
            
            # 重置帳戶
            await async_db_manager.execute(
                """
                UPDATE virtual_accounts
                SET balance = $2, total_value = $2, realized_pnl = 0,
                    unrealized_pnl = 0, total_pnl = 0, total_pnl_pct = 0,
                    max_drawdown = 0, peak_value = $2, trade_count = 0,
                    win_count = 0, loss_count = 0, win_rate = 0,
                    updated_at = CURRENT_TIMESTAMP
                WHERE model_version = $1
                """,
                model_version, INITIAL_BALANCE
            )
            
            logger.info(f"Reset virtual account for {model_version}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset account for {model_version}: {e}")
            return False

    # ==================== 持倉管理 ====================

    async def get_position(self, model_version: str, symbol: str) -> Optional[VirtualPosition]:
        """獲取虛擬持倉"""
        try:
            row = await async_db_manager.execute_one(
                """
                SELECT model_version, symbol, side, quantity, entry_price,
                       current_price, unrealized_pnl, unrealized_pnl_pct
                FROM virtual_positions
                WHERE model_version = $1 AND symbol = $2
                """,
                model_version, symbol
            )
            
            if not row or Decimal(str(row['quantity'])) <= 0:
                return None
            
            return VirtualPosition(
                model_version=row['model_version'],
                symbol=row['symbol'],
                side=row['side'],
                quantity=Decimal(str(row['quantity'])),
                entry_price=Decimal(str(row['entry_price'])),
                current_price=Decimal(str(row['current_price'] or row['entry_price'])),
                unrealized_pnl=Decimal(str(row['unrealized_pnl'] or 0)),
                unrealized_pnl_pct=float(row['unrealized_pnl_pct'] or 0),
            )
        except Exception as e:
            logger.error(f"Failed to get position for {model_version}/{symbol}: {e}")
            return None

    async def get_all_positions(self, model_version: str) -> List[VirtualPosition]:
        """獲取所有虛擬持倉"""
        try:
            rows = await async_db_manager.execute(
                """
                SELECT model_version, symbol, side, quantity, entry_price,
                       current_price, unrealized_pnl, unrealized_pnl_pct
                FROM virtual_positions
                WHERE model_version = $1 AND quantity > 0
                """,
                model_version
            )
            
            return [
                VirtualPosition(
                    model_version=row['model_version'],
                    symbol=row['symbol'],
                    side=row['side'],
                    quantity=Decimal(str(row['quantity'])),
                    entry_price=Decimal(str(row['entry_price'])),
                    current_price=Decimal(str(row['current_price'] or row['entry_price'])),
                    unrealized_pnl=Decimal(str(row['unrealized_pnl'] or 0)),
                    unrealized_pnl_pct=float(row['unrealized_pnl_pct'] or 0),
                )
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get positions for {model_version}: {e}")
            return []

    # ==================== 交易執行 ====================

    async def execute_virtual_buy(
        self,
        model_version: str,
        symbol: str,
        price: Decimal,
        quantity: Optional[Decimal] = None,
        quote_amount: Optional[Decimal] = None,
        signal_strength: float = 0,
        signal_confidence: float = 0,
    ) -> Dict[str, Any]:
        """執行虛擬買入"""
        try:
            await self.ensure_account(model_version)
            
            # 獲取帳戶
            account = await self.get_account(model_version)
            if not account:
                return {"success": False, "error": "Account not found"}
            
            # 計算數量和金額
            if quantity:
                quote_value = quantity * price
            elif quote_amount:
                quote_value = quote_amount
                quantity = (quote_amount / price).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
            else:
                # 預設使用 10% 餘額
                quote_value = account.balance * Decimal("0.1")
                quantity = (quote_value / price).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
            
            # 計算手續費
            commission = quote_value * COMMISSION_RATE
            total_cost = quote_value + commission
            
            # 檢查餘額
            if account.balance < total_cost:
                return {"success": False, "error": f"Insufficient balance: {account.balance} < {total_cost}"}
            
            # 檢查現有持倉
            existing_position = await self.get_position(model_version, symbol)
            
            if existing_position:
                # 加倉 - 計算新的平均成本
                new_quantity = existing_position.quantity + quantity
                new_entry_price = (
                    (existing_position.entry_price * existing_position.quantity + price * quantity)
                    / new_quantity
                )
                
                await async_db_manager.execute(
                    """
                    UPDATE virtual_positions
                    SET quantity = $3, entry_price = $4, current_price = $5, updated_at = CURRENT_TIMESTAMP
                    WHERE model_version = $1 AND symbol = $2
                    """,
                    model_version, symbol, new_quantity, new_entry_price, price
                )
            else:
                # 新建持倉
                await async_db_manager.execute(
                    """
                    INSERT INTO virtual_positions
                    (model_version, symbol, side, quantity, entry_price, current_price)
                    VALUES ($1, $2, 'LONG', $3, $4, $4)
                    ON CONFLICT (model_version, symbol) DO UPDATE SET
                        quantity = $3, entry_price = $4, current_price = $4,
                        side = 'LONG', updated_at = CURRENT_TIMESTAMP
                    """,
                    model_version, symbol, quantity, price
                )
            
            # 扣除餘額
            new_balance = account.balance - total_cost
            await async_db_manager.execute(
                """
                UPDATE virtual_accounts
                SET balance = $2, trade_count = trade_count + 1, updated_at = CURRENT_TIMESTAMP
                WHERE model_version = $1
                """,
                model_version, new_balance
            )
            
            # 記錄交易
            await async_db_manager.execute(
                """
                INSERT INTO virtual_trades
                (timestamp, model_version, symbol, side, quantity, price, quote_value, 
                 commission, balance_after, signal_strength, signal_confidence)
                VALUES (CURRENT_TIMESTAMP, $1, $2, 'BUY', $3, $4, $5, $6, $7, $8, $9)
                """,
                model_version, symbol, quantity, price, quote_value,
                commission, new_balance, signal_strength, signal_confidence
            )
            
            # 更新統計
            await self._update_account_stats(model_version)
            
            logger.info(f"Virtual BUY: {model_version} bought {quantity} {symbol} @ {price}")
            
            return {
                "success": True,
                "side": "BUY",
                "symbol": symbol,
                "quantity": float(quantity),
                "price": float(price),
                "quote_value": float(quote_value),
                "commission": float(commission),
                "balance_after": float(new_balance),
            }
            
        except Exception as e:
            logger.error(f"Failed to execute virtual buy: {e}")
            return {"success": False, "error": str(e)}

    async def execute_virtual_sell(
        self,
        model_version: str,
        symbol: str,
        price: Decimal,
        quantity: Optional[Decimal] = None,
        signal_strength: float = 0,
        signal_confidence: float = 0,
    ) -> Dict[str, Any]:
        """執行虛擬賣出"""
        try:
            # 獲取持倉
            position = await self.get_position(model_version, symbol)
            if not position or position.quantity <= 0:
                return {"success": False, "error": f"No position for {symbol}"}
            
            # 獲取帳戶
            account = await self.get_account(model_version)
            if not account:
                return {"success": False, "error": "Account not found"}
            
            # 決定賣出數量
            sell_quantity = quantity if quantity and quantity <= position.quantity else position.quantity
            
            # 計算金額
            quote_value = sell_quantity * price
            commission = quote_value * COMMISSION_RATE
            net_proceeds = quote_value - commission
            
            # 計算已實現盈虧
            cost_basis = sell_quantity * position.entry_price
            realized_pnl = net_proceeds - cost_basis
            
            # 更新餘額
            new_balance = account.balance + net_proceeds
            
            # 判斷勝負
            is_win = realized_pnl > 0
            
            # 更新帳戶
            await async_db_manager.execute(
                """
                UPDATE virtual_accounts
                SET balance = $2, 
                    realized_pnl = realized_pnl + $3,
                    trade_count = trade_count + 1,
                    win_count = win_count + $4,
                    loss_count = loss_count + $5,
                    updated_at = CURRENT_TIMESTAMP
                WHERE model_version = $1
                """,
                model_version, new_balance, realized_pnl,
                1 if is_win else 0, 0 if is_win else 1
            )
            
            # 更新或刪除持倉
            remaining_quantity = position.quantity - sell_quantity
            if remaining_quantity <= 0:
                await async_db_manager.execute(
                    "DELETE FROM virtual_positions WHERE model_version = $1 AND symbol = $2",
                    model_version, symbol
                )
            else:
                await async_db_manager.execute(
                    """
                    UPDATE virtual_positions
                    SET quantity = $3, current_price = $4, updated_at = CURRENT_TIMESTAMP
                    WHERE model_version = $1 AND symbol = $2
                    """,
                    model_version, symbol, remaining_quantity, price
                )
            
            # 記錄交易
            await async_db_manager.execute(
                """
                INSERT INTO virtual_trades
                (timestamp, model_version, symbol, side, quantity, price, quote_value,
                 commission, realized_pnl, balance_after, signal_strength, signal_confidence)
                VALUES (CURRENT_TIMESTAMP, $1, $2, 'SELL', $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                model_version, symbol, sell_quantity, price, quote_value,
                commission, realized_pnl, new_balance, signal_strength, signal_confidence
            )
            
            # 更新統計
            await self._update_account_stats(model_version)
            
            pnl_emoji = "🟢" if is_win else "🔴"
            logger.info(f"Virtual SELL: {model_version} sold {sell_quantity} {symbol} @ {price}, PnL: {pnl_emoji} {realized_pnl}")
            
            return {
                "success": True,
                "side": "SELL",
                "symbol": symbol,
                "quantity": float(sell_quantity),
                "price": float(price),
                "quote_value": float(quote_value),
                "commission": float(commission),
                "realized_pnl": float(realized_pnl),
                "is_win": is_win,
                "balance_after": float(new_balance),
            }
            
        except Exception as e:
            logger.error(f"Failed to execute virtual sell: {e}")
            return {"success": False, "error": str(e)}

    # ==================== 價格更新 ====================

    async def update_position_prices(self, symbol: str, current_price: Decimal) -> int:
        """更新所有模型的持倉價格"""
        try:
            # 更新持倉價格和未實現盈虧
            result = await async_db_manager.execute(
                """
                UPDATE virtual_positions
                SET current_price = $2,
                    unrealized_pnl = (quantity * $2) - (quantity * entry_price),
                    unrealized_pnl_pct = CASE 
                        WHEN entry_price > 0 THEN (($2 - entry_price) / entry_price) * 100
                        ELSE 0
                    END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE symbol = $1 AND quantity > 0
                RETURNING model_version
                """,
                symbol, current_price
            )
            
            updated_count = len(result) if result else 0
            
            # 更新相關帳戶的統計
            if result:
                for row in result:
                    await self._update_account_stats(row['model_version'])
            
            return updated_count
            
        except Exception as e:
            logger.error(f"Failed to update position prices for {symbol}: {e}")
            return 0

    async def _update_account_stats(self, model_version: str):
        """更新帳戶統計數據"""
        try:
            # 計算未實現盈虧
            unrealized_row = await async_db_manager.execute_one(
                """
                SELECT COALESCE(SUM(unrealized_pnl), 0) as total_unrealized,
                       COALESCE(SUM(quantity * current_price), 0) as position_value
                FROM virtual_positions
                WHERE model_version = $1 AND quantity > 0
                """,
                model_version
            )
            
            unrealized_pnl = Decimal(str(unrealized_row['total_unrealized'] or 0))
            position_value = Decimal(str(unrealized_row['position_value'] or 0))
            
            # 獲取餘額和已實現盈虧
            account_row = await async_db_manager.execute_one(
                """
                SELECT balance, realized_pnl, peak_value, trade_count, win_count, loss_count
                FROM virtual_accounts
                WHERE model_version = $1
                """,
                model_version
            )
            
            if not account_row:
                return
            
            balance = Decimal(str(account_row['balance']))
            realized_pnl = Decimal(str(account_row['realized_pnl']))
            peak_value = Decimal(str(account_row['peak_value']))
            trade_count = int(account_row['trade_count'])
            win_count = int(account_row['win_count'])
            loss_count = int(account_row['loss_count'])
            
            # 計算總價值和盈虧
            total_value = balance + position_value
            total_pnl = realized_pnl + unrealized_pnl
            total_pnl_pct = float((total_pnl / INITIAL_BALANCE) * 100) if INITIAL_BALANCE > 0 else 0
            
            # 更新峰值和回撤
            new_peak = max(peak_value, total_value)
            drawdown = float(((new_peak - total_value) / new_peak) * 100) if new_peak > 0 else 0
            
            # 計算勝率
            total_trades = win_count + loss_count
            win_rate = float((win_count / total_trades) * 100) if total_trades > 0 else 0
            
            # 更新帳戶
            await async_db_manager.execute(
                """
                UPDATE virtual_accounts
                SET total_value = $2, unrealized_pnl = $3, total_pnl = $4,
                    total_pnl_pct = $5, peak_value = $6, max_drawdown = GREATEST(max_drawdown, $7),
                    win_rate = $8, updated_at = CURRENT_TIMESTAMP
                WHERE model_version = $1
                """,
                model_version, total_value, unrealized_pnl, total_pnl,
                total_pnl_pct, new_peak, drawdown, win_rate
            )
            
            # 同步更新 arena_models 表
            await async_db_manager.execute(
                """
                UPDATE arena_models
                SET win_rate = $2, total_return = $3, draw_down = $4, updated_at = CURRENT_TIMESTAMP
                WHERE version = $1
                """,
                model_version, win_rate, total_pnl_pct, drawdown
            )
            
        except Exception as e:
            logger.error(f"Failed to update account stats for {model_version}: {e}")

    # ==================== 交易歷史 ====================

    async def get_trade_history(
        self,
        model_version: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """獲取交易歷史"""
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
                    "timestamp": row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                    "symbol": row['symbol'],
                    "side": row['side'],
                    "quantity": float(row['quantity']),
                    "price": float(row['price']),
                    "quote_value": float(row['quote_value']),
                    "commission": float(row['commission'] or 0),
                    "realized_pnl": float(row['realized_pnl']) if row['realized_pnl'] else None,
                    "balance_after": float(row['balance_after'] or 0),
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get trade history for {model_version}: {e}")
            return []

    # ==================== 排行榜 ====================

    async def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """獲取排行榜"""
        try:
            rows = await async_db_manager.execute(
                """
                SELECT va.model_version, am.model_type, va.balance, va.total_value,
                       va.realized_pnl, va.unrealized_pnl, va.total_pnl, va.total_pnl_pct,
                       va.max_drawdown, va.trade_count, va.win_count, va.loss_count, va.win_rate
                FROM virtual_accounts va
                JOIN arena_models am ON va.model_version = am.version
                ORDER BY va.total_pnl_pct DESC
                LIMIT $1
                """,
                limit
            )
            
            return [
                {
                    "rank": idx + 1,
                    "model_version": row['model_version'],
                    "model_type": row['model_type'],
                    "balance": float(row['balance']),
                    "total_value": float(row['total_value']),
                    "realized_pnl": float(row['realized_pnl']),
                    "unrealized_pnl": float(row['unrealized_pnl']),
                    "total_pnl": float(row['total_pnl']),
                    "total_pnl_pct": float(row['total_pnl_pct']),
                    "max_drawdown": float(row['max_drawdown']),
                    "trade_count": int(row['trade_count']),
                    "win_rate": float(row['win_rate']),
                }
                for idx, row in enumerate(rows)
            ]
        except Exception as e:
            logger.error(f"Failed to get leaderboard: {e}")
            return []


    # ==================== 別名方法（相容性）====================

    async def get_or_create_account(self, model_version: str) -> Optional[VirtualAccount]:
        """獲取或創建虛擬帳戶（相容舊 API）"""
        await self.ensure_account(model_version)
        return await self.get_account(model_version)

    async def update_positions_price(self, symbol: str, current_price: Decimal) -> int:
        """更新持倉價格（別名方法，相容舊 API）"""
        return await self.update_position_prices(symbol, current_price)


# 單例實例
virtual_trading_service = VirtualTradingService()
