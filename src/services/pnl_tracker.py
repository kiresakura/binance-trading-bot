"""
PnL Tracker Service - 盈虧追蹤服務

功能：
1. 同步 Binance 歷史交易記錄
2. 計算持倉成本（平均成本法）
3. 計算已實現/未實現盈虧
4. 記錄每日盈虧快照
5. 提供盈虧歷史查詢

Author: Trading Bot System
Version: 0.2.1
"""

import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.logger import logger
from src.services.binance_client import binance_client


@dataclass
class TradeRecord:
    """交易記錄"""
    id: int
    order_id: int
    symbol: str
    trade_time: datetime
    side: str  # BUY or SELL
    price: Decimal
    qty: Decimal
    quote_qty: Decimal
    commission: Decimal
    commission_asset: str
    is_buyer: bool
    is_maker: bool


@dataclass
class PositionCost:
    """持倉成本"""
    symbol: str
    base_asset: str
    quote_asset: str
    quantity: Decimal
    total_cost: Decimal
    average_cost: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal


@dataclass
class DailyPnLRecord:
    """每日盈虧記錄"""
    date: date
    symbol: str
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal
    pnl_percentage: Decimal
    trade_count: int
    buy_volume: Decimal
    sell_volume: Decimal
    commission_total: Decimal


@dataclass
class PnLSummary:
    """盈虧摘要"""
    total_realized_pnl: Decimal
    total_unrealized_pnl: Decimal
    total_pnl: Decimal
    total_pnl_percentage: Decimal
    today_pnl: Decimal
    today_pnl_percentage: Decimal
    total_commission: Decimal
    total_trades: int
    win_rate: Decimal
    positions: List[PositionCost]


class PnLTracker:
    """盈虧追蹤服務"""

    def __init__(self):
        self._db = None
        self._initialized = False
        self._sync_lock = asyncio.Lock()

        # 定時任務相關
        self._scheduler_running = False
        self._scheduler_task = None

        # 交易對配置（哪些需要追蹤）
        self.tracked_symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
            "ADAUSDT", "DOGEUSDT", "XRPUSDT", "DOTUSDT"
        ]

    async def init(self):
        """初始化服務"""
        if self._initialized:
            return

        try:
            from src.utils.database import async_db_manager
            self._db = async_db_manager

            if not self._db._initialized:
                await self._db.init()

            self._initialized = True
            logger.info("✅ PnL Tracker initialized")

        except Exception as e:
            logger.warning(f"⚠️ PnL Tracker init failed (will use memory mode): {e}")
            self._initialized = True  # 允許在無數據庫模式下運行

    async def sync_historical_trades(
        self,
        symbols: List[str] = None,
        days_back: int = 30
    ) -> Dict:
        """
        同步歷史交易記錄

        Args:
            symbols: 要同步的交易對列表，None 表示所有追蹤的交易對
            days_back: 往回同步多少天

        Returns:
            同步結果摘要
        """
        async with self._sync_lock:
            if symbols is None:
                symbols = self.tracked_symbols

            result = {
                "synced_symbols": [],
                "total_trades": 0,
                "new_trades": 0,
                "errors": []
            }

            # days_back=0 表示同步所有歷史交易
            if days_back > 0:
                start_time = int((datetime.utcnow() - timedelta(days=days_back)).timestamp() * 1000)
            else:
                start_time = 0  # 同步所有

            for symbol in symbols:
                try:
                    # 從 Binance 獲取交易記錄
                    trades = await binance_client.get_my_trades(symbol, limit=1000)

                    # 檢查是否為有效的列表（Binance 可能返回錯誤訊息）
                    if not trades or not isinstance(trades, list):
                        if isinstance(trades, dict) and 'msg' in trades:
                            raise Exception(f"Binance API error: {trades.get('msg')}")
                        continue

                    new_count = 0
                    for trade in trades:
                        trade_time = trade.get('time', 0)
                        # 只在設定了時間範圍時跳過舊交易
                        if start_time > 0 and trade_time < start_time:
                            continue

                        # 保存到數據庫
                        saved = await self._save_trade(symbol, trade)
                        if saved:
                            new_count += 1

                    if new_count > 0:
                        result["synced_symbols"].append(symbol)
                        result["new_trades"] += new_count

                    result["total_trades"] += len(trades)

                    logger.info(f"📊 Synced {symbol}: {len(trades)} trades, {new_count} new")

                except Exception as e:
                    logger.error(f"❌ Failed to sync {symbol}: {e}")
                    result["errors"].append({"symbol": symbol, "error": str(e)})

            # 同步後重新計算持倉成本
            await self._recalculate_all_positions()

            return result

    async def _save_trade(self, symbol: str, trade: Dict) -> bool:
        """保存單筆交易到數據庫，並計算賣出的已實現盈虧"""
        if not self._db:
            return False

        try:
            trade_id = trade.get('id')

            # 檢查是否已存在
            existing = await self._db.fetch_one(
                "SELECT id FROM binance_trades WHERE id = $1 AND symbol = $2",
                trade_id, symbol
            )

            if existing:
                return False  # 已存在

            # 轉換數據
            trade_time = datetime.utcfromtimestamp(trade.get('time', 0) / 1000)
            is_buyer = trade.get('isBuyer', False)
            price = Decimal(trade.get('price', '0'))
            qty = Decimal(trade.get('qty', '0'))
            quote_qty = Decimal(trade.get('quoteQty', '0'))
            commission = Decimal(trade.get('commission', '0'))
            commission_asset = trade.get('commissionAsset', '')

            # 計算已實現盈虧（僅賣出時）
            realized_pnl = None
            if not is_buyer:
                realized_pnl = await self._calculate_realized_pnl_for_sell(
                    symbol, trade_time, qty, price, commission, commission_asset
                )

            await self._db.execute("""
                INSERT INTO binance_trades
                (id, order_id, symbol, trade_time, side, price, qty, quote_qty,
                 commission, commission_asset, is_buyer, is_maker, is_best_match, realized_pnl)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (id, symbol) DO NOTHING
            """,
                trade_id,
                trade.get('orderId'),
                symbol,
                trade_time,
                'BUY' if is_buyer else 'SELL',
                price,
                qty,
                quote_qty,
                commission,
                commission_asset,
                is_buyer,
                trade.get('isMaker', False),
                trade.get('isBestMatch', True),
                realized_pnl
            )

            return True

        except Exception as e:
            logger.error(f"Failed to save trade {trade.get('id')}: {e}")
            return False

    async def _calculate_realized_pnl_for_sell(
        self,
        symbol: str,
        sell_time: datetime,
        sell_qty: Decimal,
        sell_price: Decimal,
        commission: Decimal,
        commission_asset: str
    ) -> Decimal:
        """
        計算賣出交易的已實現盈虧（平均成本法）

        已實現盈虧 = (賣出價格 - 平均買入價格) × 賣出數量 - 手續費
        """
        try:
            # 獲取在此賣出之前的所有買入交易
            buy_trades = await self._db.fetch_all("""
                SELECT price, qty, quote_qty
                FROM binance_trades
                WHERE symbol = $1 AND side = 'BUY' AND trade_time < $2
                ORDER BY trade_time ASC
            """, symbol, sell_time)

            # 獲取在此賣出之前的所有賣出交易（用於計算剩餘持倉）
            previous_sells = await self._db.fetch_all("""
                SELECT qty
                FROM binance_trades
                WHERE symbol = $1 AND side = 'SELL' AND trade_time < $2
                ORDER BY trade_time ASC
            """, symbol, sell_time)

            if not buy_trades:
                # 沒有買入記錄，可能是轉入的幣
                return Decimal('0')

            # 計算到賣出時點的平均成本
            total_bought_qty = sum(Decimal(str(t['qty'])) for t in buy_trades)
            total_bought_cost = sum(Decimal(str(t['quote_qty'])) for t in buy_trades)
            total_sold_qty = sum(Decimal(str(t['qty'])) for t in previous_sells) if previous_sells else Decimal('0')

            # 剩餘持倉
            remaining_qty = total_bought_qty - total_sold_qty

            if remaining_qty <= 0:
                # 沒有持倉可賣
                return Decimal('0')

            # 計算剩餘持倉的平均成本
            # 使用 FIFO 簡化：平均成本 = 總成本 / 總買入量（這是平均成本法）
            avg_cost = total_bought_cost / total_bought_qty if total_bought_qty > 0 else Decimal('0')

            # 計算已實現盈虧
            sell_revenue = sell_qty * sell_price
            cost_basis = sell_qty * avg_cost
            realized_pnl = sell_revenue - cost_basis

            # 減去手續費（如果是 USDT 計價）
            if commission_asset == 'USDT':
                realized_pnl -= commission

            logger.debug(f"📊 {symbol} SELL: qty={sell_qty}, price={sell_price}, avg_cost={avg_cost:.4f}, pnl={realized_pnl:.4f}")

            return realized_pnl

        except Exception as e:
            logger.error(f"Failed to calculate realized PnL: {e}")
            return Decimal('0')

    async def calculate_position_cost(self, symbol: str) -> Optional[PositionCost]:
        """
        計算指定交易對的持倉成本（平均成本法）

        Returns:
            PositionCost 對象，或 None 如果無持倉
        """
        if not self._db:
            return None

        try:
            # 獲取所有交易記錄，按時間排序
            trades = await self._db.fetch_all("""
                SELECT id, trade_time, side, price, qty, quote_qty, commission, commission_asset
                FROM binance_trades
                WHERE symbol = $1
                ORDER BY trade_time ASC
            """, symbol)

            if not trades:
                return None

            # 計算持倉和成本
            total_qty = Decimal('0')
            total_cost = Decimal('0')
            realized_pnl = Decimal('0')
            total_commission = Decimal('0')

            for trade in trades:
                qty = Decimal(str(trade['qty']))
                price = Decimal(str(trade['price']))
                quote_qty = Decimal(str(trade['quote_qty']))
                commission = Decimal(str(trade['commission']))

                # 如果手續費是以 quote asset 計算
                if trade['commission_asset'] == 'USDT':
                    total_commission += commission

                if trade['side'] == 'BUY':
                    # 買入增加持倉和成本
                    total_cost += quote_qty
                    total_qty += qty
                else:
                    # 賣出減少持倉，計算已實現盈虧
                    if total_qty > 0:
                        avg_cost = total_cost / total_qty
                        sell_cost = avg_cost * qty
                        realized_pnl += quote_qty - sell_cost

                        # 減少成本
                        cost_reduction = sell_cost if qty <= total_qty else total_cost
                        total_cost -= cost_reduction
                        total_qty -= qty

            # 計算平均成本
            avg_cost = total_cost / total_qty if total_qty > 0 else Decimal('0')

            # 獲取當前價格計算未實現盈虧
            unrealized_pnl = Decimal('0')
            if total_qty > 0:
                try:
                    ticker = await binance_client.get_ticker_price(symbol)
                    current_price = Decimal(str(ticker.get('price', '0')))
                    current_value = total_qty * current_price
                    unrealized_pnl = current_value - total_cost
                except:
                    pass

            # 提取 base/quote asset
            base_asset = symbol.replace('USDT', '').replace('BUSD', '').replace('BTC', '')
            quote_asset = 'USDT' if 'USDT' in symbol else 'BUSD' if 'BUSD' in symbol else 'BTC'

            position = PositionCost(
                symbol=symbol,
                base_asset=base_asset,
                quote_asset=quote_asset,
                quantity=total_qty,
                total_cost=total_cost,
                average_cost=avg_cost,
                realized_pnl=realized_pnl - total_commission,  # 減去手續費
                unrealized_pnl=unrealized_pnl
            )

            # 保存到數據庫
            await self._save_position_cost(position)

            return position

        except Exception as e:
            logger.error(f"Failed to calculate position cost for {symbol}: {e}")
            return None

    async def _save_position_cost(self, position: PositionCost):
        """保存持倉成本到數據庫"""
        if not self._db:
            return

        try:
            await self._db.execute("""
                INSERT INTO position_costs
                (symbol, base_asset, quote_asset, quantity, total_cost, average_cost,
                 realized_pnl, unrealized_pnl, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                ON CONFLICT (symbol) DO UPDATE SET
                    quantity = EXCLUDED.quantity,
                    total_cost = EXCLUDED.total_cost,
                    average_cost = EXCLUDED.average_cost,
                    realized_pnl = EXCLUDED.realized_pnl,
                    unrealized_pnl = EXCLUDED.unrealized_pnl,
                    updated_at = NOW()
            """,
                position.symbol,
                position.base_asset,
                position.quote_asset,
                position.quantity,
                position.total_cost,
                position.average_cost,
                position.realized_pnl,
                position.unrealized_pnl
            )
        except Exception as e:
            logger.error(f"Failed to save position cost: {e}")

    async def _recalculate_all_positions(self):
        """重新計算所有持倉成本"""
        for symbol in self.tracked_symbols:
            await self.calculate_position_cost(symbol)

    async def get_unrealized_pnl(self, symbol: str = None) -> Dict:
        """
        獲取未實現盈虧

        Args:
            symbol: 指定交易對，None 表示所有

        Returns:
            未實現盈虧數據
        """
        result = {
            "positions": [],
            "total_unrealized_pnl": Decimal('0'),
            "total_position_value": Decimal('0')
        }

        if not self._db:
            return result

        try:
            if symbol:
                positions = await self._db.fetch_all(
                    "SELECT * FROM position_costs WHERE symbol = $1 AND quantity > 0",
                    symbol
                )
            else:
                positions = await self._db.fetch_all(
                    "SELECT * FROM position_costs WHERE quantity > 0"
                )

            for pos in positions:
                qty = Decimal(str(pos['quantity']))
                if qty <= 0:
                    continue

                # 獲取當前價格
                try:
                    ticker = await binance_client.get_ticker_price(pos['symbol'])
                    current_price = Decimal(str(ticker.get('price', '0')))
                except:
                    current_price = Decimal(str(pos['average_cost']))

                current_value = qty * current_price
                avg_cost = Decimal(str(pos['average_cost']))
                unrealized = current_value - (qty * avg_cost)

                result["positions"].append({
                    "symbol": pos['symbol'],
                    "quantity": str(qty),
                    "average_cost": str(avg_cost),
                    "current_price": str(current_price),
                    "current_value": str(current_value),
                    "unrealized_pnl": str(unrealized),
                    "unrealized_pnl_pct": str((unrealized / (qty * avg_cost) * 100) if avg_cost > 0 else 0)
                })

                result["total_unrealized_pnl"] += unrealized
                result["total_position_value"] += current_value

            result["total_unrealized_pnl"] = str(result["total_unrealized_pnl"])
            result["total_position_value"] = str(result["total_position_value"])

        except Exception as e:
            logger.error(f"Failed to get unrealized PnL: {e}")

        return result

    async def get_total_pnl(self) -> PnLSummary:
        """
        獲取總盈虧摘要

        Returns:
            PnLSummary 對象
        """
        positions = []
        total_realized = Decimal('0')
        total_unrealized = Decimal('0')
        total_commission = Decimal('0')
        total_trades = 0
        win_trades = 0

        if self._db:
            try:
                # 獲取所有持倉
                pos_records = await self._db.fetch_all(
                    "SELECT * FROM position_costs"
                )

                for pos in pos_records:
                    realized = Decimal(str(pos['realized_pnl']))
                    total_realized += realized

                    qty = Decimal(str(pos['quantity']))
                    if qty > 0:
                        # 計算未實現盈虧
                        try:
                            ticker = await binance_client.get_ticker_price(pos['symbol'])
                            current_price = Decimal(str(ticker.get('price', '0')))
                            avg_cost = Decimal(str(pos['average_cost']))
                            unrealized = (current_price - avg_cost) * qty
                            total_unrealized += unrealized
                        except:
                            unrealized = Decimal('0')

                        positions.append(PositionCost(
                            symbol=pos['symbol'],
                            base_asset=pos['base_asset'],
                            quote_asset=pos['quote_asset'],
                            quantity=qty,
                            total_cost=Decimal(str(pos['total_cost'])),
                            average_cost=Decimal(str(pos['average_cost'])),
                            realized_pnl=realized,
                            unrealized_pnl=unrealized
                        ))

                # 獲取交易統計
                stats = await self._db.fetch_one("""
                    SELECT
                        COUNT(*) as total_trades,
                        SUM(commission) as total_commission
                    FROM binance_trades
                    WHERE commission_asset = 'USDT'
                """)

                if stats:
                    total_trades = stats['total_trades'] or 0
                    total_commission = Decimal(str(stats['total_commission'] or 0))

            except Exception as e:
                logger.error(f"Failed to get total PnL: {e}")

        total_pnl = total_realized + total_unrealized

        # 獲取今日盈虧
        today_pnl = Decimal('0')
        today = date.today()
        if self._db:
            try:
                today_record = await self._db.fetch_one(
                    "SELECT total_pnl FROM daily_pnl WHERE date = $1 AND symbol = 'ALL'",
                    today
                )
                if today_record:
                    today_pnl = Decimal(str(today_record['total_pnl']))
            except:
                pass

        # 計算勝率（這裡簡化處理）
        win_rate = Decimal('0')
        if total_trades > 0 and win_trades > 0:
            win_rate = Decimal(win_trades) / Decimal(total_trades) * 100

        return PnLSummary(
            total_realized_pnl=total_realized,
            total_unrealized_pnl=total_unrealized,
            total_pnl=total_pnl,
            total_pnl_percentage=Decimal('0'),  # 需要初始資金來計算
            today_pnl=today_pnl,
            today_pnl_percentage=Decimal('0'),
            total_commission=total_commission,
            total_trades=total_trades,
            win_rate=win_rate,
            positions=positions
        )

    async def record_daily_snapshot(self) -> bool:
        """
        記錄每日盈虧快照

        Returns:
            是否成功
        """
        if not self._db:
            return False

        try:
            # 獲取當前帳戶狀態
            balances = await binance_client.get_balances()

            # 計算總 USDT 價值
            total_usdt = Decimal('0')
            in_position_usdt = Decimal('0')
            position_count = 0

            for balance in balances:
                asset = balance['asset']
                total = Decimal(str(balance['total']))

                if total <= 0:
                    continue

                if asset == 'USDT':
                    total_usdt += total
                else:
                    # 獲取價格
                    try:
                        ticker = await binance_client.get_ticker_price(f"{asset}USDT")
                        price = Decimal(str(ticker.get('price', '0')))
                        value = total * price
                        total_usdt += value
                        in_position_usdt += value
                        position_count += 1
                    except:
                        pass

            # 獲取 BTC 價格
            btc_price = Decimal('0')
            try:
                btc_ticker = await binance_client.get_ticker_price("BTCUSDT")
                btc_price = Decimal(str(btc_ticker.get('price', '0')))
            except:
                pass

            # 獲取總盈虧
            pnl_summary = await self.get_total_pnl()

            # 插入快照
            await self._db.execute("""
                INSERT INTO pnl_snapshots
                (timestamp, snapshot_type, total_balance_usdt, available_balance_usdt,
                 in_position_usdt, total_realized_pnl, total_unrealized_pnl, total_pnl,
                 btc_price, position_count)
                VALUES (NOW(), 'daily', $1, $2, $3, $4, $5, $6, $7, $8)
            """,
                total_usdt,
                total_usdt - in_position_usdt,
                in_position_usdt,
                pnl_summary.total_realized_pnl,
                pnl_summary.total_unrealized_pnl,
                pnl_summary.total_pnl,
                btc_price,
                position_count
            )

            logger.info(f"📸 Recorded daily snapshot: ${total_usdt:.2f}")
            return True

        except Exception as e:
            logger.error(f"Failed to record daily snapshot: {e}")
            return False

    async def get_pnl_history(
        self,
        days: int = 30,
        symbol: str = None
    ) -> List[Dict]:
        """
        獲取盈虧歷史

        Args:
            days: 查詢天數
            symbol: 指定交易對，None 表示總計

        Returns:
            盈虧歷史列表
        """
        if not self._db:
            return []

        try:
            if symbol:
                records = await self._db.fetch_all("""
                    SELECT * FROM daily_pnl
                    WHERE symbol = $1 AND date >= CURRENT_DATE - $2::INTEGER
                    ORDER BY date DESC
                """, symbol, days)
            else:
                records = await self._db.fetch_all("""
                    SELECT * FROM daily_pnl
                    WHERE symbol = 'ALL' AND date >= CURRENT_DATE - $1::INTEGER
                    ORDER BY date DESC
                """, days)

            return [
                {
                    "date": str(r['date']),
                    "realized_pnl": str(r['realized_pnl']),
                    "unrealized_pnl": str(r['unrealized_pnl']),
                    "total_pnl": str(r['total_pnl']),
                    "pnl_percentage": str(r['pnl_percentage']),
                    "trade_count": r['trade_count'],
                    "buy_volume": str(r['buy_volume']),
                    "sell_volume": str(r['sell_volume']),
                    "commission_total": str(r['commission_total'])
                }
                for r in records
            ]

        except Exception as e:
            logger.error(f"Failed to get PnL history: {e}")
            return []

    async def get_trade_history(
        self,
        symbol: str = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """
        獲取交易歷史

        Args:
            symbol: 指定交易對，None 表示所有
            limit: 返回數量
            offset: 偏移量

        Returns:
            交易歷史列表
        """
        if not self._db:
            return []

        try:
            if symbol:
                records = await self._db.fetch_all("""
                    SELECT * FROM binance_trades
                    WHERE symbol = $1
                    ORDER BY trade_time DESC
                    LIMIT $2 OFFSET $3
                """, symbol, limit, offset)
            else:
                records = await self._db.fetch_all("""
                    SELECT * FROM binance_trades
                    ORDER BY trade_time DESC
                    LIMIT $1 OFFSET $2
                """, limit, offset)

            return [
                {
                    "id": r['id'],
                    "symbol": r['symbol'],
                    "side": r['side'],
                    "price": str(r['price']),
                    "quantity": str(r['qty']),
                    "quote_qty": str(r['quote_qty']),
                    "commission": str(r['commission']),
                    "commission_asset": r['commission_asset'],
                    "trade_time": r['trade_time'].isoformat(),
                    "is_maker": r['is_maker'],
                    "realized_pnl": str(r['realized_pnl']) if r.get('realized_pnl') is not None else None
                }
                for r in records
            ]

        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []

    async def get_chart_data(
        self,
        days: int = 30,
        interval: str = 'daily'
    ) -> List[Dict]:
        """
        獲取圖表數據

        Args:
            days: 查詢天數
            interval: 'hourly' 或 'daily'

        Returns:
            圖表數據點列表
        """
        if not self._db:
            return []

        try:
            records = await self._db.fetch_all("""
                SELECT
                    timestamp,
                    total_balance_usdt,
                    total_pnl,
                    pnl_percentage,
                    position_count
                FROM pnl_snapshots
                WHERE snapshot_type = $1
                  AND timestamp >= NOW() - ($2 || ' days')::INTERVAL
                ORDER BY timestamp ASC
            """, interval, str(days))

            return [
                {
                    "timestamp": r['timestamp'].isoformat(),
                    "balance": str(r['total_balance_usdt']),
                    "pnl": str(r['total_pnl']),
                    "pnl_percentage": str(r['pnl_percentage']),
                    "position_count": r['position_count']
                }
                for r in records
            ]

        except Exception as e:
            logger.error(f"Failed to get chart data: {e}")
            return []


    # ==================== 定時任務 ====================

    async def start_scheduler(self):
        """啟動定時任務調度器"""
        if self._scheduler_running:
            logger.info("ℹ️ PnL scheduler already running")
            return

        self._scheduler_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("✅ PnL scheduler started - will record snapshots daily")

    async def stop_scheduler(self):
        """停止定時任務調度器"""
        self._scheduler_running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None
        logger.info("✅ PnL scheduler stopped")

    async def _scheduler_loop(self):
        """定時任務循環"""
        # 計算下次執行時間（每天 UTC 00:05）
        while self._scheduler_running:
            try:
                now = datetime.utcnow()
                # 計算下一個 00:05 的時間
                next_run = now.replace(hour=0, minute=5, second=0, microsecond=0)
                if now >= next_run:
                    next_run += timedelta(days=1)

                wait_seconds = (next_run - now).total_seconds()
                logger.info(f"📅 Next PnL snapshot scheduled at {next_run.isoformat()}Z (in {wait_seconds/3600:.1f} hours)")

                # 等待到下次執行時間
                await asyncio.sleep(wait_seconds)

                # 執行快照
                if self._scheduler_running:
                    logger.info("⏰ Running scheduled PnL snapshot...")
                    await self.record_daily_snapshot()

                    # 同時同步最近 7 天的交易記錄
                    await self.sync_historical_trades(days_back=7)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Scheduler error: {e}")
                # 出錯後等待 1 小時再重試
                await asyncio.sleep(3600)


# 全局實例
pnl_tracker = PnLTracker()
