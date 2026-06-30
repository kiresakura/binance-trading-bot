"""
PnL API Routes - 盈虧追蹤 API 路由

Endpoints:
- POST /api/v1/pnl/sync - 同步 Binance 交易記錄
- GET /api/v1/pnl/summary - 獲取盈虧摘要
- GET /api/v1/pnl/positions - 獲取持倉盈虧
- GET /api/v1/pnl/trades - 獲取交易歷史
- GET /api/v1/pnl/daily - 獲取每日盈虧
- GET /api/v1/pnl/chart - 獲取圖表數據

Author: Trading Bot System
Version: 0.2.1
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from decimal import Decimal

from src.logger import logger
from src.services.pnl_tracker import pnl_tracker


router = APIRouter(prefix="/pnl", tags=["PnL"])


# ==================== Pydantic 模型 ====================

class SyncRequest(BaseModel):
    """同步請求"""
    symbols: Optional[List[str]] = Field(None, description="要同步的交易對列表")
    days_back: int = Field(30, ge=0, le=3650, description="同步多少天的數據（0=全部）")


class SyncResponse(BaseModel):
    """同步響應"""
    success: bool
    synced_symbols: List[str]
    total_trades: int
    new_trades: int
    errors: List[dict]


class PositionPnL(BaseModel):
    """持倉盈虧"""
    symbol: str
    quantity: str
    average_cost: str
    current_price: str
    current_value: str
    unrealized_pnl: str
    unrealized_pnl_pct: str


class PnLSummaryResponse(BaseModel):
    """盈虧摘要響應"""
    total_realized_pnl: str
    total_unrealized_pnl: str
    total_pnl: str
    total_pnl_percentage: str
    today_pnl: str
    today_pnl_percentage: str
    total_commission: str
    total_trades: int
    win_rate: str
    position_count: int


class TradeRecord(BaseModel):
    """交易記錄"""
    id: int
    symbol: str
    side: str
    price: str
    quantity: str
    quote_qty: str
    commission: str
    commission_asset: str
    trade_time: str
    is_maker: bool
    realized_pnl: Optional[str] = None  # 賣出時的已實現盈虧


class DailyPnLRecord(BaseModel):
    """每日盈虧記錄"""
    date: str
    realized_pnl: str
    unrealized_pnl: str
    total_pnl: str
    pnl_percentage: str
    trade_count: int
    buy_volume: str
    sell_volume: str
    commission_total: str


class ChartDataPoint(BaseModel):
    """圖表數據點"""
    timestamp: str
    balance: str
    pnl: str
    pnl_percentage: str
    position_count: int


# ==================== API Endpoints ====================

@router.post("/sync", response_model=SyncResponse)
async def sync_trades(request: SyncRequest = None):
    """
    同步 Binance 交易記錄

    從 Binance API 獲取歷史交易記錄並存入本地數據庫。
    這是初次設置或定期同步時使用的端點。

    - **symbols**: 要同步的交易對列表（空則同步所有追蹤的交易對）
    - **days_back**: 往回同步多少天（默認 30 天）
    """
    try:
        # 確保服務已初始化
        await pnl_tracker.init()

        symbols = None
        days_back = 30

        if request:
            symbols = request.symbols
            days_back = request.days_back

        result = await pnl_tracker.sync_historical_trades(
            symbols=symbols,
            days_back=days_back
        )

        return SyncResponse(
            success=len(result["errors"]) == 0,
            synced_symbols=result["synced_symbols"],
            total_trades=result["total_trades"],
            new_trades=result["new_trades"],
            errors=result["errors"]
        )

    except Exception as e:
        logger.error(f"Sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", response_model=PnLSummaryResponse)
async def get_pnl_summary():
    """
    獲取盈虧摘要

    返回：
    - 總已實現盈虧
    - 總未實現盈虧
    - 總盈虧
    - 今日盈虧
    - 總手續費
    - 交易次數
    - 勝率
    - 持倉數量
    """
    try:
        await pnl_tracker.init()

        summary = await pnl_tracker.get_total_pnl()

        return PnLSummaryResponse(
            total_realized_pnl=str(summary.total_realized_pnl),
            total_unrealized_pnl=str(summary.total_unrealized_pnl),
            total_pnl=str(summary.total_pnl),
            total_pnl_percentage=str(summary.total_pnl_percentage),
            today_pnl=str(summary.today_pnl),
            today_pnl_percentage=str(summary.today_pnl_percentage),
            total_commission=str(summary.total_commission),
            total_trades=summary.total_trades,
            win_rate=str(summary.win_rate),
            position_count=len(summary.positions)
        )

    except Exception as e:
        logger.error(f"Get PnL summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
async def get_positions_pnl(symbol: str = None):
    """
    獲取持倉盈虧

    返回所有持倉的未實現盈虧。

    - **symbol**: 可選，指定交易對
    """
    try:
        await pnl_tracker.init()

        result = await pnl_tracker.get_unrealized_pnl(symbol)

        return {
            "success": True,
            "positions": result["positions"],
            "total_unrealized_pnl": result["total_unrealized_pnl"],
            "total_position_value": result["total_position_value"]
        }

    except Exception as e:
        logger.error(f"Get positions PnL failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades", response_model=List[TradeRecord])
async def get_trade_history(
    symbol: str = Query(None, description="交易對"),
    limit: int = Query(50, ge=1, le=500, description="返回數量"),
    offset: int = Query(0, ge=0, description="偏移量")
):
    """
    獲取交易歷史

    返回已同步的交易記錄。

    - **symbol**: 可選，指定交易對
    - **limit**: 返回數量（默認 50，最大 500）
    - **offset**: 偏移量（用於分頁）
    """
    try:
        await pnl_tracker.init()

        trades = await pnl_tracker.get_trade_history(
            symbol=symbol,
            limit=limit,
            offset=offset
        )

        return [TradeRecord(**t) for t in trades]

    except Exception as e:
        logger.error(f"Get trade history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/daily", response_model=List[DailyPnLRecord])
async def get_daily_pnl(
    days: int = Query(30, ge=1, le=365, description="查詢天數"),
    symbol: str = Query(None, description="交易對")
):
    """
    獲取每日盈虧

    返回每日的盈虧統計。

    - **days**: 查詢天數（默認 30 天）
    - **symbol**: 可選，指定交易對
    """
    try:
        await pnl_tracker.init()

        history = await pnl_tracker.get_pnl_history(
            days=days,
            symbol=symbol
        )

        return [DailyPnLRecord(**h) for h in history]

    except Exception as e:
        logger.error(f"Get daily PnL failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chart", response_model=List[ChartDataPoint])
async def get_chart_data(
    days: int = Query(30, ge=1, le=365, description="查詢天數"),
    interval: str = Query("daily", description="間隔：hourly 或 daily")
):
    """
    獲取圖表數據

    返回用於繪製盈虧圖表的數據點。

    - **days**: 查詢天數（默認 30 天）
    - **interval**: 數據間隔（hourly 或 daily）
    """
    try:
        await pnl_tracker.init()

        if interval not in ["hourly", "daily"]:
            raise HTTPException(status_code=400, detail="interval must be 'hourly' or 'daily'")

        data = await pnl_tracker.get_chart_data(
            days=days,
            interval=interval
        )

        return [ChartDataPoint(**d) for d in data]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get chart data failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/snapshot")
async def record_snapshot():
    """
    手動記錄快照

    觸發記錄當前的盈虧快照。通常由定時任務自動執行。
    """
    try:
        await pnl_tracker.init()

        success = await pnl_tracker.record_daily_snapshot()

        return {
            "success": success,
            "message": "Snapshot recorded" if success else "Failed to record snapshot"
        }

    except Exception as e:
        logger.error(f"Record snapshot failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/migrate")
async def run_migration():
    """
    執行 PnL 表遷移

    創建 PnL 追蹤所需的數據庫表。
    這個端點是冪等的，可以安全地多次調用。
    """
    try:
        from src.utils.database import async_db_manager

        if not async_db_manager._initialized:
            await async_db_manager.init()

        # 創建 PnL 表的 SQL
        migration_sql = """
        -- binance_trades 表
        CREATE TABLE IF NOT EXISTS binance_trades (
            id BIGINT NOT NULL,
            order_id BIGINT NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            trade_time TIMESTAMPTZ NOT NULL,
            side VARCHAR(10) NOT NULL,
            price DECIMAL(20, 8) NOT NULL,
            qty DECIMAL(20, 8) NOT NULL,
            quote_qty DECIMAL(20, 8) NOT NULL,
            commission DECIMAL(20, 8) NOT NULL,
            commission_asset VARCHAR(20) NOT NULL,
            is_buyer BOOLEAN NOT NULL,
            is_maker BOOLEAN NOT NULL,
            is_best_match BOOLEAN DEFAULT TRUE,
            realized_pnl DECIMAL(20, 8) DEFAULT NULL,
            synced_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (id, symbol)
        );

        -- position_costs 表
        CREATE TABLE IF NOT EXISTS position_costs (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL UNIQUE,
            base_asset VARCHAR(20) NOT NULL,
            quote_asset VARCHAR(20) NOT NULL,
            quantity DECIMAL(20, 8) NOT NULL DEFAULT 0,
            total_cost DECIMAL(20, 8) NOT NULL DEFAULT 0,
            average_cost DECIMAL(20, 8) NOT NULL DEFAULT 0,
            realized_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
            unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
            last_trade_id BIGINT,
            last_trade_time TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );

        -- daily_pnl 表
        CREATE TABLE IF NOT EXISTS daily_pnl (
            date DATE NOT NULL,
            symbol VARCHAR(20) NOT NULL DEFAULT 'ALL',
            starting_balance DECIMAL(20, 8) NOT NULL DEFAULT 0,
            ending_balance DECIMAL(20, 8) NOT NULL DEFAULT 0,
            realized_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
            unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
            total_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
            pnl_percentage DECIMAL(10, 4) DEFAULT 0,
            trade_count INTEGER DEFAULT 0,
            buy_volume DECIMAL(20, 8) DEFAULT 0,
            sell_volume DECIMAL(20, 8) DEFAULT 0,
            commission_total DECIMAL(20, 8) DEFAULT 0,
            win_trades INTEGER DEFAULT 0,
            loss_trades INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, symbol)
        );

        -- pnl_snapshots 表
        CREATE TABLE IF NOT EXISTS pnl_snapshots (
            timestamp TIMESTAMPTZ NOT NULL,
            snapshot_type VARCHAR(20) NOT NULL DEFAULT 'hourly',
            total_balance_usdt DECIMAL(20, 8) NOT NULL,
            available_balance_usdt DECIMAL(20, 8),
            in_position_usdt DECIMAL(20, 8),
            total_realized_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
            total_unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
            total_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
            pnl_percentage DECIMAL(10, 4) DEFAULT 0,
            btc_price DECIMAL(20, 8),
            position_count INTEGER DEFAULT 0,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (timestamp, snapshot_type)
        );

        -- cost_basis_lots 表 (FIFO 用)
        CREATE TABLE IF NOT EXISTS cost_basis_lots (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            trade_id BIGINT NOT NULL,
            buy_time TIMESTAMPTZ NOT NULL,
            quantity DECIMAL(20, 8) NOT NULL,
            remaining_qty DECIMAL(20, 8) NOT NULL,
            cost_per_unit DECIMAL(20, 8) NOT NULL,
            total_cost DECIMAL(20, 8) NOT NULL,
            is_closed BOOLEAN DEFAULT FALSE,
            closed_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """

        # 分別執行每個 CREATE TABLE
        statements = [s.strip() for s in migration_sql.split(';') if s.strip() and 'CREATE TABLE' in s]

        tables_created = []
        for stmt in statements:
            try:
                await async_db_manager.execute(stmt)
                # 從語句中提取表名
                table_name = stmt.split('CREATE TABLE IF NOT EXISTS')[1].split('(')[0].strip()
                tables_created.append(table_name)
            except Exception as e:
                logger.warning(f"Table creation warning: {e}")

        # 創建索引（忽略已存在的錯誤）
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_binance_trades_symbol ON binance_trades (symbol, trade_time DESC)",
            "CREATE INDEX IF NOT EXISTS idx_binance_trades_order ON binance_trades (order_id, symbol)",
            "CREATE INDEX IF NOT EXISTS idx_position_costs_symbol ON position_costs (symbol)",
            "CREATE INDEX IF NOT EXISTS idx_daily_pnl_date ON daily_pnl (date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_pnl_snapshots_type ON pnl_snapshots (snapshot_type, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_cost_basis_lots_symbol ON cost_basis_lots (symbol, is_closed, buy_time)"
        ]

        for idx in indexes:
            try:
                await async_db_manager.execute(idx)
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")

        # 添加 realized_pnl 欄位（如果不存在）
        try:
            await async_db_manager.execute("""
                ALTER TABLE binance_trades ADD COLUMN IF NOT EXISTS realized_pnl DECIMAL(20, 8) DEFAULT NULL
            """)
            logger.info("✅ Added realized_pnl column to binance_trades")
        except Exception as e:
            logger.warning(f"realized_pnl column might already exist: {e}")

        logger.info(f"✅ PnL migration completed. Tables: {tables_created}")

        return {
            "success": True,
            "message": "PnL tables migration completed",
            "tables_created": tables_created
        }

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/position/{symbol}")
async def get_position_detail(symbol: str):
    """
    獲取單個持倉詳情

    返回指定交易對的持倉成本和盈虧詳情。

    - **symbol**: 交易對 (e.g., BTCUSDT)
    """
    try:
        await pnl_tracker.init()

        position = await pnl_tracker.calculate_position_cost(symbol.upper())

        if not position:
            return {
                "success": True,
                "symbol": symbol.upper(),
                "has_position": False,
                "position": None
            }

        return {
            "success": True,
            "symbol": symbol.upper(),
            "has_position": position.quantity > 0,
            "position": {
                "quantity": str(position.quantity),
                "total_cost": str(position.total_cost),
                "average_cost": str(position.average_cost),
                "realized_pnl": str(position.realized_pnl),
                "unrealized_pnl": str(position.unrealized_pnl)
            }
        }

    except Exception as e:
        logger.error(f"Get position detail failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recalculate-pnl")
async def recalculate_realized_pnl():
    """
    重新計算所有賣出交易的已實現盈虧

    遍歷所有賣出交易，重新計算其 realized_pnl 並更新數據庫。
    用於修復歷史數據或在算法更新後重新計算。
    """
    try:
        await pnl_tracker.init()

        # 獲取所有賣出交易
        sell_trades = await pnl_tracker._db.fetch_all("""
            SELECT id, symbol, trade_time, qty, price, commission, commission_asset
            FROM binance_trades
            WHERE side = 'SELL'
            ORDER BY trade_time ASC
        """)

        if not sell_trades:
            return {
                "success": True,
                "message": "No sell trades found",
                "updated": 0
            }

        updated = 0
        from decimal import Decimal

        for trade in sell_trades:
            try:
                realized_pnl = await pnl_tracker._calculate_realized_pnl_for_sell(
                    symbol=trade['symbol'],
                    sell_time=trade['trade_time'],
                    sell_qty=Decimal(str(trade['qty'])),
                    sell_price=Decimal(str(trade['price'])),
                    commission=Decimal(str(trade['commission'])),
                    commission_asset=trade['commission_asset']
                )

                await pnl_tracker._db.execute("""
                    UPDATE binance_trades
                    SET realized_pnl = $1
                    WHERE id = $2 AND symbol = $3
                """, realized_pnl, trade['id'], trade['symbol'])

                updated += 1
                logger.info(f"✅ Recalculated PnL for {trade['symbol']} trade {trade['id']}: {realized_pnl}")

            except Exception as e:
                logger.error(f"Failed to recalculate PnL for trade {trade['id']}: {e}")

        return {
            "success": True,
            "message": f"Recalculated realized PnL for {updated} sell trades",
            "updated": updated,
            "total_sell_trades": len(sell_trades)
        }

    except Exception as e:
        logger.error(f"Recalculate PnL failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
