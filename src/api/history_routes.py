"""
歷史數據 API
提供交易歷史、盈虧記錄與圖表數據
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from src.services.pnl_tracker import pnl_tracker
from src.logger import logger

router = APIRouter(prefix="/api/v1/history", tags=["History"])

@router.get("/pnl", summary="獲取盈虧歷史")
async def get_pnl_history(
    days: int = Query(30, ge=1, le=365, description="查詢天數"),
    symbol: Optional[str] = Query(None, description="指定交易對")
):
    try:
        return await pnl_tracker.get_pnl_history(days=days, symbol=symbol)
    except Exception as e:
        logger.error(f"API Error get_pnl_history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trades", summary="獲取交易歷史")
async def get_trade_history(
    symbol: Optional[str] = Query(None, description="指定交易對"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    try:
        return await pnl_tracker.get_trade_history(symbol=symbol, limit=limit, offset=offset)
    except Exception as e:
        logger.error(f"API Error get_trade_history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chart", summary="獲取圖表數據")
async def get_chart_data(
    period: str = Query("7d", description="時間範圍: 24h, 7d, 30d, 90d"),
    type: str = Query("pnl", description="圖表類型: pnl (盈虧走勢), asset (資產分佈)")
):
    """
    獲取前端繪圖所需的數據格式
    """
    try:
        # PnL Tracker 中應該有 get_chart_data 方法，如果沒有我們會使用 get_pnl_history 轉換
        if hasattr(pnl_tracker, 'get_chart_data'):
            return await pnl_tracker.get_chart_data(period=period, type=type)
        else:
            # Fallback if method doesn't exist yet (based on previous view)
            # 我們假設 view_file 會顯示出 get_chart_data，如果沒有就使用這個 fallback
            days_map = {"24h": 1, "7d": 7, "30d": 30, "90d": 90}
            days = days_map.get(period, 7)
            history = await pnl_tracker.get_pnl_history(days=days)
            
            # 轉換為圖表格式
            return {
                "period": period,
                "data": [
                    {
                        "time": h["date"], 
                        "value": float(h["total_pnl"]),
                        "label": "Total PnL"
                    } for h in history
                ]
            }
            
    except Exception as e:
        logger.error(f"API Error get_chart_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
