"""
市場數據 API
提供實時市場數據，如訂單簿 (Depth)、最新價格等
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from src.services.binance_client import binance_client
from src.logger import logger

router = APIRouter(prefix="/api/v1/market", tags=["Market Data"])

@router.get("/depth", summary="獲取訂單簿深度")
async def get_order_book(
    symbol: str = Query(..., description="交易對，如 BTCUSDT"),
    limit: int = Query(20, ge=5, le=100, description="深度數量")
):
    try:
        return await binance_client.get_order_book(symbol=symbol, limit=limit)
    except Exception as e:
        logger.error(f"API Error get_order_book: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/price", summary="獲取最新價格")
async def get_price(
    symbol: Optional[str] = Query(None, description="交易對")
):
    try:
        return await binance_client.get_ticker_price(symbol=symbol)
    except Exception as e:
        logger.error(f"API Error get_price: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/24hr", summary="獲取 24 小時行情")
async def get_ticker_24h(
    symbol: Optional[str] = Query(None, description="交易對")
):
    try:
        return await binance_client.get_ticker_24h(symbol=symbol)
    except Exception as e:
        logger.error(f"API Error get_ticker_24h: {e}")
        raise HTTPException(status_code=500, detail=str(e))
