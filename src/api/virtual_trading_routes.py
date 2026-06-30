"""
Virtual Trading API Routes - 虛擬交易 API 端點

提供虛擬交易查詢和管理功能
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from decimal import Decimal

from src.logger import logger
from src.services.virtual_trading_service import virtual_trading_service


router = APIRouter(prefix="/virtual", tags=["Virtual Trading"])


@router.on_event("startup")
async def startup():
    """初始化虛擬交易服務"""
    await virtual_trading_service.initialize()


@router.get("/accounts")
async def get_all_accounts():
    """獲取所有虛擬帳戶排行榜"""
    try:
        accounts = await virtual_trading_service.get_all_accounts_summary()
        return {
            "success": True,
            "count": len(accounts),
            "accounts": accounts
        }
    except Exception as e:
        logger.error(f"Failed to get accounts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/accounts/{model_version}")
async def get_account(model_version: str):
    """獲取特定模型的虛擬帳戶詳情"""
    try:
        summary = await virtual_trading_service.get_account_summary(model_version)
        if not summary:
            raise HTTPException(status_code=404, detail="Account not found")
        return {
            "success": True,
            "account": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get account: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/accounts/{model_version}/reset")
async def reset_account(model_version: str):
    """重置虛擬帳戶（恢復到 10000 USDT）"""
    try:
        success = await virtual_trading_service.reset_account(model_version)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to reset account")
        return {
            "success": True,
            "message": f"Account {model_version} reset to 10000 USDT"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset account: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/accounts/{model_version}/trades")
async def get_trades(
    model_version: str,
    limit: int = Query(default=50, ge=1, le=200)
):
    """獲取虛擬交易歷史"""
    try:
        trades = await virtual_trading_service.get_trade_history(model_version, limit)
        return {
            "success": True,
            "count": len(trades),
            "trades": trades
        }
    except Exception as e:
        logger.error(f"Failed to get trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions/{model_version}")
async def get_positions(model_version: str):
    """獲取虛擬持倉"""
    try:
        positions = await virtual_trading_service.get_all_positions(model_version)
        return {
            "success": True,
            "count": len(positions),
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "quantity": float(p.quantity),
                    "entry_price": float(p.entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pnl": float(p.unrealized_pnl),
                    "unrealized_pnl_pct": float(p.unrealized_pnl_pct)
                }
                for p in positions
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/init-all")
async def init_all_accounts():
    """為所有現有模型初始化虛擬帳戶"""
    try:
        from src.services.arena_service import arena_service
        await arena_service.initialize()
        
        models = await arena_service.list_models()
        initialized = []
        
        for model in models:
            account = await virtual_trading_service.get_or_create_account(model.version)
            if account:
                initialized.append(model.version)
        
        return {
            "success": True,
            "message": f"Initialized {len(initialized)} virtual accounts",
            "accounts": initialized
        }
    except Exception as e:
        logger.error(f"Failed to init all accounts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-all")
async def reset_all_accounts():
    """重置所有虛擬帳戶"""
    try:
        from src.services.arena_service import arena_service
        
        models = await arena_service.list_models()
        reset = []
        
        for model in models:
            success = await virtual_trading_service.reset_account(model.version)
            if success:
                reset.append(model.version)
        
        return {
            "success": True,
            "message": f"Reset {len(reset)} virtual accounts to 10000 USDT",
            "accounts": reset
        }
    except Exception as e:
        logger.error(f"Failed to reset all accounts: {e}")
        raise HTTPException(status_code=500, detail=str(e))
