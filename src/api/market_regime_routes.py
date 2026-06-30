"""
Market Regime API Routes - 市場狀態分析 API 端點

實現：
- GET  /api/v1/market/regime/{symbol}          - 獲取當前市場狀態
- GET  /api/v1/market/regime/{symbol}/history  - 獲取市場狀態歷史
"""

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.logger import logger
from src.strategies.market_regime import market_regime_analyzer, MarketRegime


router = APIRouter(tags=["Market Regime"])


# ==================== Response Models ====================

class MarketRegimeResponse(BaseModel):
    """市場狀態響應"""
    regime: str = Field(..., description="市場狀態類型")
    timestamp: Optional[str] = Field(None, description="分析時間")
    trend_strength: Optional[float] = Field(None, description="趨勢強度 (0-1)")
    volatility: Optional[float] = Field(None, description="波動率水平 (0-1)")
    confidence: Optional[float] = Field(None, description="信心度 (0-1)")
    strategy_weights: Optional[Dict[str, float]] = Field(None, description="策略權重建議")


class RegimeHistoryItem(BaseModel):
    """狀態歷史項目"""
    regime: str
    timestamp: str


class MarketRegimeDetailResponse(BaseModel):
    """市場狀態詳細響應"""
    regime: str
    confidence: float
    trend_strength: float
    volatility_level: float
    regime_duration: int
    regime_probabilities: Dict[str, float]
    strategy_weights: Dict[str, float]
    indicators: Dict[str, Any]
    analysis_time: str


# ==================== Market Regime Endpoints ====================

@router.get("/market/regime/{symbol}", response_model=MarketRegimeResponse)
async def get_market_regime(symbol: str):
    """
    獲取指定交易對的當前市場狀態

    Args:
        symbol: 交易對，如 BTCUSDT

    Returns:
        當前市場狀態及策略權重建議
    """
    try:
        # 獲取特徵數據
        from src.features.feature_store import feature_store
        from src.engines import data_engine

        symbol = symbol.upper()

        # 嘗試獲取最新價格
        try:
            current_price = data_engine.get_price(symbol)
            if current_price is None or current_price <= 0:
                current_price = await _fetch_current_price(symbol)
        except Exception:
            current_price = await _fetch_current_price(symbol)

        # 獲取特徵數據
        features = await feature_store.get_features(symbol, "1h")

        if features is None or len(features) < 30:
            # 如果沒有足夠的特徵數據，返回默認狀態
            return MarketRegimeResponse(
                regime=MarketRegime.RANGING.value,
                timestamp=datetime.utcnow().isoformat(),
                trend_strength=0.0,
                volatility=0.5,
                confidence=0.5,
                strategy_weights={
                    'trend_following': 1.0,
                    'momentum': 1.0,
                    'mean_reversion': 1.0
                }
            )

        # 分析市場狀態
        analysis = market_regime_analyzer.analyze(features, current_price)

        return MarketRegimeResponse(
            regime=analysis.regime.value,
            timestamp=analysis.analysis_time.isoformat(),
            trend_strength=analysis.trend_strength,
            volatility=analysis.volatility_level,
            confidence=analysis.confidence,
            strategy_weights=analysis.strategy_weights
        )

    except Exception as e:
        logger.error(f"Failed to get market regime for {symbol}: {e}")
        # 返回默認狀態而非拋出錯誤
        return MarketRegimeResponse(
            regime=market_regime_analyzer.get_current_regime().value,
            timestamp=datetime.utcnow().isoformat(),
            trend_strength=0.0,
            volatility=0.5,
            confidence=0.5,
            strategy_weights={
                'trend_following': 1.0,
                'momentum': 1.0,
                'mean_reversion': 1.0
            }
        )


@router.get("/market/regime/{symbol}/history")
async def get_market_regime_history(
    symbol: str,
    limit: int = Query(default=50, le=100, description="最大返回數量")
):
    """
    獲取市場狀態變化歷史

    Args:
        symbol: 交易對
        limit: 返回數量限制

    Returns:
        歷史狀態變化列表
    """
    try:
        history = market_regime_analyzer.get_regime_history(limit)

        # 轉換為前端期望的格式
        formatted_history = []
        for item in history:
            formatted_history.append({
                "regime": item.get("regime", "ranging"),
                "timestamp": item.get("timestamp"),
                "trend_strength": None,  # 歷史記錄中可能沒有這些詳細數據
                "volatility": None,
                "confidence": None,
                "strategy_weights": None
            })

        return formatted_history

    except Exception as e:
        logger.error(f"Failed to get market regime history for {symbol}: {e}")
        return []


@router.get("/market/regime/{symbol}/detail", response_model=MarketRegimeDetailResponse)
async def get_market_regime_detail(symbol: str):
    """
    獲取詳細的市場狀態分析

    包含所有指標和各狀態概率

    Args:
        symbol: 交易對

    Returns:
        完整的市場狀態分析結果
    """
    try:
        from src.features.feature_store import feature_store
        from src.engines import data_engine

        symbol = symbol.upper()

        # 獲取價格
        try:
            current_price = data_engine.get_price(symbol)
            if current_price is None or current_price <= 0:
                current_price = await _fetch_current_price(symbol)
        except Exception:
            current_price = await _fetch_current_price(symbol)

        # 獲取特徵
        features = await feature_store.get_features(symbol, "1h")

        if features is None or len(features) < 30:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Insufficient feature data for {symbol}. Need at least 30 bars."
            )

        # 完整分析
        analysis = market_regime_analyzer.analyze(features, current_price)

        return MarketRegimeDetailResponse(
            regime=analysis.regime.value,
            confidence=analysis.confidence,
            trend_strength=analysis.trend_strength,
            volatility_level=analysis.volatility_level,
            regime_duration=analysis.regime_duration,
            regime_probabilities=analysis.regime_probabilities,
            strategy_weights=analysis.strategy_weights,
            indicators={k: float(v) if isinstance(v, (int, float)) else str(v)
                       for k, v in analysis.indicators.items()},
            analysis_time=analysis.analysis_time.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get market regime detail for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze market regime: {str(e)}"
        )


@router.get("/market/regime/current")
async def get_current_regime():
    """
    快速獲取當前市場狀態（無需指定交易對）

    使用全局分析器的最新狀態
    """
    try:
        current = market_regime_analyzer.get_current_regime()
        history = market_regime_analyzer.get_regime_history(5)

        return {
            "current_regime": current.value,
            "recent_history": history
        }
    except Exception as e:
        logger.error(f"Failed to get current regime: {e}")
        return {
            "current_regime": MarketRegime.RANGING.value,
            "recent_history": []
        }


@router.get("/market/regime/should-reduce")
async def should_reduce_exposure():
    """
    檢查是否應該減少曝險

    基於市場狀態分析建議
    """
    try:
        should_reduce, reason = market_regime_analyzer.should_reduce_exposure()

        return {
            "should_reduce": should_reduce,
            "reason": reason,
            "current_regime": market_regime_analyzer.get_current_regime().value
        }
    except Exception as e:
        logger.error(f"Failed to check reduce exposure: {e}")
        return {
            "should_reduce": False,
            "reason": "",
            "current_regime": MarketRegime.RANGING.value
        }


# ==================== Helper Functions ====================

async def _fetch_current_price(symbol: str) -> float:
    """從幣安獲取當前價格"""
    try:
        from src.services.binance_client import binance_client
        ticker = await binance_client.get_ticker_price(symbol)
        if isinstance(ticker, dict) and "price" in ticker:
            return float(ticker["price"])
        return 0.0
    except Exception as e:
        logger.warning(f"Failed to fetch price for {symbol}: {e}")
        return 0.0
