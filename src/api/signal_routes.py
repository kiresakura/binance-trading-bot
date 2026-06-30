"""
策略信號和 MTF 分析 API 路由

提供給 iOS App 的即時策略數據端點：
- 當前策略信號
- 多時間框架分析
- 市場狀態分析
- 成交量分析
- 策略績效
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio

from src.logger import logger
from src.config import settings

router = APIRouter(prefix="/signals", tags=["signals"])


# ==================== Pydantic 模型 ====================

class TimeframeTrendResponse(BaseModel):
    """單一時間框架趨勢"""
    timeframe: str
    trend: int  # 1=上漲, -1=下跌, 0=中性
    strength: float  # 0-1
    rsi: Optional[float] = None
    adx: Optional[float] = None
    ema_alignment: Optional[str] = None


class MTFAnalysisResponse(BaseModel):
    """多時間框架分析響應"""
    higher_tf_trend: int
    trend_strength: float
    trend_alignment: bool
    confidence_multiplier: float
    timeframes_analyzed: List[str]
    details: Dict[str, TimeframeTrendResponse]
    timestamp: str


class StrategySignalResponse(BaseModel):
    """策略信號響應"""
    id: str
    symbol: str
    direction: int  # 1=做多, -1=做空, 0=中性
    strength: float
    confidence: float
    strategy_name: str
    signal_type: str  # BUY, SELL, STRONG_BUY, STRONG_SELL, HOLD
    reason: str
    timestamp: str
    features: Optional[Dict[str, float]] = None


class MarketRegimeResponse(BaseModel):
    """市場狀態響應"""
    regime: str  # trending, ranging, breakout, high_volatility
    regime_scores: Dict[str, float]
    active_strategies: List[str]
    strategy_weights: Dict[str, float]
    timestamp: str


class VolumeAnalysisResponse(BaseModel):
    """成交量分析響應"""
    symbol: str
    volume_ratio: float  # 相對於20日均量
    volume_status: str  # surge, confirmed, normal, dry
    signal_adjustment: float  # 信號強度調整值
    volume_history: List[float]  # 過去12個時間段的成交量比
    timestamp: str


class KellyPositionResponse(BaseModel):
    """Kelly Criterion 倉位建議"""
    symbol: str
    kelly_fraction: float  # 原始 Kelly 比例
    adjusted_fraction: float  # 調整後的倉位比例
    win_rate: float
    avg_win: float
    avg_loss: float
    confidence: float
    max_position_pct: float


class CurrentSignalsResponse(BaseModel):
    """當前所有信號的彙總響應"""
    signals: List[StrategySignalResponse]
    mtf_analysis: Optional[MTFAnalysisResponse] = None
    market_regime: Optional[MarketRegimeResponse] = None
    volume_analysis: Optional[VolumeAnalysisResponse] = None
    kelly_position: Optional[KellyPositionResponse] = None
    timestamp: str


class StrategyPerformanceResponse(BaseModel):
    """單一策略績效"""
    name: str
    display_name: str
    total_pnl: float
    win_rate: float
    profit_factor: float
    trade_count: int
    avg_profit: float
    avg_loss: float
    max_drawdown: float


class PerformanceSummaryResponse(BaseModel):
    """績效摘要響應"""
    strategies: List[StrategyPerformanceResponse]
    total_pnl: float
    total_trades: int
    overall_win_rate: float
    overall_profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_holding_time: str
    time_range: str


class DailyPnLResponse(BaseModel):
    """每日盈虧"""
    date: str
    pnl: float
    trade_count: int


# ==================== API 端點 ====================

@router.get("/current", response_model=CurrentSignalsResponse)
async def get_current_signals(symbol: str = "BTCUSDT"):
    """
    獲取當前策略信號和分析數據

    返回：
    - 所有策略的當前信號
    - MTF 分析結果
    - 市場狀態
    - 成交量分析
    - Kelly 倉位建議
    """
    try:
        from src.engines import strategy_engine, data_engine

        # 獲取策略引擎狀態
        status = strategy_engine.get_status()

        # 構建信號列表
        signals = []
        recent_signals = status.get('recent_signals', [])

        for sig in recent_signals[-5:]:  # 最近5個信號
            signals.append(StrategySignalResponse(
                id=sig.get('id', str(hash(str(sig)))),
                symbol=sig.get('symbol', symbol),
                direction=sig.get('direction', 0),
                strength=sig.get('strength', 0),
                confidence=sig.get('confidence', 0),
                strategy_name=sig.get('strategy_name', 'unknown'),
                signal_type=sig.get('signal_type', 'HOLD'),
                reason=sig.get('reason', ''),
                timestamp=sig.get('timestamp', datetime.utcnow().isoformat()),
                features=sig.get('features')
            ))

        # 獲取 MTF 分析
        mtf_analysis = None
        mtf_status = status.get('mtf_analyzer', {})
        if mtf_status:
            details = {}
            for tf, data in mtf_status.get('trend_cache', {}).items():
                if isinstance(data, dict):
                    details[tf] = TimeframeTrendResponse(
                        timeframe=tf,
                        trend=data.get('trend', 0),
                        strength=data.get('strength', 0),
                        rsi=data.get('rsi'),
                        adx=data.get('adx'),
                        ema_alignment=data.get('ema_alignment')
                    )

            mtf_analysis = MTFAnalysisResponse(
                higher_tf_trend=mtf_status.get('current_trend', 0),
                trend_strength=mtf_status.get('trend_strength', 0),
                trend_alignment=mtf_status.get('trend_alignment', False),
                confidence_multiplier=mtf_status.get('confidence_multiplier', 1.0),
                timeframes_analyzed=mtf_status.get('timeframes', ['4h', '1d']),
                details=details,
                timestamp=datetime.utcnow().isoformat()
            )

        # 獲取市場狀態
        market_regime = None
        market_state = status.get('current_market_state', 'unknown')
        regime_scores = status.get('regime_scores', {})

        if market_state != 'unknown':
            # 獲取策略權重
            strategy_weights = {}
            for strategy in status.get('enabled_strategies', []):
                weight = 1.0
                if market_state == 'trending':
                    weight = 1.2 if 'trend' in strategy else 0.8
                elif market_state == 'ranging':
                    weight = 1.2 if 'mean' in strategy else 0.8
                strategy_weights[strategy] = weight

            market_regime = MarketRegimeResponse(
                regime=market_state,
                regime_scores=regime_scores if regime_scores else {
                    'trending': 0.3,
                    'ranging': 0.3,
                    'breakout': 0.2,
                    'high_volatility': 0.2
                },
                active_strategies=status.get('enabled_strategies', []),
                strategy_weights=strategy_weights,
                timestamp=datetime.utcnow().isoformat()
            )

        # 獲取成交量分析
        volume_analysis = None
        try:
            features = await data_engine.get_features(symbol, limit=25)
            if features is not None and len(features) > 0:
                latest = features.iloc[-1]
                volume = latest.get('volume', 1)
                volume_sma = latest.get('volume_sma_20', volume)
                volume_ratio = volume / volume_sma if volume_sma > 0 else 1.0

                # 計算歷史成交量比
                volume_history = []
                for i in range(-12, 0):
                    if len(features) > abs(i):
                        row = features.iloc[i]
                        v = row.get('volume', 1)
                        v_sma = row.get('volume_sma_20', v)
                        volume_history.append(v / v_sma if v_sma > 0 else 1.0)

                # 確定狀態和調整值
                if volume_ratio >= 2.0:
                    v_status = 'surge'
                    adjustment = 0.15
                elif volume_ratio >= 1.2:
                    v_status = 'confirmed'
                    adjustment = 0.10
                elif volume_ratio < 0.8:
                    v_status = 'dry'
                    adjustment = -0.15
                else:
                    v_status = 'normal'
                    adjustment = 0

                volume_analysis = VolumeAnalysisResponse(
                    symbol=symbol,
                    volume_ratio=volume_ratio,
                    volume_status=v_status,
                    signal_adjustment=adjustment,
                    volume_history=volume_history,
                    timestamp=datetime.utcnow().isoformat()
                )
        except Exception as e:
            logger.warning(f"Volume analysis error: {e}")

        # 獲取 Kelly 倉位建議
        kelly_position = None
        kelly_status = status.get('kelly_calculator', {})
        if kelly_status:
            kelly_position = KellyPositionResponse(
                symbol=symbol,
                kelly_fraction=kelly_status.get('kelly_fraction', 0),
                adjusted_fraction=kelly_status.get('adjusted_fraction', 0),
                win_rate=kelly_status.get('win_rate', 0),
                avg_win=kelly_status.get('avg_win', 0),
                avg_loss=kelly_status.get('avg_loss', 0),
                confidence=kelly_status.get('confidence', 0),
                max_position_pct=kelly_status.get('max_position_pct', 0.25)
            )

        return CurrentSignalsResponse(
            signals=signals,
            mtf_analysis=mtf_analysis,
            market_regime=market_regime,
            volume_analysis=volume_analysis,
            kelly_position=kelly_position,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Get current signals error: {e}")
        return CurrentSignalsResponse(
            signals=[],
            mtf_analysis=None,
            market_regime=None,
            volume_analysis=None,
            kelly_position=None,
            timestamp=datetime.utcnow().isoformat()
        )


@router.get("/mtf/{symbol}", response_model=MTFAnalysisResponse)
async def get_mtf_analysis(symbol: str):
    """
    獲取指定交易對的多時間框架分析

    分析 5m, 15m, 1h, 4h, 1d 時間框架的趨勢
    """
    try:
        from src.engines import strategy_engine, data_engine

        # 嘗試從策略引擎獲取 MTF 分析
        status = strategy_engine.get_status()
        mtf_status = status.get('mtf_analyzer', {})

        # 如果沒有緩存數據，實時計算
        details = {}
        timeframes = ['5m', '15m', '1h', '4h', '1d']

        for tf in timeframes:
            try:
                # 從緩存獲取或計算
                cached = mtf_status.get('trend_cache', {}).get(tf)
                if cached:
                    details[tf] = TimeframeTrendResponse(
                        timeframe=tf,
                        trend=cached.get('trend', 0),
                        strength=cached.get('strength', 0),
                        rsi=cached.get('rsi'),
                        adx=cached.get('adx'),
                        ema_alignment=cached.get('ema_alignment')
                    )
                else:
                    # 使用默認值
                    details[tf] = TimeframeTrendResponse(
                        timeframe=tf,
                        trend=0,
                        strength=0.5,
                        rsi=50,
                        adx=20,
                        ema_alignment='neutral'
                    )
            except Exception as e:
                logger.warning(f"MTF analysis for {tf} failed: {e}")
                details[tf] = TimeframeTrendResponse(
                    timeframe=tf, trend=0, strength=0.5
                )

        # 計算綜合趨勢
        weights = {'5m': 0.1, '15m': 0.2, '1h': 0.3, '4h': 0.5, '1d': 0.7}
        weighted_trend = 0
        total_weight = 0
        for tf, data in details.items():
            w = weights.get(tf, 0.2)
            weighted_trend += data.trend * data.strength * w
            total_weight += w

        avg_trend = weighted_trend / total_weight if total_weight > 0 else 0
        higher_tf_trend = 1 if avg_trend > 0.2 else (-1 if avg_trend < -0.2 else 0)
        trend_strength = min(abs(avg_trend) * 2, 1.0)

        return MTFAnalysisResponse(
            higher_tf_trend=higher_tf_trend,
            trend_strength=trend_strength,
            trend_alignment=True,  # 會由 iOS 端根據信號方向判斷
            confidence_multiplier=1.0 + (trend_strength * 0.2) if higher_tf_trend != 0 else 1.0,
            timeframes_analyzed=timeframes,
            details=details,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"MTF analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regime/{symbol}", response_model=MarketRegimeResponse)
async def get_market_regime(symbol: str):
    """
    獲取指定交易對的市場狀態

    分析當前市場是：
    - trending: 趨勢行情
    - ranging: 震盪行情
    - breakout: 突破行情
    - high_volatility: 高波動
    """
    try:
        from src.services.market_analyzer import market_analyzer

        # 獲取市場分析
        market_data = await market_analyzer.get_market_state(symbol)
        regime = market_data.get('state', 'unknown')

        # 計算各狀態分數
        regime_scores = {
            'trending': market_data.get('trend_score', 0.3),
            'ranging': market_data.get('range_score', 0.3),
            'breakout': market_data.get('breakout_score', 0.2),
            'high_volatility': market_data.get('volatility_score', 0.2)
        }

        # 根據市場狀態決定策略權重
        strategy_weights = {
            'trend_following': 1.0,
            'momentum': 1.0,
            'mean_reversion': 1.0
        }

        if regime == 'trending':
            strategy_weights['trend_following'] = 1.3
            strategy_weights['momentum'] = 1.2
            strategy_weights['mean_reversion'] = 0.6
        elif regime == 'ranging':
            strategy_weights['trend_following'] = 0.7
            strategy_weights['momentum'] = 0.8
            strategy_weights['mean_reversion'] = 1.3
        elif regime == 'breakout':
            strategy_weights['trend_following'] = 1.1
            strategy_weights['momentum'] = 1.3
            strategy_weights['mean_reversion'] = 0.7
        elif regime == 'high_volatility':
            strategy_weights['trend_following'] = 0.8
            strategy_weights['momentum'] = 0.8
            strategy_weights['mean_reversion'] = 0.8

        # 獲取活躍策略
        active_strategies = [k for k, v in strategy_weights.items() if v >= 0.8]

        return MarketRegimeResponse(
            regime=regime,
            regime_scores=regime_scores,
            active_strategies=active_strategies,
            strategy_weights=strategy_weights,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Market regime error: {e}")
        # 返回默認值
        return MarketRegimeResponse(
            regime='unknown',
            regime_scores={'trending': 0.25, 'ranging': 0.25, 'breakout': 0.25, 'high_volatility': 0.25},
            active_strategies=['trend_following', 'momentum', 'mean_reversion'],
            strategy_weights={'trend_following': 1.0, 'momentum': 1.0, 'mean_reversion': 1.0},
            timestamp=datetime.utcnow().isoformat()
        )


@router.get("/volume/{symbol}", response_model=VolumeAnalysisResponse)
async def get_volume_analysis(symbol: str):
    """獲取成交量分析"""
    try:
        from src.engines import data_engine

        features = await data_engine.get_features(symbol, limit=25)

        if features is None or len(features) == 0:
            raise HTTPException(status_code=404, detail="No data available")

        latest = features.iloc[-1]
        volume = latest.get('volume', 1)
        volume_sma = latest.get('volume_sma_20', volume)
        volume_ratio = volume / volume_sma if volume_sma > 0 else 1.0

        # 計算歷史
        volume_history = []
        for i in range(-12, 0):
            if len(features) > abs(i):
                row = features.iloc[i]
                v = row.get('volume', 1)
                v_sma = row.get('volume_sma_20', v)
                volume_history.append(round(v / v_sma if v_sma > 0 else 1.0, 2))

        # 確定狀態
        if volume_ratio >= 2.0:
            v_status = 'surge'
            adjustment = 0.15
        elif volume_ratio >= 1.2:
            v_status = 'confirmed'
            adjustment = 0.10
        elif volume_ratio < 0.8:
            v_status = 'dry'
            adjustment = -0.15
        else:
            v_status = 'normal'
            adjustment = 0

        return VolumeAnalysisResponse(
            symbol=symbol,
            volume_ratio=round(volume_ratio, 2),
            volume_status=v_status,
            signal_adjustment=adjustment,
            volume_history=volume_history,
            timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Volume analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/kelly/{symbol}", response_model=KellyPositionResponse)
async def get_kelly_position(symbol: str):
    """獲取 Kelly Criterion 倉位建議"""
    try:
        from src.engines import strategy_engine

        status = strategy_engine.get_status()
        kelly_status = status.get('kelly_calculator', {})

        if not kelly_status:
            # 返回默認保守倉位
            return KellyPositionResponse(
                symbol=symbol,
                kelly_fraction=0.1,
                adjusted_fraction=0.05,
                win_rate=0.5,
                avg_win=0.02,
                avg_loss=0.01,
                confidence=0.5,
                max_position_pct=0.25
            )

        return KellyPositionResponse(
            symbol=symbol,
            kelly_fraction=kelly_status.get('kelly_fraction', 0.1),
            adjusted_fraction=kelly_status.get('adjusted_fraction', 0.05),
            win_rate=kelly_status.get('win_rate', 0.5),
            avg_win=kelly_status.get('avg_win', 0.02),
            avg_loss=kelly_status.get('avg_loss', 0.01),
            confidence=kelly_status.get('confidence', 0.5),
            max_position_pct=kelly_status.get('max_position_pct', 0.25)
        )

    except Exception as e:
        logger.error(f"Kelly calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance", response_model=PerformanceSummaryResponse)
async def get_strategy_performance(days: int = 30):
    """
    獲取策略績效摘要

    Args:
        days: 統計天數 (7, 14, 30, 60, 90)
    """
    try:
        from src.services.pnl_tracker import pnl_tracker

        # 從 PnL tracker 獲取數據
        summary = await pnl_tracker.get_pnl_summary()

        # 構建策略績效（如果有詳細數據）
        strategies = []

        # 這裡使用模擬數據，實際應從數據庫獲取
        strategy_names = ['trend_following', 'momentum', 'mean_reversion']
        display_names = {'trend_following': '趨勢跟蹤', 'momentum': '動量', 'mean_reversion': '均值回歸'}

        for name in strategy_names:
            strategies.append(StrategyPerformanceResponse(
                name=name,
                display_name=display_names.get(name, name),
                total_pnl=summary.get(f'{name}_pnl', 0),
                win_rate=summary.get(f'{name}_win_rate', 0.5),
                profit_factor=summary.get(f'{name}_profit_factor', 1.0),
                trade_count=summary.get(f'{name}_trades', 0),
                avg_profit=summary.get(f'{name}_avg_profit', 0),
                avg_loss=summary.get(f'{name}_avg_loss', 0),
                max_drawdown=summary.get(f'{name}_max_drawdown', 0)
            ))

        return PerformanceSummaryResponse(
            strategies=strategies,
            total_pnl=summary.get('total_realized_pnl', 0) + summary.get('total_unrealized_pnl', 0),
            total_trades=summary.get('total_trades', 0),
            overall_win_rate=summary.get('win_rate', 0.5),
            overall_profit_factor=summary.get('profit_factor', 1.0),
            sharpe_ratio=summary.get('sharpe_ratio', 0),
            max_drawdown=summary.get('max_drawdown', 0),
            avg_holding_time=summary.get('avg_holding_time', '0h'),
            time_range=f"{days}D"
        )

    except Exception as e:
        logger.error(f"Performance summary error: {e}")
        # 返回空數據
        return PerformanceSummaryResponse(
            strategies=[],
            total_pnl=0,
            total_trades=0,
            overall_win_rate=0,
            overall_profit_factor=0,
            sharpe_ratio=0,
            max_drawdown=0,
            avg_holding_time="0h",
            time_range=f"{days}D"
        )


@router.get("/daily-pnl", response_model=List[DailyPnLResponse])
async def get_daily_pnl(days: int = 30):
    """獲取每日盈虧數據（用於熱力圖）"""
    try:
        from src.services.pnl_tracker import pnl_tracker

        # 獲取每日盈虧數據
        daily_data = await pnl_tracker.get_daily_pnl_history(days=days)

        result = []
        for entry in daily_data:
            result.append(DailyPnLResponse(
                date=entry.get('date', ''),
                pnl=entry.get('pnl', 0),
                trade_count=entry.get('trade_count', 0)
            ))

        return result

    except Exception as e:
        logger.error(f"Daily PnL error: {e}")
        # 返回空數據
        return []
