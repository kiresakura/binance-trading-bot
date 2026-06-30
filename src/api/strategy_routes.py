"""
策略管理 API 路由

v0.4.0 - Enterprise Edition

提供策略超市功能的 REST API：
- 策略列表和詳情
- 策略啟用/禁用
- 策略配置管理
- 信號聚合設置
- 績效查詢
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.logger import logger
from src.strategies.strategy_manager import (
    strategy_manager,
    StrategyCategory,
    AggregationMethod,
    StrategyMetadata
)
from src.strategies.community import COMMUNITY_STRATEGIES

router = APIRouter(prefix="/api/v1/strategies", tags=["strategies"])


# ==================== Pydantic 模型 ====================

class StrategyInfo(BaseModel):
    """策略信息"""
    name: str
    display_name: str
    category: str
    description: str
    author: str
    version: str
    timeframe: str
    risk_level: int
    tags: List[str]
    enabled: bool
    weight: float
    hyperopt_enabled: bool
    performance: Optional[Dict[str, Any]] = None


class StrategyListResponse(BaseModel):
    """策略列表響應"""
    total: int
    strategies: List[StrategyInfo]


class StrategyStoreResponse(BaseModel):
    """策略超市響應"""
    total_strategies: int
    categories: List[Dict[str, Any]]
    featured: List[Dict[str, Any]]
    top_performers: List[Dict[str, Any]]
    recently_added: List[Dict[str, Any]]


class StrategyConfigRequest(BaseModel):
    """策略配置請求"""
    enabled: Optional[bool] = None
    weight: Optional[float] = Field(None, ge=0, le=10)
    max_position_size: Optional[float] = Field(None, ge=0, le=1)
    stop_loss_pct: Optional[float] = Field(None, ge=0, le=0.5)
    take_profit_pct: Optional[float] = Field(None, ge=0, le=1)
    min_signal_strength: Optional[float] = Field(None, ge=0, le=1)
    min_confidence: Optional[float] = Field(None, ge=0, le=1)
    cooldown_minutes: Optional[int] = Field(None, ge=0)
    params: Optional[Dict[str, Any]] = None


class StrategyConfigResponse(BaseModel):
    """策略配置響應"""
    name: str
    enabled: bool
    weight: float
    max_position_size: float
    stop_loss_pct: float
    take_profit_pct: float
    min_signal_strength: float
    min_confidence: float
    cooldown_minutes: int
    params: Dict[str, Any]


class AggregationSettingsRequest(BaseModel):
    """聚合設置請求"""
    method: Optional[str] = Field(None, description="聚合方法")
    min_agreement: Optional[float] = Field(None, ge=0, le=1)


class AggregationSettingsResponse(BaseModel):
    """聚合設置響應"""
    method: str
    min_agreement: float
    signal_decay_minutes: int
    available_methods: List[str]


class PerformanceResponse(BaseModel):
    """績效響應"""
    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float


class StrategyCountResponse(BaseModel):
    """策略數量響應"""
    total: int
    registered: int
    enabled: int


# ==================== 初始化社區策略 ====================

def initialize_community_strategies():
    """初始化社區策略"""
    for strategy_info in COMMUNITY_STRATEGIES:
        try:
            # 創建元數據
            metadata = StrategyMetadata(
                name=strategy_info['name'],
                display_name=strategy_info['display_name'],
                category=StrategyCategory(strategy_info['category']),
                description=strategy_info['description'],
                risk_level=strategy_info['risk_level'],
                timeframe=strategy_info['timeframe'],
                author='Freqtrade Community',
                version='1.0.0',
                tags=[strategy_info['category'], strategy_info['timeframe']],
                hyperopt_enabled=True
            )

            # 註冊策略類
            strategy_manager.register_strategy_class(
                strategy_info['class'],
                metadata
            )

            # 創建並註冊實例
            strategy_instance = strategy_info['class']()
            strategy_manager.register_strategy(strategy_instance, metadata)

            logger.info(f"Initialized community strategy: {strategy_info['name']}")

        except Exception as e:
            logger.error(f"Failed to initialize strategy {strategy_info.get('name', 'unknown')}: {e}")

# 應用啟動時初始化
initialize_community_strategies()


# ==================== API 端點 ====================

@router.get("/store", response_model=StrategyStoreResponse)
async def get_strategy_store():
    """
    獲取策略超市信息

    返回策略超市的完整概覽，包括：
    - 策略總數和分類
    - 推薦策略
    - 績效最佳策略
    - 最近添加的策略
    """
    try:
        store_info = strategy_manager.get_strategy_store()
        return StrategyStoreResponse(
            total_strategies=store_info['total_strategies'],
            categories=store_info['categories'],
            featured=store_info['featured'],
            top_performers=store_info['top_performers'],
            recently_added=store_info['recently_added']
        )
    except Exception as e:
        logger.error(f"Failed to get strategy store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=StrategyListResponse)
async def list_strategies(
    category: Optional[str] = Query(None, description="按分類過濾"),
    enabled_only: bool = Query(False, description="只顯示啟用的策略")
):
    """
    列出所有策略

    Args:
        category: 策略分類過濾
        enabled_only: 是否只顯示啟用的策略
    """
    try:
        cat = StrategyCategory(category) if category else None
        strategies = strategy_manager.list_strategies(
            category=cat,
            enabled_only=enabled_only
        )

        return StrategyListResponse(
            total=len(strategies),
            strategies=[
                StrategyInfo(
                    name=s['name'],
                    display_name=s['display_name'],
                    category=s['category'],
                    description=s['description'],
                    author=s.get('author', 'Unknown'),
                    version=s.get('version', '1.0.0'),
                    timeframe=s.get('timeframe', '1h'),
                    risk_level=s.get('risk_level', 3),
                    tags=s.get('tags', []),
                    enabled=s.get('enabled', False),
                    weight=s.get('weight', 1.0),
                    hyperopt_enabled=s.get('hyperopt_enabled', False),
                    performance=s.get('performance')
                )
                for s in strategies
            ]
        )
    except Exception as e:
        logger.error(f"Failed to list strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def get_categories():
    """獲取所有策略分類"""
    try:
        categories = strategy_manager.get_available_categories()
        return {
            "categories": categories,
            "total": len(categories)
        }
    except Exception as e:
        logger.error(f"Failed to get categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/count", response_model=StrategyCountResponse)
async def get_strategy_count():
    """獲取策略數量統計"""
    try:
        count = strategy_manager.get_strategy_count()
        return StrategyCountResponse(**count)
    except Exception as e:
        logger.error(f"Failed to get strategy count: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{strategy_name}")
async def get_strategy(strategy_name: str):
    """獲取策略詳情"""
    try:
        metadata = strategy_manager.get_strategy_metadata(strategy_name)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Strategy not found: {strategy_name}")

        config = strategy_manager.get_strategy_config(strategy_name)
        performance = strategy_manager.get_strategy_performance(strategy_name)

        return {
            "metadata": metadata,
            "config": config,
            "performance": performance
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get strategy {strategy_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{strategy_name}/config", response_model=StrategyConfigResponse)
async def get_strategy_config(strategy_name: str):
    """獲取策略配置"""
    try:
        config = strategy_manager.get_strategy_config(strategy_name)
        if not config:
            raise HTTPException(status_code=404, detail=f"Strategy not found: {strategy_name}")
        return StrategyConfigResponse(**config)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get strategy config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{strategy_name}/config")
async def update_strategy_config(
    strategy_name: str,
    config: StrategyConfigRequest
):
    """更新策略配置"""
    try:
        config_dict = config.dict(exclude_none=True)

        if not config_dict:
            raise HTTPException(status_code=400, detail="No configuration provided")

        success = strategy_manager.update_strategy_config(strategy_name, config_dict)

        if not success:
            raise HTTPException(status_code=404, detail=f"Strategy not found: {strategy_name}")

        return {
            "success": True,
            "message": f"Strategy {strategy_name} configuration updated",
            "updated_fields": list(config_dict.keys())
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update strategy config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{strategy_name}/enable")
async def enable_strategy(strategy_name: str):
    """啟用策略"""
    try:
        success = strategy_manager.enable_strategy(strategy_name)
        if not success:
            raise HTTPException(status_code=404, detail=f"Strategy not found: {strategy_name}")

        return {
            "success": True,
            "message": f"Strategy {strategy_name} enabled",
            "strategy_name": strategy_name,
            "enabled": True
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{strategy_name}/disable")
async def disable_strategy(strategy_name: str):
    """禁用策略"""
    try:
        success = strategy_manager.disable_strategy(strategy_name)
        if not success:
            raise HTTPException(status_code=404, detail=f"Strategy not found: {strategy_name}")

        return {
            "success": True,
            "message": f"Strategy {strategy_name} disabled",
            "strategy_name": strategy_name,
            "enabled": False
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disable strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{strategy_name}/weight")
async def set_strategy_weight(
    strategy_name: str,
    weight: float = Query(..., ge=0, le=10, description="策略權重 (0-10)")
):
    """設置策略權重"""
    try:
        success = strategy_manager.set_strategy_weight(strategy_name, weight)
        if not success:
            raise HTTPException(status_code=404, detail=f"Strategy not found: {strategy_name}")

        return {
            "success": True,
            "message": f"Strategy {strategy_name} weight set to {weight}",
            "strategy_name": strategy_name,
            "weight": weight
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set strategy weight: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{strategy_name}/performance", response_model=PerformanceResponse)
async def get_strategy_performance(strategy_name: str):
    """獲取策略績效"""
    try:
        performance = strategy_manager.get_strategy_performance(strategy_name)
        if not performance:
            # 返回空績效
            return PerformanceResponse(
                strategy_name=strategy_name,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0
            )
        return PerformanceResponse(**performance)
    except Exception as e:
        logger.error(f"Failed to get strategy performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/all")
async def get_all_performance():
    """獲取所有策略績效"""
    try:
        performances = strategy_manager.get_all_performance()
        return {
            "total": len(performances),
            "performances": performances
        }
    except Exception as e:
        logger.error(f"Failed to get all performances: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 聚合設置 ====================

@router.get("/aggregation/settings", response_model=AggregationSettingsResponse)
async def get_aggregation_settings():
    """獲取信號聚合設置"""
    try:
        settings = strategy_manager.get_aggregation_settings()
        return AggregationSettingsResponse(**settings)
    except Exception as e:
        logger.error(f"Failed to get aggregation settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/aggregation/settings")
async def update_aggregation_settings(settings: AggregationSettingsRequest):
    """更新信號聚合設置"""
    try:
        if settings.method:
            success = strategy_manager.set_aggregation_method(settings.method)
            if not success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid aggregation method: {settings.method}"
                )

        if settings.min_agreement is not None:
            strategy_manager._min_agreement = settings.min_agreement
            strategy_manager._save_config()

        current_settings = strategy_manager.get_aggregation_settings()
        return {
            "success": True,
            "message": "Aggregation settings updated",
            "settings": current_settings
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update aggregation settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 批量操作 ====================

@router.post("/batch/enable")
async def batch_enable_strategies(
    strategy_names: List[str] = Body(..., description="策略名稱列表")
):
    """批量啟用策略"""
    try:
        results = {}
        for name in strategy_names:
            results[name] = strategy_manager.enable_strategy(name)

        enabled_count = sum(1 for v in results.values() if v)
        return {
            "success": True,
            "message": f"Enabled {enabled_count}/{len(strategy_names)} strategies",
            "results": results
        }
    except Exception as e:
        logger.error(f"Failed to batch enable strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch/disable")
async def batch_disable_strategies(
    strategy_names: List[str] = Body(..., description="策略名稱列表")
):
    """批量禁用策略"""
    try:
        results = {}
        for name in strategy_names:
            results[name] = strategy_manager.disable_strategy(name)

        disabled_count = sum(1 for v in results.values() if v)
        return {
            "success": True,
            "message": f"Disabled {disabled_count}/{len(strategy_names)} strategies",
            "results": results
        }
    except Exception as e:
        logger.error(f"Failed to batch disable strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Hyperopt 空間 ====================

@router.get("/{strategy_name}/hyperopt-space")
async def get_hyperopt_space(strategy_name: str):
    """獲取策略的 Hyperopt 搜索空間"""
    try:
        strategy = strategy_manager.get_strategy(strategy_name)
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy not found: {strategy_name}")

        space = strategy.get_hyperopt_space()
        return {
            "strategy_name": strategy_name,
            "hyperopt_space": space,
            "parameter_count": len(space)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get hyperopt space: {e}")
        raise HTTPException(status_code=500, detail=str(e))
