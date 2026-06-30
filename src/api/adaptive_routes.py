"""
自適應訓練 API 路由

提供自適應訓練器和 Hyperopt 優化器的 REST API 端點

v0.3.0 新增
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.logger import logger
from src.engines.adaptive_trainer import adaptive_trainer, AdaptiveConfig, RetrainTrigger

# 嘗試導入 Hyperopt 優化器
try:
    from src.engines.hyperopt_optimizer import hyperopt_optimizer, HyperoptConfig, LossFunction
    HYPEROPT_AVAILABLE = hyperopt_optimizer is not None
except ImportError:
    HYPEROPT_AVAILABLE = False
    hyperopt_optimizer = None

router = APIRouter(prefix="/api/v1/adaptive", tags=["Adaptive Training"])


# ==================== Request Models ====================

class AdaptiveConfigRequest(BaseModel):
    """自適應訓練配置請求"""
    retrain_trigger: Optional[str] = Field(None, description="重訓練觸發方式: time_based, candle_based, performance_based")
    retrain_interval_hours: Optional[int] = Field(None, ge=1, le=168, description="重訓練間隔（小時）")
    retrain_candle_count: Optional[int] = Field(None, ge=10, le=1000, description="K線數量觸發閾值")
    training_window_days: Optional[int] = Field(None, ge=7, le=365, description="訓練數據天數")
    weight_factor: Optional[float] = Field(None, ge=0.1, le=1.0, description="近期數據權重係數")
    continual_learning: Optional[bool] = Field(None, description="是否啟用持續學習")
    auto_register_to_arena: Optional[bool] = Field(None, description="是否自動註冊到Arena")
    auto_start_shadow_trading: Optional[bool] = Field(None, description="是否自動開始影子交易")
    model_types: Optional[List[str]] = Field(None, description="要訓練的模型類型")
    symbol: Optional[str] = Field(None, description="交易對")
    interval: Optional[str] = Field(None, description="K線間隔")


class TrainRequest(BaseModel):
    """訓練請求"""
    symbol: Optional[str] = Field(None, description="交易對")
    force: bool = Field(False, description="強制訓練（忽略重訓練檢查）")


class HyperoptConfigRequest(BaseModel):
    """Hyperopt 配置請求"""
    loss_function: Optional[str] = Field(None, description="損失函數: sharpe, sortino, profit_drawdown, calmar")
    n_trials: Optional[int] = Field(None, ge=10, le=1000, description="優化試驗次數")
    timeout: Optional[int] = Field(None, ge=60, le=86400, description="超時時間（秒）")
    data_days: Optional[int] = Field(None, ge=7, le=365, description="回測數據天數")
    starting_balance: Optional[float] = Field(None, ge=100, description="起始資金")
    strategy: Optional[str] = Field(None, description="策略名稱")
    symbol: Optional[str] = Field(None, description="交易對")
    interval: Optional[str] = Field(None, description="K線間隔")


class HyperoptRunRequest(BaseModel):
    """Hyperopt 運行請求"""
    strategy: Optional[str] = Field(None, description="策略名稱")
    n_trials: Optional[int] = Field(None, ge=10, le=1000, description="優化試驗次數")
    timeout: Optional[int] = Field(None, ge=60, le=86400, description="超時時間（秒）")


# ==================== Adaptive Training Endpoints ====================

@router.get("/status")
async def get_adaptive_status():
    """獲取自適應訓練狀態"""
    try:
        return {
            "success": True,
            "data": adaptive_trainer.get_status()
        }
    except Exception as e:
        logger.error(f"Failed to get adaptive status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_adaptive_config():
    """獲取自適應訓練配置"""
    try:
        return {
            "success": True,
            "data": adaptive_trainer.get_config()
        }
    except Exception as e:
        logger.error(f"Failed to get adaptive config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config")
async def update_adaptive_config(request: AdaptiveConfigRequest):
    """更新自適應訓練配置"""
    try:
        # 過濾掉 None 值
        updates = {k: v for k, v in request.dict().items() if v is not None}

        if not updates:
            raise HTTPException(status_code=400, detail="No configuration provided")

        config = adaptive_trainer.update_config(**updates)

        return {
            "success": True,
            "message": "Configuration updated",
            "data": config.to_dict()
        }
    except Exception as e:
        logger.error(f"Failed to update adaptive config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def trigger_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """觸發模型訓練"""
    try:
        # 在後台執行訓練
        async def run_training():
            return await adaptive_trainer.train_model(
                symbol=request.symbol,
                force=request.force
            )

        background_tasks.add_task(run_training)

        return {
            "success": True,
            "message": "Training started in background",
            "data": {
                "symbol": request.symbol or adaptive_trainer.config.symbol,
                "force": request.force
            }
        }
    except Exception as e:
        logger.error(f"Failed to trigger training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_training_history(limit: int = 20):
    """獲取訓練歷史"""
    try:
        return {
            "success": True,
            "data": adaptive_trainer.get_training_history(limit)
        }
    except Exception as e:
        logger.error(f"Failed to get training history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_adaptive_models():
    """列出所有自適應訓練的模型"""
    try:
        return {
            "success": True,
            "data": adaptive_trainer.list_models()
        }
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_background_scanning(symbols: Optional[List[str]] = None):
    """啟動後台掃描（自動重訓練）"""
    try:
        adaptive_trainer.start_background_scanning(symbols)

        return {
            "success": True,
            "message": "Background scanning started",
            "data": {
                "symbols": symbols or [adaptive_trainer.config.symbol]
            }
        }
    except Exception as e:
        logger.error(f"Failed to start background scanning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_background_scanning():
    """停止後台掃描"""
    try:
        adaptive_trainer.stop_background_scanning()

        return {
            "success": True,
            "message": "Background scanning stopped"
        }
    except Exception as e:
        logger.error(f"Failed to stop background scanning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Hyperopt Endpoints ====================

@router.get("/hyperopt/status")
async def get_hyperopt_status():
    """獲取 Hyperopt 優化狀態"""
    if not HYPEROPT_AVAILABLE:
        return {
            "success": False,
            "message": "Hyperopt not available. Install optuna: pip install optuna"
        }

    try:
        return {
            "success": True,
            "data": hyperopt_optimizer.get_status()
        }
    except Exception as e:
        logger.error(f"Failed to get hyperopt status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hyperopt/config")
async def get_hyperopt_config():
    """獲取 Hyperopt 配置"""
    if not HYPEROPT_AVAILABLE:
        return {
            "success": False,
            "message": "Hyperopt not available"
        }

    try:
        return {
            "success": True,
            "data": hyperopt_optimizer.get_config()
        }
    except Exception as e:
        logger.error(f"Failed to get hyperopt config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hyperopt/config")
async def update_hyperopt_config(request: HyperoptConfigRequest):
    """更新 Hyperopt 配置"""
    if not HYPEROPT_AVAILABLE:
        raise HTTPException(status_code=400, detail="Hyperopt not available")

    try:
        updates = {k: v for k, v in request.dict().items() if v is not None}

        if not updates:
            raise HTTPException(status_code=400, detail="No configuration provided")

        config = hyperopt_optimizer.update_config(**updates)

        return {
            "success": True,
            "message": "Hyperopt configuration updated",
            "data": config.to_dict()
        }
    except Exception as e:
        logger.error(f"Failed to update hyperopt config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hyperopt/run")
async def run_hyperopt(request: HyperoptRunRequest, background_tasks: BackgroundTasks):
    """運行 Hyperopt 優化"""
    if not HYPEROPT_AVAILABLE:
        raise HTTPException(status_code=400, detail="Hyperopt not available")

    try:
        if hyperopt_optimizer._is_running:
            raise HTTPException(status_code=400, detail="Optimization already running")

        # 在後台執行優化
        async def run_optimization():
            return await hyperopt_optimizer.optimize(
                strategy=request.strategy,
                n_trials=request.n_trials,
                timeout=request.timeout
            )

        background_tasks.add_task(run_optimization)

        return {
            "success": True,
            "message": "Hyperopt optimization started in background",
            "data": {
                "strategy": request.strategy or hyperopt_optimizer.config.strategy,
                "n_trials": request.n_trials or hyperopt_optimizer.config.n_trials,
                "timeout": request.timeout or hyperopt_optimizer.config.timeout
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to run hyperopt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hyperopt/history")
async def get_hyperopt_history(limit: int = 20):
    """獲取 Hyperopt 優化歷史"""
    if not HYPEROPT_AVAILABLE:
        return {
            "success": False,
            "message": "Hyperopt not available"
        }

    try:
        return {
            "success": True,
            "data": hyperopt_optimizer.get_history(limit)
        }
    except Exception as e:
        logger.error(f"Failed to get hyperopt history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hyperopt/best")
async def get_hyperopt_best_params():
    """獲取 Hyperopt 最佳參數"""
    if not HYPEROPT_AVAILABLE:
        return {
            "success": False,
            "message": "Hyperopt not available"
        }

    try:
        best_params = hyperopt_optimizer.get_best_params()

        if best_params is None:
            return {
                "success": False,
                "message": "No optimization results available"
            }

        return {
            "success": True,
            "data": best_params
        }
    except Exception as e:
        logger.error(f"Failed to get best params: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hyperopt/results")
async def get_hyperopt_results():
    """獲取保存的 Hyperopt 結果"""
    if not HYPEROPT_AVAILABLE:
        return {
            "success": False,
            "message": "Hyperopt not available"
        }

    try:
        return {
            "success": True,
            "data": hyperopt_optimizer.load_results()
        }
    except Exception as e:
        logger.error(f"Failed to get hyperopt results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hyperopt/visualization")
async def get_hyperopt_visualization():
    """獲取 Hyperopt 可視化數據"""
    if not HYPEROPT_AVAILABLE:
        return {
            "success": False,
            "message": "Hyperopt not available"
        }

    try:
        return {
            "success": True,
            "data": hyperopt_optimizer.visualize_optimization()
        }
    except Exception as e:
        logger.error(f"Failed to get visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))
