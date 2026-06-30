"""
回測和 ML 訓練 API 路由 (v0.5.0 - 新增同步回測和歷史查詢)
"""

import uuid
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from src.logger import logger
from src.engines.backtest_engine import (
    BacktestEngine, BacktestConfig, BacktestMode,
    sma_crossover_strategy, rsi_strategy, macd_strategy, bollinger_strategy
)
from src.engines.ml_trainer import MLTrainer, ModelConfig, ModelType
from src.api.auth import require_scope, TokenData

router = APIRouter(tags=["Backtest & ML"])


# ==================== Pydantic 模型 ====================

class BacktestRequest(BaseModel):
    """回測請求 (原始格式)"""
    symbol: str = Field(default="BTCUSDT", description="交易對")
    interval: str = Field(default="1h", description="K線間隔")
    limit: int = Field(default=500, ge=100, le=2000, description="數據量")
    strategy: str = Field(default="sma_crossover", description="策略名稱")
    initial_capital: float = Field(default=10000, gt=0, description="初始資金")
    commission_rate: float = Field(default=0.001, ge=0, le=0.01, description="手續費率")
    stop_loss_pct: float = Field(default=0.02, ge=0, le=0.1, description="止損百分比")
    take_profit_pct: float = Field(default=0.04, ge=0, le=0.2, description="止盈百分比")


class iOSBacktestRequest(BaseModel):
    """回測請求 (iOS App 格式 v0.5.0)"""
    strategy: str = Field(default="momentum", description="策略名稱")
    symbol: str = Field(default="BTCUSDT", description="交易對")
    days: int = Field(default=60, ge=7, le=365, description="回測天數")
    starting_balance: float = Field(default=10000, gt=0, description="起始資金")
    commission_rate: float = Field(default=0.001, ge=0, le=0.01, description="手續費率")
    slippage_rate: float = Field(default=0.0005, ge=0, le=0.01, description="滑點率")


class BacktestTrade(BaseModel):
    """回測交易記錄"""
    timestamp: str
    side: str
    price: float
    quantity: float
    pnl: float
    pnl_pct: float


class iOSBacktestResult(BaseModel):
    """回測結果 (iOS App 格式 v0.5.0)"""
    id: str
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    initial_balance: float
    final_balance: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    trades: Optional[List[BacktestTrade]] = None


class BacktestHistoryResponse(BaseModel):
    """回測歷史響應"""
    results: List[iOSBacktestResult]
    total: int


class MLTrainRequest(BaseModel):
    """ML 訓練請求"""
    symbol: str = Field(default="BTCUSDT", description="交易對")
    interval: str = Field(default="1h", description="K線間隔")
    limit: int = Field(default=2000, ge=500, le=5000, description="數據量")
    model_types: List[str] = Field(default=["random_forest"], description="模型類型")
    prediction_horizon: int = Field(default=5, ge=1, le=20, description="預測週期")
    threshold: float = Field(default=0.01, ge=0.001, le=0.05, description="漲跌閾值")


class BatchTrainRequest(BaseModel):
    """批次訓練請求"""
    symbols: List[str] = Field(..., description="要訓練的交易對列表")
    model_type: str = Field(default="random_forest", description="模型類型")
    days_back: int = Field(default=30, ge=7, le=365, description="訓練數據天數")
    interval: str = Field(default="1h", description="K線間隔")


class BatchTrainJob(BaseModel):
    """批次訓練任務"""
    symbol: str
    job_id: Optional[str] = None
    status: str  # "queued", "running", "completed", "failed"
    error: Optional[str] = None


class BatchTrainResponse(BaseModel):
    """批次訓練響應"""
    success: bool
    jobs: List[BatchTrainJob]
    message: str


class PredictRequest(BaseModel):
    """預測請求"""
    symbol: str = Field(default="BTCUSDT", description="交易對")
    interval: str = Field(default="1h", description="K線間隔")
    model_type: str = Field(default="random_forest", description="模型類型")


# ==================== 策略映射 ====================

STRATEGIES = {
    # 原始名稱
    "sma_crossover": sma_crossover_strategy,
    "rsi": rsi_strategy,
    "macd": macd_strategy,
    "bollinger": bollinger_strategy,
    # iOS App 使用的名稱 (v0.5.0)
    "sma": sma_crossover_strategy,
    "momentum": sma_crossover_strategy,  # 使用 SMA 作為動量策略的基礎
    "mean_reversion": rsi_strategy,  # 使用 RSI 作為均值回歸基礎
    "trend_following": macd_strategy,  # 使用 MACD 作為趨勢跟蹤
    "breakout": bollinger_strategy,  # 使用布林帶作為突破策略
}

# 策略描述（支持中英文）
STRATEGY_DESCRIPTIONS = {
    "sma_crossover": "SMA 20/50 交叉策略",
    "sma": "SMA 均線交叉策略",
    "rsi": "RSI 超買超賣策略",
    "macd": "MACD 交叉策略",
    "bollinger": "布林帶突破策略",
    "momentum": "動量策略 (基於均線)",
    "mean_reversion": "均值回歸策略 (基於 RSI)",
    "trend_following": "趨勢跟蹤策略 (基於 MACD)",
    "breakout": "突破策略 (基於布林帶)",
}


# ==================== 全局狀態 ====================

# 回測任務狀態（異步任務）
backtest_tasks: Dict[str, Dict[str, Any]] = {}

# ML 訓練任務狀態
ml_tasks: Dict[str, Dict[str, Any]] = {}

# 回測歷史記錄 (v0.5.0 - 支持 iOS App 歷史查詢)
backtest_history: List[iOSBacktestResult] = []
MAX_HISTORY_SIZE = 100  # 最多保存 100 條記錄


# ==================== 回測 API ====================

@router.post("/backtest/run-async")
async def run_backtest_async(
    request: BacktestRequest,
    background_tasks: BackgroundTasks
):
    """執行回測"""
    task_id = f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 檢查策略是否存在
    if request.strategy not in STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy: {request.strategy}. Available: {list(STRATEGIES.keys())}"
        )

    # 初始化任務狀態
    backtest_tasks[task_id] = {
        "status": "pending",
        "progress": 0,
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat()
    }

    # 在背景執行回測
    background_tasks.add_task(
        _run_backtest_task,
        task_id,
        request
    )

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Backtest started"
    }


async def _run_backtest_task(task_id: str, request: BacktestRequest):
    """背景執行回測任務"""
    try:
        backtest_tasks[task_id]["status"] = "running"
        backtest_tasks[task_id]["progress"] = 10

        # 創建回測引擎
        config = BacktestConfig(
            initial_capital=request.initial_capital,
            commission_rate=request.commission_rate,
            stop_loss_pct=request.stop_loss_pct,
            take_profit_pct=request.take_profit_pct,
            symbol=request.symbol,
            interval=request.interval
        )
        engine = BacktestEngine(config)

        backtest_tasks[task_id]["progress"] = 20

        # 載入數據
        df = await engine.load_data(
            symbol=request.symbol,
            interval=request.interval,
            limit=request.limit
        )

        backtest_tasks[task_id]["progress"] = 40

        # 添加指標
        df = engine.add_indicators(df)

        backtest_tasks[task_id]["progress"] = 60

        # 執行回測
        strategy = STRATEGIES[request.strategy]
        result = await engine.run(strategy, df)

        backtest_tasks[task_id]["progress"] = 100
        backtest_tasks[task_id]["status"] = "completed"
        backtest_tasks[task_id]["result"] = result.to_dict()

        logger.info(f"Backtest {task_id} completed: {result.total_return:.2%}")

    except Exception as e:
        logger.error(f"Backtest {task_id} failed: {e}")
        backtest_tasks[task_id]["status"] = "failed"
        backtest_tasks[task_id]["error"] = str(e)


@router.get("/backtest/status/{task_id}")
async def get_backtest_status(task_id: str):
    """獲取回測任務狀態"""
    if task_id not in backtest_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return backtest_tasks[task_id]


@router.get("/backtest/strategies")
async def get_available_strategies():
    """獲取可用策略列表"""
    return {
        "strategies": list(set(STRATEGIES.keys())),
        "descriptions": STRATEGY_DESCRIPTIONS
    }


# ==================== iOS App 兼容回測 API (v0.5.0) ====================

@router.post("/backtest/run", response_model=iOSBacktestResult)
async def run_ios_backtest(request: iOSBacktestRequest):
    """
    執行回測 (iOS App 同步版本)

    與異步版本不同，這個端點會同步執行回測並直接返回結果
    """
    # 檢查策略是否存在
    if request.strategy not in STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy: {request.strategy}. Available: {list(set(STRATEGIES.keys()))}"
        )

    try:
        # 計算數據量（根據天數和間隔）
        # 假設 1h 間隔，每天 24 根 K 線
        limit = min(request.days * 24, 2000)

        # 創建回測引擎
        config = BacktestConfig(
            initial_capital=request.starting_balance,
            commission_rate=request.commission_rate,
            slippage_rate=request.slippage_rate,
            symbol=request.symbol,
            interval="1h"
        )
        engine = BacktestEngine(config)

        # 載入數據
        df = await engine.load_data(
            symbol=request.symbol,
            interval="1h",
            limit=limit
        )

        # 添加指標
        df = engine.add_indicators(df)

        # 執行回測
        strategy = STRATEGIES[request.strategy]
        result = await engine.run(strategy, df)

        # 轉換為 iOS 格式
        backtest_id = str(uuid.uuid4())[:8]
        ios_result = iOSBacktestResult(
            id=backtest_id,
            strategy=request.strategy,
            symbol=request.symbol,
            start_date=result.start_date.isoformat() if result.start_date else "",
            end_date=result.end_date.isoformat() if result.end_date else "",
            initial_balance=result.initial_capital,
            final_balance=result.final_capital,
            total_return=result.total_return * 100,  # 轉為百分比
            total_trades=result.total_trades,
            winning_trades=result.winning_trades,
            losing_trades=result.losing_trades,
            win_rate=result.win_rate,  # 已經是 0-1 範圍
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown,  # 已經是 0-1 範圍
            profit_factor=result.profit_factor,
            avg_win=result.avg_win * 100,  # 轉為百分比
            avg_loss=abs(result.avg_loss) * 100,  # 轉為正數百分比
            trades=[
                BacktestTrade(
                    timestamp=t.timestamp.isoformat() if hasattr(t, 'timestamp') else "",
                    side=t.side if hasattr(t, 'side') else "",
                    price=t.price if hasattr(t, 'price') else 0,
                    quantity=t.quantity if hasattr(t, 'quantity') else 0,
                    pnl=t.pnl if hasattr(t, 'pnl') else 0,
                    pnl_pct=t.pnl_pct * 100 if hasattr(t, 'pnl_pct') else 0
                ) for t in (result.trades or [])[:50]  # 最多返回 50 筆交易記錄
            ] if result.trades else None
        )

        # 保存到歷史記錄
        backtest_history.insert(0, ios_result)
        if len(backtest_history) > MAX_HISTORY_SIZE:
            backtest_history.pop()

        logger.info(f"iOS backtest completed: {request.strategy} on {request.symbol}, return={ios_result.total_return:.2f}%")
        return ios_result

    except Exception as e:
        logger.error(f"iOS backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest/history", response_model=BacktestHistoryResponse)
async def get_backtest_history(limit: int = Query(default=20, ge=1, le=100)):
    """獲取回測歷史記錄"""
    results = backtest_history[:limit]
    return BacktestHistoryResponse(
        results=results,
        total=len(backtest_history)
    )


@router.get("/backtest/result/{backtest_id}", response_model=iOSBacktestResult)
async def get_backtest_result(backtest_id: str):
    """獲取單個回測結果"""
    for result in backtest_history:
        if result.id == backtest_id:
            return result

    raise HTTPException(status_code=404, detail=f"Backtest result not found: {backtest_id}")


@router.get("/backtest/compare")
async def compare_backtests(ids: str = Query(..., description="Comma-separated backtest IDs")):
    """比較多個回測結果"""
    id_list = [id.strip() for id in ids.split(",")]
    results = []

    for backtest_id in id_list:
        for result in backtest_history:
            if result.id == backtest_id:
                results.append(result)
                break

    if not results:
        raise HTTPException(status_code=404, detail="No matching backtests found")

    return results


# ==================== ML 訓練 API ====================

@router.post("/ml/train")
async def train_ml_model(
    request: MLTrainRequest,
    background_tasks: BackgroundTasks
):
    """訓練 ML 模型"""
    task_id = f"ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 驗證模型類型
    valid_types = [mt.value for mt in ModelType]
    for mt in request.model_types:
        if mt not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model type: {mt}. Available: {valid_types}"
            )

    # 初始化任務狀態
    ml_tasks[task_id] = {
        "status": "pending",
        "progress": 0,
        "results": None,
        "error": None,
        "created_at": datetime.now().isoformat()
    }

    # 在背景執行訓練
    background_tasks.add_task(
        _run_ml_training_task,
        task_id,
        request
    )

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "ML training started"
    }


async def _run_ml_training_task(task_id: str, request: MLTrainRequest):
    """背景執行 ML 訓練任務"""
    try:
        ml_tasks[task_id]["status"] = "running"
        ml_tasks[task_id]["progress"] = 10

        # 創建訓練器
        config = ModelConfig(
            prediction_horizon=request.prediction_horizon,
            threshold=request.threshold
        )
        trainer = MLTrainer(config)

        ml_tasks[task_id]["progress"] = 20

        # 訓練模型
        model_types = [ModelType(mt) for mt in request.model_types]
        results = await trainer.train(
            symbol=request.symbol,
            interval=request.interval,
            limit=request.limit,
            model_types=model_types
        )

        ml_tasks[task_id]["progress"] = 100
        ml_tasks[task_id]["status"] = "completed"
        ml_tasks[task_id]["results"] = {k: v.to_dict() for k, v in results.items()}

        logger.info(f"ML training {task_id} completed")

    except Exception as e:
        logger.error(f"ML training {task_id} failed: {e}")
        ml_tasks[task_id]["status"] = "failed"
        ml_tasks[task_id]["error"] = str(e)


# ==================== 批次訓練 API ====================

# 批次訓練任務狀態
batch_train_tasks: Dict[str, Dict] = {}


@router.post("/ml/train/batch", response_model=BatchTrainResponse)
async def batch_train_models(
    request: BatchTrainRequest,
    background_tasks: BackgroundTasks
):
    """
    批次訓練多個交易對的模型

    依序加入背景任務，避免同時跑太多任務導致記憶體爆掉
    """
    import time

    jobs = []
    batch_id = f"batch_{int(time.time() * 1000)}"

    # 初始化批次任務狀態
    batch_train_tasks[batch_id] = {
        "status": "running",
        "total": len(request.symbols),
        "completed": 0,
        "failed": 0,
        "jobs": {}
    }

    for symbol in request.symbols:
        try:
            job_id = f"train_{symbol}_{int(time.time() * 1000)}"

            # 初始化單個任務狀態
            batch_train_tasks[batch_id]["jobs"][job_id] = {
                "symbol": symbol,
                "status": "queued",
                "error": None
            }

            # 加入背景任務（會依序執行）
            background_tasks.add_task(
                _run_batch_train_task,
                batch_id=batch_id,
                job_id=job_id,
                symbol=symbol,
                model_type=request.model_type,
                days_back=request.days_back,
                interval=request.interval
            )

            jobs.append(BatchTrainJob(
                symbol=symbol,
                job_id=job_id,
                status="queued"
            ))

        except Exception as e:
            jobs.append(BatchTrainJob(
                symbol=symbol,
                job_id=None,
                status="failed",
                error=str(e)
            ))

    return BatchTrainResponse(
        success=True,
        jobs=jobs,
        message=f"已排程 {len(request.symbols)} 個模型訓練任務"
    )


async def _run_batch_train_task(
    batch_id: str,
    job_id: str,
    symbol: str,
    model_type: str,
    days_back: int,
    interval: str
):
    """背景執行批次訓練任務"""
    try:
        # 更新狀態為 running
        if batch_id in batch_train_tasks:
            batch_train_tasks[batch_id]["jobs"][job_id]["status"] = "running"

        logger.info(f"🚀 Starting batch training for {symbol}")

        # 計算 limit（根據 days_back 和 interval）
        interval_hours = {
            "1m": 1/60, "5m": 5/60, "15m": 0.25, "30m": 0.5,
            "1h": 1, "4h": 4, "1d": 24
        }
        hours_per_candle = interval_hours.get(interval, 1)
        limit = min(int(days_back * 24 / hours_per_candle), 5000)

        # 創建訓練器
        config = ModelConfig(
            prediction_horizon=5,
            threshold=0.01
        )
        trainer = MLTrainer(config)

        # 訓練模型
        model_types = [ModelType(model_type)]
        results = await trainer.train(
            symbol=symbol,
            interval=interval,
            limit=limit,
            model_types=model_types
        )

        # 更新狀態為 completed
        if batch_id in batch_train_tasks:
            batch_train_tasks[batch_id]["jobs"][job_id]["status"] = "completed"
            batch_train_tasks[batch_id]["completed"] += 1

            # 檢查是否全部完成
            total = batch_train_tasks[batch_id]["total"]
            completed = batch_train_tasks[batch_id]["completed"]
            failed = batch_train_tasks[batch_id]["failed"]
            if completed + failed >= total:
                batch_train_tasks[batch_id]["status"] = "completed"

        logger.info(f"✅ Batch training completed for {symbol}")

    except Exception as e:
        logger.error(f"❌ Batch training failed for {symbol}: {e}")
        if batch_id in batch_train_tasks:
            batch_train_tasks[batch_id]["jobs"][job_id]["status"] = "failed"
            batch_train_tasks[batch_id]["jobs"][job_id]["error"] = str(e)
            batch_train_tasks[batch_id]["failed"] += 1

            # 檢查是否全部完成
            total = batch_train_tasks[batch_id]["total"]
            completed = batch_train_tasks[batch_id]["completed"]
            failed = batch_train_tasks[batch_id]["failed"]
            if completed + failed >= total:
                batch_train_tasks[batch_id]["status"] = "completed"


@router.get("/ml/train/batch/status")
async def get_batch_train_status():
    """獲取所有批次訓練任務狀態"""
    return {
        "tasks": batch_train_tasks,
        "active_count": sum(1 for t in batch_train_tasks.values() if t["status"] == "running")
    }


@router.get("/ml/status/{task_id}")
async def get_ml_training_status(task_id: str):
    """獲取 ML 訓練任務狀態"""
    if task_id not in ml_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return ml_tasks[task_id]


@router.get("/ml/model-types")
async def get_available_model_types():
    """獲取可用模型類型（用於後端訓練）"""
    return {
        "models": [mt.value for mt in ModelType],
        "descriptions": {
            "random_forest": "隨機森林分類器",
            "gradient_boosting": "梯度提升分類器",
            "xgboost": "XGBoost 分類器",
            "ensemble": "集成模型（結合多個模型）"
        }
    }


@router.get("/ml/history")
async def get_training_history():
    """獲取訓練歷史"""
    from src.engines.ml_trainer import ml_trainer
    return {
        "history": ml_trainer.get_training_history(),
        "best_model": ml_trainer.get_best_model()
    }


@router.post("/ml/predict")
async def predict(request: PredictRequest):
    """使用訓練好的模型進行預測"""
    from src.engines.ml_trainer import ml_trainer, FeatureEngineer
    from src.services.binance_client import binance_client

    try:
        # 獲取最新數據
        klines = await binance_client.get_klines(
            symbol=request.symbol,
            interval=request.interval,
            limit=100
        )

        if not klines:
            raise HTTPException(status_code=500, detail="Failed to fetch market data")

        # 轉換為 DataFrame
        import pandas as pd
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df.set_index('timestamp', inplace=True)

        # 創建特徵
        features = FeatureEngineer.create_features(df)

        # 取最後一行（最新數據）
        latest_features = features.iloc[[-1]].dropna(axis=1)

        # 預測
        if request.model_type not in ml_trainer.models:
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.model_type} not loaded. Train a model first."
            )

        predictions, probabilities = ml_trainer.predict(
            latest_features,
            model_type=request.model_type
        )

        prediction = int(predictions[0])
        probability = float(probabilities[0].max()) if probabilities is not None else None

        return {
            "symbol": request.symbol,
            "prediction": prediction,
            "prediction_label": "UP" if prediction == 1 else "DOWN",
            "probability": probability,
            "model_type": request.model_type,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== iOS 模型回測 API ====================

@router.post("/backtest/ios-model")
async def backtest_ios_model(
    model_id: str,
    symbol: str = "BTCUSDT",
    days: int = 30,
    initial_capital: float = 10000.0,
    background_tasks: BackgroundTasks = None
):
    """
    使用 iOS 訓練的模型進行回測

    對指定的 iOS 訓練模型執行歷史回測，評估其實際表現。
    """
    from src.services.backtest_service import backtest_service, BacktestConfig
    from pathlib import Path
    import json as json_module

    # 獲取模型數據
    model_file = Path("models/ios_trained") / f"{model_id}.json"
    model_data = None
    if model_file.exists():
        with open(model_file) as f:
            model_data = json_module.load(f)

    if not model_data:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # 創建回測配置
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    config = BacktestConfig(
        symbol=symbol,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        initial_capital=initial_capital,
        commission_rate=0.001,
        slippage=0.0005,
        position_size=0.3,
        confidence_threshold=0.6,
        strength_threshold=0.5,
        stop_loss_pct=5.0,
        take_profit_pct=10.0,
        max_hold_hours=168
    )

    # 執行回測
    result = await backtest_service.run_backtest(model_id, config, model_data)

    return {
        "backtest_id": result.backtest_id,
        "model_id": model_id,
        "symbol": symbol,
        "days": days,
        "status": result.status,
        "metrics": {
            "total_return_pct": result.total_return_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown_pct": result.max_drawdown_pct,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "total_trades": result.total_trades
        },
        "duration_seconds": result.duration_seconds,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/backtest/compare-models")
async def compare_ios_models(
    model_ids: List[str],
    symbol: str = "BTCUSDT",
    days: int = 30
):
    """
    對比多個 iOS 訓練模型的回測結果

    同時回測多個模型，返回比較結果和排名。
    """
    from src.services.backtest_service import backtest_service, BacktestConfig

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    config = BacktestConfig(
        symbol=symbol,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat()
    )

    comparison = await backtest_service.compare_models(model_ids, config)

    return comparison
