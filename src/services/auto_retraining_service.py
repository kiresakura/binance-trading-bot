"""
Auto-Retraining Service - 自動再訓練服務

提供：
- 模型性能監控
- 自動觸發再訓練
- 數據漂移檢測
- 定期再訓練排程
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json

from src.logger import logger


class RetrainingTrigger(str, Enum):
    """再訓練觸發條件"""
    SCHEDULED = "scheduled"           # 定期排程
    PERFORMANCE_DECAY = "performance_decay"  # 性能下降
    DATA_DRIFT = "data_drift"         # 數據漂移
    MANUAL = "manual"                 # 手動觸發
    MODEL_EXPIRY = "model_expiry"     # 模型過期


class RetrainingStatus(str, Enum):
    """再訓練狀態"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RetrainingConfig:
    """再訓練配置"""
    # 定期排程
    enabled: bool = True
    schedule_interval_days: int = 7      # 每 7 天自動訓練
    min_training_samples: int = 1000     # 最小訓練樣本數

    # 性能監控閾值
    performance_threshold: float = 0.6   # 性能低於此值觸發再訓練
    win_rate_threshold: float = 0.45     # 勝率低於此值觸發再訓練
    max_consecutive_losses: int = 5      # 連續虧損次數閾值

    # 數據漂移檢測
    drift_detection_enabled: bool = True
    drift_threshold: float = 0.2         # 漂移閾值

    # 訓練配置
    lookback_days: int = 30              # 訓練數據回溯天數
    validation_split: float = 0.2        # 驗證集比例
    max_training_time_minutes: int = 60  # 最大訓練時間

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrainingTask:
    """再訓練任務"""
    task_id: str
    model_version: str
    symbol: str
    trigger: RetrainingTrigger
    status: RetrainingStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model_version": self.model_version,
            "symbol": self.symbol,
            "trigger": self.trigger.value if isinstance(self.trigger, RetrainingTrigger) else self.trigger,
            "status": self.status.value if isinstance(self.status, RetrainingStatus) else self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error
        }


@dataclass
class PerformanceMetrics:
    """性能指標"""
    model_version: str
    timestamp: str
    accuracy: float
    win_rate: float
    total_pnl_pct: float
    sharpe_ratio: float
    consecutive_losses: int
    total_trades: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AutoRetrainingService:
    """自動再訓練服務"""

    def __init__(self):
        self.config = RetrainingConfig()
        self._tasks: Dict[str, RetrainingTask] = {}
        self._performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self._last_training: Dict[str, datetime] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        self._task_counter = 0

        logger.info("AutoRetrainingService initialized")

    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Retraining config updated: {key} = {value}")

    async def start_monitoring(self, interval_minutes: int = 30):
        """啟動性能監控"""
        if self._is_monitoring:
            logger.warning("Monitoring already running")
            return

        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval_minutes)
        )
        logger.info(f"Auto-retraining monitoring started (interval: {interval_minutes}m)")

    async def stop_monitoring(self):
        """停止性能監控"""
        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Auto-retraining monitoring stopped")

    async def _monitoring_loop(self, interval_minutes: int):
        """監控循環"""
        while self._is_monitoring:
            try:
                await self._check_all_models()
                await asyncio.sleep(interval_minutes * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)

    async def _check_all_models(self):
        """檢查所有模型"""
        try:
            # 從 Arena 服務獲取模型列表
            from src.services.arena_service import arena_service
            models = await arena_service.get_leaderboard()

            for model in models:
                model_version = model.get("model_version")
                if model_version:
                    await self.check_model_performance(model_version)

        except Exception as e:
            logger.error(f"Failed to check all models: {e}")

    async def check_model_performance(
        self,
        model_version: str
    ) -> Optional[RetrainingTrigger]:
        """
        檢查模型性能，決定是否需要再訓練

        Returns:
            RetrainingTrigger if retraining needed, None otherwise
        """
        try:
            # 獲取模型性能數據
            metrics = await self._get_model_metrics(model_version)

            if not metrics:
                return None

            # 記錄性能歷史
            if model_version not in self._performance_history:
                self._performance_history[model_version] = []
            self._performance_history[model_version].append(metrics)

            # 保留最近 100 條記錄
            if len(self._performance_history[model_version]) > 100:
                self._performance_history[model_version].pop(0)

            # 檢查定期排程
            if self.config.enabled:
                last_train = self._last_training.get(model_version)
                if last_train:
                    days_since_training = (datetime.utcnow() - last_train).days
                    if days_since_training >= self.config.schedule_interval_days:
                        logger.info(f"Model {model_version} scheduled for retraining")
                        return RetrainingTrigger.SCHEDULED

            # 檢查性能下降
            if metrics.win_rate < self.config.win_rate_threshold:
                logger.warning(f"Model {model_version} win rate below threshold: {metrics.win_rate}")
                return RetrainingTrigger.PERFORMANCE_DECAY

            if metrics.consecutive_losses >= self.config.max_consecutive_losses:
                logger.warning(f"Model {model_version} consecutive losses: {metrics.consecutive_losses}")
                return RetrainingTrigger.PERFORMANCE_DECAY

            # 檢查數據漂移
            if self.config.drift_detection_enabled:
                drift = await self._detect_data_drift(model_version)
                if drift > self.config.drift_threshold:
                    logger.warning(f"Model {model_version} data drift detected: {drift}")
                    return RetrainingTrigger.DATA_DRIFT

            return None

        except Exception as e:
            logger.error(f"Failed to check model performance: {e}")
            return None

    async def _get_model_metrics(self, model_version: str) -> Optional[PerformanceMetrics]:
        """獲取模型性能指標"""
        try:
            from src.services.arena_service import arena_service

            # 獲取模型統計
            stats = await arena_service.get_model_stats(model_version)
            if not stats:
                return None

            # 計算連續虧損
            trades = await arena_service.get_model_trades(model_version, limit=20)
            consecutive_losses = 0
            for trade in trades:
                if trade.get("pnl_pct", 0) < 0:
                    consecutive_losses += 1
                else:
                    break

            return PerformanceMetrics(
                model_version=model_version,
                timestamp=datetime.utcnow().isoformat(),
                accuracy=stats.get("accuracy", 0),
                win_rate=stats.get("win_rate", 0),
                total_pnl_pct=stats.get("total_pnl_pct", 0),
                sharpe_ratio=stats.get("sharpe_ratio", 0),
                consecutive_losses=consecutive_losses,
                total_trades=stats.get("total_trades", 0)
            )

        except Exception as e:
            logger.error(f"Failed to get model metrics: {e}")
            return None

    async def _detect_data_drift(self, model_version: str) -> float:
        """
        檢測數據漂移

        使用簡化的統計方法比較最近數據與訓練數據分佈

        Returns:
            float: 漂移分數 (0-1, 越高漂移越大)
        """
        try:
            # 獲取最近預測的特徵統計
            history = self._performance_history.get(model_version, [])

            if len(history) < 10:
                return 0.0

            # 比較最近 10 次和之前 10 次的性能變化
            recent = history[-10:]
            previous = history[-20:-10] if len(history) >= 20 else history[:10]

            if not previous:
                return 0.0

            recent_win_rate = sum(m.win_rate for m in recent) / len(recent)
            previous_win_rate = sum(m.win_rate for m in previous) / len(previous)

            # 計算相對變化
            if previous_win_rate > 0:
                drift = abs(recent_win_rate - previous_win_rate) / previous_win_rate
            else:
                drift = 0.0

            return min(drift, 1.0)

        except Exception as e:
            logger.error(f"Failed to detect data drift: {e}")
            return 0.0

    async def trigger_retraining(
        self,
        model_version: str,
        symbol: str = "BTCUSDT",
        trigger: RetrainingTrigger = RetrainingTrigger.MANUAL
    ) -> RetrainingTask:
        """
        觸發模型再訓練

        Args:
            model_version: 模型版本
            symbol: 交易對
            trigger: 觸發原因

        Returns:
            RetrainingTask: 再訓練任務
        """
        self._task_counter += 1
        task_id = f"retrain_{self._task_counter}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        task = RetrainingTask(
            task_id=task_id,
            model_version=model_version,
            symbol=symbol,
            trigger=trigger,
            status=RetrainingStatus.PENDING,
            created_at=datetime.utcnow().isoformat()
        )

        self._tasks[task_id] = task

        # 異步執行訓練
        asyncio.create_task(self._execute_retraining(task))

        logger.info(f"Retraining triggered for {model_version}: {trigger.value}")
        return task

    async def _execute_retraining(self, task: RetrainingTask):
        """執行再訓練"""
        try:
            task.status = RetrainingStatus.RUNNING
            task.started_at = datetime.utcnow().isoformat()

            # 準備訓練數據
            training_data = await self._prepare_training_data(
                task.symbol,
                self.config.lookback_days
            )

            if len(training_data) < self.config.min_training_samples:
                raise ValueError(
                    f"Insufficient training data: {len(training_data)} < {self.config.min_training_samples}"
                )

            # 執行訓練（調用現有的訓練 API）
            result = await self._train_model(
                task.model_version,
                task.symbol,
                training_data
            )

            task.status = RetrainingStatus.COMPLETED
            task.completed_at = datetime.utcnow().isoformat()
            task.result = result

            # 記錄訓練時間
            self._last_training[task.model_version] = datetime.utcnow()

            # 發送通知
            await self._send_notification(task, success=True)

            logger.info(f"Retraining completed for {task.model_version}")

        except Exception as e:
            task.status = RetrainingStatus.FAILED
            task.completed_at = datetime.utcnow().isoformat()
            task.error = str(e)

            await self._send_notification(task, success=False)

            logger.error(f"Retraining failed for {task.model_version}: {e}")

    async def _prepare_training_data(
        self,
        symbol: str,
        lookback_days: int
    ) -> List[Dict[str, Any]]:
        """準備訓練數據"""
        try:
            from src.services.binance_client import BinanceClient
            client = BinanceClient()

            # 獲取歷史 K 線數據
            klines = await client.get_klines(
                symbol=symbol,
                interval="1h",
                limit=lookback_days * 24
            )

            if not klines:
                return []

            # 轉換為訓練數據格式
            training_data = []
            for kline in klines:
                training_data.append({
                    "timestamp": kline[0],
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5])
                })

            return training_data

        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return []

    async def _train_model(
        self,
        model_version: str,
        symbol: str,
        training_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """執行模型訓練"""
        # 這裡調用現有的訓練邏輯
        # 實際實現會根據具體的訓練框架

        # 模擬訓練結果
        await asyncio.sleep(5)  # 模擬訓練時間

        return {
            "new_model_version": f"{model_version}_retrained",
            "training_samples": len(training_data),
            "accuracy": 0.65 + (len(training_data) / 10000) * 0.1,
            "training_time_seconds": 5
        }

    async def _send_notification(self, task: RetrainingTask, success: bool):
        """發送通知"""
        try:
            from src.services.notification_service import notification_service

            if success:
                await notification_service.notify_model_update(
                    model_id=task.model_version,
                    update_type="retrained",
                    details={
                        "trigger": task.trigger.value if isinstance(task.trigger, RetrainingTrigger) else task.trigger,
                        "result": task.result
                    }
                )
            else:
                await notification_service.notify_risk_alert(
                    alert_type="retraining_failed",
                    model_version=task.model_version,
                    message=f"Model retraining failed: {task.error}",
                    severity="warning"
                )

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    def get_task(self, task_id: str) -> Optional[RetrainingTask]:
        """獲取任務"""
        return self._tasks.get(task_id)

    def get_all_tasks(
        self,
        status: Optional[RetrainingStatus] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """獲取所有任務"""
        tasks = list(self._tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]

        # 按創建時間倒序
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return [t.to_dict() for t in tasks[:limit]]

    def get_performance_history(
        self,
        model_version: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """獲取性能歷史"""
        history = self._performance_history.get(model_version, [])
        return [m.to_dict() for m in history[-limit:]]

    def cancel_task(self, task_id: str) -> bool:
        """取消任務"""
        task = self._tasks.get(task_id)
        if task and task.status == RetrainingStatus.PENDING:
            task.status = RetrainingStatus.CANCELLED
            return True
        return False


# 單例實例
auto_retraining_service = AutoRetrainingService()
