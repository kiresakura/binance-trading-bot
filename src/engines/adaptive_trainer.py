"""
自適應訓練器 (Adaptive Trainer)

靈感來源：FreqAI 自適應 ML 系統
功能：
1. 定期自動重訓練（可配置間隔：每天/每週/每N根K線）
2. 滑動窗口數據選擇
3. 模型版本自動管理
4. 與 Arena 系統整合：新模型自動註冊為 Challenger
5. 持續學習支援
6. 近期數據加權

作者：Binance Trading Bot v0.3.0
"""

import asyncio
import threading
import time
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import deque

from src.config import settings
from src.logger import logger
from src.services.binance_client import binance_client
from src.engines.ml_trainer import MLTrainer, ModelType, FeatureEngineer, ModelConfig

# 嘗試導入 ML 庫
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class RetrainTrigger(str, Enum):
    """重訓練觸發方式"""
    TIME_BASED = "time_based"      # 基於時間間隔
    CANDLE_BASED = "candle_based"  # 基於 K 線數量
    PERFORMANCE_BASED = "performance_based"  # 基於性能下降
    MANUAL = "manual"              # 手動觸發


@dataclass
class AdaptiveConfig:
    """自適應訓練配置"""
    # 重訓練設定
    retrain_trigger: RetrainTrigger = RetrainTrigger.TIME_BASED
    retrain_interval_hours: int = 24        # 每 24 小時重訓練
    retrain_candle_count: int = 100         # 每 100 根 K 線重訓練
    performance_threshold: float = 0.1      # 性能下降超過 10% 觸發重訓練

    # 數據窗口
    training_window_days: int = 30          # 訓練數據天數
    min_training_samples: int = 500         # 最小訓練樣本數

    # 數據權重
    weight_factor: float = 0.9              # 近期數據權重係數（指數衰減）
    use_weighted_samples: bool = True       # 是否使用加權樣本

    # 持續學習
    continual_learning: bool = False        # 是否從舊模型繼續訓練
    max_model_versions: int = 10            # 保留的最大模型版本數

    # Arena 整合
    auto_register_to_arena: bool = True     # 自動註冊到 Arena
    auto_start_shadow_trading: bool = True  # 自動開始影子交易

    # 模型配置
    model_types: List[str] = field(default_factory=lambda: ["random_forest", "xgboost"])
    symbol: str = "BTCUSDT"
    interval: str = "1h"

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "retrain_trigger": self.retrain_trigger.value
        }


@dataclass
class TrainingJob:
    """訓練任務"""
    job_id: str
    symbol: str
    status: str  # pending, running, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    model_version: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "symbol": self.symbol,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "model_version": self.model_version,
            "metrics": self.metrics,
            "error": self.error
        }


class AdaptiveTrainer:
    """自適應訓練器"""

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()
        self.ml_trainer = MLTrainer()

        # 訓練狀態
        self._is_running = False
        self._stop_event = threading.Event()
        self._background_task: Optional[asyncio.Task] = None

        # 訓練記錄
        self.last_training_time: Dict[str, datetime] = {}  # symbol -> last_train_time
        self.candle_count_since_train: Dict[str, int] = {}  # symbol -> candle_count
        self.training_history: List[TrainingJob] = []
        self.current_job: Optional[TrainingJob] = None

        # 訓練隊列（按優先級排序）
        self.train_queue: deque = deque()

        # 模型版本管理
        self.model_dir = Path("models/adaptive")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Arena 服務（延遲導入避免循環依賴）
        self._arena_service = None

        logger.info("AdaptiveTrainer initialized")

    @property
    def arena_service(self):
        """延遲導入 Arena 服務"""
        if self._arena_service is None:
            from src.services.arena_service import arena_service
            self._arena_service = arena_service
        return self._arena_service

    # ==================== 配置管理 ====================

    def update_config(self, **kwargs) -> AdaptiveConfig:
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                if key == "retrain_trigger" and isinstance(value, str):
                    value = RetrainTrigger(value)
                setattr(self.config, key, value)

        logger.info(f"AdaptiveTrainer config updated: {kwargs}")
        return self.config

    def get_config(self) -> Dict[str, Any]:
        """獲取配置"""
        return self.config.to_dict()

    # ==================== 數據處理 ====================

    def set_weights_higher_recent(self, num_samples: int) -> np.ndarray:
        """
        設置近期數據更高權重（指數衰減）
        借鑒自 FreqAI 的 data_kitchen.py

        Args:
            num_samples: 樣本數量

        Returns:
            權重數組
        """
        wfactor = self.config.weight_factor
        weights = np.exp(-np.arange(num_samples) / (wfactor * num_samples))[::-1]
        return weights

    async def load_training_data(
        self,
        symbol: str,
        interval: str,
        days: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        載入訓練數據

        Args:
            symbol: 交易對
            interval: K 線間隔
            days: 數據天數

        Returns:
            (原始數據, 特徵, 標籤)
        """
        # 計算需要的 K 線數量
        interval_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        minutes_per_day = 24 * 60
        candles_per_day = minutes_per_day // interval_minutes.get(interval, 60)
        limit = days * candles_per_day

        logger.info(f"Loading {limit} candles for {symbol} {interval}")

        # 獲取數據
        klines = await binance_client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=min(limit, 1000)  # Binance 限制
        )

        if not klines:
            raise ValueError(f"No data returned for {symbol}")

        # 轉換為 DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df.set_index('timestamp', inplace=True)

        # 創建特徵和標籤
        features = FeatureEngineer.create_features(df)
        labels = FeatureEngineer.create_labels(
            df,
            horizon=self.ml_trainer.config.prediction_horizon,
            threshold=self.ml_trainer.config.threshold
        )

        # 清理數據
        features = features.dropna()
        labels = labels.loc[features.index]
        valid_mask = labels.notna()
        features = features[valid_mask]
        labels = labels[valid_mask].astype(int)

        logger.info(f"Loaded {len(features)} valid samples")

        return df, features, labels

    # ==================== 訓練邏輯 ====================

    def check_if_retrain_required(self, symbol: str) -> Tuple[bool, str]:
        """
        檢查是否需要重訓練
        借鑒自 FreqAI 的 check_if_new_training_required

        Args:
            symbol: 交易對

        Returns:
            (是否需要重訓練, 原因)
        """
        trigger = self.config.retrain_trigger

        if trigger == RetrainTrigger.TIME_BASED:
            last_train = self.last_training_time.get(symbol)
            if last_train is None:
                return True, "No previous training"

            hours_since_train = (datetime.now() - last_train).total_seconds() / 3600
            if hours_since_train >= self.config.retrain_interval_hours:
                return True, f"Time elapsed: {hours_since_train:.1f}h >= {self.config.retrain_interval_hours}h"
            return False, f"Time elapsed: {hours_since_train:.1f}h < {self.config.retrain_interval_hours}h"

        elif trigger == RetrainTrigger.CANDLE_BASED:
            candles = self.candle_count_since_train.get(symbol, 0)
            if candles >= self.config.retrain_candle_count:
                return True, f"Candle count: {candles} >= {self.config.retrain_candle_count}"
            return False, f"Candle count: {candles} < {self.config.retrain_candle_count}"

        elif trigger == RetrainTrigger.PERFORMANCE_BASED:
            # TODO: 實現基於性能的重訓練邏輯
            return False, "Performance-based trigger not implemented"

        return False, "Unknown trigger"

    async def train_model(
        self,
        symbol: Optional[str] = None,
        force: bool = False
    ) -> Optional[TrainingJob]:
        """
        訓練模型

        Args:
            symbol: 交易對（默認使用配置中的 symbol）
            force: 是否強制訓練（忽略重訓練檢查）

        Returns:
            訓練任務
        """
        symbol = symbol or self.config.symbol

        # 檢查是否需要重訓練
        if not force:
            needs_retrain, reason = self.check_if_retrain_required(symbol)
            if not needs_retrain:
                logger.info(f"Skipping training for {symbol}: {reason}")
                return None

        # 創建訓練任務
        job_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job = TrainingJob(
            job_id=job_id,
            symbol=symbol,
            status="running",
            started_at=datetime.now()
        )
        self.current_job = job

        try:
            logger.info(f"Starting adaptive training for {symbol}")

            # 載入數據
            df, features, labels = await self.load_training_data(
                symbol=symbol,
                interval=self.config.interval,
                days=self.config.training_window_days
            )

            # 檢查樣本數
            if len(features) < self.config.min_training_samples:
                raise ValueError(
                    f"Insufficient samples: {len(features)} < {self.config.min_training_samples}"
                )

            # 計算樣本權重
            if self.config.use_weighted_samples:
                sample_weights = self.set_weights_higher_recent(len(features))
            else:
                sample_weights = np.ones(len(features))

            # 標準化
            scaler = StandardScaler()
            X = scaler.fit_transform(features.values)
            y = labels.values

            # 分割數據
            label_counts = labels.value_counts()
            use_stratify = label_counts.min() >= 5

            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, sample_weights,
                test_size=0.2,
                random_state=42,
                stratify=y if use_stratify else None
            )

            # 訓練多種模型
            best_model = None
            best_f1 = 0
            best_model_type = None
            results = {}

            for model_type_str in self.config.model_types:
                try:
                    model_type = ModelType(model_type_str)

                    if model_type == ModelType.RANDOM_FOREST:
                        model = self.ml_trainer.train_random_forest(X_train, y_train)
                    elif model_type == ModelType.XGBOOST:
                        model = self.ml_trainer.train_xgboost(X_train, y_train)
                    elif model_type == ModelType.GRADIENT_BOOSTING:
                        model = self.ml_trainer.train_gradient_boosting(X_train, y_train)
                    else:
                        continue

                    # 評估
                    from sklearn.metrics import accuracy_score, f1_score
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                    results[model_type_str] = {
                        "accuracy": round(accuracy * 100, 2),
                        "f1": round(f1 * 100, 2)
                    }

                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = model
                        best_model_type = model_type_str

                    logger.info(f"{model_type_str}: Accuracy={accuracy:.2%}, F1={f1:.2%}")

                except Exception as e:
                    logger.error(f"Failed to train {model_type_str}: {e}")

            if best_model is None:
                raise ValueError("All model training failed")

            # 生成模型版本號
            model_version = f"adaptive_{best_model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 保存模型
            model_path = self.model_dir / f"{model_version}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': best_model,
                    'scaler': scaler,
                    'feature_names': features.columns.tolist(),
                    'model_type': best_model_type,
                    'config': self.config.to_dict(),
                    'trained_at': datetime.now().isoformat(),
                    'metrics': results
                }, f)

            logger.info(f"Model saved: {model_path}")

            # 更新訓練記錄
            self.last_training_time[symbol] = datetime.now()
            self.candle_count_since_train[symbol] = 0

            # 更新任務
            job.status = "completed"
            job.completed_at = datetime.now()
            job.model_version = model_version
            job.metrics = {
                "best_model": best_model_type,
                "best_f1": round(best_f1 * 100, 2),
                "all_results": results,
                "sample_count": len(features)
            }

            # Arena 整合
            if self.config.auto_register_to_arena:
                await self._register_to_arena(model_version, best_model_type, results)

            # 清理舊模型
            self._cleanup_old_models()

            return job

        except Exception as e:
            logger.error(f"Training failed: {e}")
            job.status = "failed"
            job.completed_at = datetime.now()
            job.error = str(e)
            return job

        finally:
            self.training_history.append(job)
            self.current_job = None

    async def _register_to_arena(
        self,
        model_version: str,
        model_type: str,
        metrics: Dict
    ):
        """將新模型註冊到 Arena"""
        try:
            await self.arena_service.initialize()

            # 註冊模型
            success = await self.arena_service.register_model(
                version=model_version,
                model_type=model_type,
                config={
                    "source": "adaptive_trainer",
                    "metrics": metrics,
                    "trained_at": datetime.now().isoformat()
                }
            )

            if success:
                logger.info(f"Model {model_version} registered to Arena")

                # 自動開始影子交易
                if self.config.auto_start_shadow_trading:
                    await self.arena_service.start_shadow_trading(model_version)
                    logger.info(f"Shadow trading started for {model_version}")

        except Exception as e:
            logger.error(f"Failed to register model to Arena: {e}")

    def _cleanup_old_models(self):
        """清理舊模型，保留最新的 N 個版本"""
        try:
            model_files = sorted(
                self.model_dir.glob("*.pkl"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            for old_model in model_files[self.config.max_model_versions:]:
                old_model.unlink()
                logger.info(f"Removed old model: {old_model}")

        except Exception as e:
            logger.error(f"Failed to cleanup old models: {e}")

    # ==================== 後台掃描 ====================

    def start_background_scanning(self, symbols: Optional[List[str]] = None):
        """
        啟動後台掃描（檢查是否需要重訓練）
        借鑒自 FreqAI 的 _start_scanning

        Args:
            symbols: 要監控的交易對列表
        """
        if self._is_running:
            logger.warning("Background scanning already running")
            return

        symbols = symbols or [self.config.symbol]
        self.train_queue = deque(symbols)

        self._is_running = True
        self._stop_event.clear()

        # 使用 asyncio.create_task 而非 threading (避免跨 event loop 問題)
        self._background_task = asyncio.create_task(
            self._background_scan_loop_async()
        )

        logger.info(f"Background scanning started for {symbols}")

    async def _background_scan_loop_async(self):
        """後台掃描循環 (async 版本 - 避免跨 event loop 問題)"""
        while not self._stop_event.is_set():
            try:
                if not self.train_queue:
                    await asyncio.sleep(60)
                    continue

                symbol = self.train_queue[0]

                # 檢查是否需要重訓練
                needs_retrain, reason = self.check_if_retrain_required(symbol)

                if needs_retrain:
                    logger.info(f"Triggering retrain for {symbol}: {reason}")

                    # 直接 await (在同一個 event loop 中執行)
                    await self.train_model(symbol, force=True)

                    # 輪轉隊列
                    self.train_queue.rotate(-1)

                # 等待一段時間再檢查
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                logger.info("Background scan task cancelled")
                break
            except Exception as e:
                logger.error(f"Background scan error: {e}")
                await asyncio.sleep(60)

    def stop_background_scanning(self):
        """停止後台掃描"""
        if not self._is_running:
            return

        self._stop_event.set()
        if hasattr(self, '_background_task') and self._background_task:
            self._background_task.cancel()

        self._is_running = False
        logger.info("Background scanning stopped")

    def increment_candle_count(self, symbol: str, count: int = 1):
        """增加 K 線計數（由外部調用）"""
        self.candle_count_since_train[symbol] = \
            self.candle_count_since_train.get(symbol, 0) + count

    # ==================== 狀態查詢 ====================

    def get_status(self) -> Dict[str, Any]:
        """獲取訓練器狀態"""
        return {
            "is_running": self._is_running,
            "config": self.config.to_dict(),
            "current_job": self.current_job.to_dict() if self.current_job else None,
            "last_training": {
                symbol: ts.isoformat()
                for symbol, ts in self.last_training_time.items()
            },
            "candle_counts": self.candle_count_since_train,
            "train_queue": list(self.train_queue),
            "model_versions": [f.stem for f in self.model_dir.glob("*.pkl")]
        }

    def get_training_history(self, limit: int = 20) -> List[Dict]:
        """獲取訓練歷史"""
        return [job.to_dict() for job in self.training_history[-limit:]]

    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有模型"""
        models = []
        for model_file in sorted(
            self.model_dir.glob("*.pkl"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        ):
            try:
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)

                models.append({
                    "version": model_file.stem,
                    "model_type": data.get('model_type'),
                    "trained_at": data.get('trained_at'),
                    "metrics": data.get('metrics'),
                    "file_size": model_file.stat().st_size
                })
            except Exception as e:
                logger.error(f"Failed to load model info {model_file}: {e}")

        return models


# 全局實例
adaptive_trainer = AdaptiveTrainer()
