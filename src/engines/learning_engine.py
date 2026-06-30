"""
學習引擎

負責：
- 模型訓練調度
- 在線學習
- 模型評估和選擇
- 持續優化
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import json
import pickle

from src.config import settings
from src.logger import logger
from src.features.technical import TechnicalFeatures
from src.features.feature_store import feature_store


@dataclass
class ModelVersion:
    """模型版本信息"""
    version: str
    model_type: str
    created_at: datetime
    metrics: Dict[str, float]
    params: Dict[str, Any]
    path: str
    is_active: bool = False


@dataclass
class LearningConfig:
    """學習配置"""
    # 訓練參數
    training_interval_hours: int = 24  # 訓練間隔
    min_training_samples: int = 1000  # 最小訓練樣本數
    validation_split: float = 0.2  # 驗證集比例

    # 模型選擇
    min_sharpe_ratio: float = 0.5  # 最小 Sharpe 要求
    max_drawdown_threshold: float = 0.15  # 最大回撤閾值

    # 在線學習
    online_learning_enabled: bool = True
    online_update_frequency: int = 100  # 每 N 筆交易更新

    # 概念漂移檢測
    drift_detection_enabled: bool = True
    drift_window_size: int = 500
    drift_threshold: float = 0.1

    # 保存
    model_dir: str = "models"
    max_versions: int = 10  # 保留的最大版本數


class LearningEngine:
    """
    學習引擎

    功能：
    - 調度定期訓練
    - 管理模型版本
    - 評估模型性能
    - 檢測概念漂移
    - 自動模型切換
    """

    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()

        # 模型管理
        self._models: Dict[str, Any] = {}
        self._model_versions: Dict[str, List[ModelVersion]] = {}
        self._active_version: Dict[str, str] = {}

        # 訓練狀態
        self._is_training = False
        self._last_training_time: Optional[datetime] = None
        self._training_history: List[Dict[str, Any]] = []

        # 性能追蹤
        self._performance_buffer: List[float] = []
        self._trade_count = 0

        # 特徵計算
        self.feature_calculator = TechnicalFeatures()

        # 確保目錄存在
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)

        logger.info("LearningEngine initialized")

    async def start(self):
        """啟動學習引擎"""
        logger.info("Starting LearningEngine...")

        # 啟動定期訓練任務
        asyncio.create_task(self._training_loop())

        # 啟動概念漂移檢測
        if self.config.drift_detection_enabled:
            asyncio.create_task(self._drift_detection_loop())

    async def _training_loop(self):
        """定期訓練循環"""
        while True:
            try:
                # 檢查是否需要訓練
                if self._should_train():
                    await self.train_models()

                # 等待到下一個訓練時間
                await asyncio.sleep(3600)  # 每小時檢查一次

            except Exception as e:
                logger.error(f"Training loop error: {e}")
                await asyncio.sleep(60)

    def _should_train(self) -> bool:
        """檢查是否應該訓練"""
        if self._is_training:
            return False

        if self._last_training_time is None:
            return True

        elapsed = datetime.utcnow() - self._last_training_time
        return elapsed >= timedelta(hours=self.config.training_interval_hours)

    async def train_models(
        self,
        training_data: Optional[pd.DataFrame] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        訓練所有模型

        Args:
            training_data: 訓練數據（如果為 None，從特徵存儲獲取）
            force: 強制訓練

        Returns:
            訓練結果
        """
        if self._is_training and not force:
            return {'status': 'already_training'}

        self._is_training = True
        self._last_training_time = datetime.utcnow()

        results = {}

        try:
            logger.info("Starting model training...")

            # 獲取訓練數據
            if training_data is None:
                training_data = await self._fetch_training_data()

            if training_data is None or len(training_data) < self.config.min_training_samples:
                logger.warning("Insufficient training data")
                return {'status': 'insufficient_data'}

            # 計算特徵
            if 'rsi_14' not in training_data.columns:
                training_data = self.feature_calculator.calculate_all(training_data)

            # 分割數據
            split_idx = int(len(training_data) * (1 - self.config.validation_split))
            train_df = training_data.iloc[:split_idx]
            val_df = training_data.iloc[split_idx:]

            # 訓練 PPO 模型
            ppo_result = await self._train_ppo(train_df, val_df)
            results['ppo'] = ppo_result

            # 訓練 XGBoost 模型（如果需要）
            # xgb_result = await self._train_xgboost(train_df, val_df)
            # results['xgboost'] = xgb_result

            # 記錄訓練歷史
            self._training_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'data_size': len(training_data),
                'results': results
            })

            logger.info(f"Training completed: {results}")

        except Exception as e:
            logger.error(f"Training error: {e}")
            results['error'] = str(e)

        finally:
            self._is_training = False

        return results

    async def _fetch_training_data(self) -> Optional[pd.DataFrame]:
        """從特徵存儲獲取訓練數據"""
        try:
            # 嘗試從 Redis 獲取
            df = await feature_store.get_klines("BTCUSDT", "1m", limit=10000)

            if df is not None and len(df) > 0:
                return df

            # 如果沒有緩存數據，嘗試從 Binance API 獲取
            from src.services.binance_client import binance_client
            klines = await binance_client.get_historical_klines("BTCUSDT", "1m", limit=5000)
            
            # 將 list 轉換為 DataFrame
            if klines and isinstance(klines, list):
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                # 轉換數據類型
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            return None

        except Exception as e:
            logger.error(f"Failed to fetch training data: {e}")
            return None

    async def _train_ppo(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """訓練 PPO 模型"""
        try:
            from src.strategies.rl_based.ppo_agent import PPOTradingAgent

            agent = PPOTradingAgent()

            # 訓練 (使用 thread pool 避免阻塞 event loop)
            import asyncio
            result = await asyncio.to_thread(
                agent.train,
                training_data=train_df,
                total_timesteps=50000,
                eval_data=val_df
            )

            # 評估
            metrics = result.get('evaluation', {})

            # 檢查是否達到標準
            sharpe = metrics.get('mean_return', 0) / (metrics.get('std_return', 1) + 1e-10)

            if sharpe >= self.config.min_sharpe_ratio:
                # 保存為新版本
                version = self._create_version(
                    model_type='ppo',
                    metrics=metrics,
                    params=agent.config.params,
                    path=result.get('model_path', '')
                )

                # 激活新版本
                self._activate_version('ppo', version.version)

                return {
                    'status': 'success',
                    'version': version.version,
                    'metrics': metrics,
                    'activated': True
                }
            else:
                return {
                    'status': 'below_threshold',
                    'sharpe': sharpe,
                    'metrics': metrics
                }

        except Exception as e:
            logger.error(f"PPO training error: {e}")
            return {'status': 'error', 'error': str(e)}

    def _create_version(
        self,
        model_type: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        path: str
    ) -> ModelVersion:
        """創建模型版本"""
        version_id = f"v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        version = ModelVersion(
            version=version_id,
            model_type=model_type,
            created_at=datetime.utcnow(),
            metrics=metrics,
            params=params,
            path=path
        )

        # 添加到版本列表
        if model_type not in self._model_versions:
            self._model_versions[model_type] = []

        self._model_versions[model_type].append(version)

        # 清理舊版本
        if len(self._model_versions[model_type]) > self.config.max_versions:
            self._model_versions[model_type] = self._model_versions[model_type][-self.config.max_versions:]

        logger.info(f"Created model version: {model_type}/{version_id}")

        return version

    def _activate_version(self, model_type: str, version: str):
        """激活模型版本"""
        versions = self._model_versions.get(model_type, [])

        for v in versions:
            v.is_active = (v.version == version)

        self._active_version[model_type] = version
        logger.info(f"Activated model version: {model_type}/{version}")

    # ==================== 在線學習 ====================

    def record_trade_result(self, pnl_percent: float):
        """記錄交易結果用於在線學習"""
        self._performance_buffer.append(pnl_percent)
        self._trade_count += 1

        # 檢查是否需要在線更新
        if (self.config.online_learning_enabled and
            self._trade_count % self.config.online_update_frequency == 0):
            asyncio.create_task(self._online_update())

    async def _online_update(self):
        """執行在線學習更新"""
        if len(self._performance_buffer) < 50:
            return

        try:
            # 計算最近績效
            recent_perf = self._performance_buffer[-50:]
            avg_return = np.mean(recent_perf)
            sharpe = np.mean(recent_perf) / (np.std(recent_perf) + 1e-10)

            logger.info(f"Online update - Avg return: {avg_return:.4f}, Sharpe: {sharpe:.2f}")

            # 如果績效下降，考慮回退到舊版本
            if sharpe < self.config.min_sharpe_ratio * 0.5:
                await self._consider_rollback()

        except Exception as e:
            logger.error(f"Online update error: {e}")

    async def _consider_rollback(self):
        """考慮回退到舊版本"""
        for model_type, versions in self._model_versions.items():
            if len(versions) < 2:
                continue

            # 獲取前一個版本
            current_idx = -1
            for i, v in enumerate(versions):
                if v.is_active:
                    current_idx = i
                    break

            if current_idx > 0:
                previous = versions[current_idx - 1]
                logger.warning(f"Considering rollback to {model_type}/{previous.version}")
                # 實際回退邏輯...

    # ==================== 概念漂移檢測 ====================

    async def _drift_detection_loop(self):
        """概念漂移檢測循環"""
        while True:
            await asyncio.sleep(3600)  # 每小時檢測

            try:
                if len(self._performance_buffer) >= self.config.drift_window_size:
                    drift_detected = self._detect_drift()

                    if drift_detected:
                        logger.warning("Concept drift detected!")
                        await self.train_models(force=True)

            except Exception as e:
                logger.error(f"Drift detection error: {e}")

    def _detect_drift(self) -> bool:
        """檢測概念漂移"""
        if len(self._performance_buffer) < self.config.drift_window_size * 2:
            return False

        # 比較前後窗口的性能分布
        window = self.config.drift_window_size
        old_perf = self._performance_buffer[-window*2:-window]
        new_perf = self._performance_buffer[-window:]

        old_mean = np.mean(old_perf)
        new_mean = np.mean(new_perf)

        # 簡單的均值漂移檢測
        if abs(new_mean - old_mean) > self.config.drift_threshold:
            return True

        return False

    # ==================== 狀態和信息 ====================

    def get_status(self) -> Dict[str, Any]:
        """獲取學習引擎狀態"""
        return {
            'is_training': self._is_training,
            'last_training': self._last_training_time.isoformat() if self._last_training_time else None,
            'trade_count': self._trade_count,
            'performance_buffer_size': len(self._performance_buffer),
            'active_versions': self._active_version,
            'model_versions': {
                model_type: [
                    {
                        'version': v.version,
                        'created_at': v.created_at.isoformat(),
                        'is_active': v.is_active,
                        'metrics': v.metrics
                    }
                    for v in versions
                ]
                for model_type, versions in self._model_versions.items()
            },
            'training_history_length': len(self._training_history)
        }

    def get_training_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """獲取訓練歷史"""
        return self._training_history[-limit:]

    def get_recent_performance(self, window: int = 100) -> Dict[str, float]:
        """獲取最近績效"""
        if not self._performance_buffer:
            return {}

        recent = self._performance_buffer[-window:]
        returns = np.array(recent)

        return {
            'count': len(recent),
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'sharpe': float(np.mean(returns) / (np.std(returns) + 1e-10)),
            'win_rate': float(np.sum(returns > 0) / len(returns)),
            'total_return': float(np.sum(returns))
        }


# 全局實例
learning_engine = LearningEngine()
