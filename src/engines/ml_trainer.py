"""
機器學習模型訓練引擎

功能：
1. 特徵工程
2. 多種模型訓練（隨機森林、XGBoost、LSTM）
3. 模型評估和選擇
4. 超參數優化
5. 模型持久化
"""

import asyncio
import numpy as np
import pandas as pd
import pickle
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from src.config import settings
from src.logger import logger
from src.services.binance_client import binance_client

# 嘗試導入 ML 庫（可選依賴）
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. ML features will be limited.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed. XGBoost models will not be available.")


class ModelType(str, Enum):
    """模型類型"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"


@dataclass
class ModelConfig:
    """模型配置"""
    model_type: ModelType = ModelType.RANDOM_FOREST
    prediction_horizon: int = 5  # 預測未來 N 根 K 線
    threshold: float = 0.01     # 漲跌閾值 1%
    test_size: float = 0.2     # 測試集比例
    random_state: int = 42


@dataclass
class TrainingResult:
    """訓練結果"""
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    feature_importance: Dict[str, float]
    training_time: float
    model_path: str
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "accuracy": round(self.accuracy * 100, 2),
            "precision": round(self.precision * 100, 2),
            "recall": round(self.recall * 100, 2),
            "f1": round(self.f1 * 100, 2),
            "feature_importance": {k: round(v, 4) for k, v in self.feature_importance.items()},
            "training_time": round(self.training_time, 2),
            "model_path": self.model_path,
            "timestamp": self.timestamp.isoformat()
        }


class FeatureEngineer:
    """特徵工程"""

    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """創建特徵"""
        features = pd.DataFrame(index=df.index)

        # 價格特徵
        features['returns'] = df['close'].pct_change()
        features['returns_2'] = df['close'].pct_change(2)
        features['returns_5'] = df['close'].pct_change(5)
        features['returns_10'] = df['close'].pct_change(10)

        # 波動率特徵
        features['volatility_5'] = df['close'].rolling(5).std() / df['close'].rolling(5).mean()
        features['volatility_10'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()
        features['volatility_20'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()

        # 動量特徵
        features['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        features['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        features['momentum_20'] = df['close'] / df['close'].shift(20) - 1

        # 均線特徵
        for period in [5, 10, 20, 50]:
            sma = df['close'].rolling(period).mean()
            features[f'sma_{period}_ratio'] = df['close'] / sma - 1
            features[f'sma_{period}_slope'] = sma.pct_change()

        # EMA 特徵
        for period in [12, 26]:
            ema = df['close'].ewm(span=period).mean()
            features[f'ema_{period}_ratio'] = df['close'] / ema - 1

        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        features['rsi_normalized'] = (features['rsi'] - 50) / 50

        # 布林帶
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        features['bb_position'] = (df['close'] - bb_middle) / (2 * bb_std)

        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr'] = tr.rolling(14).mean() / df['close']

        # 成交量特徵
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_momentum'] = df['volume'].pct_change(5)

        # K 線形態特徵
        features['candle_body'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.0001)
        features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 0.0001)
        features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 0.0001)

        # 趨勢特徵
        features['trend_strength'] = (df['close'].rolling(20).mean() - df['close'].rolling(50).mean()) / df['close']

        # 時間特徵（如果有時間索引）
        if isinstance(df.index, pd.DatetimeIndex):
            features['hour'] = df.index.hour / 24
            features['day_of_week'] = df.index.dayofweek / 7

        return features

    @staticmethod
    def create_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.01) -> pd.Series:
        """
        創建標籤

        Args:
            df: 原始數據
            horizon: 預測週期
            threshold: 漲跌閾值

        Returns:
            標籤 Series (1=漲, 0=跌, NaN=無效)
        """
        future_returns = df['close'].shift(-horizon) / df['close'] - 1

        # 使用 NaN 作為初始值，之後可以方便地過濾
        labels = pd.Series(index=df.index, dtype=float)
        labels[:] = np.nan  # 預設為 NaN

        # 設置漲跌標籤
        labels[future_returns > threshold] = 1   # 漲
        labels[future_returns < -threshold] = 0  # 跌
        # 橫盤保持 NaN，會被過濾掉

        return labels


class MLTrainer:
    """機器學習訓練器"""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_names: List[str] = []

        # 模型保存路徑
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

        # 訓練歷史
        self.training_history: List[TrainingResult] = []

        logger.info("MLTrainer initialized")

    async def load_data(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 1000
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        載入並準備訓練數據

        Returns:
            (原始數據, 特徵, 標籤)
        """
        logger.info(f"Loading training data: {symbol} {interval}")

        # 從 Binance 獲取數據
        klines = await binance_client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
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
            horizon=self.config.prediction_horizon,
            threshold=self.config.threshold
        )

        # 移除無效樣本 - 處理 NaN 值
        # 1. 先移除特徵中的 NaN
        features = features.dropna()

        # 2. 對齊標籤
        labels = labels.loc[features.index]

        # 3. 移除標籤中的 NaN（包括橫盤和無效數據）
        valid_mask = labels.notna()
        features = features[valid_mask]
        labels = labels[valid_mask]

        # 4. 確保標籤為整數類型
        labels = labels.astype(int)

        self.feature_names = features.columns.tolist()

        if len(features) == 0:
            raise ValueError("No valid samples after preprocessing. Try increasing data limit or adjusting threshold.")

        logger.info(f"Prepared {len(features)} samples with {len(self.feature_names)} features")
        logger.info(f"Label distribution: {labels.value_counts().to_dict()}")

        return df, features, labels

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimize: bool = False
    ) -> RandomForestClassifier:
        """訓練隨機森林模型"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Random Forest")

        logger.info("Training Random Forest model...")

        if optimize:
            # 超參數優化
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = GridSearchCV(
                RandomForestClassifier(random_state=self.config.random_state),
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            logger.info(f"Best params: {model.best_params_}")
            return model.best_estimator_
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            return model

    def train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> GradientBoostingClassifier:
        """訓練梯度提升模型"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Gradient Boosting")

        logger.info("Training Gradient Boosting model...")

        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.config.random_state
        )
        model.fit(X_train, y_train)
        return model

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ):
        """訓練 XGBoost 模型"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required for XGBoost model")

        logger.info("Training XGBoost model...")

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.config.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        return model

    async def train(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 2000,
        model_types: List[ModelType] = None
    ) -> Dict[str, TrainingResult]:
        """
        訓練模型

        Args:
            symbol: 交易對
            interval: K 線間隔
            limit: 數據量
            model_types: 要訓練的模型類型

        Returns:
            各模型的訓練結果
        """
        if model_types is None:
            model_types = [ModelType.RANDOM_FOREST]
            if XGBOOST_AVAILABLE:
                model_types.append(ModelType.XGBOOST)

        # 載入數據
        df, features, labels = await self.load_data(symbol, interval, limit)

        # 檢查最小樣本數
        min_samples = 100
        if len(features) < min_samples:
            raise ValueError(
                f"Insufficient samples: {len(features)} (minimum: {min_samples}). "
                "Try increasing data limit or lowering threshold."
            )

        # 檢查標籤分佈
        label_counts = labels.value_counts()
        if len(label_counts) < 2:
            raise ValueError(
                f"Insufficient label diversity: {label_counts.to_dict()}. "
                "Need both UP (1) and DOWN (0) samples."
            )

        # 標準化
        scaler = StandardScaler()
        X = scaler.fit_transform(features.values)
        y = labels.values

        # 分割數據（如果某類樣本太少，不使用 stratify）
        min_class_count = label_counts.min()
        use_stratify = min_class_count >= 5

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if use_stratify else None
        )

        results = {}

        for model_type in model_types:
            start_time = datetime.now()

            try:
                # 訓練模型
                if model_type == ModelType.RANDOM_FOREST:
                    model = self.train_random_forest(X_train, y_train)
                elif model_type == ModelType.GRADIENT_BOOSTING:
                    model = self.train_gradient_boosting(X_train, y_train)
                elif model_type == ModelType.XGBOOST:
                    model = self.train_xgboost(X_train, y_train)
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    continue

                # 評估模型
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                # 特徵重要性
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(self.feature_names, model.feature_importances_))
                    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
                else:
                    importance = {}

                # 保存模型
                model_path = self.model_dir / f"{model_type.value}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': model,
                        'scaler': scaler,
                        'feature_names': self.feature_names,
                        'config': self.config
                    }, f)

                training_time = (datetime.now() - start_time).total_seconds()

                result = TrainingResult(
                    model_type=model_type.value,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    feature_importance=importance,
                    training_time=training_time,
                    model_path=str(model_path),
                    timestamp=datetime.now()
                )

                results[model_type.value] = result
                self.models[model_type.value] = model
                self.scalers[model_type.value] = scaler
                self.training_history.append(result)

                logger.info(f"{model_type.value} - Accuracy: {accuracy:.2%}, F1: {f1:.2%}")

            except Exception as e:
                logger.error(f"Failed to train {model_type.value}: {e}")

        return results

    def load_model(self, model_path: str) -> bool:
        """載入已保存的模型"""
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)

            model_type = Path(model_path).stem.split('_')[0]
            self.models[model_type] = data['model']
            self.scalers[model_type] = data['scaler']
            self.feature_names = data['feature_names']

            logger.info(f"Loaded model from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def predict(
        self,
        features: pd.DataFrame,
        model_type: str = "random_forest"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        預測

        Args:
            features: 特徵 DataFrame
            model_type: 模型類型

        Returns:
            (預測類別, 預測概率)
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not loaded")

        model = self.models[model_type]
        scaler = self.scalers[model_type]

        # 確保特徵順序正確
        features = features[self.feature_names]

        # 標準化
        X = scaler.transform(features.values)

        # 預測
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None

        return predictions, probabilities

    def get_training_history(self) -> List[Dict]:
        """獲取訓練歷史"""
        return [r.to_dict() for r in self.training_history]

    def get_best_model(self) -> Optional[str]:
        """獲取最佳模型"""
        if not self.training_history:
            return None

        best = max(self.training_history, key=lambda x: x.f1)
        return best.model_type


# 全局實例
ml_trainer = MLTrainer()
