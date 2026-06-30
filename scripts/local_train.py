#!/usr/bin/env python3
"""
本地訓練腳本

在本地 Mac 上運行 ML 模型訓練，不需要啟動整個 API 服務。
訓練完成後，可以用 sync_models.py 將模型上傳到 Railway。

使用方式：
    python scripts/local_train.py --symbol BTCUSDT --days 30
    python scripts/local_train.py --symbol BTCUSDT --model-type ensemble --epochs 100
    python scripts/local_train.py --backtest --strategy momentum
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 添加專案根目錄到 Python 路徑
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 設置環境變數
os.environ.setdefault("TRAINING_ENABLED", "true")
os.environ.setdefault("DRY_RUN", "true")  # 本地訓練不執行實際交易

from src.logger import logger


def setup_device():
    """設置訓練設備（優先使用 Mac MPS GPU）"""
    import torch

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("🚀 Using Apple MPS GPU for training")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("🚀 Using NVIDIA CUDA GPU for training")
    else:
        device = torch.device("cpu")
        logger.info("💻 Using CPU for training")

    return device


async def fetch_training_data(symbol: str, days: int, source: str = "binance"):
    """獲取訓練數據

    Args:
        symbol: 交易對 (e.g., BTCUSDT)
        days: 獲取過去幾天的數據
        source: 數據來源 ('binance' 或 'database')
    """
    logger.info(f"📊 Fetching {days} days of data for {symbol} from {source}...")

    if source == "binance":
        # 直接從幣安 API 獲取數據
        from src.services.binance_client import binance_client

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # 獲取 1 小時 K 線數據
        klines = await binance_client.get_historical_klines(
            symbol=symbol,
            interval="1h",
            start_time=start_time,
            end_time=end_time
        )

        logger.info(f"✅ Fetched {len(klines)} klines from Binance")
        return klines

    elif source == "database":
        # 從 Railway PostgreSQL 獲取數據（需要設置 DATABASE_URL）
        from src.utils.database import async_db_manager

        if not async_db_manager._initialized:
            await async_db_manager.init()

        # 這裡需要根據你的資料表結構調整
        query = """
            SELECT * FROM market_data
            WHERE symbol = $1 AND timestamp > $2
            ORDER BY timestamp ASC
        """
        start_time = datetime.now() - timedelta(days=days)

        async with async_db_manager.pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, start_time)

        logger.info(f"✅ Fetched {len(rows)} rows from database")
        return rows

    else:
        raise ValueError(f"Unknown data source: {source}")


async def train_ml_model(
    symbol: str,
    model_type: str = "ensemble",
    days: int = 30,
    epochs: int = 50,
    output_dir: str = None
):
    """訓練 ML 模型

    Args:
        symbol: 交易對
        model_type: 模型類型 ('random_forest', 'xgboost', 'lstm', 'ensemble')
        days: 訓練數據天數
        epochs: 訓練輪數
        output_dir: 模型輸出目錄
    """
    device = setup_device()

    if output_dir is None:
        output_dir = PROJECT_ROOT / "models"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"🎯 Training {model_type} model for {symbol}")
    logger.info(f"📁 Output directory: {output_dir}")

    # 獲取訓練數據
    data = await fetch_training_data(symbol, days, source="binance")

    if not data:
        logger.error("❌ No training data available")
        return None

    # 數據預處理
    logger.info("🔧 Preprocessing data...")
    from src.features.technical import TechnicalFeatureGenerator
    import pandas as pd

    # 轉換 K 線數據為 DataFrame
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')

    # 轉換數值類型
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 生成技術特徵
    feature_generator = TechnicalFeatureGenerator()
    df_features = feature_generator.add_all_features(df)

    logger.info(f"📈 Generated {len(df_features.columns)} features")

    # 根據模型類型訓練
    model_path = None

    if model_type in ["random_forest", "xgboost", "gradient_boosting", "ensemble"]:
        model_path = await train_sklearn_model(
            df_features, symbol, model_type, output_dir
        )
    elif model_type == "lstm":
        model_path = await train_lstm_model(
            df_features, symbol, epochs, device, output_dir
        )
    elif model_type == "ppo":
        model_path = await train_rl_model(
            df_features, symbol, epochs, device, output_dir
        )
    else:
        logger.error(f"❌ Unknown model type: {model_type}")
        return None

    if model_path:
        logger.info(f"✅ Model saved to: {model_path}")

        # 保存模型元數據
        metadata = {
            "symbol": symbol,
            "model_type": model_type,
            "trained_at": datetime.now().isoformat(),
            "training_days": days,
            "epochs": epochs,
            "model_path": str(model_path),
            "device": str(device)
        }

        import json
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"📝 Metadata saved to: {metadata_path}")

    return model_path


async def train_sklearn_model(df, symbol, model_type, output_dir):
    """訓練 Scikit-learn 類型的模型"""
    import joblib
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

    logger.info(f"🤖 Training {model_type} model...")

    # 準備特徵和標籤
    feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]

    # 創建標籤：未來收益是否為正
    df = df.copy()
    df['future_return'] = df['close'].pct_change(5).shift(-5)
    df['label'] = (df['future_return'] > 0).astype(int)
    df = df.dropna()

    X = df[feature_cols].values
    y = df['label'].values

    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 訓練模型
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        except ImportError:
            logger.warning("XGBoost not installed, using GradientBoosting instead")
            model = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
    elif model_type == "ensemble":
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
        model = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)

    # 評估
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    logger.info(f"📊 Train accuracy: {train_score:.4f}")
    logger.info(f"📊 Test accuracy: {test_score:.4f}")

    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_type}_{symbol}_{timestamp}.joblib"
    model_path = output_dir / model_filename

    # 保存模型和 scaler
    model_bundle = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "train_score": train_score,
        "test_score": test_score
    }

    joblib.dump(model_bundle, model_path)

    return model_path


async def train_lstm_model(df, symbol, epochs, device, output_dir):
    """訓練 LSTM 模型"""
    import torch
    import torch.nn as nn
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    logger.info(f"🧠 Training LSTM model on {device}...")

    # 準備序列數據
    feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]

    df = df.copy()
    df['future_return'] = df['close'].pct_change(5).shift(-5)
    df['label'] = (df['future_return'] > 0).astype(int)
    df = df.dropna()

    X = df[feature_cols].values
    y = df['label'].values

    # 創建序列
    seq_length = 24  # 24 小時序列
    X_seq, y_seq = [], []

    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # 分割數據
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    # 轉換為 Tensor
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.LongTensor(y_test).to(device)

    # 定義 LSTM 模型
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    model = LSTMModel(input_size=X_train.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 訓練
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_pred = test_outputs.argmax(dim=1)
                test_acc = (test_pred == y_test).float().mean()
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Test Acc: {test_acc:.4f}")

    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"lstm_{symbol}_{timestamp}.pt"
    model_path = output_dir / model_filename

    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "seq_length": seq_length,
        "input_size": X_train.shape[2]
    }, model_path)

    return model_path


async def train_rl_model(df, symbol, epochs, device, output_dir):
    """訓練強化學習 (PPO) 模型"""
    logger.info(f"🎮 Training PPO model on {device}...")

    try:
        from src.engines.learning_engine import learning_engine

        # 使用現有的學習引擎訓練
        result = await learning_engine.train_models(
            symbols=[symbol],
            model_types=["ppo"],
            training_days=30
        )

        if result and "ppo" in result:
            model_info = result["ppo"]
            if model_info.get("model_path"):
                return Path(model_info["model_path"])

        logger.warning("PPO training did not produce a model")
        return None

    except Exception as e:
        logger.error(f"Failed to train PPO model: {e}")
        return None


async def run_backtest(symbol: str, strategy: str, days: int = 30):
    """運行回測

    Args:
        symbol: 交易對
        strategy: 策略名稱
        days: 回測天數
    """
    logger.info(f"📈 Running backtest for {strategy} on {symbol} ({days} days)")

    try:
        from src.services.backtest_engine import backtest_engine

        result = await backtest_engine.run_backtest(
            symbol=symbol,
            strategy=strategy,
            days=days
        )

        if result:
            logger.info("=" * 50)
            logger.info("📊 Backtest Results:")
            logger.info(f"  Total Return: {result.get('total_return', 0):.2%}")
            logger.info(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
            logger.info(f"  Max Drawdown: {result.get('max_drawdown', 0):.2%}")
            logger.info(f"  Win Rate: {result.get('win_rate', 0):.2%}")
            logger.info(f"  Total Trades: {result.get('total_trades', 0)}")
            logger.info("=" * 50)

        return result

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return None


async def run_hyperopt(symbol: str, strategy: str, n_trials: int = 50):
    """運行超參數優化

    Args:
        symbol: 交易對
        strategy: 策略名稱
        n_trials: 優化試驗次數
    """
    logger.info(f"🔍 Running hyperopt for {strategy} ({n_trials} trials)")

    try:
        from src.services.hyperopt_optimizer import hyperopt_optimizer

        result = await hyperopt_optimizer.optimize(
            strategy=strategy,
            symbol=symbol,
            n_trials=n_trials
        )

        if result:
            logger.info("=" * 50)
            logger.info("🏆 Best Parameters Found:")
            for param, value in result.get("best_params", {}).items():
                logger.info(f"  {param}: {value}")
            logger.info(f"  Best Score: {result.get('best_score', 0):.4f}")
            logger.info("=" * 50)

        return result

    except Exception as e:
        logger.error(f"Hyperopt failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="本地訓練腳本 - 在 Mac 上訓練 ML 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  # 訓練 ensemble 模型
  python scripts/local_train.py --symbol BTCUSDT --model-type ensemble

  # 訓練 LSTM 模型（使用 MPS GPU）
  python scripts/local_train.py --symbol BTCUSDT --model-type lstm --epochs 100

  # 運行回測
  python scripts/local_train.py --backtest --strategy momentum --days 60

  # 運行超參數優化
  python scripts/local_train.py --hyperopt --strategy momentum --trials 100
        """
    )

    # 基本參數
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易對")
    parser.add_argument("--days", type=int, default=30, help="數據天數")

    # 訓練參數
    parser.add_argument("--model-type", type=str, default="ensemble",
                        choices=["random_forest", "xgboost", "gradient_boosting", "ensemble", "lstm", "ppo"],
                        help="模型類型")
    parser.add_argument("--epochs", type=int, default=50, help="訓練輪數（用於深度學習模型）")
    parser.add_argument("--output-dir", type=str, help="模型輸出目錄")

    # 回測參數
    parser.add_argument("--backtest", action="store_true", help="運行回測")
    parser.add_argument("--strategy", type=str, default="momentum", help="回測策略")

    # 超參數優化
    parser.add_argument("--hyperopt", action="store_true", help="運行超參數優化")
    parser.add_argument("--trials", type=int, default=50, help="優化試驗次數")

    args = parser.parse_args()

    # 顯示配置
    logger.info("=" * 50)
    logger.info("🏠 Local Training Environment")
    logger.info("=" * 50)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Days: {args.days}")

    # 運行對應任務
    if args.backtest:
        logger.info(f"Mode: Backtest ({args.strategy})")
        asyncio.run(run_backtest(args.symbol, args.strategy, args.days))
    elif args.hyperopt:
        logger.info(f"Mode: Hyperopt ({args.strategy}, {args.trials} trials)")
        asyncio.run(run_hyperopt(args.symbol, args.strategy, args.trials))
    else:
        logger.info(f"Mode: Training ({args.model_type})")
        logger.info(f"Epochs: {args.epochs}")
        asyncio.run(train_ml_model(
            symbol=args.symbol,
            model_type=args.model_type,
            days=args.days,
            epochs=args.epochs,
            output_dir=args.output_dir
        ))


if __name__ == "__main__":
    main()
