# 本地訓練指南

本指南說明如何在本地 Mac 上訓練 ML 模型，並將訓練好的模型同步到 Railway 線上環境。

## 架構概覽

```
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│         本地 Mac                 │     │         Railway                  │
│      （訓練環境）                 │     │      （線上交易環境）              │
├─────────────────────────────────┤     ├─────────────────────────────────┤
│  • ML 模型訓練                   │     │  • 即時交易引擎                   │
│  • 回測引擎                      │────▶│  • API 服務                      │
│  • 超參數優化                    │     │  • WebSocket 即時數據             │
│  • 自適應訓練                    │     │  • Telegram 通知                  │
│                                 │     │  • 只用模型預測，不訓練            │
└─────────────────────────────────┘     └─────────────────────────────────┘
         │                                         ▲
         │         模型同步                         │
         └────────────────────────────────────────┘
```

## 快速開始

### 1. 設置本地環境

```bash
# 進入專案目錄
cd ~/Documents/BinanceProject/binancetradebot

# 創建虛擬環境
python3 -m venv venv
source venv/bin/activate

# 安裝依賴
pip install -r requirements.txt

# 安裝訓練相關套件
pip install torch torchvision xgboost optuna
```

### 2. 配置環境變數

```bash
# 複製範例配置
cp .env.local.example .env.local

# 編輯配置（填入你的 API Key 等）
nano .env.local

# 載入配置
export $(cat .env.local | xargs)
```

### 3. 運行訓練

```bash
# 訓練 ensemble 模型（預設）
python scripts/local_train.py --symbol BTCUSDT --days 30

# 訓練 LSTM 模型（使用 MPS GPU）
python scripts/local_train.py --symbol BTCUSDT --model-type lstm --epochs 100

# 訓練 PPO 強化學習模型
python scripts/local_train.py --symbol BTCUSDT --model-type ppo --epochs 50000
```

### 4. 同步模型到 Railway

```bash
# 上傳最新訓練的模型
python scripts/sync_models.py --upload-latest

# 或上傳指定模型
python scripts/sync_models.py --upload models/ensemble_BTCUSDT_20241212.joblib

# 通知 Railway 重新載入模型
python scripts/sync_models.py --reload
```

## 訓練命令詳解

### 模型訓練

```bash
# 基本用法
python scripts/local_train.py --symbol <交易對> --model-type <模型類型> --days <天數>

# 參數說明：
#   --symbol        交易對，如 BTCUSDT, ETHUSDT
#   --model-type    模型類型：random_forest, xgboost, ensemble, lstm, ppo
#   --days          訓練數據天數（預設 30）
#   --epochs        訓練輪數（深度學習模型用，預設 50）
#   --output-dir    模型輸出目錄（預設 ./models）
```

### 模型類型比較

| 模型類型 | 訓練速度 | 預測速度 | GPU 加速 | 適用場景 |
|---------|---------|---------|---------|---------|
| random_forest | 快 | 快 | 否 | 穩定基準 |
| xgboost | 快 | 快 | 是 | 表格數據 |
| gradient_boosting | 中 | 快 | 否 | 穩定基準 |
| ensemble | 中 | 快 | 否 | 綜合表現 |
| lstm | 慢 | 中 | 是 (MPS) | 時序預測 |
| ppo | 很慢 | 快 | 是 (MPS) | 強化學習 |

### 回測

```bash
# 運行回測
python scripts/local_train.py --backtest --strategy momentum --days 60

# 可用策略：
#   momentum        動量策略
#   mean_reversion  均值回歸
#   trend_following 趨勢跟蹤
#   breakout        突破策略
```

### 超參數優化

```bash
# 運行 Hyperopt 優化
python scripts/local_train.py --hyperopt --strategy momentum --trials 100
```

## 模型同步命令詳解

### 列出模型

```bash
# 列出本地模型
python scripts/sync_models.py --list

# 列出 Railway 上的模型
python scripts/sync_models.py --list-remote
```

### 上傳模型

```bash
# 上傳最新模型
python scripts/sync_models.py --upload-latest

# 上傳指定類型的最新模型
python scripts/sync_models.py --upload-type ensemble

# 上傳指定檔案
python scripts/sync_models.py --upload models/lstm_BTCUSDT_20241212.pt

# 強制覆蓋
python scripts/sync_models.py --upload-latest --force

# 同步所有本地模型
python scripts/sync_models.py --sync-all
```

### 重載模型

```bash
# 通知 Railway 重新載入所有模型
python scripts/sync_models.py --reload
```

### 刪除遠端模型

```bash
python scripts/sync_models.py --delete ensemble_BTCUSDT_20241212.joblib
```

## 使用 Apple MPS GPU

在 Apple Silicon Mac（M1/M2/M3）上，可以使用 MPS 加速 PyTorch 訓練：

```python
import torch

# 檢查 MPS 是否可用
print(torch.backends.mps.is_available())  # True

# 腳本會自動檢測並使用 MPS
# 如果遇到問題，可以強制使用 CPU：
export PYTORCH_DEVICE=cpu
```

## 模型檔案格式

| 副檔名 | 框架 | 說明 |
|-------|------|------|
| .joblib | Scikit-learn | 隨機森林、XGBoost、Ensemble |
| .pt | PyTorch | LSTM、PPO |
| .pkl | Pickle | 通用 Python 物件 |
| .h5 | TensorFlow/Keras | 深度學習模型 |

每個模型檔案會附帶一個 `.json` 元數據檔案，包含：
- 模型類型
- 交易對
- 訓練時間
- 訓練參數
- 評估指標

## Railway 環境配置

Railway 環境需要設置以下環境變數：

```bash
# 禁用訓練功能
TRAINING_ENABLED=false

# 啟動時自動載入模型
AUTO_LOAD_MODELS=true

# 模型目錄
MODEL_DIR=models
```

## 常見問題

### Q: 為什麼要分離訓練和交易？

A: 幾個原因：
1. **資源節省**：Railway 的小型實例資源有限，不適合訓練
2. **成本控制**：訓練消耗大量 CPU/GPU，本地訓練免費
3. **穩定性**：訓練過程不會影響線上交易服務
4. **靈活性**：可以使用更強大的本地硬體（如 MPS GPU）

### Q: 模型多久需要重新訓練？

A: 建議：
- **每週一次**：定期更新以適應市場變化
- **市場劇變時**：大幅波動後立即重新訓練
- **績效下降時**：當策略表現明顯下滑時

### Q: 如何自動化訓練和同步？

A: 可以設置 cron job：

```bash
# 編輯 crontab
crontab -e

# 每週日凌晨 3 點自動訓練和同步
0 3 * * 0 cd ~/Documents/BinanceProject/binancetradebot && \
    source venv/bin/activate && \
    python scripts/local_train.py --symbol BTCUSDT --model-type ensemble && \
    python scripts/sync_models.py --upload-latest --reload
```

### Q: 同步失敗怎麼辦？

A: 檢查：
1. Railway API 是否正常運行：`curl https://your-app.up.railway.app/health`
2. API Key 是否正確
3. 網路連接是否正常
4. 模型檔案是否損壞

## 進階用法

### 自定義訓練數據源

修改 `scripts/local_train.py` 中的 `fetch_training_data` 函數，可以從不同來源獲取數據：
- Binance API（即時數據）
- Railway PostgreSQL（歷史數據）
- 本地 CSV 檔案
- 其他交易所 API

### 添加新的模型類型

在 `scripts/local_train.py` 中添加新的訓練函數，然後更新 `train_ml_model` 函數的條件分支。

### 自定義模型評估指標

修改訓練函數中的評估邏輯，添加更多指標如：
- Sharpe Ratio
- Maximum Drawdown
- Profit Factor
- Win Rate

## 相關文件

- [scripts/local_train.py](../scripts/local_train.py) - 本地訓練腳本
- [scripts/sync_models.py](../scripts/sync_models.py) - 模型同步腳本
- [src/api/model_routes.py](../src/api/model_routes.py) - 模型管理 API
- [.env.local.example](../.env.local.example) - 本地環境配置範例
