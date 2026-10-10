# 🤖 Binance 自動交易機器人

[![CI](https://github.com/kiresakura/binance-trading-bot/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kiresakura/binance-trading-bot/actions/workflows/ci.yml)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)

[English](README.md) | 繁體中文

以教育與研究為目的的 FastAPI 後端，用來在 **Binance 現貨測試網** 上實驗自動化加密貨幣交易。
它以測試網的即時行情驅動紙上交易引擎，下單前經過一連串風控檢查，並把操作寫進以 SHA-256
串成雜湊鏈的稽核日誌。

- **不需要帳號就能試用：** 下方示範不需任何 API 金鑰，使用 SQLite、乾跑（dry-run）模式。
- **160 個離線測試**（pytest；Binance HTTP 呼叫皆以 mock 取代），CI 在 Python 3.11 與 3.12
  上執行，並附 flake8 語法錯誤／未定義名稱檢查。
- 這是學習用程式庫，不是可上線的交易系統——使用任何功能前請先看[專案現況](#-專案現況)。

---

## ⚠️ 免責聲明

> **本專案為教育與研究用途的開源軟體，不構成任何投資建議。**

- **風險自負：** 自動化加密貨幣交易風險極高，可能造成**部分或全部資金損失**。使用本軟體進行的每一筆交易都由你自行負責。
- **不提供任何擔保：** 本軟體依 [MIT 授權](LICENSE)以「現狀」提供，不附任何形式的擔保；作者**不對任何損失或損害負責**。
- **與 Binance 無隸屬關係：** 本專案為獨立的第三方開源專案，**與 Binance 沒有隸屬、背書或授權關係**。「Binance」為其所有者的商標。
- **法規遵循：** 各司法管轄區對自動化交易的規範不同，部分地區限制使用特定交易所。**使用前請確認當地法律及交易所的服務條款。**
- **安全性：** 部署前請閱讀 [SECURITY.md](SECURITY.md)。切勿提交任何密鑰；Binance API Key 請關閉提幣權限並設定 IP 白名單；務必設定高強度的 `SECRET_KEY`。

---

## 🚀 快速開始：測試網乾跑示範（不需 API 金鑰）

需要 Python 3.11 或 3.12，以及網路連線（讀取測試網公開行情）。不需要 Binance 帳號、資料庫或 Redis。

```bash
git clone https://github.com/kiresakura/binance-trading-bot.git
cd binance-trading-bot
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 開發模式：開啟 /docs、使用 SQLite、自動產生一次性的隨機 SECRET_KEY
ENVIRONMENT=development uvicorn src.api.app:app    # 8000 埠被占用時加上 --port 8080
```

另開一個終端機：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/v1/trading/status       # 引擎以 "paper"（紙上交易）模式運作
curl -X POST http://127.0.0.1:8000/api/v1/orders/validate \
     -H 'Content-Type: application/json' \
     -d '{"symbol":"BTCUSDT","side":"BUY","order_type":"LIMIT","quantity":0.001,"price":60000}'
open http://127.0.0.1:8000/docs                         # 互動式 API 文件
```

啟動後會發生的事：建立 `data/trading_bot.db`（SQLite）與 `logs/`，訂閱測試網 K 線
WebSocket 串流，並以紙上交易模式啟動自動交易引擎（`DRY_RUN=true`，成交為模擬）。
沒有 API 金鑰時，帳戶與餘額查詢會在日誌中報錯，屬正常現象。開發模式會放寬驗證
（見[已知限制](#已知限制)），請只在 `127.0.0.1` 上執行。

想用自己的測試網帳號：把 `.env.example` 複製成 `.env`，填入在
<https://testnet.binance.vision/> 申請的金鑰。

## 🧪 測試

```bash
pip install -r requirements.txt -r requirements-test.txt
flake8 --count --select=E9,F63,F7,F82 --show-source --statistics src tests scripts
pytest                       # 160 個測試，離線執行，每個測試逾時上限 60 秒
```

`requirements-dev.txt` 在 `requirements-test.txt` 之上再加格式化、lint 與 notebook 工具。
執行測試不需要重型的選用 ML 套件（torch、stable-baselines3、gymnasium）。

## 📋 專案現況

| 項目 | 狀態 | 實際內容 |
|---|---|---|
| 下單前風控檢查 | 已實作、有單元測試 | 依序執行五道檢查：部位大小、自程序啟動起算的回撤、相關性、黑天鵝（波動／成交量異常）、交易頻率熔斷。 |
| 下單執行 | 部分完成、以 mock 做單元測試 | 依流動性拆成 3／5／10 筆子單；滑價與市場衝擊估計為簡單的經驗公式；TWAP／VWAP 尚未實作。 |
| 容錯 | 部分完成、以 mock 做單元測試 | 重試、狀態快照，連續 3 次執行失敗自動緊急停止。「備援」連線使用同一個 Binance 端點。 |
| 稽核日誌 | 部分完成、以 mock 資料庫做單元測試 | SHA-256 雜湊鏈存於資料庫（備援寫入 JSONL 檔）。鏈完整性驗證目前不可靠（見下方）。 |
| 紙上交易引擎 | 已實作、無測試 | 啟動時自動執行；`DRY_RUN=true` 時以模擬成交取代實際下單。 |
| 行情資料 | 已實作、無測試 | Binance K 線 WebSocket 串流與 REST K 線（預設為測試網）。 |
| 監控指標 | 部分完成 | Prometheus `/metrics`（系統、交易、風控拒絕指標）；範例 Grafana 儀表板（`config/grafana_dashboard.json`）需手動匯入，多數面板目前沒有資料。 |
| 儲存 | 已實作 | 預設 SQLite；設定 `DATABASE_URL` 時用 PostgreSQL；設定 `REDIS_URL` 時才啟用 Redis 快取。 |
| Telegram | 部分完成、預設關閉 | 啟動與訊號通知；下單／成交通知尚未正確串接。 |
| 強化學習 | 實驗性 | 安裝 torch、stable-baselines3、gymnasium 後可用 PPO 代理（Stable-Baselines3，見 `requirements-railway-full.txt`）；未安裝時訓練會記錄錯誤並略過。沒有 SAC 代理。 |
| 多交易所 | 未實作 | `requirements.txt` 列有 `ccxt`，但程式只對接 Binance。 |

### 已知限制

- **`DRY_RUN` 只涵蓋自動交易引擎。** `POST /api/v1/orders`（需要 trading 權限的憑證）
  一律把訂單送到 Binance——除非設定 `BINANCE_TESTNET=false`，否則送往測試網。
- 風控門檻為寫死的預設值，`.env` 裡的 `MAX_*` 變數尚未套用。`SELL` 訂單不經風控檢查，
  熔斷器的交易計數目前只有測試會呼叫。
- `GET /api/v1/audit/verify` 在 SQLite（預設資料庫）上一律回報違規，且程序重啟後雜湊鏈不會接續。
- `ENVIRONMENT=development` 時 `/api/v1/models/*` 的 API Key 檢查會被略過（包含會載入 pickle
  檔的模型上傳），另有數個管理端點在任何環境都沒有驗證。切勿把執行個體暴露到網際網路。
- `API_KEY_*` 憑證從真正的環境變數讀取，不會從 `.env` 讀取。`AUTO_START_TRADING` 與
  `TRAINING_ENABLED` 沒有被讀取：每次啟動都會啟動交易引擎與模型訓練。
- Docker 映像與 `docker-compose.yml` 未經 CI 驗證。compose 中的應用程式實際使用 SQLite
  （沒有傳入 `DATABASE_URL`），也沒有使用 Redis 容器；Grafana 未自動佈建。

## 🔐 安全預設設定

| 設定 | 預設值 | 說明 |
|---|---|---|
| `ENVIRONMENT` | `production` | 關閉 `/docs`；啟動時要求 `SECRET_KEY`、`BINANCE_API_KEY`／`BINANCE_API_SECRET`，以及 `DB_USER`／`DB_PASSWORD`（設定 `USE_SQLITE=true` 時免） |
| `BINANCE_TESTNET` | `true` | 使用 Binance 測試網（虛擬資金）——程式、`docker-compose.yml` 與兩份 env 範例皆以此為預設 |
| `DRY_RUN` | `true` | 自動交易引擎以模擬成交運作；`POST /api/v1/orders` 的例外見上方 |
| `SECRET_KEY` | *（production 必填）* | JWT 簽章金鑰；非 production 環境未設定時自動產生一次性隨機金鑰 |

涉及資金、帳戶與管理的端點以 JWT（`Authorization: Bearer <token>`）或 API Key（`X-API-Key`）
驗證。API Key 透過 `API_KEY_<名稱>=<金鑰>:<scope>` 環境變數注入——原始碼中**沒有任何內建憑證**。

### `.env` 範例

```bash
# 安全（production 必填）
SECRET_KEY=                       # python -c "import secrets; print(secrets.token_urlsafe(48))"
ENVIRONMENT=production            # 本機開發請用 development
DRY_RUN=true

# Binance API
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=true              # true=測試網，false=真實交易
```

完整設定說明見 [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)。

### Docker

```bash
cp .env.example .env       # 接著編輯 .env
docker-compose up -d
docker-compose logs -f trading-bot
```

啟動 API（8000 埠），以及 TimescaleDB、Redis、RabbitMQ、Prometheus（9090）與 Grafana（3000）容器。
未經 CI 驗證。

## 📊 主要 API 端點

| 方法 | 路徑 | 驗證 | 說明 |
|---|---|---|---|
| `POST` | `/api/v1/orders` | trading | 建立訂單（`BUY` 經風控檢查；即使 `DRY_RUN=true` 仍會送到 Binance） |
| `POST` | `/api/v1/orders/validate` | 公開 | 只執行風控檢查、不下單 |
| `GET`  | `/api/v1/orders` | 需驗證 | 訂單歷史 |
| `GET`  | `/api/v1/portfolio/summary` | 需驗證 | 投資組合摘要（餘額） |
| `GET`  | `/api/v1/audit/trail` | 需驗證 | 稽核軌跡 |
| `GET`  | `/api/v1/audit/verify` | 需驗證 | 驗證稽核鏈完整性（見已知限制） |
| `POST` | `/api/v1/system/emergency-stop` | trading | 觸發緊急停止 |
| `GET`  | `/api/v1/trading/status` | 公開 | 交易引擎狀態 |
| `GET`  | `/health`、`/metrics` | 公開 | 健康檢查、Prometheus 指標 |

只有 `ENVIRONMENT` 不是 `production` 時才提供 `/docs`。

## 🔧 技術棧

- **後端：** Python 3.11／3.12、FastAPI、asyncio
- **儲存：** SQLite（預設）或透過 SQLAlchemy／asyncpg 使用 PostgreSQL；選用 Redis
- **交易所 API：** Binance REST 與 WebSocket
- **監控：** prometheus-client；範例 Prometheus／Grafana 設定
- **ML（選用）：** scikit-learn、XGBoost、Optuna；強化學習用 PyTorch + Stable-Baselines3
- **測試：** pytest、pytest-asyncio、pytest-timeout；GitHub Actions

## 📁 專案結構

```
.
├── src/
│   ├── api/            # FastAPI 應用、路由、驗證、限流
│   ├── services/       # risk_manager、order_executor、fault_tolerance、
│   │                   #   audit_logger、monitoring、binance_client …
│   ├── engines/        # 交易／資料／學習／回測引擎
│   ├── strategies/     # 規則型、強化學習、社群策略
│   ├── config.py       # 設定（pydantic-settings）
│   └── logger.py
├── tests/              # 單元與整合測試
├── config/             # prometheus、grafana、告警規則、資料庫遷移
├── docker-compose.yml
├── requirements.txt        # 執行期依賴
├── requirements-test.txt   # 測試與 CI 依賴
├── requirements-dev.txt    # 測試依賴加上開發工具
└── COMPLETE_GUIDE.md       # 詳細指南
```

## 🤝 參與貢獻

歡迎提出 Issue 與 Pull Request，請先閱讀 [CONTRIBUTING.md](CONTRIBUTING.md)。安全性問題請依
[SECURITY.md](SECURITY.md) 私下回報（**不要**為漏洞開公開 Issue）。

## 📄 授權

[MIT](LICENSE) © kiresakura
