# Security Policy

## 回報安全漏洞 (Reporting a Vulnerability)

**請勿**透過公開的 GitHub Issue 回報安全漏洞。

若你發現安全問題，請以私下方式回報：

- 使用 GitHub 的 **Security Advisories**（本倉庫 → Security → Report a vulnerability），或
- 透過倉庫所有者於個人檔案公開的聯絡方式私訊。

回報時請盡量包含：
- 受影響的檔案／端點／版本
- 重現步驟或 PoC
- 影響評估（可能造成的後果）

我們會盡快回覆並協調修補與揭露時程。

## 支援的版本 (Supported Versions)

本專案目前處於活躍開發階段，僅維護 `main` 分支的最新版本。

## 使用者安全須知 (Operational Security)

此為**自動化交易**軟體，部署不當可能導致資金損失或帳戶被接管。請務必：

1. **絕不**將任何金鑰（Binance API Key/Secret、`SECRET_KEY`、資料庫密碼、Telegram Token）寫入原始碼或提交到版本控制；一律使用環境變數 / `.env`（已被 `.gitignore` 排除）。
2. **務必設定 `SECRET_KEY`** 為高強度隨機字串；正式環境未設定時程式會拒絕啟動。
3. Binance API Key 權限**僅開啟現貨交易與讀取**，**切勿開啟提幣權限**，並設定 **IP 白名單**。
4. 先以 `BINANCE_TESTNET=true` 與 `DRY_RUN=true`（皆為預設）充分測試，再考慮真實交易。
5. 對外服務務必置於反向代理／防火牆之後，並為所有寫入端點提供有效認證憑證（JWT 或 `X-API-Key`）。
6. 正式環境請設定 `ENVIRONMENT=production`（預設值）以關閉互動式 API 文件與除錯端點。

## 已知設計界線

- API 認證採 JWT（HS256）與 `X-API-Key`；金鑰經由 `API_KEY_*` 環境變數注入，原始碼不含任何內建金鑰。
- 請求日誌會遮蔽 `password`、`token`、`api_key`、`api_secret` 等敏感欄位。
