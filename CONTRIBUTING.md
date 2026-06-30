# Contributing Guide

感謝你願意貢獻！請先閱讀以下規範。

## 開發環境

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env        # 填入你自己的設定（勿提交）
```

本地開發請於 `.env` 設定 `ENVIRONMENT=development` 並提供 `SECRET_KEY`。

## 提交前檢查

```bash
# 格式與靜態檢查
black src tests
isort src tests
flake8 src
mypy src

# 測試
pytest

# 安全：確認沒有把密鑰寫進變更
pip install pip-audit && pip-audit -r requirements.txt
git diff --staged   # 自我檢查不含任何金鑰
```

## 規範

- **絕不**提交任何真實憑證（API 金鑰、密碼、token、連線字串）。所有機密一律走環境變數。
- 新增行為請附上測試；修 bug 請先寫可重現的測試再修。
- 不得為了通過 CI 而刪除、跳過或弱化既有測試。
- Commit message 建議採用 Conventional Commits（`feat:`、`fix:`、`docs:`、`refactor:` 等）。
- 涉及交易執行、風控、認證的變更請在 PR 描述中特別說明風險。

## 回報安全問題

安全漏洞請依 [SECURITY.md](SECURITY.md) 私下回報，**勿**開公開 Issue。
