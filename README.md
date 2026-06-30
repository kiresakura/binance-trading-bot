# 🤖 Binance Automated Trading Bot

**Enterprise-grade hybrid-architecture crypto trading system**

![CI](https://github.com/kiresakura/binance-trading-bot/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)

An educational/research automated cryptocurrency trading system built on FastAPI,
featuring a five-layer risk-defense pipeline, smart order execution, automatic
fault-tolerant recovery, a tamper-evident hash-chained audit log, and full
observability.

---

## ⚠️ Disclaimer

> **This is open-source software for educational and research purposes. It is NOT financial advice.**

- **Use at your own risk:** Automated cryptocurrency trading is highly risky and may lead to **partial or total loss of funds**. You are solely responsible for every trade made with this software.
- **No warranty:** This software is provided "AS IS" under the [MIT License](LICENSE), without warranty of any kind. The authors are **not liable for any loss or damage**.
- **Not affiliated with Binance:** This is an independent, third-party open-source project. It is **not affiliated with, endorsed by, or authorized by Binance**. "Binance" is a trademark of its respective owner.
- **Regulatory compliance:** Automated trading is regulated differently across jurisdictions, and some regions restrict the use of certain exchanges. **Confirm your local laws and the exchange's Terms of Service before use.**
- **Security:** Read [SECURITY.md](SECURITY.md) before deploying. Never commit secrets; disable withdrawal permission on your Binance API key and set an IP allowlist; always configure a strong `SECRET_KEY`.

---

## ✨ Features

### 🛡️ Enterprise capabilities
- **Five-layer risk defense** — position sizing, drawdown protection, correlation analysis, black-swan protection, circuit breaker
- **Smart order execution** — order splitting, slippage prediction, market-impact modeling, adaptive strategy
- **Fault tolerance** — primary/backup failover, state snapshots, automatic emergency stop on repeated failure
- **Tamper-evident audit log** — SHA-256 hash chain with integrity verification
- **Full observability** — Prometheus metrics + Grafana dashboards

### 🚀 Trading
- **Real-time market data** via Binance WebSocket
- **Multi-exchange ready** through CCXT
- **Telegram notifications** for trade alerts
- **Reinforcement learning (optional)** — PPO/SAC agents (requires the optional ML extras)

## 🔧 Tech stack

- **Backend:** Python 3.10+, FastAPI, asyncio
- **Database:** PostgreSQL (with SQLite fallback), SQLAlchemy, asyncpg
- **Cache:** Redis (with in-memory fallback)
- **Monitoring:** Prometheus + Grafana
- **Containerization:** Docker + Docker Compose
- **Exchange API:** Binance REST/WebSocket, CCXT
- **ML (optional):** scikit-learn, XGBoost; PyTorch + Stable-Baselines3 for RL
- **Testing:** pytest, pytest-asyncio (155 passing tests)

## 🚀 Quick start

### Option 1 — Local development (recommended for testing)

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate            # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your environment
cp .env.example .env
# Edit .env: set a strong SECRET_KEY and your Binance API key
# Generate a key: python -c "import secrets; print(secrets.token_urlsafe(48))"
# For local dev, set ENVIRONMENT=development

# 4. Start the API server
uvicorn src.api.app:app --reload

# 5. Open the API docs (non-production only)
open http://localhost:8000/docs
```

### Option 2 — Docker (production)

```bash
cp .env.example .env       # then edit .env
docker-compose up -d
docker-compose logs -f trading-bot
```

### Endpoints

- **Health:** http://localhost:8000/health
- **Prometheus metrics:** http://localhost:8000/metrics
- **API docs:** http://localhost:8000/docs *(disabled when `ENVIRONMENT=production`)*
- **Grafana:** http://localhost:3000
- **Prometheus:** http://localhost:9090

## 🔐 Safe-by-default configuration

The defaults are intentionally conservative — you must explicitly opt in to real trading:

| Setting | Default | Meaning |
|---|---|---|
| `ENVIRONMENT` | `production` | Disables `/docs` and debug endpoints; enforces auth |
| `BINANCE_TESTNET` | `true` | Uses the Binance testnet (virtual funds) |
| `DRY_RUN` | `true` | Logs intended trades without placing real orders |
| `SECRET_KEY` | *(required)* | JWT signing key; the app refuses to start in production without it |

Authentication is required on all money/account/admin endpoints via JWT
(`Authorization: Bearer <token>`) or API key (`X-API-Key`). API keys are
injected through `API_KEY_*` environment variables — there are **no built-in
credentials** in the source.

### Example `.env`

```bash
# Security (required)
SECRET_KEY=                       # python -c "import secrets; print(secrets.token_urlsafe(48))"
ENVIRONMENT=production            # use 'development' for local work
DRY_RUN=true

# Binance API
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=true              # true=testnet, false=real trading

# Risk management
MAX_POSITION_SIZE=0.1             # max 10% per position
MAX_DAILY_LOSS=0.02               # 2% daily loss limit
MAX_DRAWDOWN=0.10                 # 10% max drawdown
```

See [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md) for full configuration details.

## 📊 Selected API endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/orders` | trading | Create an order (five-layer risk validation) |
| `POST` | `/api/v1/orders/validate` | public | Validate risk without executing |
| `GET`  | `/api/v1/orders` | auth | Order history |
| `GET`  | `/api/v1/portfolio/summary` | auth | Portfolio summary (balances) |
| `GET`  | `/api/v1/audit/trail` | auth | Audit trail |
| `GET`  | `/api/v1/audit/verify` | auth | Verify audit-chain integrity |
| `POST` | `/api/v1/system/emergency-stop` | trading | Trigger emergency stop |
| `GET`  | `/api/v1/trading/status` | public | Trading-engine status |

## 🧪 Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

The suite (155 tests) runs fully offline — all exchange/network calls are mocked.
A per-test timeout is enforced via `pytest-timeout` to prevent hangs.

## 📁 Project structure

```
.
├── src/
│   ├── api/            # FastAPI app, routes, auth, rate limiting
│   ├── services/       # risk_manager, order_executor, fault_tolerance,
│   │                   #   audit_logger, monitoring, binance_client, ...
│   ├── engines/        # trading / data / learning / backtest engines
│   ├── strategies/     # rule-based, RL-based, community strategies
│   ├── config.py       # configuration (pydantic-settings)
│   └── logger.py
├── tests/              # unit + integration tests
├── config/             # prometheus, grafana, alert rules, migrations
├── docker-compose.yml
├── requirements.txt        # runtime dependencies
├── requirements-dev.txt    # dev/test/tooling
└── COMPLETE_GUIDE.md       # detailed guide
```

## 🤝 Contributing

Issues and pull requests are welcome — please read [CONTRIBUTING.md](CONTRIBUTING.md)
first. Report security issues privately per [SECURITY.md](SECURITY.md) (do **not**
open a public issue for vulnerabilities).

## 📄 License

[MIT](LICENSE) © kiresakura
