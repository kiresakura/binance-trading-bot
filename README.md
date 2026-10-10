# 🤖 Binance Automated Trading Bot

[![CI](https://github.com/kiresakura/binance-trading-bot/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kiresakura/binance-trading-bot/actions/workflows/ci.yml)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)

English | [繁體中文](README.zh-TW.md)

An educational/research FastAPI backend for experimenting with automated crypto
trading against the **Binance spot testnet**. It runs a paper-trading engine on
live testnet market data, passes orders through a chain of pre-trade risk
checks, and records actions in a SHA-256 hash-chained audit log.

- **No account needed to try it:** the demo below runs without any API key, on
  SQLite, in dry-run mode.
- **160 offline tests** (pytest; Binance HTTP calls are mocked), run by CI on
  Python 3.11 and 3.12 together with a flake8 check for syntax errors and
  undefined names.
- It is a learning codebase, not a production trading system — read
  [Project status](#-project-status) before relying on any feature.

---

## ⚠️ Disclaimer

> **This is open-source software for educational and research purposes. It is NOT financial advice.**

- **Use at your own risk:** Automated cryptocurrency trading is highly risky and may lead to **partial or total loss of funds**. You are solely responsible for every trade made with this software.
- **No warranty:** This software is provided "AS IS" under the [MIT License](LICENSE), without warranty of any kind. The authors are **not liable for any loss or damage**.
- **Not affiliated with Binance:** This is an independent, third-party open-source project. It is **not affiliated with, endorsed by, or authorized by Binance**. "Binance" is a trademark of its respective owner.
- **Regulatory compliance:** Automated trading is regulated differently across jurisdictions, and some regions restrict the use of certain exchanges. **Confirm your local laws and the exchange's Terms of Service before use.**
- **Security:** Read [SECURITY.md](SECURITY.md) before deploying. Never commit secrets; disable withdrawal permission on your Binance API key and set an IP allowlist; always configure a strong `SECRET_KEY`.

---

## 🚀 Quick start: dry-run demo on the testnet (no API key)

Requires Python 3.11 or 3.12 and an internet connection (for public testnet
market data). No Binance account, database or Redis is needed.

```bash
git clone https://github.com/kiresakura/binance-trading-bot.git
cd binance-trading-bot
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# development mode: /docs enabled, SQLite, one-off random SECRET_KEY
ENVIRONMENT=development uvicorn src.api.app:app    # add --port 8080 if 8000 is taken
```

In a second terminal:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/v1/trading/status       # engine running in "paper" mode
curl -X POST http://127.0.0.1:8000/api/v1/orders/validate \
     -H 'Content-Type: application/json' \
     -d '{"symbol":"BTCUSDT","side":"BUY","order_type":"LIMIT","quantity":0.001,"price":60000}'
open http://127.0.0.1:8000/docs                         # interactive API docs
```

What happens: the app creates `data/trading_bot.db` (SQLite) and `logs/`,
subscribes to testnet kline WebSocket streams, and starts the automated engine
in paper mode (`DRY_RUN=true`, simulated fills). Without API keys, account and
balance lookups log errors — that is expected. Keep a development instance on
`127.0.0.1`: development mode relaxes authentication (see
[Known limitations](#known-limitations)).

To use your own testnet account, copy `.env.example` to `.env` and fill in a key
from <https://testnet.binance.vision/>.

## 🧪 Testing

```bash
pip install -r requirements.txt -r requirements-test.txt
flake8 --count --select=E9,F63,F7,F82 --show-source --statistics src tests scripts
pytest                       # 160 tests, offline, per-test timeout of 60 s
```

`requirements-dev.txt` adds formatters, linters and notebook tooling on top of
`requirements-test.txt`. The heavy optional ML packages (torch,
stable-baselines3, gymnasium) are not needed for the tests.

## 📋 Project status

| Area | Status | What is actually there |
|---|---|---|
| Pre-trade risk checks | Implemented, unit-tested | Five checks run in sequence: position size, drawdown since process start, correlation, black-swan (volatility/volume spike), trade-rate circuit breaker. |
| Order execution | Partial, unit-tested with mocks | Liquidity-based splitting into 3/5/10 chunks; slippage and market-impact figures are simple heuristics; TWAP/VWAP are not implemented. |
| Fault tolerance | Partial, unit-tested with mocks | Retries, state snapshots, and an automatic emergency stop after 3 consecutive failed executions. The "backup" connection uses the same Binance endpoint. |
| Audit log | Partial, unit-tested with a mocked DB | SHA-256 hash chain stored in the database (JSONL file fallback). Chain verification is not reliable yet (see below). |
| Paper-trading engine | Implemented, no tests | Starts automatically; with `DRY_RUN=true` it simulates fills instead of placing orders. |
| Market data | Implemented, no tests | Binance kline WebSocket streams and REST klines (testnet by default). |
| Metrics | Partial | Prometheus `/metrics` with system, trade and risk-rejection metrics; a sample Grafana dashboard (`config/grafana_dashboard.json`) that must be imported by hand — most of its panels have no data yet. |
| Storage | Implemented | SQLite by default; PostgreSQL when `DATABASE_URL` is set; Redis caching only when `REDIS_URL` is set. |
| Telegram | Partial, off by default | Startup and signal notifications; order/fill alerts are not wired up correctly yet. |
| Reinforcement learning | Experimental | A PPO agent (Stable-Baselines3) when torch, stable-baselines3 and gymnasium are installed (see `requirements-railway-full.txt`); otherwise training logs an error and is skipped. There is no SAC agent. |
| Multi-exchange | Not implemented | `ccxt` is in `requirements.txt` but the code only talks to Binance. |

### Known limitations

- **`DRY_RUN` covers the automated engine only.** `POST /api/v1/orders` (which
  needs a trading-scope credential) always sends the order to Binance — the
  testnet unless you set `BINANCE_TESTNET=false`.
- Risk thresholds are hard-coded defaults; the `MAX_*` variables in `.env` are
  not applied yet. `SELL` orders skip the risk checks, and the circuit breaker's
  trade counter is only fed by tests.
- `GET /api/v1/audit/verify` reports a violation on SQLite (the default
  database), and the chain is not re-linked after a process restart.
- In `ENVIRONMENT=development` the API-key check on `/api/v1/models/*` is
  bypassed (including model upload, which loads pickled files), and several
  management endpoints have no authentication in any environment. Never expose
  an instance to the internet.
- `API_KEY_*` credentials are read from real environment variables, not from
  `.env`. `AUTO_START_TRADING` and `TRAINING_ENABLED` are not read: the engine
  and model training start on every boot.
- Docker images and `docker-compose.yml` are not exercised by CI. In compose the
  app runs on SQLite (no `DATABASE_URL` is passed) and does not use the Redis
  container; Grafana is not provisioned.

## 🔐 Safe-by-default configuration

| Setting | Default | Meaning |
|---|---|---|
| `ENVIRONMENT` | `production` | Disables `/docs`; startup requires `SECRET_KEY`, `BINANCE_API_KEY`/`BINANCE_API_SECRET`, and `DB_USER`/`DB_PASSWORD` unless `USE_SQLITE=true` |
| `BINANCE_TESTNET` | `true` | Binance testnet (virtual funds) — the default in code, `docker-compose.yml` and both env examples |
| `DRY_RUN` | `true` | The automated engine simulates fills; see the `POST /api/v1/orders` caveat above |
| `SECRET_KEY` | *(required in production)* | JWT signing key; outside production a random one-off key is generated |

Authentication on money/account/admin endpoints uses JWT
(`Authorization: Bearer <token>`) or an API key (`X-API-Key`). API keys come from
`API_KEY_<NAME>=<key>:<scope>` environment variables — there are **no built-in
credentials** in the source.

### Example `.env`

```bash
# Security (required in production)
SECRET_KEY=                       # python -c "import secrets; print(secrets.token_urlsafe(48))"
ENVIRONMENT=production            # use 'development' for local work
DRY_RUN=true

# Binance API
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=true              # true=testnet, false=real trading
```

See [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md) for the full configuration reference.

### Docker

```bash
cp .env.example .env       # then edit .env
docker-compose up -d
docker-compose logs -f trading-bot
```

Starts the API (port 8000) next to TimescaleDB, Redis, RabbitMQ, Prometheus
(9090) and Grafana (3000) containers. Not covered by CI.

## 📊 Selected API endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/orders` | trading | Create an order (risk checks on `BUY`; sent to Binance even when `DRY_RUN=true`) |
| `POST` | `/api/v1/orders/validate` | public | Run the risk checks without placing an order |
| `GET`  | `/api/v1/orders` | auth | Order history |
| `GET`  | `/api/v1/portfolio/summary` | auth | Portfolio summary (balances) |
| `GET`  | `/api/v1/audit/trail` | auth | Audit trail |
| `GET`  | `/api/v1/audit/verify` | auth | Verify audit-chain integrity (see limitations) |
| `POST` | `/api/v1/system/emergency-stop` | trading | Trigger emergency stop |
| `GET`  | `/api/v1/trading/status` | public | Trading-engine status |
| `GET`  | `/health`, `/metrics` | public | Health check, Prometheus metrics |

`/docs` is available only when `ENVIRONMENT` is not `production`.

## 🔧 Tech stack

- **Backend:** Python 3.11/3.12, FastAPI, asyncio
- **Storage:** SQLite (default) or PostgreSQL via SQLAlchemy/asyncpg; optional Redis
- **Exchange API:** Binance REST and WebSocket
- **Monitoring:** prometheus-client; sample Prometheus/Grafana config
- **ML (optional):** scikit-learn, XGBoost, Optuna; PyTorch + Stable-Baselines3 for RL
- **Testing:** pytest, pytest-asyncio, pytest-timeout; GitHub Actions

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
├── requirements-test.txt   # test + CI dependencies
├── requirements-dev.txt    # test deps plus dev tooling
└── COMPLETE_GUIDE.md       # detailed guide
```

## 🤝 Contributing

Issues and pull requests are welcome — please read [CONTRIBUTING.md](CONTRIBUTING.md)
first. Report security issues privately per [SECURITY.md](SECURITY.md) (do **not**
open a public issue for vulnerabilities).

## 📄 License

[MIT](LICENSE) © kiresakura
