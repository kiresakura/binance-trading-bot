"""
Backtest Service - 策略回測服務

提供：
- 歷史數據回測
- 多策略對比
- 回測報告生成
- 性能指標計算
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import json
import numpy as np

from src.logger import logger
from src.utils.database import async_db_manager


class BacktestStatus(str, Enum):
    """回測狀態"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BacktestConfig:
    """回測配置"""
    symbol: str
    start_date: str  # ISO format
    end_date: str
    initial_capital: float = 10000.0
    commission_rate: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    position_size: float = 0.3  # 30% per trade
    # 策略參數
    confidence_threshold: float = 0.6
    strength_threshold: float = 0.5
    # 風險管理
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 10.0
    max_hold_hours: int = 168

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestTrade:
    """回測交易記錄"""
    timestamp: str
    symbol: str
    side: str  # BUY or SELL
    price: float
    quantity: float
    value: float
    commission: float
    signal_confidence: float
    signal_strength: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    hold_time_hours: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    """回測結果"""
    backtest_id: str
    model_id: str
    config: Dict[str, Any]
    status: str
    # 績效指標
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    # 交易統計
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_hold_time_hours: float = 0.0
    # 詳細數據
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    # 元數據
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BacktestService:
    """回測服務"""

    def __init__(self):
        self._running_backtests: Dict[str, BacktestResult] = {}
        logger.info("BacktestService initialized")

    async def run_backtest(
        self,
        model_id: str,
        config: BacktestConfig,
        model_data: Optional[Dict[str, Any]] = None
    ) -> BacktestResult:
        """
        執行回測

        Args:
            model_id: 模型 ID
            config: 回測配置
            model_data: 模型數據（用於預測）

        Returns:
            BacktestResult: 回測結果
        """
        backtest_id = f"bt-{model_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        started_at = datetime.utcnow()

        result = BacktestResult(
            backtest_id=backtest_id,
            model_id=model_id,
            config=config.to_dict(),
            status=BacktestStatus.RUNNING,
            started_at=started_at.isoformat()
        )

        self._running_backtests[backtest_id] = result

        try:
            # 獲取歷史數據
            klines = await self._get_historical_data(
                config.symbol,
                config.start_date,
                config.end_date
            )

            if not klines or len(klines) < 50:
                result.status = BacktestStatus.FAILED
                logger.error(f"Insufficient historical data for backtest: {len(klines)} candles")
                return result

            # 執行模擬交易
            trades, equity_curve = await self._simulate_trading(
                klines=klines,
                config=config,
                model_data=model_data
            )

            # 計算績效指標
            result = self._calculate_metrics(
                result=result,
                trades=trades,
                equity_curve=equity_curve,
                initial_capital=config.initial_capital
            )

            result.status = BacktestStatus.COMPLETED
            result.completed_at = datetime.utcnow().isoformat()
            result.duration_seconds = (datetime.utcnow() - started_at).total_seconds()

            # 保存到數據庫
            await self._save_backtest_result(result)

            logger.info(f"Backtest completed: {backtest_id}, "
                       f"Return: {result.total_return_pct:.2f}%, "
                       f"Win Rate: {result.win_rate:.2f}%, "
                       f"Sharpe: {result.sharpe_ratio:.2f}")

        except Exception as e:
            result.status = BacktestStatus.FAILED
            result.completed_at = datetime.utcnow().isoformat()
            logger.error(f"Backtest failed: {e}")

        finally:
            self._running_backtests.pop(backtest_id, None)

        return result

    async def _get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """獲取歷史 K 線數據"""
        try:
            from src.services.binance_client import BinanceClient

            client = BinanceClient()

            # 計算需要的 K 線數量
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            hours = int((end - start).total_seconds() / 3600)

            # 分批獲取數據（每次最多 1000 根）
            all_klines = []
            current_start = start

            while current_start < end:
                try:
                    klines = await client.get_klines(
                        symbol=symbol,
                        interval="1h",
                        limit=min(1000, hours),
                        start_time=int(current_start.timestamp() * 1000)
                    )

                    if not klines:
                        break

                    for k in klines:
                        all_klines.append({
                            "timestamp": datetime.fromtimestamp(k[0] / 1000).isoformat(),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5])
                        })

                    # 移動到下一批
                    last_ts = klines[-1][0]
                    current_start = datetime.fromtimestamp(last_ts / 1000) + timedelta(hours=1)

                except Exception as e:
                    logger.warning(f"Failed to get klines batch: {e}")
                    break

            return all_klines

        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            # 嘗試從數據庫獲取
            try:
                rows = await async_db_manager.execute(
                    """
                    SELECT timestamp, open, high, low, close, volume
                    FROM kline_cache
                    WHERE symbol = $1 AND interval = '1h'
                    AND timestamp >= $2 AND timestamp <= $3
                    ORDER BY timestamp
                    """,
                    symbol, start_date, end_date
                )
                return [dict(row) for row in rows]
            except Exception:
                return []

    async def _simulate_trading(
        self,
        klines: List[Dict[str, Any]],
        config: BacktestConfig,
        model_data: Optional[Dict[str, Any]]
    ) -> Tuple[List[BacktestTrade], List[Dict[str, Any]]]:
        """模擬交易"""
        trades = []
        equity_curve = []

        capital = config.initial_capital
        position = None  # {"quantity": 0, "entry_price": 0, "entry_time": "", "peak_price": 0}

        for i in range(50, len(klines)):  # 需要至少 50 根 K 線計算指標
            current = klines[i]
            timestamp = current["timestamp"]
            price = current["close"]

            # 計算技術指標
            features = self._calculate_features(klines[i-50:i+1])

            # 生成信號
            if model_data:
                signal = self._generate_signal_from_model(model_data, features)
            else:
                signal = self._generate_signal_from_features(features)

            confidence = signal.get("confidence", 0)
            strength = signal.get("strength", 0)
            prediction = signal.get("prediction", "HOLD")

            # 檢查現有持倉的風險
            if position:
                entry_price = position["entry_price"]
                peak_price = max(position.get("peak_price", price), price)
                position["peak_price"] = peak_price

                pnl_pct = (price - entry_price) / entry_price * 100
                drawdown_from_peak = (peak_price - price) / peak_price * 100

                # 計算持倉時間
                entry_time = datetime.fromisoformat(position["entry_time"].replace('Z', '+00:00'))
                current_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hold_hours = (current_time - entry_time).total_seconds() / 3600

                # 風險管理觸發平倉
                should_close = False
                close_reason = ""

                if pnl_pct <= -config.stop_loss_pct:
                    should_close = True
                    close_reason = "stop_loss"
                elif pnl_pct >= config.take_profit_pct:
                    should_close = True
                    close_reason = "take_profit"
                elif drawdown_from_peak >= 3.0 and pnl_pct > 0:  # 追蹤止損
                    should_close = True
                    close_reason = "trailing_stop"
                elif hold_hours >= config.max_hold_hours:
                    should_close = True
                    close_reason = "time_stop"
                elif prediction == "SELL" and confidence >= config.confidence_threshold:
                    should_close = True
                    close_reason = "signal"

                if should_close:
                    # 執行賣出
                    sell_value = position["quantity"] * price * (1 - config.commission_rate - config.slippage)
                    pnl = sell_value - position["cost"]

                    trade = BacktestTrade(
                        timestamp=timestamp,
                        symbol=config.symbol,
                        side="SELL",
                        price=price,
                        quantity=position["quantity"],
                        value=sell_value,
                        commission=position["quantity"] * price * config.commission_rate,
                        signal_confidence=confidence,
                        signal_strength=strength,
                        pnl=pnl,
                        pnl_pct=pnl / position["cost"] * 100,
                        hold_time_hours=hold_hours
                    )
                    trades.append(trade)

                    capital += sell_value
                    position = None

            # 開新倉
            if not position and prediction == "BUY":
                if confidence >= config.confidence_threshold and strength >= config.strength_threshold:
                    # 執行買入
                    buy_value = capital * config.position_size
                    commission = buy_value * config.commission_rate
                    slippage_cost = buy_value * config.slippage
                    actual_value = buy_value - commission - slippage_cost
                    quantity = actual_value / price

                    trade = BacktestTrade(
                        timestamp=timestamp,
                        symbol=config.symbol,
                        side="BUY",
                        price=price,
                        quantity=quantity,
                        value=buy_value,
                        commission=commission,
                        signal_confidence=confidence,
                        signal_strength=strength
                    )
                    trades.append(trade)

                    capital -= buy_value
                    position = {
                        "quantity": quantity,
                        "entry_price": price,
                        "entry_time": timestamp,
                        "peak_price": price,
                        "cost": buy_value
                    }

            # 記錄權益曲線
            total_value = capital
            if position:
                total_value += position["quantity"] * price

            equity_curve.append({
                "timestamp": timestamp,
                "capital": capital,
                "position_value": position["quantity"] * price if position else 0,
                "total_value": total_value
            })

        return trades, equity_curve

    def _calculate_features(self, klines: List[Dict[str, Any]]) -> Dict[str, float]:
        """計算技術指標特徵"""
        closes = [k["close"] for k in klines]
        highs = [k["high"] for k in klines]
        lows = [k["low"] for k in klines]
        volumes = [k["volume"] for k in klines]

        # RSI
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema12 = np.mean(closes[-12:])
        ema26 = np.mean(closes[-26:])
        macd = ema12 - ema26

        # Bollinger Bands Position
        sma20 = np.mean(closes[-20:])
        std20 = np.std(closes[-20:])
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        bb_position = (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5

        # Volume Ratio
        avg_volume = np.mean(volumes[-20:])
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0

        # Price Change
        price_change = (closes[-1] - closes[-24]) / closes[-24] * 100 if len(closes) >= 24 else 0

        # EMA Cross
        ema9 = np.mean(closes[-9:])
        ema21 = np.mean(closes[-21:])
        ema_cross = (ema9 - ema21) / ema21 * 100 if ema21 > 0 else 0

        # Volatility (ATR)
        tr_list = []
        for j in range(1, min(14, len(closes))):
            tr = max(
                highs[j] - lows[j],
                abs(highs[j] - closes[j-1]),
                abs(lows[j] - closes[j-1])
            )
            tr_list.append(tr)
        atr = np.mean(tr_list) if tr_list else 0
        volatility = (atr / closes[-1] * 100) if closes[-1] > 0 else 0

        # Momentum
        momentum = (closes[-1] - closes[-10]) / closes[-10] * 100 if len(closes) >= 10 else 0

        return {
            "rsi": rsi,
            "macd": macd,
            "bbPosition": bb_position,
            "volumeRatio": volume_ratio,
            "priceChange": price_change,
            "emaCross": ema_cross,
            "volatility": volatility,
            "momentum": momentum
        }

    def _generate_signal_from_features(
        self,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """從特徵生成簡單信號（無模型時使用）"""
        score = 0

        # RSI
        if features["rsi"] < 30:
            score += 2  # 超賣，買入信號
        elif features["rsi"] > 70:
            score -= 2  # 超買，賣出信號

        # MACD
        if features["macd"] > 0:
            score += 1
        else:
            score -= 1

        # BB Position
        if features["bbPosition"] < 0.2:
            score += 1
        elif features["bbPosition"] > 0.8:
            score -= 1

        # Volume
        if features["volumeRatio"] > 1.5:
            score += 0.5

        # EMA Cross
        if features["emaCross"] > 0:
            score += 1
        else:
            score -= 1

        # 決定信號
        if score >= 2:
            prediction = "BUY"
        elif score <= -2:
            prediction = "SELL"
        else:
            prediction = "HOLD"

        confidence = min(1.0, abs(score) / 5)
        strength = min(1.0, abs(score) / 4)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "strength": strength
        }

    def _generate_signal_from_model(
        self,
        model_data: Dict[str, Any],
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """使用模型生成信號"""
        # 這裡可以整合實際的模型預測邏輯
        # 簡化版本：使用特徵重要性加權
        weights = model_data.get("feature_importance", {})

        score = 0
        total_weight = 0

        for feature, importance in weights.items():
            if feature.lower() in features:
                value = features[feature.lower()]
                # 正規化特徵值
                normalized = self._normalize_feature(feature.lower(), value)
                score += normalized * importance
                total_weight += abs(importance)

        if total_weight > 0:
            score = score / total_weight

        if score > 0.15:
            prediction = "BUY"
        elif score < -0.15:
            prediction = "SELL"
        else:
            prediction = "HOLD"

        confidence = min(0.95, 0.5 + abs(score) * 0.5)
        strength = min(1.0, abs(score))

        return {
            "prediction": prediction,
            "confidence": confidence,
            "strength": strength
        }

    def _normalize_feature(self, feature: str, value: float) -> float:
        """正規化特徵值到 [-1, 1]"""
        if feature == "rsi":
            return (value - 50) / 50
        elif feature == "macd":
            return max(-1, min(1, value / 50))
        elif feature == "bbposition":
            return (value - 0.5) * 2
        elif feature == "volumeratio":
            return max(-1, min(1, (value - 1) * 2))
        elif feature == "pricechange":
            return max(-1, min(1, value / 5))
        elif feature == "emacross":
            return max(-1, min(1, value))
        elif feature == "volatility":
            return -max(-1, min(1, (value - 20) / 30))
        elif feature == "momentum":
            return max(-1, min(1, value / 10))
        return max(-1, min(1, value))

    def _calculate_metrics(
        self,
        result: BacktestResult,
        trades: List[BacktestTrade],
        equity_curve: List[Dict[str, Any]],
        initial_capital: float
    ) -> BacktestResult:
        """計算績效指標"""
        result.trades = [t.to_dict() for t in trades]
        result.equity_curve = equity_curve

        if not trades:
            return result

        # 基本統計
        sell_trades = [t for t in trades if t.side == "SELL" and t.pnl is not None]
        wins = [t for t in sell_trades if t.pnl > 0]
        losses = [t for t in sell_trades if t.pnl <= 0]

        result.total_trades = len(sell_trades)
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)

        if result.total_trades > 0:
            result.win_rate = len(wins) / result.total_trades * 100

        # PnL 統計
        if wins:
            result.avg_win = np.mean([t.pnl for t in wins])
            result.largest_win = max([t.pnl for t in wins])
        if losses:
            result.avg_loss = np.mean([t.pnl for t in losses])
            result.largest_loss = min([t.pnl for t in losses])

        # Profit Factor
        total_wins = sum([t.pnl for t in wins]) if wins else 0
        total_losses = abs(sum([t.pnl for t in losses])) if losses else 0
        if total_losses > 0:
            result.profit_factor = total_wins / total_losses

        # 平均持倉時間
        hold_times = [t.hold_time_hours for t in sell_trades if t.hold_time_hours]
        if hold_times:
            result.avg_hold_time_hours = np.mean(hold_times)

        # 總回報
        if equity_curve:
            final_value = equity_curve[-1]["total_value"]
            result.total_return = final_value - initial_capital
            result.total_return_pct = result.total_return / initial_capital * 100

            # 最大回撤
            peak = initial_capital
            max_dd = 0
            for point in equity_curve:
                value = point["total_value"]
                if value > peak:
                    peak = value
                dd = (peak - value) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            result.max_drawdown_pct = max_dd
            result.max_drawdown = peak * max_dd / 100

        # 計算年化回報和夏普比率
        if len(equity_curve) > 1:
            # 計算日回報率
            returns = []
            for i in range(1, len(equity_curve)):
                prev_value = equity_curve[i-1]["total_value"]
                curr_value = equity_curve[i]["total_value"]
                if prev_value > 0:
                    returns.append((curr_value - prev_value) / prev_value)

            if returns:
                # 假設每根 K 線是 1 小時
                hours_per_year = 8760
                periods = len(returns)

                # 年化回報
                total_return = (equity_curve[-1]["total_value"] / initial_capital) - 1
                result.annualized_return = ((1 + total_return) ** (hours_per_year / periods) - 1) * 100

                # 夏普比率 (假設無風險利率為 0)
                returns_std = np.std(returns)
                if returns_std > 0:
                    result.sharpe_ratio = np.mean(returns) / returns_std * np.sqrt(hours_per_year)

                # Sortino 比率（只考慮下行風險）
                downside_returns = [r for r in returns if r < 0]
                if downside_returns:
                    downside_std = np.std(downside_returns)
                    if downside_std > 0:
                        result.sortino_ratio = np.mean(returns) / downside_std * np.sqrt(hours_per_year)

        # 月度回報
        monthly = {}
        for point in equity_curve:
            month = point["timestamp"][:7]  # YYYY-MM
            monthly[month] = point["total_value"]

        prev_value = initial_capital
        for month, value in sorted(monthly.items()):
            if month not in result.monthly_returns:
                result.monthly_returns[month] = (value - prev_value) / prev_value * 100
                prev_value = value

        return result

    async def _save_backtest_result(self, result: BacktestResult):
        """保存回測結果到數據庫"""
        try:
            await async_db_manager.execute(
                """
                INSERT INTO backtest_results
                (backtest_id, model_id, config, status, total_return, total_return_pct,
                 sharpe_ratio, max_drawdown_pct, win_rate, total_trades,
                 started_at, completed_at, duration_seconds, result_data)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (backtest_id) DO UPDATE SET
                    status = excluded.status,
                    total_return = excluded.total_return,
                    total_return_pct = excluded.total_return_pct,
                    sharpe_ratio = excluded.sharpe_ratio,
                    max_drawdown_pct = excluded.max_drawdown_pct,
                    win_rate = excluded.win_rate,
                    total_trades = excluded.total_trades,
                    completed_at = excluded.completed_at,
                    duration_seconds = excluded.duration_seconds,
                    result_data = excluded.result_data
                """,
                result.backtest_id, result.model_id,
                json.dumps(result.config), result.status,
                result.total_return, result.total_return_pct,
                result.sharpe_ratio, result.max_drawdown_pct,
                result.win_rate, result.total_trades,
                result.started_at, result.completed_at,
                result.duration_seconds, json.dumps(result.to_dict())
            )
        except Exception as e:
            logger.warning(f"Failed to save backtest result: {e}")

    async def get_backtest_result(self, backtest_id: str) -> Optional[BacktestResult]:
        """獲取回測結果"""
        # 先檢查運行中的
        if backtest_id in self._running_backtests:
            return self._running_backtests[backtest_id]

        # 從數據庫獲取
        try:
            row = await async_db_manager.execute_one(
                "SELECT result_data FROM backtest_results WHERE backtest_id = $1",
                backtest_id
            )
            if row and row["result_data"]:
                data = json.loads(row["result_data"]) if isinstance(row["result_data"], str) else row["result_data"]
                return BacktestResult(**data)
        except Exception as e:
            logger.error(f"Failed to get backtest result: {e}")

        return None

    async def list_backtests(
        self,
        model_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """列出回測記錄"""
        try:
            if model_id:
                rows = await async_db_manager.execute(
                    """
                    SELECT backtest_id, model_id, status, total_return_pct,
                           sharpe_ratio, win_rate, total_trades, completed_at
                    FROM backtest_results
                    WHERE model_id = $1
                    ORDER BY completed_at DESC
                    LIMIT $2
                    """,
                    model_id, limit
                )
            else:
                rows = await async_db_manager.execute(
                    """
                    SELECT backtest_id, model_id, status, total_return_pct,
                           sharpe_ratio, win_rate, total_trades, completed_at
                    FROM backtest_results
                    ORDER BY completed_at DESC
                    LIMIT $1
                    """,
                    limit
                )

            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to list backtests: {e}")
            return []

    async def compare_models(
        self,
        model_ids: List[str],
        config: BacktestConfig
    ) -> Dict[str, Any]:
        """對比多個模型的回測結果"""
        results = {}

        for model_id in model_ids:
            try:
                # 獲取模型數據
                from pathlib import Path
                model_file = Path("models/ios_trained") / f"{model_id}.json"
                model_data = None
                if model_file.exists():
                    with open(model_file) as f:
                        model_data = json.load(f)

                result = await self.run_backtest(model_id, config, model_data)
                results[model_id] = {
                    "total_return_pct": result.total_return_pct,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "total_trades": result.total_trades
                }
            except Exception as e:
                logger.error(f"Failed to backtest model {model_id}: {e}")
                results[model_id] = {"error": str(e)}

        # 計算排名
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if valid_results:
            # 按夏普比率排名
            sorted_by_sharpe = sorted(
                valid_results.items(),
                key=lambda x: x[1].get("sharpe_ratio", 0),
                reverse=True
            )
            for rank, (model_id, _) in enumerate(sorted_by_sharpe, 1):
                results[model_id]["sharpe_rank"] = rank

        return {
            "config": config.to_dict(),
            "model_count": len(model_ids),
            "results": results,
            "best_model": sorted_by_sharpe[0][0] if valid_results else None,
            "timestamp": datetime.utcnow().isoformat()
        }


# 單例實例
backtest_service = BacktestService()
