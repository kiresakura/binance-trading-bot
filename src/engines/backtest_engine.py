"""
回測引擎

功能：
1. 歷史數據回測
2. 策略績效評估
3. 風險指標計算
4. 交易記錄生成
5. 可視化報告
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from src.config import settings
from src.logger import logger
from src.services.binance_client import binance_client


class BacktestMode(str, Enum):
    """回測模式"""
    SIMPLE = "simple"           # 簡單回測（收盤價成交）
    REALISTIC = "realistic"     # 現實回測（考慮滑點、手續費）
    MONTE_CARLO = "monte_carlo" # 蒙特卡羅模擬


@dataclass
class BacktestConfig:
    """回測配置"""
    # 時間範圍
    start_date: datetime = None
    end_date: datetime = None

    # 交易配置
    initial_capital: float = 10000.0
    commission_rate: float = 0.001  # 0.1% 手續費
    slippage_rate: float = 0.0005   # 0.05% 滑點

    # 風險配置
    max_position_size: float = 0.2   # 最大單筆倉位 20%
    stop_loss_pct: float = 0.02      # 止損 2%
    take_profit_pct: float = 0.04    # 止盈 4%

    # 回測模式
    mode: BacktestMode = BacktestMode.REALISTIC

    # 數據配置
    symbol: str = "BTCUSDT"
    interval: str = "1h"


@dataclass
class Trade:
    """交易記錄"""
    id: int
    timestamp: datetime
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float
    commission: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""


@dataclass
class Position:
    """持倉"""
    symbol: str
    side: str  # "LONG" or "SHORT"
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class BacktestResult:
    """回測結果"""
    # 基本信息
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float

    # 績效指標
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # 交易統計
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_duration: timedelta

    # 風險指標
    volatility: float
    var_95: float  # 95% VaR
    cvar_95: float # 95% CVaR (Expected Shortfall)

    # 詳細數據
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "symbol": self.symbol,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "initial_capital": self.initial_capital,
            "final_capital": round(self.final_capital, 2),
            "total_return": round(self.total_return * 100, 2),
            "annual_return": round(self.annual_return * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "max_drawdown": round(self.max_drawdown * 100, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate * 100, 2),
            "avg_win": round(self.avg_win * 100, 2),
            "avg_loss": round(self.avg_loss * 100, 2),
            "profit_factor": round(self.profit_factor, 2),
            "volatility": round(self.volatility * 100, 2),
            "var_95": round(self.var_95 * 100, 2),
            "cvar_95": round(self.cvar_95 * 100, 2),
        }


class BacktestEngine:
    """回測引擎"""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

        # 狀態
        self._capital = self.config.initial_capital
        self._position: Optional[Position] = None
        self._trades: List[Trade] = []
        self._equity_curve: List[float] = []
        self._trade_id = 0

        # 數據
        self._data: Optional[pd.DataFrame] = None

        logger.info("BacktestEngine initialized")

    async def load_data(
        self,
        symbol: str = None,
        interval: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """載入歷史數據"""
        symbol = symbol or self.config.symbol
        interval = interval or self.config.interval

        logger.info(f"Loading historical data: {symbol} {interval}")

        try:
            # 從 Binance 獲取 K 線數據
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

            # 處理數據類型
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            # 設置索引
            df.set_index('timestamp', inplace=True)

            # 過濾日期範圍
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            self._data = df
            self.config.symbol = symbol
            self.config.start_date = df.index[0].to_pydatetime()
            self.config.end_date = df.index[-1].to_pydatetime()

            logger.info(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
            return df

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def add_indicators(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """添加技術指標"""
        df = df if df is not None else self._data
        if df is None:
            raise ValueError("No data loaded")

        # SMA
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # EMA
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std

        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()

        # 成交量指標
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        self._data = df
        return df

    async def run(
        self,
        strategy: Callable[[pd.DataFrame, int], int],
        data: pd.DataFrame = None
    ) -> BacktestResult:
        """
        執行回測

        Args:
            strategy: 策略函數，接收 (data, current_index)，返回信號 (1=買入, -1=賣出, 0=持有)
            data: 歷史數據 DataFrame

        Returns:
            BacktestResult: 回測結果
        """
        df = data if data is not None else self._data
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # 重置狀態
        self._capital = self.config.initial_capital
        self._position = None
        self._trades = []
        self._equity_curve = [self.config.initial_capital]
        self._trade_id = 0

        logger.info(f"Starting backtest: {self.config.symbol}")
        logger.info(f"Period: {df.index[0]} to {df.index[-1]}")
        logger.info(f"Initial capital: ${self.config.initial_capital:,.2f}")

        # 遍歷每根 K 線
        for i in range(50, len(df)):  # 跳過前 50 根用於指標計算
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            high_price = df['high'].iloc[i]
            low_price = df['low'].iloc[i]

            # 檢查止損止盈
            if self._position:
                self._check_stop_loss_take_profit(
                    current_time, current_price, high_price, low_price
                )

            # 獲取策略信號
            signal = strategy(df, i)

            # 執行交易
            if signal == 1 and not self._position:
                # 開多倉
                self._open_position(current_time, current_price, "LONG")
            elif signal == -1 and not self._position:
                # 開空倉
                self._open_position(current_time, current_price, "SHORT")
            elif signal == 0 and self._position:
                # 平倉
                self._close_position(current_time, current_price, "Signal neutral")
            elif signal == 1 and self._position and self._position.side == "SHORT":
                # 空轉多
                self._close_position(current_time, current_price, "Signal reversal")
                self._open_position(current_time, current_price, "LONG")
            elif signal == -1 and self._position and self._position.side == "LONG":
                # 多轉空
                self._close_position(current_time, current_price, "Signal reversal")
                self._open_position(current_time, current_price, "SHORT")

            # 更新淨值
            equity = self._calculate_equity(current_price)
            self._equity_curve.append(equity)

        # 最後平倉
        if self._position:
            final_price = df['close'].iloc[-1]
            self._close_position(df.index[-1], final_price, "Backtest ended")

        # 計算結果
        result = self._calculate_results(df)

        logger.info(f"Backtest completed: Total return {result.total_return:.2%}")
        logger.info(f"Sharpe ratio: {result.sharpe_ratio:.2f}, Max drawdown: {result.max_drawdown:.2%}")

        return result

    def _open_position(self, timestamp: datetime, price: float, side: str):
        """開倉"""
        # 計算滑點
        if self.config.mode == BacktestMode.REALISTIC:
            slippage = price * self.config.slippage_rate
            price = price + slippage if side == "LONG" else price - slippage

        # 計算倉位大小
        position_value = self._capital * self.config.max_position_size
        quantity = position_value / price

        # 計算手續費
        commission = position_value * self.config.commission_rate
        self._capital -= commission

        # 計算止損止盈價格
        if side == "LONG":
            stop_loss = price * (1 - self.config.stop_loss_pct)
            take_profit = price * (1 + self.config.take_profit_pct)
        else:
            stop_loss = price * (1 + self.config.stop_loss_pct)
            take_profit = price * (1 - self.config.take_profit_pct)

        # 創建持倉
        self._position = Position(
            symbol=self.config.symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        # 記錄交易
        self._trade_id += 1
        self._trades.append(Trade(
            id=self._trade_id,
            timestamp=timestamp,
            symbol=self.config.symbol,
            side="BUY" if side == "LONG" else "SELL",
            quantity=quantity,
            price=price,
            commission=commission,
            reason=f"Open {side}"
        ))

    def _close_position(self, timestamp: datetime, price: float, reason: str):
        """平倉"""
        if not self._position:
            return

        # 計算滑點
        if self.config.mode == BacktestMode.REALISTIC:
            slippage = price * self.config.slippage_rate
            price = price - slippage if self._position.side == "LONG" else price + slippage

        # 計算盈虧
        if self._position.side == "LONG":
            pnl = (price - self._position.entry_price) * self._position.quantity
        else:
            pnl = (self._position.entry_price - price) * self._position.quantity

        pnl_pct = pnl / (self._position.entry_price * self._position.quantity)

        # 計算手續費
        position_value = price * self._position.quantity
        commission = position_value * self.config.commission_rate

        # 更新資金
        self._capital += pnl - commission

        # 記錄交易
        self._trade_id += 1
        self._trades.append(Trade(
            id=self._trade_id,
            timestamp=timestamp,
            symbol=self.config.symbol,
            side="SELL" if self._position.side == "LONG" else "BUY",
            quantity=self._position.quantity,
            price=price,
            commission=commission,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason
        ))

        # 清除持倉
        self._position = None

    def _check_stop_loss_take_profit(
        self,
        timestamp: datetime,
        current_price: float,
        high_price: float,
        low_price: float
    ):
        """檢查止損止盈"""
        if not self._position:
            return

        if self._position.side == "LONG":
            # 多倉：檢查止損（最低價）和止盈（最高價）
            if self._position.stop_loss and low_price <= self._position.stop_loss:
                self._close_position(timestamp, self._position.stop_loss, "Stop loss")
            elif self._position.take_profit and high_price >= self._position.take_profit:
                self._close_position(timestamp, self._position.take_profit, "Take profit")
        else:
            # 空倉：檢查止損（最高價）和止盈（最低價）
            if self._position.stop_loss and high_price >= self._position.stop_loss:
                self._close_position(timestamp, self._position.stop_loss, "Stop loss")
            elif self._position.take_profit and low_price <= self._position.take_profit:
                self._close_position(timestamp, self._position.take_profit, "Take profit")

    def _calculate_equity(self, current_price: float) -> float:
        """計算當前淨值"""
        if not self._position:
            return self._capital

        if self._position.side == "LONG":
            unrealized_pnl = (current_price - self._position.entry_price) * self._position.quantity
        else:
            unrealized_pnl = (self._position.entry_price - current_price) * self._position.quantity

        return self._capital + unrealized_pnl

    def _calculate_results(self, df: pd.DataFrame) -> BacktestResult:
        """計算回測結果"""
        equity = np.array(self._equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # 基本績效
        total_return = (equity[-1] - equity[0]) / equity[0]

        # 年化收益
        days = (df.index[-1] - df.index[0]).days
        annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1

        # 波動率
        volatility = np.std(returns) * np.sqrt(365 * 24)  # 假設小時數據

        # 夏普比率
        risk_free_rate = 0.02  # 假設 2% 無風險利率
        excess_returns = returns - risk_free_rate / (365 * 24)
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365 * 24) if np.std(excess_returns) > 0 else 0

        # Sortino 比率
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0001
        sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(365 * 24) if downside_std > 0 else 0

        # 最大回撤
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown)

        # Calmar 比率
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

        # 交易統計
        closing_trades = [t for t in self._trades if t.pnl != 0]
        winning_trades = [t for t in closing_trades if t.pnl > 0]
        losing_trades = [t for t in closing_trades if t.pnl < 0]

        total_trades = len(closing_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0

        # 盈虧比
        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # 平均持倉時間
        durations = []
        for i in range(0, len(self._trades) - 1, 2):
            if i + 1 < len(self._trades):
                duration = self._trades[i + 1].timestamp - self._trades[i].timestamp
                durations.append(duration)
        avg_duration = sum(durations, timedelta()) / len(durations) if durations else timedelta()

        # VaR 和 CVaR
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else 0

        return BacktestResult(
            symbol=self.config.symbol,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=self.config.initial_capital,
            final_capital=equity[-1],
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_trade_duration=avg_duration,
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            trades=self._trades,
            equity_curve=self._equity_curve.copy(),
            drawdown_curve=drawdown.tolist()
        )


# ==================== 內建策略 ====================

def sma_crossover_strategy(df: pd.DataFrame, i: int) -> int:
    """SMA 交叉策略"""
    if 'sma_20' not in df.columns or 'sma_50' not in df.columns:
        return 0

    sma_20 = df['sma_20'].iloc[i]
    sma_50 = df['sma_50'].iloc[i]
    sma_20_prev = df['sma_20'].iloc[i - 1]
    sma_50_prev = df['sma_50'].iloc[i - 1]

    if pd.isna(sma_20) or pd.isna(sma_50):
        return 0

    # 金叉
    if sma_20_prev <= sma_50_prev and sma_20 > sma_50:
        return 1
    # 死叉
    elif sma_20_prev >= sma_50_prev and sma_20 < sma_50:
        return -1

    return 0


def rsi_strategy(df: pd.DataFrame, i: int) -> int:
    """RSI 策略"""
    if 'rsi' not in df.columns:
        return 0

    rsi = df['rsi'].iloc[i]

    if pd.isna(rsi):
        return 0

    # 超賣買入
    if rsi < 30:
        return 1
    # 超買賣出
    elif rsi > 70:
        return -1

    return 0


def macd_strategy(df: pd.DataFrame, i: int) -> int:
    """MACD 策略"""
    if 'macd' not in df.columns or 'macd_signal' not in df.columns:
        return 0

    macd = df['macd'].iloc[i]
    signal = df['macd_signal'].iloc[i]
    macd_prev = df['macd'].iloc[i - 1]
    signal_prev = df['macd_signal'].iloc[i - 1]

    if pd.isna(macd) or pd.isna(signal):
        return 0

    # MACD 上穿信號線
    if macd_prev <= signal_prev and macd > signal:
        return 1
    # MACD 下穿信號線
    elif macd_prev >= signal_prev and macd < signal:
        return -1

    return 0


def bollinger_strategy(df: pd.DataFrame, i: int) -> int:
    """布林帶策略"""
    if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
        return 0

    close = df['close'].iloc[i]
    bb_upper = df['bb_upper'].iloc[i]
    bb_lower = df['bb_lower'].iloc[i]

    if pd.isna(bb_upper) or pd.isna(bb_lower):
        return 0

    # 價格觸及下軌買入
    if close <= bb_lower:
        return 1
    # 價格觸及上軌賣出
    elif close >= bb_upper:
        return -1

    return 0


# 全局實例
backtest_engine = BacktestEngine()
