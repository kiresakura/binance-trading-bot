"""
策略基類

定義所有策略的通用接口和行為

v0.3.0 更新：
- 增加 Hyperopt 搜索空間定義支援
- 增加特徵工程方法
- 增加多時間框架支援
- 借鑒 Freqtrade IStrategy 設計

作者：Binance Trading Bot v0.3.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
import pandas as pd
import numpy as np

from src.logger import logger


# ==================== 時間過濾器 (v0.3.1) ====================

class TimeFilter:
    """
    交易時間過濾器

    避免在低流動性時段（如週末、假日、非活躍時間）交易
    加密貨幣市場 24/7 運行，但不同時段流動性差異很大
    """

    # 高流動性時段（UTC）
    HIGH_LIQUIDITY_HOURS = [
        (8, 12),   # 亞洲交易時段
        (12, 16),  # 歐洲交易時段開始
        (14, 22),  # 美歐交易重疊時段（最活躍）
    ]

    # 低流動性時段（UTC）- 避免交易
    LOW_LIQUIDITY_HOURS = [
        (3, 7),    # 亞洲深夜
    ]

    def __init__(
        self,
        enabled: bool = True,
        avoid_weekend: bool = False,  # 加密貨幣週末也交易
        high_liquidity_hours: Optional[List[Tuple[int, int]]] = None,
        low_liquidity_hours: Optional[List[Tuple[int, int]]] = None,
        reduce_size_low_liquidity: float = 0.5,  # 低流動性時段減倉
    ):
        self.enabled = enabled
        self.avoid_weekend = avoid_weekend
        self.high_liquidity_hours = high_liquidity_hours or self.HIGH_LIQUIDITY_HOURS
        self.low_liquidity_hours = low_liquidity_hours or self.LOW_LIQUIDITY_HOURS
        self.reduce_size_low_liquidity = reduce_size_low_liquidity

    def is_trading_allowed(self, timestamp: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        檢查當前時間是否允許交易

        Returns:
            (是否允許交易, 原因說明)
        """
        if not self.enabled:
            return True, "時間過濾器已禁用"

        if timestamp is None:
            timestamp = datetime.utcnow()

        # 週末檢查
        if self.avoid_weekend and timestamp.weekday() >= 5:
            return False, f"週末不交易 (weekday={timestamp.weekday()})"

        hour = timestamp.hour

        # 低流動性時段檢查
        for start, end in self.low_liquidity_hours:
            if start <= hour < end:
                return False, f"低流動性時段 ({start}:00-{end}:00 UTC)"

        return True, "正常交易時段"

    def get_liquidity_multiplier(self, timestamp: Optional[datetime] = None) -> float:
        """
        獲取流動性倉位調整係數

        高流動性時段: 1.0 (正常倉位)
        中等流動性: 0.8
        低流動性: reduce_size_low_liquidity

        Returns:
            倉位調整係數 (0.5-1.0)
        """
        if not self.enabled:
            return 1.0

        if timestamp is None:
            timestamp = datetime.utcnow()

        hour = timestamp.hour

        # 高流動性時段
        for start, end in self.high_liquidity_hours:
            if start <= hour < end:
                return 1.0

        # 低流動性時段
        for start, end in self.low_liquidity_hours:
            if start <= hour < end:
                return self.reduce_size_low_liquidity

        # 中等流動性
        return 0.8

    def get_status(self) -> Dict[str, Any]:
        """獲取時間過濾器狀態"""
        now = datetime.utcnow()
        allowed, reason = self.is_trading_allowed(now)
        return {
            'enabled': self.enabled,
            'current_hour_utc': now.hour,
            'is_trading_allowed': allowed,
            'reason': reason,
            'liquidity_multiplier': self.get_liquidity_multiplier(now),
            'weekday': now.weekday()
        }


class SignalType(Enum):
    """信號類型"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class Signal:
    """交易信號"""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    direction: int  # 1=做多, -1=做空, 0=觀望
    strength: float  # 0-1 信號強度
    confidence: float  # 0-1 置信度
    strategy_name: str

    # 建議參數
    suggested_size: float = 0.0  # 建議倉位比例 (0-1)
    stop_loss: Optional[float] = None  # 建議止損價
    take_profit: Optional[float] = None  # 建議止盈價

    # 附加信息
    features: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'direction': self.direction,
            'strength': self.strength,
            'confidence': self.confidence,
            'strategy_name': self.strategy_name,
            'suggested_size': self.suggested_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'features': self.features,
            'reason': self.reason,
            'metadata': self.metadata
        }


@dataclass
class StrategyConfig:
    """策略配置"""
    name: str
    enabled: bool = True
    weight: float = 1.0  # 策略權重（用於集成）

    # 風險參數
    max_position_size: float = 0.1  # 最大倉位
    stop_loss_pct: float = 0.02  # 止損百分比
    take_profit_pct: float = 0.04  # 止盈百分比

    # 信號參數
    min_signal_strength: float = 0.3  # 最小信號強度
    min_confidence: float = 0.5  # 最小置信度

    # 冷卻期
    cooldown_minutes: int = 5  # 信號冷卻期

    # 資金管理參數
    consecutive_loss_reduce_2: float = 0.3   # 連續虧損 2 次後減倉比例
    consecutive_loss_reduce_3: float = 0.5   # 連續虧損 3 次後減倉比例
    min_win_rate_threshold: float = 0.4      # 最低勝率門檻
    win_rate_lookback: int = 20              # 勝率計算回看交易數

    # 時間過濾器參數 (v0.3.1)
    time_filter_enabled: bool = True         # 啟用時間過濾
    avoid_low_liquidity: bool = True         # 避免低流動性時段
    low_liquidity_hours: List[Tuple[int, int]] = field(
        default_factory=lambda: [(3, 7)]     # UTC 3:00-7:00 低流動性
    )

    # Kelly Criterion 參數 (v0.3.2)
    kelly_enabled: bool = True               # 啟用 Kelly 倉位管理
    kelly_fraction: float = 0.25             # Kelly 係數（0.25 = 四分之一 Kelly，更保守）
    kelly_min_trades: int = 10               # 計算 Kelly 前的最少交易數
    kelly_max_position: float = 0.3          # Kelly 計算的最大倉位上限
    kelly_min_position: float = 0.05         # Kelly 計算的最小倉位下限

    # 自定義參數
    params: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    策略基類

    所有交易策略必須繼承此類並實現以下方法：
    - generate_signal: 生成交易信號
    - update: 更新策略狀態
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig(name=self.__class__.__name__)
        self.name = self.config.name

        # 狀態
        self._last_signal_time: Optional[datetime] = None
        self._last_signal: Optional[Signal] = None
        self._trade_count = 0
        self._win_count = 0

        # 績效追蹤
        self._returns: List[float] = []
        self._signals_history: List[Signal] = []

        # 資金管理狀態
        self._consecutive_losses = 0          # 連續虧損次數
        self._position_size_multiplier = 1.0  # 倉位調整係數
        self._recent_results: List[bool] = [] # 最近交易結果（True=盈利）

        # 時間過濾器 (v0.3.1)
        self._time_filter = TimeFilter(
            enabled=self.config.time_filter_enabled,
            avoid_weekend=False,  # 加密貨幣週末交易
            low_liquidity_hours=self.config.low_liquidity_hours if self.config.avoid_low_liquidity else []
        )

        logger.info(f"Strategy {self.name} initialized (TimeFilter: {'ON' if self.config.time_filter_enabled else 'OFF'})")

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        features: pd.DataFrame,
        current_price: float,
        position: float = 0.0
    ) -> Optional[Signal]:
        """
        生成交易信號

        Args:
            symbol: 交易對
            features: 特徵 DataFrame（包含技術指標）
            current_price: 當前價格
            position: 當前持倉 (-1到1，0為無倉位)

        Returns:
            Signal 對象或 None
        """
        pass

    @abstractmethod
    def update(self, market_data: Dict[str, Any]):
        """
        更新策略內部狀態

        Args:
            market_data: 市場數據字典
        """
        pass

    def should_generate_signal(self) -> bool:
        """檢查是否應該生成新信號（冷卻期 + 時間過濾）"""
        if not self.config.enabled:
            return False

        # 時間過濾器檢查 (v0.3.1)
        now = datetime.utcnow()
        is_allowed, reason = self._time_filter.is_trading_allowed(now)
        if not is_allowed:
            return False

        if self._last_signal_time is None:
            return True

        elapsed = (now - self._last_signal_time).total_seconds()
        return elapsed >= self.config.cooldown_minutes * 60

    def get_time_filter_status(self) -> Dict[str, Any]:
        """獲取時間過濾器狀態"""
        return self._time_filter.get_status()

    def get_liquidity_multiplier(self) -> float:
        """獲取流動性倉位調整係數"""
        return self._time_filter.get_liquidity_multiplier()

    # ==================== Kelly Criterion 倉位管理 (v0.3.2) ====================

    def calculate_kelly_fraction(self) -> float:
        """
        計算 Kelly Criterion 最佳倉位比例

        Kelly 公式: f* = (p * b - q) / b
        其中:
        - f* = 最佳下注比例
        - p = 勝率
        - q = 敗率 = 1 - p
        - b = 賠率 = 平均獲利 / 平均虧損

        為了更保守，使用 fractional Kelly（如 0.25 Kelly）

        Returns:
            建議的倉位比例 (0.0-1.0)
        """
        if not self.config.kelly_enabled:
            return self.config.max_position_size

        # 需要足夠的交易數據
        if len(self._returns) < self.config.kelly_min_trades:
            return self.config.max_position_size

        returns = np.array(self._returns[-50:])  # 使用最近 50 筆交易

        # 計算勝率
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            return self.config.max_position_size

        win_rate = len(wins) / len(returns)
        loss_rate = 1 - win_rate

        # 計算平均獲利和虧損
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_loss == 0:
            return self.config.max_position_size

        # 賠率（風報比）
        odds = avg_win / avg_loss

        # Kelly 公式
        kelly = (win_rate * odds - loss_rate) / odds

        # 應用 fractional Kelly
        kelly *= self.config.kelly_fraction

        # 限制範圍
        kelly = max(self.config.kelly_min_position, min(self.config.kelly_max_position, kelly))

        # 如果 Kelly 為負（預期虧損），返回最小倉位
        if kelly <= 0:
            logger.warning(f"[{self.name}] Kelly negative ({kelly:.4f}), using min position")
            return self.config.kelly_min_position

        return kelly

    def get_kelly_stats(self) -> Dict[str, Any]:
        """
        獲取 Kelly Criterion 統計數據

        Returns:
            包含 Kelly 相關統計的字典
        """
        if len(self._returns) < self.config.kelly_min_trades:
            return {
                'enabled': self.config.kelly_enabled,
                'has_enough_data': False,
                'trades_needed': self.config.kelly_min_trades - len(self._returns),
                'kelly_fraction': None,
                'recommended_position': self.config.max_position_size
            }

        returns = np.array(self._returns[-50:])
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            return {
                'enabled': self.config.kelly_enabled,
                'has_enough_data': True,
                'win_rate': len(wins) / len(returns) if len(returns) > 0 else 0,
                'avg_win': 0,
                'avg_loss': 0,
                'odds': 0,
                'raw_kelly': 0,
                'kelly_fraction': 0,
                'recommended_position': self.config.kelly_min_position
            }

        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        odds = avg_win / avg_loss if avg_loss > 0 else 0
        raw_kelly = (win_rate * odds - (1 - win_rate)) / odds if odds > 0 else 0
        fractional_kelly = self.calculate_kelly_fraction()

        return {
            'enabled': self.config.kelly_enabled,
            'has_enough_data': True,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'odds': odds,
            'raw_kelly': raw_kelly,
            'applied_fraction': self.config.kelly_fraction,
            'kelly_fraction': fractional_kelly,
            'recommended_position': fractional_kelly
        }

    def record_signal(self, signal: Signal):
        """記錄信號"""
        self._last_signal = signal
        self._last_signal_time = signal.timestamp
        self._signals_history.append(signal)

        # 保留最近 1000 個信號
        if len(self._signals_history) > 1000:
            self._signals_history = self._signals_history[-1000:]

    def record_trade_result(self, pnl_pct: float):
        """記錄交易結果並更新資金管理狀態"""
        self._trade_count += 1
        is_win = pnl_pct > 0

        if is_win:
            self._win_count += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        self._returns.append(pnl_pct)
        self._recent_results.append(is_win)

        # 保留最近記錄
        if len(self._returns) > 1000:
            self._returns = self._returns[-1000:]
        if len(self._recent_results) > self.config.win_rate_lookback:
            self._recent_results = self._recent_results[-self.config.win_rate_lookback:]

        # 更新倉位調整係數
        self._update_position_multiplier()

    def get_performance(self) -> Dict[str, float]:
        """獲取策略績效"""
        if not self._returns:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }

        returns = np.array(self._returns)

        return {
            'total_trades': self._trade_count,
            'win_rate': self._win_count / self._trade_count if self._trade_count > 0 else 0,
            'avg_return': np.mean(returns),
            'total_return': np.sum(returns),
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'profit_factor': self._calculate_profit_factor(returns)
        }

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """計算最大回撤"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return float(np.min(drawdown))

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """計算盈虧比"""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return gains / (losses + 1e-10)

    def _update_position_multiplier(self):
        """
        更新倉位調整係數（資金管理核心邏輯）

        規則：
        1. 連續虧損 2 次：減倉 30%
        2. 連續虧損 3 次或以上：減倉 50%
        3. 近期勝率低於 40%：自動減倉
        4. 恢復後逐步恢復倉位
        """
        multiplier = 1.0
        reason = ""

        # 連續虧損減倉
        if self._consecutive_losses >= 3:
            multiplier = 1.0 - self.config.consecutive_loss_reduce_3
            reason = f"連續虧損{self._consecutive_losses}次，減倉{self.config.consecutive_loss_reduce_3*100:.0f}%"
        elif self._consecutive_losses >= 2:
            multiplier = 1.0 - self.config.consecutive_loss_reduce_2
            reason = f"連續虧損{self._consecutive_losses}次，減倉{self.config.consecutive_loss_reduce_2*100:.0f}%"

        # 勝率過低減倉
        if len(self._recent_results) >= 10:  # 至少 10 筆交易才計算
            recent_win_rate = sum(self._recent_results) / len(self._recent_results)
            if recent_win_rate < self.config.min_win_rate_threshold:
                win_rate_multiplier = recent_win_rate / self.config.min_win_rate_threshold
                multiplier = min(multiplier, win_rate_multiplier)
                reason = f"勝率過低({recent_win_rate*100:.1f}%)，減倉至{win_rate_multiplier*100:.0f}%"

        # 限制最小倉位
        multiplier = max(0.2, multiplier)

        if multiplier != self._position_size_multiplier:
            logger.info(f"[{self.name}] 倉位調整: {self._position_size_multiplier:.2f} -> {multiplier:.2f} ({reason})")
            self._position_size_multiplier = multiplier

    def get_position_multiplier(self) -> float:
        """獲取當前倉位調整係數"""
        return self._position_size_multiplier

    def get_recent_win_rate(self) -> float:
        """獲取近期勝率"""
        if not self._recent_results:
            return 0.5  # 默認 50%
        return sum(self._recent_results) / len(self._recent_results)

    def get_consecutive_losses(self) -> int:
        """獲取連續虧損次數"""
        return self._consecutive_losses

    def get_money_management_status(self) -> Dict[str, Any]:
        """獲取資金管理狀態"""
        return {
            'consecutive_losses': self._consecutive_losses,
            'position_multiplier': self._position_size_multiplier,
            'recent_win_rate': self.get_recent_win_rate(),
            'recent_trades_count': len(self._recent_results),
            'is_reduced': self._position_size_multiplier < 1.0
        }

    def calculate_position_size(
        self,
        signal: Signal,
        account_balance: float,
        current_price: float
    ) -> float:
        """
        計算建議的倉位大小（含資金管理）

        Args:
            signal: 交易信號
            account_balance: 賬戶餘額
            current_price: 當前價格

        Returns:
            建議的數量（如果資金不足返回 0）
        """
        from src.config import settings
        
        # 基於信號強度和置信度調整倉位
        size_factor = signal.strength * signal.confidence

        # 應用資金管理倉位調整
        size_factor *= self._position_size_multiplier

        # 應用流動性倉位調整 (v0.3.1)
        size_factor *= self.get_liquidity_multiplier()

        # 使用 Kelly Criterion 計算基礎倉位 (v0.3.2)
        kelly_position = self.calculate_kelly_fraction()

        # 計算最大倉位金額（使用 Kelly 或默認最大倉位的較小值）
        base_position = min(kelly_position, self.config.max_position_size)
        max_position_value = account_balance * base_position * size_factor

        # 🆕 最低訂單金額檢查
        min_order_value = getattr(settings, 'MIN_ORDER_VALUE', 6.0)
        
        if max_position_value < min_order_value:
            # 如果計算出的金額低於最低值，嘗試使用最低值
            if account_balance >= min_order_value * 1.2:  # 確保有足夠餘額（加 20% buffer）
                max_position_value = min_order_value
                logger.debug(f"Position value adjusted to minimum: ${min_order_value}")
            else:
                # 資金真的不足，返回 0 讓上層跳過此訂單
                logger.warning(f"Insufficient balance for minimum order: ${account_balance:.2f} < ${min_order_value * 1.2:.2f}")
                return 0.0

        # 轉換為數量
        quantity = max_position_value / current_price

        return quantity

    def calculate_stop_loss(self, entry_price: float, direction: int) -> float:
        """計算止損價"""
        if direction > 0:  # 做多
            return entry_price * (1 - self.config.stop_loss_pct)
        else:  # 做空
            return entry_price * (1 + self.config.stop_loss_pct)

    def calculate_take_profit(self, entry_price: float, direction: int) -> float:
        """計算止盈價"""
        if direction > 0:  # 做多
            return entry_price * (1 + self.config.take_profit_pct)
        else:  # 做空
            return entry_price * (1 - self.config.take_profit_pct)

    def reset(self):
        """重置策略狀態"""
        self._last_signal_time = None
        self._last_signal = None
        self._trade_count = 0
        self._win_count = 0
        self._returns = []
        self._signals_history = []
        # 重置資金管理狀態
        self._consecutive_losses = 0
        self._position_size_multiplier = 1.0
        self._recent_results = []
        logger.info(f"Strategy {self.name} reset")

    def get_state(self) -> Dict[str, Any]:
        """獲取策略狀態"""
        return {
            'name': self.name,
            'enabled': self.config.enabled,
            'weight': self.config.weight,
            'last_signal_time': self._last_signal_time.isoformat() if self._last_signal_time else None,
            'trade_count': self._trade_count,
            'performance': self.get_performance(),
            'money_management': self.get_money_management_status(),
            'time_filter': self.get_time_filter_status(),  # v0.3.1
            'kelly_stats': self.get_kelly_stats()  # v0.3.2
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, enabled={self.config.enabled})"

    # ==================== Hyperopt 支援 (v0.3.0) ====================

    @classmethod
    def get_hyperopt_space(cls) -> Dict[str, Any]:
        """
        定義 Hyperopt 搜索空間

        子類應該覆寫此方法以定義可優化的參數

        Returns:
            搜索空間定義字典
            格式: {
                "param_name": {
                    "type": "int" | "float" | "categorical",
                    "low": min_value,
                    "high": max_value,
                    "step": step_size (optional),
                    "choices": [list] (for categorical),
                    "log": bool (for float, log scale)
                }
            }

        Example:
            return {
                "rsi_period": {"type": "int", "low": 5, "high": 30},
                "rsi_overbought": {"type": "int", "low": 60, "high": 90},
                "position_ratio": {"type": "float", "low": 0.1, "high": 1.0, "step": 0.1},
                "signal_mode": {"type": "categorical", "choices": ["aggressive", "conservative"]}
            }
        """
        return {}

    @classmethod
    def suggest_params(cls, trial) -> Dict[str, Any]:
        """
        使用 Optuna trial 建議參數

        Args:
            trial: Optuna trial 對象

        Returns:
            建議的參數字典
        """
        space = cls.get_hyperopt_space()
        params = {}

        for name, config in space.items():
            param_type = config.get("type", "float")

            if param_type == "int":
                params[name] = trial.suggest_int(
                    name,
                    config["low"],
                    config["high"],
                    step=config.get("step", 1)
                )
            elif param_type == "float":
                if config.get("log", False):
                    params[name] = trial.suggest_float(
                        name,
                        config["low"],
                        config["high"],
                        log=True
                    )
                else:
                    params[name] = trial.suggest_float(
                        name,
                        config["low"],
                        config["high"],
                        step=config.get("step")
                    )
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(
                    name,
                    config["choices"]
                )

        return params

    def apply_params(self, params: Dict[str, Any]):
        """
        應用優化後的參數

        Args:
            params: 參數字典
        """
        for key, value in params.items():
            if hasattr(self.config, 'params') and isinstance(self.config.params, dict):
                self.config.params[key] = value
            elif hasattr(self.config, key):
                setattr(self.config, key, value)

        logger.info(f"[{self.name}] Applied params: {params}")

    # ==================== 特徵工程支援 (v0.3.0) ====================

    def populate_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        填充技術指標（供子類覆寫）
        借鑒自 Freqtrade IStrategy

        Args:
            dataframe: 包含 OHLCV 數據的 DataFrame

        Returns:
            添加了指標的 DataFrame
        """
        return dataframe

    def populate_entry_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        填充入場信號（供子類覆寫）

        Args:
            dataframe: 包含指標的 DataFrame

        Returns:
            添加了 'enter_long', 'enter_short' 列的 DataFrame
        """
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        return dataframe

    def populate_exit_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        填充出場信號（供子類覆寫）

        Args:
            dataframe: 包含指標的 DataFrame

        Returns:
            添加了 'exit_long', 'exit_short' 列的 DataFrame
        """
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe

    def analyze(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        完整分析流程

        Args:
            dataframe: 原始 OHLCV DataFrame

        Returns:
            完整分析後的 DataFrame
        """
        # 1. 填充指標
        dataframe = self.populate_indicators(dataframe)

        # 2. 填充入場信號
        dataframe = self.populate_entry_signal(dataframe)

        # 3. 填充出場信號
        dataframe = self.populate_exit_signal(dataframe)

        return dataframe

    # ==================== 多時間框架支援 (v0.3.0) ====================

    @property
    def timeframe(self) -> str:
        """主時間框架"""
        return self.config.params.get('timeframe', '1h')

    @property
    def informative_timeframes(self) -> List[str]:
        """
        輔助時間框架列表
        子類可覆寫以支援多時間框架分析

        Returns:
            時間框架列表，如 ['4h', '1d']
        """
        return []

    def merge_informative(
        self,
        dataframe: pd.DataFrame,
        informative: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """
        合併輔助時間框架數據

        Args:
            dataframe: 主時間框架數據
            informative: 輔助時間框架數據
            timeframe: 輔助時間框架

        Returns:
            合併後的 DataFrame
        """
        # 為輔助時間框架的列添加前綴
        suffix = f"_{timeframe}"
        informative_cols = [col for col in informative.columns if col not in ['timestamp', 'date', 'open_time']]

        for col in informative_cols:
            informative = informative.rename(columns={col: f"{col}{suffix}"})

        # 按時間合併（使用 merge_asof 進行時間對齊）
        if 'timestamp' in dataframe.columns and 'timestamp' in informative.columns:
            dataframe = pd.merge_asof(
                dataframe.sort_values('timestamp'),
                informative.sort_values('timestamp'),
                on='timestamp',
                direction='backward'
            )

        return dataframe

    # ==================== 回測支援 (v0.3.0) ====================

    def backtest_generate_signals(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        為回測生成信號（向量化操作，更高效）

        Args:
            dataframe: 包含 OHLCV 數據的 DataFrame

        Returns:
            添加了信號列的 DataFrame
        """
        # 分析（填充指標和信號）
        dataframe = self.analyze(dataframe)

        # 計算信號強度（如果子類沒有提供）
        if 'signal_strength' not in dataframe.columns:
            dataframe['signal_strength'] = 0.5

        if 'signal_confidence' not in dataframe.columns:
            dataframe['signal_confidence'] = 0.5

        return dataframe

    def get_backtest_params(self) -> Dict[str, Any]:
        """
        獲取回測參數

        Returns:
            回測配置字典
        """
        return {
            "strategy_name": self.name,
            "timeframe": self.timeframe,
            "max_position_size": self.config.max_position_size,
            "stop_loss_pct": self.config.stop_loss_pct,
            "take_profit_pct": self.config.take_profit_pct,
            "min_signal_strength": self.config.min_signal_strength,
            "min_confidence": self.config.min_confidence,
            "params": self.config.params
        }
