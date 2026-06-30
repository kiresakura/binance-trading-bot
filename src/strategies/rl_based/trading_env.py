"""
強化學習交易環境

基於 Gymnasium 的加密貨幣交易環境
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from src.logger import logger


@dataclass
class TradingEnvConfig:
    """環境配置"""
    # 觀察空間
    lookback_window: int = 50  # 歷史窗口大小
    feature_columns: List[str] = None  # 使用的特徵列

    # 動作空間
    action_type: str = "discrete"  # discrete 或 continuous
    discrete_actions: int = 5  # 離散動作數量

    # 交易參數
    initial_balance: float = 10000.0
    commission_rate: float = 0.001  # 0.1% 手續費
    slippage_rate: float = 0.0005  # 0.05% 滑點

    # 風險限制
    max_position: float = 1.0  # 最大倉位
    max_drawdown: float = 0.2  # 最大回撤停止

    # 獎勵參數
    reward_scaling: float = 100.0
    sharpe_weight: float = 0.3  # Sharpe ratio 在獎勵中的權重

    def __post_init__(self):
        if self.feature_columns is None:
            # 默認特徵
            self.feature_columns = [
                'returns_1', 'returns_5', 'returns_10',
                'volatility_10', 'volatility_20',
                'rsi_14', 'rsi_7',
                'macd', 'macd_signal', 'macd_histogram',
                'bb_position', 'bb_width',
                'volume_ratio',
                'trend_ema_9_21', 'adx',
                'stoch_k', 'stoch_d',
                'momentum_5', 'momentum_10'
            ]


class TradingEnv(gym.Env):
    """
    加密貨幣交易強化學習環境

    狀態空間：
    - 歷史價格和特徵 (lookback_window x n_features)
    - 當前持倉狀態
    - 未實現盈虧
    - 賬戶餘額比例

    動作空間：
    - 離散: [強力做空, 做空, 持有, 做多, 強力做多]
    - 連續: [-1, 1] 目標倉位

    獎勵：
    - 每步收益率
    - Sharpe ratio 增量
    - 交易成本懲罰
    - 回撤懲罰
    """

    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[TradingEnvConfig] = None,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        self.config = config or TradingEnvConfig()
        self.render_mode = render_mode

        # 數據準備
        self.df = df.copy()
        self._prepare_data()

        # 空間定義
        self._define_spaces()

        # 環境狀態
        self._reset_state()

        logger.info(f"TradingEnv initialized with {len(self.df)} samples")

    def _prepare_data(self):
        """準備數據"""
        # 確保有需要的列
        required_cols = ['close'] + self.config.feature_columns
        missing_cols = [c for c in required_cols if c not in self.df.columns]

        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}, will use zeros")
            for col in missing_cols:
                self.df[col] = 0.0

        # 填充 NaN
        self.df = self.df.fillna(0)

        # 正規化特徵
        self.feature_means = {}
        self.feature_stds = {}

        for col in self.config.feature_columns:
            self.feature_means[col] = self.df[col].mean()
            self.feature_stds[col] = self.df[col].std() + 1e-10

        # 價格數據
        self.prices = self.df['close'].values
        self.n_samples = len(self.prices)

    def _define_spaces(self):
        """定義觀察和動作空間"""
        n_features = len(self.config.feature_columns)

        # 觀察空間
        # [特徵窗口 (lookback x features) + 賬戶狀態 (position, pnl, balance_ratio)]
        obs_shape = (self.config.lookback_window * n_features + 3,)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )

        # 動作空間
        if self.config.action_type == "discrete":
            self.action_space = spaces.Discrete(self.config.discrete_actions)
            # 動作映射: 0=強力做空, 1=做空, 2=持有, 3=做多, 4=強力做多
            self.action_to_position = {
                0: -1.0,   # 強力做空
                1: -0.5,   # 做空
                2: 0.0,    # 持有
                3: 0.5,    # 做多
                4: 1.0     # 強力做多
            }
        else:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            )

    def _reset_state(self):
        """重置環境狀態"""
        self.current_step = self.config.lookback_window
        self.position = 0.0  # 當前倉位 (-1 到 1)
        self.entry_price = 0.0  # 進場價格
        self.balance = self.config.initial_balance
        self.portfolio_value = self.config.initial_balance
        self.max_portfolio_value = self.config.initial_balance

        # 績效追蹤
        self.returns_history = []
        self.portfolio_history = [self.config.initial_balance]
        self.position_history = []
        self.trade_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """重置環境"""
        super().reset(seed=seed)

        self._reset_state()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        執行一步

        Args:
            action: 動作（離散或連續）

        Returns:
            observation, reward, terminated, truncated, info
        """
        # 解析動作
        target_position = self._parse_action(action)

        # 當前和下一個價格
        current_price = self.prices[self.current_step]
        next_price = self.prices[self.current_step + 1] if self.current_step + 1 < self.n_samples else current_price

        # 執行交易
        reward = self._execute_trade(target_position, current_price, next_price)

        # 更新步數
        self.current_step += 1

        # 檢查終止條件
        terminated = self._check_termination()
        truncated = self.current_step >= self.n_samples - 1

        # 獲取新觀察
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _parse_action(self, action: Any) -> float:
        """解析動作為目標倉位"""
        if self.config.action_type == "discrete":
            return self.action_to_position.get(int(action), 0.0)
        else:
            return float(np.clip(action, -1.0, 1.0))

    def _execute_trade(
        self,
        target_position: float,
        current_price: float,
        next_price: float
    ) -> float:
        """
        執行交易並計算獎勵

        Args:
            target_position: 目標倉位
            current_price: 當前價格
            next_price: 下一個價格

        Returns:
            獎勵值
        """
        # 倉位變化
        position_change = target_position - self.position

        # 交易成本
        trade_cost = abs(position_change) * current_price * self.balance * (
            self.config.commission_rate + self.config.slippage_rate
        )

        # 更新倉位
        if position_change != 0:
            self.trade_count += 1
            if target_position != 0 and self.position == 0:
                self.entry_price = current_price

        old_position = self.position
        self.position = target_position

        # 計算收益
        price_return = (next_price - current_price) / current_price

        # 持倉盈虧（使用舊倉位計算當期收益）
        position_pnl = old_position * price_return * self.portfolio_value

        # 更新組合價值
        self.portfolio_value = self.portfolio_value + position_pnl - trade_cost
        self.balance = self.portfolio_value  # 簡化：餘額等於組合價值

        # 更新最高點
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)

        # 記錄
        step_return = (self.portfolio_value - self.portfolio_history[-1]) / self.portfolio_history[-1]
        self.returns_history.append(step_return)
        self.portfolio_history.append(self.portfolio_value)
        self.position_history.append(self.position)

        # 計算獎勵
        reward = self._calculate_reward(step_return, trade_cost, position_change)

        return reward

    def _calculate_reward(
        self,
        step_return: float,
        trade_cost: float,
        position_change: float
    ) -> float:
        """
        計算獎勵

        獎勵組成：
        1. 收益獎勵（主要）
        2. Sharpe ratio 獎勵（風險調整）
        3. 交易成本懲罰
        4. 回撤懲罰
        """
        reward = 0.0

        # 1. 收益獎勵
        reward += step_return * self.config.reward_scaling

        # 2. Sharpe ratio 獎勵（如果有足夠歷史）
        if len(self.returns_history) > 10:
            recent_returns = np.array(self.returns_history[-20:])
            sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-10)
            reward += sharpe * self.config.sharpe_weight

        # 3. 交易成本懲罰
        reward -= trade_cost / self.config.initial_balance * 10

        # 4. 回撤懲罰
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        if current_drawdown > 0.1:  # 超過 10% 回撤
            reward -= current_drawdown * 5

        # 5. 過度交易懲罰
        if abs(position_change) > 0.5:  # 大幅度調倉
            reward -= 0.1

        return float(reward)

    def _check_termination(self) -> bool:
        """檢查是否終止"""
        # 破產
        if self.portfolio_value <= self.config.initial_balance * 0.5:
            return True

        # 最大回撤
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        if current_drawdown >= self.config.max_drawdown:
            return True

        return False

    def _get_observation(self) -> np.ndarray:
        """獲取當前觀察"""
        # 特徵窗口
        start_idx = max(0, self.current_step - self.config.lookback_window)
        end_idx = self.current_step

        features = []
        for col in self.config.feature_columns:
            col_data = self.df[col].values[start_idx:end_idx]
            # 正規化
            normalized = (col_data - self.feature_means[col]) / self.feature_stds[col]
            features.extend(normalized)

        # 填充不足的部分
        expected_len = self.config.lookback_window * len(self.config.feature_columns)
        if len(features) < expected_len:
            features = [0.0] * (expected_len - len(features)) + features

        # 賬戶狀態
        balance_ratio = self.portfolio_value / self.config.initial_balance - 1
        unrealized_pnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            current_price = self.prices[self.current_step]
            unrealized_pnl = self.position * (current_price - self.entry_price) / self.entry_price

        account_state = [self.position, unrealized_pnl, balance_ratio]

        # 組合觀察
        obs = np.array(features + account_state, dtype=np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """獲取額外信息"""
        return {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'total_return': (self.portfolio_value - self.config.initial_balance) / self.config.initial_balance,
            'trade_count': self.trade_count,
            'max_drawdown': (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value,
            'current_price': self.prices[self.current_step] if self.current_step < len(self.prices) else 0
        }

    def render(self):
        """渲染環境"""
        if self.render_mode == 'human':
            info = self._get_info()
            print(f"Step: {info['step']}, "
                  f"Portfolio: ${info['portfolio_value']:.2f}, "
                  f"Position: {info['position']:.2f}, "
                  f"Return: {info['total_return']*100:.2f}%, "
                  f"Trades: {info['trade_count']}")

    def get_performance_summary(self) -> Dict[str, float]:
        """獲取績效摘要"""
        if not self.returns_history:
            return {}

        returns = np.array(self.returns_history)
        portfolio = np.array(self.portfolio_history)

        # 計算最大回撤
        running_max = np.maximum.accumulate(portfolio)
        drawdowns = (running_max - portfolio) / running_max
        max_drawdown = np.max(drawdowns)

        return {
            'total_return': (self.portfolio_value - self.config.initial_balance) / self.config.initial_balance,
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 24),  # 年化
            'max_drawdown': max_drawdown,
            'win_rate': np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0,
            'total_trades': self.trade_count,
            'avg_return_per_trade': np.mean(returns) if len(returns) > 0 else 0,
            'final_portfolio': self.portfolio_value,
            'profit_factor': np.sum(returns[returns > 0]) / (abs(np.sum(returns[returns < 0])) + 1e-10)
        }


def create_trading_env(
    df: pd.DataFrame,
    lookback: int = 50,
    initial_balance: float = 10000.0
) -> TradingEnv:
    """
    創建交易環境的便捷函數

    Args:
        df: 包含特徵的 DataFrame
        lookback: 歷史窗口大小
        initial_balance: 初始資金

    Returns:
        TradingEnv 實例
    """
    config = TradingEnvConfig(
        lookback_window=lookback,
        initial_balance=initial_balance
    )
    return TradingEnv(df, config)
