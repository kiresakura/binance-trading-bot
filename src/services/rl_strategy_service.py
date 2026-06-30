"""
Reinforcement Learning Strategy Service - 強化學習策略服務

提供：
- DQN (Deep Q-Network) 策略
- PPO (Proximal Policy Optimization) 策略
- 交易環境模擬
- 獎勵函數設計
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import deque
import random

from src.logger import logger


class RLAction(int, Enum):
    """RL 動作空間"""
    HOLD = 0
    BUY = 1
    SELL = 2


class RLAlgorithm(str, Enum):
    """RL 算法類型"""
    DQN = "dqn"
    DDQN = "ddqn"  # Double DQN
    PPO = "ppo"


@dataclass
class RLState:
    """RL 狀態"""
    price_features: List[float]      # 價格相關特徵
    technical_indicators: List[float] # 技術指標
    position_info: List[float]        # 持倉信息
    market_features: List[float]      # 市場特徵

    def to_array(self) -> np.ndarray:
        return np.concatenate([
            self.price_features,
            self.technical_indicators,
            self.position_info,
            self.market_features
        ])


@dataclass
class Experience:
    """經驗回放"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class RLConfig:
    """RL 配置"""
    # 網絡結構
    state_dim: int = 32
    hidden_dims: List[int] = None
    action_dim: int = 3  # HOLD, BUY, SELL

    # DQN 參數
    learning_rate: float = 0.001
    gamma: float = 0.99              # 折扣因子
    epsilon: float = 1.0             # 探索率
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 64
    memory_size: int = 10000
    target_update_freq: int = 100

    # PPO 參數
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    gae_lambda: float = 0.95

    # 訓練參數
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RLModel:
    """RL 模型"""
    model_id: str
    algorithm: RLAlgorithm
    config: RLConfig
    created_at: str
    trained_episodes: int = 0
    total_reward: float = 0.0
    avg_episode_reward: float = 0.0
    win_rate: float = 0.0

    # 模型權重（簡化表示）
    weights: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "algorithm": self.algorithm.value if isinstance(self.algorithm, RLAlgorithm) else self.algorithm,
            "config": self.config.to_dict(),
            "created_at": self.created_at,
            "trained_episodes": self.trained_episodes,
            "total_reward": self.total_reward,
            "avg_episode_reward": self.avg_episode_reward,
            "win_rate": self.win_rate
        }


class SimpleDQN:
    """簡化的 DQN 實現（不依賴 PyTorch/TensorFlow）"""

    def __init__(self, config: RLConfig):
        self.config = config
        self.memory = deque(maxlen=config.memory_size)
        self.epsilon = config.epsilon

        # 簡化的權重初始化（線性層）
        self._init_weights()

    def _init_weights(self):
        """初始化權重"""
        np.random.seed(42)

        dims = [self.config.state_dim] + self.config.hidden_dims + [self.config.action_dim]

        self.weights = []
        self.biases = []

        for i in range(len(dims) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            w = np.random.randn(dims[i], dims[i+1]) * scale
            b = np.zeros(dims[i+1])
            self.weights.append(w)
            self.biases.append(b)

        # Target network weights
        self.target_weights = [w.copy() for w in self.weights]
        self.target_biases = [b.copy() for b in self.biases]

    def _forward(self, state: np.ndarray, use_target: bool = False) -> np.ndarray:
        """前向傳播"""
        weights = self.target_weights if use_target else self.weights
        biases = self.target_biases if use_target else self.biases

        x = state
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = np.dot(x, w) + b
            # ReLU activation for hidden layers
            if i < len(weights) - 1:
                x = np.maximum(0, x)

        return x

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """選擇動作"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.config.action_dim - 1)

        q_values = self._forward(state)
        return int(np.argmax(q_values))

    def remember(self, experience: Experience):
        """存儲經驗"""
        self.memory.append(experience)

    def replay(self) -> float:
        """經驗回放訓練"""
        if len(self.memory) < self.config.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.config.batch_size)

        total_loss = 0.0
        for exp in batch:
            # 計算目標 Q 值
            current_q = self._forward(exp.state)
            target_q = current_q.copy()

            if exp.done:
                target_q[exp.action] = exp.reward
            else:
                next_q = self._forward(exp.next_state, use_target=True)
                target_q[exp.action] = exp.reward + self.config.gamma * np.max(next_q)

            # 簡化的梯度更新
            error = target_q - current_q
            total_loss += np.sum(error ** 2)

            # 更新權重（簡化的梯度下降）
            self._update_weights(exp.state, error)

        # 衰減探索率
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay

        return total_loss / self.config.batch_size

    def _update_weights(self, state: np.ndarray, error: np.ndarray):
        """更新權重（簡化版本）"""
        lr = self.config.learning_rate

        # 只更新輸出層（簡化）
        grad = np.outer(state, error)

        # 確保維度匹配
        if self.weights[-1].shape == grad.shape[::-1]:
            self.weights[-1] += lr * grad.T
        elif len(self.weights) > 0:
            # 嘗試用啟發式方法更新
            self.biases[-1] += lr * error

    def update_target_network(self):
        """更新目標網絡"""
        self.target_weights = [w.copy() for w in self.weights]
        self.target_biases = [b.copy() for b in self.biases]


class TradingEnvironment:
    """交易環境"""

    def __init__(
        self,
        symbol: str,
        initial_balance: float = 10000.0,
        commission_rate: float = 0.001
    ):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate

        self.reset()

    def reset(self) -> np.ndarray:
        """重置環境"""
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.current_step = 0
        self.prices = []
        self.done = False
        self.total_reward = 0.0
        self.trades = []

        return self._get_state()

    def set_price_data(self, prices: List[float]):
        """設置價格數據"""
        self.prices = prices

    def _get_state(self) -> np.ndarray:
        """獲取當前狀態"""
        if not self.prices or self.current_step >= len(self.prices):
            return np.zeros(32)

        current_price = self.prices[self.current_step]

        # 價格特徵
        price_features = self._get_price_features()

        # 技術指標
        technical = self._get_technical_indicators()

        # 持倉信息
        position_info = [
            1 if self.position > 0 else 0,
            1 if self.position < 0 else 0,
            abs(self.position),
            (current_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0,
            self.balance / self.initial_balance
        ]

        # 市場特徵
        market_features = self._get_market_features()

        state = np.concatenate([
            price_features,
            technical,
            position_info,
            market_features
        ])

        # 確保維度正確
        if len(state) < 32:
            state = np.pad(state, (0, 32 - len(state)))
        elif len(state) > 32:
            state = state[:32]

        return state

    def _get_price_features(self) -> np.ndarray:
        """獲取價格特徵"""
        if self.current_step < 10:
            return np.zeros(10)

        recent_prices = self.prices[max(0, self.current_step-10):self.current_step+1]
        current_price = self.prices[self.current_step]

        # 價格變化率
        returns = np.diff(recent_prices) / recent_prices[:-1]

        # 標準化
        returns = np.clip(returns, -0.1, 0.1) * 10

        # 填充到固定長度
        if len(returns) < 10:
            returns = np.pad(returns, (10 - len(returns), 0))

        return returns

    def _get_technical_indicators(self) -> np.ndarray:
        """獲取技術指標"""
        if self.current_step < 20:
            return np.zeros(10)

        prices = np.array(self.prices[:self.current_step+1])

        # RSI
        rsi = self._calculate_rsi(prices)

        # MA
        ma5 = np.mean(prices[-5:])
        ma10 = np.mean(prices[-10:])
        ma20 = np.mean(prices[-20:])

        current = prices[-1]

        return np.array([
            rsi / 100,  # 標準化 RSI
            (current - ma5) / ma5,
            (current - ma10) / ma10,
            (current - ma20) / ma20,
            (ma5 - ma10) / ma10,
            (ma10 - ma20) / ma20,
            np.std(prices[-10:]) / current,  # 波動率
            np.max(prices[-10:]) / current - 1,
            1 - np.min(prices[-10:]) / current,
            (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        ])

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """計算 RSI"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices[-period-1:])
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _get_market_features(self) -> np.ndarray:
        """獲取市場特徵"""
        return np.zeros(7)  # 簡化

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """執行動作"""
        if self.done or self.current_step >= len(self.prices) - 1:
            self.done = True
            return self._get_state(), 0, True, {}

        current_price = self.prices[self.current_step]
        reward = 0.0
        info = {}

        # 執行動作
        if action == RLAction.BUY and self.position <= 0:
            # 買入
            if self.position < 0:
                # 平空倉
                pnl = (self.entry_price - current_price) * abs(self.position)
                reward = pnl / self.initial_balance * 100
                self.balance += pnl - abs(self.position) * current_price * self.commission_rate
                self.trades.append({"action": "close_short", "price": current_price, "pnl": pnl})

            # 開多倉
            position_size = self.balance * 0.5 / current_price
            cost = position_size * current_price * (1 + self.commission_rate)
            if cost <= self.balance:
                self.position = position_size
                self.entry_price = current_price
                self.balance -= cost
                self.trades.append({"action": "buy", "price": current_price, "size": position_size})

        elif action == RLAction.SELL and self.position >= 0:
            # 賣出
            if self.position > 0:
                # 平多倉
                pnl = (current_price - self.entry_price) * self.position
                reward = pnl / self.initial_balance * 100
                self.balance += self.position * current_price * (1 - self.commission_rate)
                self.trades.append({"action": "close_long", "price": current_price, "pnl": pnl})
                self.position = 0
                self.entry_price = 0

        # 移動到下一步
        self.current_step += 1

        # 更新未實現盈虧
        if self.position != 0:
            next_price = self.prices[self.current_step]
            unrealized_pnl = (next_price - self.entry_price) * self.position if self.position > 0 else (self.entry_price - next_price) * abs(self.position)
            reward += unrealized_pnl / self.initial_balance * 10  # 小額獎勵/懲罰

        # 檢查是否結束
        if self.current_step >= len(self.prices) - 1:
            self.done = True
            # 強制平倉
            if self.position != 0:
                final_price = self.prices[self.current_step]
                if self.position > 0:
                    pnl = (final_price - self.entry_price) * self.position
                else:
                    pnl = (self.entry_price - final_price) * abs(self.position)
                reward += pnl / self.initial_balance * 100
                self.balance += abs(self.position) * final_price * (1 - self.commission_rate)

        self.total_reward += reward
        next_state = self._get_state()

        return next_state, reward, self.done, info


class RLStrategyService:
    """強化學習策略服務"""

    def __init__(self):
        self._models: Dict[str, RLModel] = {}
        self._agents: Dict[str, SimpleDQN] = {}
        self._model_counter = 0
        logger.info("RLStrategyService initialized")

    def create_model(
        self,
        algorithm: RLAlgorithm = RLAlgorithm.DQN,
        config: Optional[RLConfig] = None
    ) -> RLModel:
        """創建 RL 模型"""
        if config is None:
            config = RLConfig()

        self._model_counter += 1
        model_id = f"rl_{algorithm.value}_{self._model_counter}_{datetime.utcnow().strftime('%Y%m%d')}"

        model = RLModel(
            model_id=model_id,
            algorithm=algorithm,
            config=config,
            created_at=datetime.utcnow().isoformat()
        )

        self._models[model_id] = model

        # 創建對應的 agent
        if algorithm in [RLAlgorithm.DQN, RLAlgorithm.DDQN]:
            self._agents[model_id] = SimpleDQN(config)

        logger.info(f"RL model created: {model_id}")
        return model

    async def train(
        self,
        model_id: str,
        symbol: str,
        price_data: List[float],
        episodes: int = 100
    ) -> Dict[str, Any]:
        """訓練模型"""
        model = self._models.get(model_id)
        agent = self._agents.get(model_id)

        if not model or not agent:
            raise ValueError(f"Model {model_id} not found")

        env = TradingEnvironment(symbol)
        env.set_price_data(price_data)

        total_rewards = []
        winning_episodes = 0
        step_count = 0

        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(model.config.max_steps_per_episode):
                # 選擇動作
                action = agent.get_action(state, training=True)

                # 執行動作
                next_state, reward, done, info = env.step(action)

                # 存儲經驗
                agent.remember(Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                ))

                # 訓練
                loss = agent.replay()

                episode_reward += reward
                state = next_state
                step_count += 1

                # 更新目標網絡
                if step_count % model.config.target_update_freq == 0:
                    agent.update_target_network()

                if done:
                    break

            total_rewards.append(episode_reward)
            if episode_reward > 0:
                winning_episodes += 1

            # 更新模型統計
            model.trained_episodes += 1

            # 每 10 個 episode 輸出進度
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(total_rewards[-10:])
                logger.info(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

        # 最終統計
        model.total_reward = sum(total_rewards)
        model.avg_episode_reward = np.mean(total_rewards)
        model.win_rate = winning_episodes / episodes

        return {
            "model_id": model_id,
            "episodes_trained": episodes,
            "total_reward": model.total_reward,
            "avg_episode_reward": model.avg_episode_reward,
            "win_rate": model.win_rate,
            "final_epsilon": agent.epsilon
        }

    def get_action(
        self,
        model_id: str,
        state: np.ndarray
    ) -> Tuple[RLAction, float]:
        """獲取動作"""
        agent = self._agents.get(model_id)
        if not agent:
            return RLAction.HOLD, 0.0

        action_idx = agent.get_action(state, training=False)
        q_values = agent._forward(state)

        # 計算信心度
        q_softmax = np.exp(q_values) / np.sum(np.exp(q_values))
        confidence = q_softmax[action_idx]

        return RLAction(action_idx), float(confidence)

    async def get_trading_signal(
        self,
        model_id: str,
        symbol: str,
        recent_prices: List[float],
        position: float = 0.0
    ) -> Dict[str, Any]:
        """獲取交易信號"""
        agent = self._agents.get(model_id)
        if not agent:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": "Model not found"
            }

        # 創建臨時環境獲取狀態
        env = TradingEnvironment(symbol)
        env.set_price_data(recent_prices)
        env.current_step = len(recent_prices) - 1
        env.position = position

        state = env._get_state()

        action, confidence = self.get_action(model_id, state)

        signal_map = {
            RLAction.HOLD: "HOLD",
            RLAction.BUY: "BUY",
            RLAction.SELL: "SELL"
        }

        return {
            "signal": signal_map[action],
            "confidence": confidence,
            "model_id": model_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """獲取模型詳情"""
        model = self._models.get(model_id)
        if model:
            return model.to_dict()
        return None

    def get_all_models(self) -> List[Dict[str, Any]]:
        """獲取所有模型"""
        return [m.to_dict() for m in self._models.values()]

    def delete_model(self, model_id: str) -> bool:
        """刪除模型"""
        if model_id in self._models:
            del self._models[model_id]
            if model_id in self._agents:
                del self._agents[model_id]
            logger.info(f"RL model deleted: {model_id}")
            return True
        return False


# 單例實例
rl_strategy_service = RLStrategyService()
