"""
PPO 強化學習交易代理

使用 Stable-Baselines3 的 PPO 算法
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import pandas as pd

from src.logger import logger
from src.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from src.features.technical import TechnicalFeatures

# 嘗試導入 RL 庫
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    logger.warning("stable_baselines3 not available, PPO agent will be limited")

from .trading_env import TradingEnv, TradingEnvConfig


class TradingCallback(BaseCallback):
    """訓練回調：記錄和評估"""

    def __init__(self, eval_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # 記錄獎勵
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info['r'])
                self.episode_lengths.append(info['l'])

        # 定期評估
        if self.n_calls % self.eval_freq == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])
            logger.info(f"Step {self.n_calls}: Mean reward (last 100 ep) = {mean_reward:.2f}")

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward

        return True


class PPOTradingAgent(BaseStrategy):
    """
    PPO 強化學習交易代理

    使用 PPO 算法學習最優交易策略
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        model_path: Optional[str] = None
    ):
        if config is None:
            config = StrategyConfig(
                name="PPO_Agent",
                params={
                    'learning_rate': 3e-4,
                    'n_steps': 2048,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_range': 0.2,
                    'ent_coef': 0.01,
                    'vf_coef': 0.5,
                    'max_grad_norm': 0.5
                }
            )

        super().__init__(config)

        self.model: Optional['PPO'] = None
        self.env: Optional[TradingEnv] = None
        self.vec_env = None
        self.feature_calculator = TechnicalFeatures()

        # 模型路徑
        self.model_dir = Path("models/ppo")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 如果提供了模型路徑，嘗試加載
        if model_path and RL_AVAILABLE:
            self.load_model(model_path)

        self._is_trained = False

    def train(
        self,
        training_data: pd.DataFrame,
        total_timesteps: int = 100000,
        eval_data: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        訓練 PPO 模型

        Args:
            training_data: 訓練數據（需包含特徵）
            total_timesteps: 總訓練步數
            eval_data: 評估數據（可選）
            save_path: 模型保存路徑

        Returns:
            訓練結果字典
        """
        if not RL_AVAILABLE:
            logger.error("stable_baselines3 not available, cannot train")
            return {'error': 'RL library not available'}

        logger.info(f"Starting PPO training with {len(training_data)} samples, {total_timesteps} timesteps")

        # 計算特徵（如果還沒有）
        if 'rsi_14' not in training_data.columns:
            training_data = self.feature_calculator.calculate_all(training_data)

        # 創建環境
        env_config = TradingEnvConfig(
            lookback_window=50,
            initial_balance=10000.0
        )

        def make_env():
            env = TradingEnv(training_data, env_config)
            env = Monitor(env)
            return env

        self.vec_env = DummyVecEnv([make_env])
        self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True)

        # 創建模型
        params = self.config.params
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=params.get('learning_rate', 3e-4),
            n_steps=params.get('n_steps', 2048),
            batch_size=params.get('batch_size', 64),
            n_epochs=params.get('n_epochs', 10),
            gamma=params.get('gamma', 0.99),
            gae_lambda=params.get('gae_lambda', 0.95),
            clip_range=params.get('clip_range', 0.2),
            ent_coef=params.get('ent_coef', 0.01),
            vf_coef=params.get('vf_coef', 0.5),
            max_grad_norm=params.get('max_grad_norm', 0.5),
            verbose=1,
            tensorboard_log="./logs/tensorboard/"
        )

        # 回調
        callback = TradingCallback(eval_freq=5000)

        # 訓練
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )

        self._is_trained = True

        # 保存模型
        if save_path is None:
            save_path = self.model_dir / f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.model.save(str(save_path))
        self.vec_env.save(f"{save_path}_vecnorm.pkl")
        logger.info(f"Model saved to {save_path}")

        # 評估
        results = self._evaluate(eval_data if eval_data is not None else training_data)

        return {
            'model_path': str(save_path),
            'total_timesteps': total_timesteps,
            'best_mean_reward': callback.best_mean_reward,
            'total_episodes': len(callback.episode_rewards),
            'evaluation': results
        }

    def _evaluate(self, data: pd.DataFrame, n_episodes: int = 5) -> Dict[str, float]:
        """評估模型"""
        if not self._is_trained or self.model is None:
            return {}

        # 計算特徵
        if 'rsi_14' not in data.columns:
            data = self.feature_calculator.calculate_all(data)

        env = TradingEnv(data, TradingEnvConfig())

        all_rewards = []
        all_returns = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                # 正規化觀察
                if self.vec_env is not None:
                    obs = self.vec_env.normalize_obs(obs)

                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward

            all_rewards.append(episode_reward)
            all_returns.append(info.get('total_return', 0))

        return {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_return': np.mean(all_returns),
            'std_return': np.std(all_returns)
        }

    def load_model(self, model_path: str):
        """加載模型"""
        if not RL_AVAILABLE:
            logger.error("stable_baselines3 not available")
            return

        try:
            self.model = PPO.load(model_path)

            # 嘗試加載 VecNormalize
            vecnorm_path = f"{model_path}_vecnorm.pkl"
            if os.path.exists(vecnorm_path):
                # 需要先創建環境
                dummy_data = pd.DataFrame({
                    'open': [100] * 200,
                    'high': [101] * 200,
                    'low': [99] * 200,
                    'close': [100.5] * 200,
                    'volume': [1000] * 200
                })
                dummy_data = self.feature_calculator.calculate_all(dummy_data)

                def make_env():
                    return TradingEnv(dummy_data, TradingEnvConfig())

                self.vec_env = DummyVecEnv([make_env])
                self.vec_env = VecNormalize.load(vecnorm_path, self.vec_env)

            self._is_trained = True
            logger.info(f"Model loaded from {model_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def generate_signal(
        self,
        symbol: str,
        features: pd.DataFrame,
        current_price: float,
        position: float = 0.0
    ) -> Optional[Signal]:
        """
        使用訓練好的模型生成交易信號
        """
        if not self._is_trained or self.model is None:
            logger.warning("Model not trained, cannot generate signal")
            return None

        if not self.should_generate_signal():
            return None

        try:
            # 準備觀察
            obs = self._prepare_observation(features, position, current_price)

            # 正規化
            if self.vec_env is not None:
                obs = self.vec_env.normalize_obs(obs)

            # 預測
            action, _ = self.model.predict(obs, deterministic=True)

            # 解析動作
            action_to_position = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
            target_position = action_to_position.get(int(action), 0.0)

            # 生成信號
            if target_position > 0.3:
                signal_type = SignalType.STRONG_BUY if target_position > 0.7 else SignalType.BUY
                direction = 1
            elif target_position < -0.3:
                signal_type = SignalType.STRONG_SELL if target_position < -0.7 else SignalType.SELL
                direction = -1
            else:
                signal_type = SignalType.HOLD
                direction = 0

            signal = Signal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                signal_type=signal_type,
                direction=direction,
                strength=abs(target_position),
                confidence=0.7,  # PPO 模型置信度
                strategy_name=self.name,
                suggested_size=abs(target_position) * self.config.max_position_size,
                stop_loss=self.calculate_stop_loss(current_price, direction) if direction != 0 else None,
                take_profit=self.calculate_take_profit(current_price, direction) if direction != 0 else None,
                features={'action': int(action), 'target_position': target_position},
                reason=f"PPO model action: {action}, target position: {target_position:.2f}"
            )

            self.record_signal(signal)
            return signal

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None

    def _prepare_observation(
        self,
        features: pd.DataFrame,
        position: float,
        current_price: float
    ) -> np.ndarray:
        """準備觀察向量"""
        feature_cols = TradingEnvConfig().feature_columns

        # 取最後 50 行
        recent = features.tail(50)

        # 提取特徵
        obs_features = []
        for col in feature_cols:
            if col in recent.columns:
                col_data = recent[col].values
            else:
                col_data = np.zeros(len(recent))

            # 簡單正規化
            col_data = (col_data - np.mean(col_data)) / (np.std(col_data) + 1e-10)
            obs_features.extend(col_data)

        # 賬戶狀態（簡化）
        account_state = [position, 0.0, 0.0]  # position, pnl, balance_ratio

        obs = np.array(obs_features + account_state, dtype=np.float32)
        return obs

    def update(self, market_data: Dict[str, Any]):
        """更新策略狀態"""
        # PPO 代理的在線更新
        # 可以在這裡收集經驗用於後續重訓練
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型信息"""
        return {
            'name': self.name,
            'is_trained': self._is_trained,
            'model_type': 'PPO',
            'params': self.config.params,
            'performance': self.get_performance()
        }


# 便捷函數
def train_ppo_agent(
    data: pd.DataFrame,
    timesteps: int = 50000,
    save_path: Optional[str] = None
) -> PPOTradingAgent:
    """
    訓練 PPO 代理的便捷函數

    Args:
        data: OHLCV 數據
        timesteps: 訓練步數
        save_path: 保存路徑

    Returns:
        訓練好的 PPOTradingAgent
    """
    agent = PPOTradingAgent()
    agent.train(data, total_timesteps=timesteps, save_path=save_path)
    return agent
