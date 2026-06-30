"""
剝頭皮策略

高頻短線交易策略
- 使用快速指標捕捉微小價格波動
- 小止損小止盈，高頻交易

v0.4.0 - Enterprise Edition
"""

from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from src.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from src.strategies.indicators import ema, rsi, stoch_rsi, atr, bollinger_bands, sma
from src.strategies.strategy_manager import StrategyMetadata, StrategyCategory
from src.logger import logger


class ScalpingStrategy(BaseStrategy):
    """剝頭皮策略"""

    METADATA = StrategyMetadata(
        name='scalping',
        display_name='剝頭皮策略',
        category=StrategyCategory.SCALPING,
        description='高頻短線交易策略，適合波動性市場',
        author='Freqtrade Community',
        version='1.0.0',
        timeframe='5m',
        min_candles=50,
        tags=['scalping', 'high_frequency', 'short_term'],
        risk_level=4,
        recommended_pairs=['BTCUSDT', 'ETHUSDT'],
        hyperopt_enabled=True
    )

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="scalping",
                enabled=True,
                weight=1.0,
                max_position_size=0.1,
                stop_loss_pct=0.008,  # 0.8% 止損
                take_profit_pct=0.012,  # 1.2% 止盈
                min_signal_strength=0.5,
                min_confidence=0.55,
                cooldown_minutes=3,  # 3 分鐘冷卻
                params={
                    'ema_fast': 5,
                    'ema_medium': 13,
                    'ema_slow': 21,
                    'rsi_period': 7,
                    'rsi_oversold': 25,
                    'rsi_overbought': 75,
                    'stoch_period': 14,
                    'stoch_smooth': 3,
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'min_volatility': 0.002,  # 最小波動率
                }
            )
        super().__init__(config)

    @classmethod
    def get_hyperopt_space(cls) -> Dict[str, Any]:
        return {
            'ema_fast': {'type': 'int', 'low': 3, 'high': 8},
            'ema_medium': {'type': 'int', 'low': 8, 'high': 15},
            'ema_slow': {'type': 'int', 'low': 15, 'high': 30},
            'rsi_period': {'type': 'int', 'low': 5, 'high': 10},
            'rsi_oversold': {'type': 'int', 'low': 20, 'high': 35},
            'rsi_overbought': {'type': 'int', 'low': 65, 'high': 80},
        }

    def populate_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params

        # EMA 系統
        dataframe['ema_fast'] = ema(dataframe['close'], params.get('ema_fast', 5))
        dataframe['ema_medium'] = ema(dataframe['close'], params.get('ema_medium', 13))
        dataframe['ema_slow'] = ema(dataframe['close'], params.get('ema_slow', 21))

        # RSI
        dataframe['rsi'] = rsi(dataframe['close'], params.get('rsi_period', 7))

        # Stochastic RSI
        dataframe['stoch_k'], dataframe['stoch_d'] = stoch_rsi(
            dataframe['close'],
            params.get('stoch_period', 14),
            params.get('stoch_smooth', 3),
            params.get('stoch_smooth', 3)
        )

        # 布林帶
        dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = bollinger_bands(
            dataframe['close'],
            params.get('bb_period', 20),
            params.get('bb_std', 2.0)
        )
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']

        # ATR
        dataframe['atr'] = atr(dataframe, 14)

        # 波動率
        dataframe['volatility'] = dataframe['close'].pct_change().rolling(20).std()

        # 成交量
        dataframe['volume_sma'] = sma(dataframe['volume'], 20)

        # EMA 排列
        dataframe['ema_bullish'] = (
            (dataframe['ema_fast'] > dataframe['ema_medium']) &
            (dataframe['ema_medium'] > dataframe['ema_slow'])
        )
        dataframe['ema_bearish'] = (
            (dataframe['ema_fast'] < dataframe['ema_medium']) &
            (dataframe['ema_medium'] < dataframe['ema_slow'])
        )

        return dataframe

    def populate_entry_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params
        rsi_oversold = params.get('rsi_oversold', 25)
        rsi_overbought = params.get('rsi_overbought', 75)
        min_volatility = params.get('min_volatility', 0.002)

        # 做多條件
        long_condition = (
            dataframe['ema_bullish'] &
            (dataframe['rsi'] < 40) &
            (dataframe['stoch_k'] < 30) &
            (dataframe['close'] < dataframe['bb_middle']) &
            (dataframe['volatility'] > min_volatility) &
            (dataframe['volume'] > dataframe['volume_sma'] * 0.8)
        )

        # 做空條件
        short_condition = (
            dataframe['ema_bearish'] &
            (dataframe['rsi'] > 60) &
            (dataframe['stoch_k'] > 70) &
            (dataframe['close'] > dataframe['bb_middle']) &
            (dataframe['volatility'] > min_volatility) &
            (dataframe['volume'] > dataframe['volume_sma'] * 0.8)
        )

        dataframe['enter_long'] = long_condition.astype(int)
        dataframe['enter_short'] = short_condition.astype(int)

        return dataframe

    def populate_exit_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params
        rsi_overbought = params.get('rsi_overbought', 75)
        rsi_oversold = params.get('rsi_oversold', 25)

        # 做多出場
        dataframe['exit_long'] = (
            (dataframe['rsi'] > rsi_overbought) |
            (dataframe['stoch_k'] > 80) |
            (dataframe['close'] > dataframe['bb_upper'])
        ).astype(int)

        # 做空出場
        dataframe['exit_short'] = (
            (dataframe['rsi'] < rsi_oversold) |
            (dataframe['stoch_k'] < 20) |
            (dataframe['close'] < dataframe['bb_lower'])
        ).astype(int)

        return dataframe

    def generate_signal(
        self,
        symbol: str,
        features: pd.DataFrame,
        current_price: float,
        position: float = 0.0
    ) -> Optional[Signal]:

        if not self.should_generate_signal():
            return None

        if len(features) < 50:
            return None

        try:
            df = self.analyze(features.copy())
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest

            direction = 0
            strength = 0.0
            confidence = 0.0
            reasons = []

            rsi_val = latest.get('rsi', 50)
            stoch_k = latest.get('stoch_k', 50)
            stoch_d = latest.get('stoch_d', 50)
            bb_middle = latest.get('bb_middle', current_price)
            volatility = latest.get('volatility', 0.01)
            volume_ratio = latest['volume'] / (latest.get('volume_sma', latest['volume']) + 1e-10)

            # 做多信號
            if latest.get('enter_long', 0) == 1:
                direction = 1
                strength = 0.55
                confidence = 0.6

                reasons.append("EMA多頭排列")

                # RSI 超賣程度
                if rsi_val < 25:
                    strength += 0.15
                    reasons.append(f"RSI極度超賣({rsi_val:.0f})")
                else:
                    reasons.append(f"RSI偏低({rsi_val:.0f})")

                # Stochastic 確認
                if stoch_k < 20 and stoch_k > stoch_d:
                    strength += 0.1
                    confidence += 0.1
                    reasons.append(f"Stoch金叉({stoch_k:.0f})")

                # 成交量
                if volume_ratio > 1.2:
                    strength += 0.1
                    reasons.append(f"放量({volume_ratio:.1f}x)")

            # 做空信號
            elif latest.get('enter_short', 0) == 1:
                direction = -1
                strength = 0.55
                confidence = 0.6

                reasons.append("EMA空頭排列")

                if rsi_val > 75:
                    strength += 0.15
                    reasons.append(f"RSI極度超買({rsi_val:.0f})")
                else:
                    reasons.append(f"RSI偏高({rsi_val:.0f})")

                if stoch_k > 80 and stoch_k < stoch_d:
                    strength += 0.1
                    confidence += 0.1
                    reasons.append(f"Stoch死叉({stoch_k:.0f})")

                if volume_ratio > 1.2:
                    strength += 0.1
                    reasons.append(f"放量({volume_ratio:.1f}x)")

            # 平倉信號
            elif position != 0:
                if position > 0 and latest.get('exit_long', 0) == 1:
                    direction = -1
                    strength = 0.65
                    confidence = 0.75
                    reasons.append("剝頭皮止盈平多")
                elif position < 0 and latest.get('exit_short', 0) == 1:
                    direction = 1
                    strength = 0.65
                    confidence = 0.75
                    reasons.append("剝頭皮止盈平空")

            if direction == 0:
                return None

            strength = min(1.0, max(0.0, strength))
            confidence = min(1.0, max(0.0, confidence))

            if strength < self.config.min_signal_strength:
                return None

            if direction > 0:
                signal_type = SignalType.STRONG_BUY if strength > 0.7 else SignalType.BUY
            else:
                signal_type = SignalType.STRONG_SELL if strength > 0.7 else SignalType.SELL

            # 剝頭皮使用緊密的止損止盈
            atr_val = latest.get('atr', current_price * 0.01)
            if direction > 0:
                stop_loss = current_price - atr_val * 1.2
                take_profit = current_price + atr_val * 1.5
            else:
                stop_loss = current_price + atr_val * 1.2
                take_profit = current_price - atr_val * 1.5

            signal = Signal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                signal_type=signal_type,
                direction=direction,
                strength=strength,
                confidence=confidence,
                strategy_name=self.name,
                suggested_size=self.config.max_position_size * strength * confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                features={
                    'rsi': float(rsi_val),
                    'stoch_k': float(stoch_k),
                    'stoch_d': float(stoch_d),
                    'volatility': float(volatility),
                    'volume_ratio': float(volume_ratio),
                },
                reason=" | ".join(reasons)
            )

            self.record_signal(signal)
            logger.info(f"[Scalping] {symbol}: {signal_type.value} "
                       f"strength={strength:.2f} conf={confidence:.2f} | {signal.reason}")

            return signal

        except Exception as e:
            logger.error(f"Scalping error: {e}")
            return None

    def update(self, market_data: Dict[str, Any]):
        pass
