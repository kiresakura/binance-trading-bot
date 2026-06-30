"""
BB + RSI 策略

經典的布林帶與 RSI 結合策略
- 價格觸及布林帶下軌 + RSI 超賣 = 做多
- 價格觸及布林帶上軌 + RSI 超買 = 做空

適合震盪市場

v0.4.0 - Enterprise Edition
"""

from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from src.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from src.strategies.indicators import bollinger_bands, rsi, atr, sma
from src.strategies.strategy_manager import StrategyMetadata, StrategyCategory
from src.logger import logger


class BBRSIStrategy(BaseStrategy):
    """布林帶 + RSI 策略"""

    METADATA = StrategyMetadata(
        name='bb_rsi',
        display_name='BB + RSI 策略',
        category=StrategyCategory.MEAN_REVERSION,
        description='結合布林帶和 RSI 指標的經典均值回歸策略，適合震盪市場',
        author='Freqtrade Community',
        version='1.0.0',
        timeframe='1h',
        min_candles=50,
        tags=['bollinger', 'rsi', 'mean_reversion', 'classic'],
        risk_level=2,
        recommended_pairs=['BTCUSDT', 'ETHUSDT'],
        hyperopt_enabled=True
    )

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="bb_rsi",
                enabled=True,
                weight=1.0,
                max_position_size=0.15,
                stop_loss_pct=0.025,
                take_profit_pct=0.04,
                min_signal_strength=0.4,
                min_confidence=0.5,
                cooldown_minutes=30,
                params={
                    # 布林帶參數
                    'bb_period': 20,
                    'bb_std': 2.0,
                    # RSI 參數
                    'rsi_period': 14,
                    'rsi_oversold': 30,
                    'rsi_overbought': 70,
                    # 信號確認
                    'require_volume': True,
                    'volume_factor': 1.2,
                }
            )
        super().__init__(config)

    @classmethod
    def get_hyperopt_space(cls) -> Dict[str, Any]:
        """Hyperopt 搜索空間"""
        return {
            'bb_period': {'type': 'int', 'low': 10, 'high': 50},
            'bb_std': {'type': 'float', 'low': 1.5, 'high': 3.0, 'step': 0.1},
            'rsi_period': {'type': 'int', 'low': 7, 'high': 21},
            'rsi_oversold': {'type': 'int', 'low': 20, 'high': 40},
            'rsi_overbought': {'type': 'int', 'low': 60, 'high': 80},
            'volume_factor': {'type': 'float', 'low': 1.0, 'high': 2.0, 'step': 0.1},
        }

    def populate_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """填充指標"""
        params = self.config.params

        # 布林帶
        dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = bollinger_bands(
            dataframe['close'],
            params.get('bb_period', 20),
            params.get('bb_std', 2.0)
        )

        # 布林帶寬度和位置
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lower']) / (
            dataframe['bb_upper'] - dataframe['bb_lower'] + 1e-10
        )

        # RSI
        dataframe['rsi'] = rsi(dataframe['close'], params.get('rsi_period', 14))

        # ATR
        dataframe['atr'] = atr(dataframe, 14)

        # 成交量 SMA
        dataframe['volume_sma'] = sma(dataframe['volume'], 20)

        return dataframe

    def populate_entry_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """入場信號"""
        params = self.config.params
        rsi_oversold = params.get('rsi_oversold', 30)
        rsi_overbought = params.get('rsi_overbought', 70)
        volume_factor = params.get('volume_factor', 1.2)
        require_volume = params.get('require_volume', True)

        # 做多條件
        long_condition = (
            (dataframe['close'] <= dataframe['bb_lower']) &
            (dataframe['rsi'] < rsi_oversold)
        )
        if require_volume:
            long_condition = long_condition & (
                dataframe['volume'] > dataframe['volume_sma'] * volume_factor
            )

        # 做空條件
        short_condition = (
            (dataframe['close'] >= dataframe['bb_upper']) &
            (dataframe['rsi'] > rsi_overbought)
        )
        if require_volume:
            short_condition = short_condition & (
                dataframe['volume'] > dataframe['volume_sma'] * volume_factor
            )

        dataframe['enter_long'] = long_condition.astype(int)
        dataframe['enter_short'] = short_condition.astype(int)

        return dataframe

    def populate_exit_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """出場信號"""
        # 做多出場：回到中軌或 RSI 超買
        dataframe['exit_long'] = (
            (dataframe['close'] >= dataframe['bb_middle']) |
            (dataframe['rsi'] > 70)
        ).astype(int)

        # 做空出場：回到中軌或 RSI 超賣
        dataframe['exit_short'] = (
            (dataframe['close'] <= dataframe['bb_middle']) |
            (dataframe['rsi'] < 30)
        ).astype(int)

        return dataframe

    def generate_signal(
        self,
        symbol: str,
        features: pd.DataFrame,
        current_price: float,
        position: float = 0.0
    ) -> Optional[Signal]:
        """生成交易信號"""

        if not self.should_generate_signal():
            return None

        if len(features) < 50:
            return None

        try:
            # 分析數據
            df = self.analyze(features.copy())
            latest = df.iloc[-1]

            direction = 0
            strength = 0.0
            confidence = 0.0
            reasons = []

            params = self.config.params
            rsi_oversold = params.get('rsi_oversold', 30)
            rsi_overbought = params.get('rsi_overbought', 70)

            rsi_val = latest.get('rsi', 50)
            bb_percent = latest.get('bb_percent', 0.5)
            volume_ratio = latest['volume'] / (latest.get('volume_sma', latest['volume']) + 1e-10)

            # 做多信號
            if latest.get('enter_long', 0) == 1:
                direction = 1
                strength = 0.5

                # RSI 越低，信號越強
                rsi_strength = (rsi_oversold - rsi_val) / rsi_oversold if rsi_val < rsi_oversold else 0
                strength += rsi_strength * 0.3

                # 布林帶位置
                if bb_percent < 0:
                    strength += 0.2
                    reasons.append(f"跌破下軌(BB%:{bb_percent:.2f})")
                else:
                    reasons.append(f"觸及下軌(BB%:{bb_percent:.2f})")

                reasons.append(f"RSI超賣({rsi_val:.0f})")

                # 成交量確認
                if volume_ratio > 1.5:
                    confidence = 0.7
                    reasons.append(f"放量確認({volume_ratio:.1f}x)")
                else:
                    confidence = 0.55

            # 做空信號
            elif latest.get('enter_short', 0) == 1:
                direction = -1
                strength = 0.5

                # RSI 越高，信號越強
                rsi_strength = (rsi_val - rsi_overbought) / (100 - rsi_overbought) if rsi_val > rsi_overbought else 0
                strength += rsi_strength * 0.3

                # 布林帶位置
                if bb_percent > 1:
                    strength += 0.2
                    reasons.append(f"突破上軌(BB%:{bb_percent:.2f})")
                else:
                    reasons.append(f"觸及上軌(BB%:{bb_percent:.2f})")

                reasons.append(f"RSI超買({rsi_val:.0f})")

                if volume_ratio > 1.5:
                    confidence = 0.7
                    reasons.append(f"放量確認({volume_ratio:.1f}x)")
                else:
                    confidence = 0.55

            # 平倉信號
            elif position != 0:
                if position > 0 and latest.get('exit_long', 0) == 1:
                    direction = -1
                    strength = 0.6
                    confidence = 0.7
                    reasons.append("回歸中軌平多")
                elif position < 0 and latest.get('exit_short', 0) == 1:
                    direction = 1
                    strength = 0.6
                    confidence = 0.7
                    reasons.append("回歸中軌平空")

            if direction == 0:
                return None

            strength = min(1.0, max(0.0, strength))
            confidence = min(1.0, max(0.0, confidence))

            if strength < self.config.min_signal_strength:
                return None

            # 確定信號類型
            if direction > 0:
                signal_type = SignalType.STRONG_BUY if strength > 0.65 else SignalType.BUY
            else:
                signal_type = SignalType.STRONG_SELL if strength > 0.65 else SignalType.SELL

            # 動態止損止盈
            atr_val = latest.get('atr', current_price * 0.02)
            if direction > 0:
                stop_loss = current_price - atr_val * 2
                take_profit = latest.get('bb_middle', current_price * 1.02)
            else:
                stop_loss = current_price + atr_val * 2
                take_profit = latest.get('bb_middle', current_price * 0.98)

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
                    'bb_percent': float(bb_percent),
                    'bb_width': float(latest.get('bb_width', 0)),
                    'volume_ratio': float(volume_ratio),
                },
                reason=" | ".join(reasons)
            )

            self.record_signal(signal)
            logger.info(f"[BB_RSI] {symbol}: {signal_type.value} "
                       f"strength={strength:.2f} conf={confidence:.2f} | {signal.reason}")

            return signal

        except Exception as e:
            logger.error(f"BB_RSI error: {e}")
            return None

    def update(self, market_data: Dict[str, Any]):
        """更新策略狀態"""
        pass
