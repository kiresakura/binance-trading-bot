"""
超級趨勢策略

基於 SuperTrend 指標的趨勢跟蹤策略
- SuperTrend 翻綠 = 做多
- SuperTrend 翻紅 = 做空

v0.4.0 - Enterprise Edition
"""

from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from src.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from src.strategies.indicators import supertrend, atr, rsi, ema, adx, sma
from src.strategies.strategy_manager import StrategyMetadata, StrategyCategory
from src.logger import logger


class SuperTrendStrategy(BaseStrategy):
    """超級趨勢策略"""

    METADATA = StrategyMetadata(
        name='supertrend',
        display_name='超級趨勢策略',
        category=StrategyCategory.TREND_FOLLOWING,
        description='基於 SuperTrend 指標的趨勢跟蹤策略',
        author='Freqtrade Community',
        version='1.0.0',
        timeframe='1h',
        min_candles=50,
        tags=['supertrend', 'trend', 'atr'],
        risk_level=2,
        recommended_pairs=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
        hyperopt_enabled=True
    )

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="supertrend",
                enabled=True,
                weight=1.0,
                max_position_size=0.15,
                stop_loss_pct=0.025,
                take_profit_pct=0.05,
                min_signal_strength=0.45,
                min_confidence=0.5,
                cooldown_minutes=30,
                params={
                    'st_period': 10,
                    'st_multiplier': 3.0,
                    'ema_filter': True,
                    'ema_period': 50,
                    'adx_filter': True,
                    'adx_threshold': 20,
                }
            )
        super().__init__(config)

    @classmethod
    def get_hyperopt_space(cls) -> Dict[str, Any]:
        return {
            'st_period': {'type': 'int', 'low': 7, 'high': 20},
            'st_multiplier': {'type': 'float', 'low': 2.0, 'high': 4.0, 'step': 0.1},
            'ema_period': {'type': 'int', 'low': 20, 'high': 100},
            'adx_threshold': {'type': 'int', 'low': 15, 'high': 30},
        }

    def populate_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params

        # SuperTrend
        dataframe['supertrend'], dataframe['st_direction'] = supertrend(
            dataframe,
            params.get('st_period', 10),
            params.get('st_multiplier', 3.0)
        )

        # 方向變化
        dataframe['st_flip_up'] = (
            (dataframe['st_direction'] == 1) &
            (dataframe['st_direction'].shift(1) == -1)
        )
        dataframe['st_flip_down'] = (
            (dataframe['st_direction'] == -1) &
            (dataframe['st_direction'].shift(1) == 1)
        )

        # EMA
        dataframe['ema'] = ema(dataframe['close'], params.get('ema_period', 50))

        # ADX
        dataframe['adx'] = adx(dataframe, 14)

        # RSI
        dataframe['rsi'] = rsi(dataframe['close'], 14)

        # ATR
        dataframe['atr'] = atr(dataframe, 14)

        # 成交量
        dataframe['volume_sma'] = sma(dataframe['volume'], 20)

        return dataframe

    def populate_entry_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params
        ema_filter = params.get('ema_filter', True)
        adx_filter = params.get('adx_filter', True)
        adx_threshold = params.get('adx_threshold', 20)

        # 做多：SuperTrend 翻多
        long_condition = dataframe['st_flip_up'].copy()

        if ema_filter:
            long_condition = long_condition & (dataframe['close'] > dataframe['ema'])

        if adx_filter:
            long_condition = long_condition & (dataframe['adx'] > adx_threshold)

        # 做空：SuperTrend 翻空
        short_condition = dataframe['st_flip_down'].copy()

        if ema_filter:
            short_condition = short_condition & (dataframe['close'] < dataframe['ema'])

        if adx_filter:
            short_condition = short_condition & (dataframe['adx'] > adx_threshold)

        dataframe['enter_long'] = long_condition.astype(int)
        dataframe['enter_short'] = short_condition.astype(int)

        return dataframe

    def populate_exit_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # 出場：SuperTrend 反向
        dataframe['exit_long'] = dataframe['st_flip_down'].astype(int)
        dataframe['exit_short'] = dataframe['st_flip_up'].astype(int)

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

            direction = 0
            strength = 0.0
            confidence = 0.0
            reasons = []

            params = self.config.params
            st_val = latest.get('supertrend', current_price)
            st_direction = latest.get('st_direction', 0)
            ema_val = latest.get('ema', current_price)
            adx_val = latest.get('adx', 20)
            rsi_val = latest.get('rsi', 50)
            volume_ratio = latest['volume'] / (latest.get('volume_sma', latest['volume']) + 1e-10)

            # 做多信號
            if latest.get('enter_long', 0) == 1:
                direction = 1
                strength = 0.55
                confidence = 0.6

                reasons.append("SuperTrend翻多")

                # 價格與 SuperTrend 距離
                distance_pct = (current_price - st_val) / st_val * 100
                if distance_pct > 0:
                    strength += min(distance_pct * 0.05, 0.15)
                    reasons.append(f"價格在ST上方({distance_pct:.1f}%)")

                # EMA 確認
                if current_price > ema_val:
                    strength += 0.1
                    confidence += 0.05
                    reasons.append("EMA支撐")

                # ADX 確認
                if adx_val > 25:
                    strength += 0.1
                    confidence += 0.1
                    reasons.append(f"強趨勢(ADX:{adx_val:.0f})")

                # 成交量
                if volume_ratio > 1.5:
                    strength += 0.1
                    reasons.append(f"放量({volume_ratio:.1f}x)")

            # 做空信號
            elif latest.get('enter_short', 0) == 1:
                direction = -1
                strength = 0.55
                confidence = 0.6

                reasons.append("SuperTrend翻空")

                distance_pct = (st_val - current_price) / st_val * 100
                if distance_pct > 0:
                    strength += min(distance_pct * 0.05, 0.15)
                    reasons.append(f"價格在ST下方({distance_pct:.1f}%)")

                if current_price < ema_val:
                    strength += 0.1
                    confidence += 0.05
                    reasons.append("EMA壓力")

                if adx_val > 25:
                    strength += 0.1
                    confidence += 0.1
                    reasons.append(f"強趨勢(ADX:{adx_val:.0f})")

                if volume_ratio > 1.5:
                    strength += 0.1
                    reasons.append(f"放量({volume_ratio:.1f}x)")

            # 平倉信號
            elif position != 0:
                if position > 0 and latest.get('exit_long', 0) == 1:
                    direction = -1
                    strength = 0.65
                    confidence = 0.75
                    reasons.append("SuperTrend翻空平多")
                elif position < 0 and latest.get('exit_short', 0) == 1:
                    direction = 1
                    strength = 0.65
                    confidence = 0.75
                    reasons.append("SuperTrend翻多平空")

            if direction == 0:
                return None

            strength = min(1.0, max(0.0, strength))
            confidence = min(1.0, max(0.0, confidence))

            if strength < self.config.min_signal_strength:
                return None

            if direction > 0:
                signal_type = SignalType.STRONG_BUY if strength > 0.65 else SignalType.BUY
            else:
                signal_type = SignalType.STRONG_SELL if strength > 0.65 else SignalType.SELL

            # 止損設在 SuperTrend
            if direction > 0:
                stop_loss = st_val * 0.99
                take_profit = current_price + (current_price - stop_loss) * 2
            else:
                stop_loss = st_val * 1.01
                take_profit = current_price - (stop_loss - current_price) * 2

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
                    'supertrend': float(st_val),
                    'st_direction': int(st_direction),
                    'adx': float(adx_val),
                    'rsi': float(rsi_val),
                },
                reason=" | ".join(reasons)
            )

            self.record_signal(signal)
            logger.info(f"[SuperTrend] {symbol}: {signal_type.value} "
                       f"strength={strength:.2f} conf={confidence:.2f} | {signal.reason}")

            return signal

        except Exception as e:
            logger.error(f"SuperTrend error: {e}")
            return None

    def update(self, market_data: Dict[str, Any]):
        pass
