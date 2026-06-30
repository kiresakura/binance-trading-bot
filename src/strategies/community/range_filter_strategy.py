"""
範圍過濾策略

基於價格範圍過濾的波動率策略
- 價格突破上範圍 + 低波動後 = 做多
- 價格跌破下範圍 + 低波動後 = 做空

適合盤整後的突破行情

v0.4.0 - Enterprise Edition
"""

from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from src.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from src.strategies.indicators import atr, bollinger_bands, rsi, ema, adx, sma, stddev
from src.strategies.strategy_manager import StrategyMetadata, StrategyCategory
from src.logger import logger


class RangeFilterStrategy(BaseStrategy):
    """範圍過濾策略"""

    METADATA = StrategyMetadata(
        name='range_filter',
        display_name='範圍過濾策略',
        category=StrategyCategory.VOLATILITY,
        description='基於價格範圍和波動率的策略，適合盤整突破',
        author='Freqtrade Community',
        version='1.0.0',
        timeframe='1h',
        min_candles=50,
        tags=['range', 'volatility', 'breakout'],
        risk_level=3,
        recommended_pairs=['BTCUSDT', 'ETHUSDT'],
        hyperopt_enabled=True
    )

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="range_filter",
                enabled=True,
                weight=1.0,
                max_position_size=0.15,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
                min_signal_strength=0.45,
                min_confidence=0.5,
                cooldown_minutes=30,
                params={
                    'range_period': 20,
                    'range_multiplier': 2.5,
                    'volatility_lookback': 20,
                    'low_volatility_threshold': 0.015,
                    'high_volatility_threshold': 0.03,
                    'ema_period': 50,
                }
            )
        super().__init__(config)

    @classmethod
    def get_hyperopt_space(cls) -> Dict[str, Any]:
        return {
            'range_period': {'type': 'int', 'low': 10, 'high': 30},
            'range_multiplier': {'type': 'float', 'low': 2.0, 'high': 4.0, 'step': 0.1},
            'low_volatility_threshold': {'type': 'float', 'low': 0.01, 'high': 0.025, 'step': 0.001},
            'ema_period': {'type': 'int', 'low': 20, 'high': 100},
        }

    def _calculate_range_filter(self, df: pd.DataFrame, period: int, multiplier: float) -> tuple:
        """計算範圍過濾器"""
        # 使用 ATR 計算範圍
        smooth_range = atr(df, period) * multiplier

        # 範圍過濾器
        filt = df['close'].ewm(span=period, adjust=False).mean()
        filt_upper = filt + smooth_range
        filt_lower = filt - smooth_range

        # 方向
        upward = (df['close'] > filt_upper.shift(1)).astype(int)
        downward = (df['close'] < filt_lower.shift(1)).astype(int)

        return filt, filt_upper, filt_lower, upward, downward

    def populate_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params
        range_period = params.get('range_period', 20)
        range_multiplier = params.get('range_multiplier', 2.5)
        volatility_lookback = params.get('volatility_lookback', 20)

        # 範圍過濾器
        filt, filt_upper, filt_lower, upward, downward = self._calculate_range_filter(
            dataframe, range_period, range_multiplier
        )
        dataframe['range_filt'] = filt
        dataframe['range_upper'] = filt_upper
        dataframe['range_lower'] = filt_lower
        dataframe['range_upward'] = upward
        dataframe['range_downward'] = downward

        # 波動率
        dataframe['volatility'] = dataframe['close'].pct_change().rolling(volatility_lookback).std()
        dataframe['volatility_sma'] = sma(dataframe['volatility'], volatility_lookback)

        # 布林帶寬度作為波動率參考
        bb_upper, bb_middle, bb_lower = bollinger_bands(dataframe['close'], 20, 2.0)
        dataframe['bb_width'] = (bb_upper - bb_lower) / bb_middle

        # ATR
        dataframe['atr'] = atr(dataframe, 14)

        # EMA
        dataframe['ema'] = ema(dataframe['close'], params.get('ema_period', 50))

        # ADX
        dataframe['adx'] = adx(dataframe, 14)

        # RSI
        dataframe['rsi'] = rsi(dataframe['close'], 14)

        # 成交量
        dataframe['volume_sma'] = sma(dataframe['volume'], 20)

        return dataframe

    def populate_entry_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params
        low_vol_threshold = params.get('low_volatility_threshold', 0.015)

        # 波動率收縮後的條件
        volatility_squeeze = (
            dataframe['volatility'].shift(1) < low_vol_threshold
        )

        # 做多：價格突破範圍上軌 + 之前低波動
        long_condition = (
            (dataframe['range_upward'] == 1) &
            (dataframe['range_upward'].shift(1) == 0) &
            (dataframe['close'] > dataframe['ema']) &
            volatility_squeeze
        )

        # 做空：價格跌破範圍下軌 + 之前低波動
        short_condition = (
            (dataframe['range_downward'] == 1) &
            (dataframe['range_downward'].shift(1) == 0) &
            (dataframe['close'] < dataframe['ema']) &
            volatility_squeeze
        )

        dataframe['enter_long'] = long_condition.astype(int)
        dataframe['enter_short'] = short_condition.astype(int)

        return dataframe

    def populate_exit_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params
        high_vol_threshold = params.get('high_volatility_threshold', 0.03)

        # 出場：回到範圍內或波動率過高
        dataframe['exit_long'] = (
            (dataframe['close'] < dataframe['range_filt']) |
            (dataframe['volatility'] > high_vol_threshold * 1.5) |
            (dataframe['rsi'] > 80)
        ).astype(int)

        dataframe['exit_short'] = (
            (dataframe['close'] > dataframe['range_filt']) |
            (dataframe['volatility'] > high_vol_threshold * 1.5) |
            (dataframe['rsi'] < 20)
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

            params = self.config.params
            low_vol_threshold = params.get('low_volatility_threshold', 0.015)

            range_filt = latest.get('range_filt', current_price)
            range_upper = latest.get('range_upper', current_price)
            range_lower = latest.get('range_lower', current_price)
            volatility = latest.get('volatility', 0.02)
            prev_volatility = prev.get('volatility', 0.02)
            bb_width = latest.get('bb_width', 0.04)
            adx_val = latest.get('adx', 20)
            rsi_val = latest.get('rsi', 50)
            volume_ratio = latest['volume'] / (latest.get('volume_sma', latest['volume']) + 1e-10)

            # 做多信號
            if latest.get('enter_long', 0) == 1:
                direction = 1
                strength = 0.5

                reasons.append("突破範圍上軌")

                # 波動率收縮程度
                if prev_volatility < low_vol_threshold * 0.8:
                    strength += 0.15
                    confidence = 0.65
                    reasons.append(f"極低波動({prev_volatility*100:.1f}%)")
                else:
                    confidence = 0.55
                    reasons.append(f"低波動({prev_volatility*100:.1f}%)")

                # BB 寬度確認
                if bb_width < 0.03:
                    strength += 0.1
                    reasons.append("BB收縮")

                # ADX 確認
                if adx_val > 20:
                    strength += 0.1
                    confidence += 0.05
                    reasons.append(f"ADX確認({adx_val:.0f})")

                # 成交量
                if volume_ratio > 1.5:
                    strength += 0.1
                    confidence += 0.05
                    reasons.append(f"放量({volume_ratio:.1f}x)")

            # 做空信號
            elif latest.get('enter_short', 0) == 1:
                direction = -1
                strength = 0.5

                reasons.append("跌破範圍下軌")

                if prev_volatility < low_vol_threshold * 0.8:
                    strength += 0.15
                    confidence = 0.65
                    reasons.append(f"極低波動({prev_volatility*100:.1f}%)")
                else:
                    confidence = 0.55
                    reasons.append(f"低波動({prev_volatility*100:.1f}%)")

                if bb_width < 0.03:
                    strength += 0.1
                    reasons.append("BB收縮")

                if adx_val > 20:
                    strength += 0.1
                    confidence += 0.05
                    reasons.append(f"ADX確認({adx_val:.0f})")

                if volume_ratio > 1.5:
                    strength += 0.1
                    confidence += 0.05
                    reasons.append(f"放量({volume_ratio:.1f}x)")

            # 平倉信號
            elif position != 0:
                if position > 0 and latest.get('exit_long', 0) == 1:
                    direction = -1
                    strength = 0.6
                    confidence = 0.7
                    reasons.append("回到範圍內平多")
                elif position < 0 and latest.get('exit_short', 0) == 1:
                    direction = 1
                    strength = 0.6
                    confidence = 0.7
                    reasons.append("回到範圍內平空")

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

            # 止損設在範圍邊界
            atr_val = latest.get('atr', current_price * 0.02)
            if direction > 0:
                stop_loss = range_lower
                take_profit = current_price + (current_price - range_lower) * 1.5
            else:
                stop_loss = range_upper
                take_profit = current_price - (range_upper - current_price) * 1.5

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
                    'range_filt': float(range_filt),
                    'range_upper': float(range_upper),
                    'range_lower': float(range_lower),
                    'volatility': float(volatility),
                    'bb_width': float(bb_width),
                },
                reason=" | ".join(reasons)
            )

            self.record_signal(signal)
            logger.info(f"[RangeFilter] {symbol}: {signal_type.value} "
                       f"strength={strength:.2f} conf={confidence:.2f} | {signal.reason}")

            return signal

        except Exception as e:
            logger.error(f"RangeFilter error: {e}")
            return None

    def update(self, market_data: Dict[str, Any]):
        pass
