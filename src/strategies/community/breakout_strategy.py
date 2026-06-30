"""
突破策略

基於唐奇安通道的突破交易策略
- 價格突破近期高點 = 做多
- 價格跌破近期低點 = 做空

v0.4.0 - Enterprise Edition
"""

from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from src.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from src.strategies.indicators import donchian_channel, atr, adx, rsi, sma
from src.strategies.strategy_manager import StrategyMetadata, StrategyCategory
from src.logger import logger


class BreakoutStrategy(BaseStrategy):
    """突破策略"""

    METADATA = StrategyMetadata(
        name='breakout',
        display_name='突破策略',
        category=StrategyCategory.BREAKOUT,
        description='基於唐奇安通道的突破交易策略',
        author='Freqtrade Community',
        version='1.0.0',
        timeframe='1h',
        min_candles=60,
        tags=['breakout', 'donchian', 'momentum'],
        risk_level=3,
        recommended_pairs=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'DOGEUSDT'],
        hyperopt_enabled=True
    )

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="breakout",
                enabled=True,
                weight=1.0,
                max_position_size=0.15,
                stop_loss_pct=0.025,
                take_profit_pct=0.05,
                min_signal_strength=0.45,
                min_confidence=0.5,
                cooldown_minutes=30,
                params={
                    'donchian_period': 20,
                    'exit_period': 10,  # 出場用較短週期
                    'atr_multiplier': 2.0,
                    'volume_confirm': True,
                    'volume_factor': 1.5,
                    'adx_filter': True,
                    'adx_threshold': 20,
                    'false_breakout_filter': True,
                }
            )
        super().__init__(config)
        self._recent_breakouts = []

    @classmethod
    def get_hyperopt_space(cls) -> Dict[str, Any]:
        return {
            'donchian_period': {'type': 'int', 'low': 10, 'high': 50},
            'exit_period': {'type': 'int', 'low': 5, 'high': 25},
            'atr_multiplier': {'type': 'float', 'low': 1.5, 'high': 3.0, 'step': 0.1},
            'volume_factor': {'type': 'float', 'low': 1.0, 'high': 2.5, 'step': 0.1},
            'adx_threshold': {'type': 'int', 'low': 15, 'high': 30},
        }

    def populate_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params
        donchian_period = params.get('donchian_period', 20)
        exit_period = params.get('exit_period', 10)

        # 唐奇安通道（入場）
        dataframe['dc_upper'], dataframe['dc_middle'], dataframe['dc_lower'] = donchian_channel(
            dataframe, donchian_period
        )

        # 出場唐奇安通道
        dataframe['dc_exit_upper'], _, dataframe['dc_exit_lower'] = donchian_channel(
            dataframe, exit_period
        )

        # 通道寬度
        dataframe['dc_width'] = (dataframe['dc_upper'] - dataframe['dc_lower']) / dataframe['dc_middle']

        # ATR
        dataframe['atr'] = atr(dataframe, 14)

        # ADX
        dataframe['adx'] = adx(dataframe, 14)

        # RSI
        dataframe['rsi'] = rsi(dataframe['close'], 14)

        # 成交量
        dataframe['volume_sma'] = sma(dataframe['volume'], 20)

        # 前一根 K 線收盤價
        dataframe['prev_close'] = dataframe['close'].shift(1)
        dataframe['prev_dc_upper'] = dataframe['dc_upper'].shift(1)
        dataframe['prev_dc_lower'] = dataframe['dc_lower'].shift(1)

        return dataframe

    def populate_entry_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params
        volume_confirm = params.get('volume_confirm', True)
        volume_factor = params.get('volume_factor', 1.5)
        adx_filter = params.get('adx_filter', True)
        adx_threshold = params.get('adx_threshold', 20)
        false_breakout_filter = params.get('false_breakout_filter', True)

        # 向上突破
        breakout_up = (
            (dataframe['close'] > dataframe['prev_dc_upper']) &
            (dataframe['prev_close'] <= dataframe['prev_dc_upper'])
        )

        # 向下突破
        breakout_down = (
            (dataframe['close'] < dataframe['prev_dc_lower']) &
            (dataframe['prev_close'] >= dataframe['prev_dc_lower'])
        )

        # 過濾條件
        if volume_confirm:
            volume_ok = dataframe['volume'] > dataframe['volume_sma'] * volume_factor
            breakout_up = breakout_up & volume_ok
            breakout_down = breakout_down & volume_ok

        if adx_filter:
            adx_ok = dataframe['adx'] > adx_threshold
            breakout_up = breakout_up & adx_ok
            breakout_down = breakout_down & adx_ok

        if false_breakout_filter:
            # 確認收盤價在突破方向足夠遠
            strong_breakout_up = dataframe['close'] > dataframe['dc_upper'] * 1.001
            strong_breakout_down = dataframe['close'] < dataframe['dc_lower'] * 0.999
            breakout_up = breakout_up & strong_breakout_up
            breakout_down = breakout_down & strong_breakout_down

        dataframe['enter_long'] = breakout_up.astype(int)
        dataframe['enter_short'] = breakout_down.astype(int)

        return dataframe

    def populate_exit_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # 出場：回到較短週期通道內
        dataframe['exit_long'] = (
            dataframe['close'] < dataframe['dc_exit_lower']
        ).astype(int)

        dataframe['exit_short'] = (
            dataframe['close'] > dataframe['dc_exit_upper']
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

        if len(features) < 60:
            return None

        try:
            df = self.analyze(features.copy())
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest

            direction = 0
            strength = 0.0
            confidence = 0.0
            reasons = []

            dc_upper = latest.get('dc_upper', current_price)
            dc_lower = latest.get('dc_lower', current_price)
            dc_width = latest.get('dc_width', 0.02)
            adx_val = latest.get('adx', 20)
            rsi_val = latest.get('rsi', 50)
            volume_ratio = latest['volume'] / (latest.get('volume_sma', latest['volume']) + 1e-10)

            # 向上突破
            if latest.get('enter_long', 0) == 1:
                direction = 1
                strength = 0.55

                # 突破幅度
                breakout_pct = (current_price - dc_upper) / dc_upper * 100
                if breakout_pct > 0.5:
                    strength += 0.15
                    reasons.append(f"強勢突破({breakout_pct:.2f}%)")
                else:
                    reasons.append(f"突破上軌({dc_upper:.0f})")

                # ADX 確認
                if adx_val > 25:
                    strength += 0.1
                    confidence = 0.65
                    reasons.append(f"趨勢確認(ADX:{adx_val:.0f})")
                else:
                    confidence = 0.55

                # 成交量
                if volume_ratio > 2:
                    strength += 0.15
                    confidence += 0.1
                    reasons.append(f"巨量突破({volume_ratio:.1f}x)")
                elif volume_ratio > 1.5:
                    strength += 0.1
                    reasons.append(f"放量確認({volume_ratio:.1f}x)")

                # 通道寬度（較寬通道突破更有意義）
                if dc_width > 0.03:
                    strength += 0.05
                    reasons.append("寬幅通道")

            # 向下突破
            elif latest.get('enter_short', 0) == 1:
                direction = -1
                strength = 0.55

                breakout_pct = (dc_lower - current_price) / dc_lower * 100
                if breakout_pct > 0.5:
                    strength += 0.15
                    reasons.append(f"強勢跌破({breakout_pct:.2f}%)")
                else:
                    reasons.append(f"跌破下軌({dc_lower:.0f})")

                if adx_val > 25:
                    strength += 0.1
                    confidence = 0.65
                    reasons.append(f"趨勢確認(ADX:{adx_val:.0f})")
                else:
                    confidence = 0.55

                if volume_ratio > 2:
                    strength += 0.15
                    confidence += 0.1
                    reasons.append(f"巨量跌破({volume_ratio:.1f}x)")
                elif volume_ratio > 1.5:
                    strength += 0.1
                    reasons.append(f"放量確認({volume_ratio:.1f}x)")

                if dc_width > 0.03:
                    strength += 0.05
                    reasons.append("寬幅通道")

            # 平倉信號
            elif position != 0:
                if position > 0 and latest.get('exit_long', 0) == 1:
                    direction = -1
                    strength = 0.65
                    confidence = 0.7
                    reasons.append("回到通道內平多")
                elif position < 0 and latest.get('exit_short', 0) == 1:
                    direction = 1
                    strength = 0.65
                    confidence = 0.7
                    reasons.append("回到通道內平空")

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

            # 止損在通道內
            atr_val = latest.get('atr', current_price * 0.02)
            if direction > 0:
                stop_loss = dc_lower  # 跌破下軌止損
                take_profit = current_price + (current_price - dc_lower) * 1.5
            else:
                stop_loss = dc_upper  # 突破上軌止損
                take_profit = current_price - (dc_upper - current_price) * 1.5

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
                    'dc_upper': float(dc_upper),
                    'dc_lower': float(dc_lower),
                    'dc_width': float(dc_width),
                    'adx': float(adx_val),
                    'volume_ratio': float(volume_ratio),
                },
                reason=" | ".join(reasons)
            )

            self.record_signal(signal)
            logger.info(f"[Breakout] {symbol}: {signal_type.value} "
                       f"strength={strength:.2f} conf={confidence:.2f} | {signal.reason}")

            return signal

        except Exception as e:
            logger.error(f"Breakout error: {e}")
            return None

    def update(self, market_data: Dict[str, Any]):
        pass
