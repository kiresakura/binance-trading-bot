"""
ADX 動量策略

結合 ADX 趨勢強度和動量指標
- ADX > 閾值 + 正動量 = 做多
- ADX > 閾值 + 負動量 = 做空

v0.4.0 - Enterprise Edition
"""

from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from src.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from src.strategies.indicators import adx, di_plus, di_minus, rsi, ema, atr, momentum, sma
from src.strategies.strategy_manager import StrategyMetadata, StrategyCategory
from src.logger import logger


class ADXMomentumStrategy(BaseStrategy):
    """ADX 動量策略"""

    METADATA = StrategyMetadata(
        name='adx_momentum',
        display_name='ADX 動量策略',
        category=StrategyCategory.MOMENTUM,
        description='結合 ADX 趨勢強度和動量指標的策略',
        author='Freqtrade Community',
        version='1.0.0',
        timeframe='1h',
        min_candles=50,
        tags=['adx', 'momentum', 'trend'],
        risk_level=3,
        recommended_pairs=['BTCUSDT', 'ETHUSDT'],
        hyperopt_enabled=True
    )

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="adx_momentum",
                enabled=True,
                weight=1.0,
                max_position_size=0.15,
                stop_loss_pct=0.025,
                take_profit_pct=0.05,
                min_signal_strength=0.45,
                min_confidence=0.5,
                cooldown_minutes=20,
                params={
                    'adx_period': 14,
                    'adx_threshold': 25,
                    'adx_strong': 40,
                    'momentum_period': 10,
                    'rsi_period': 14,
                    'ema_period': 50,
                }
            )
        super().__init__(config)

    @classmethod
    def get_hyperopt_space(cls) -> Dict[str, Any]:
        return {
            'adx_period': {'type': 'int', 'low': 10, 'high': 20},
            'adx_threshold': {'type': 'int', 'low': 20, 'high': 35},
            'adx_strong': {'type': 'int', 'low': 35, 'high': 50},
            'momentum_period': {'type': 'int', 'low': 5, 'high': 20},
            'ema_period': {'type': 'int', 'low': 20, 'high': 100},
        }

    def populate_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params
        adx_period = params.get('adx_period', 14)

        # ADX 和 DI
        dataframe['adx'] = adx(dataframe, adx_period)
        dataframe['di_plus'] = di_plus(dataframe, adx_period)
        dataframe['di_minus'] = di_minus(dataframe, adx_period)

        # DI 差異
        dataframe['di_diff'] = dataframe['di_plus'] - dataframe['di_minus']

        # 動量
        dataframe['momentum'] = momentum(dataframe['close'], params.get('momentum_period', 10))
        dataframe['momentum_pct'] = dataframe['momentum'] / dataframe['close'].shift(params.get('momentum_period', 10)) * 100

        # RSI
        dataframe['rsi'] = rsi(dataframe['close'], params.get('rsi_period', 14))

        # EMA 趨勢
        dataframe['ema'] = ema(dataframe['close'], params.get('ema_period', 50))

        # ATR
        dataframe['atr'] = atr(dataframe, 14)

        # 成交量
        dataframe['volume_sma'] = sma(dataframe['volume'], 20)

        return dataframe

    def populate_entry_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params
        adx_threshold = params.get('adx_threshold', 25)

        # 做多：ADX 強 + DI+ > DI- + 正動量 + 價格在 EMA 上方
        long_condition = (
            (dataframe['adx'] > adx_threshold) &
            (dataframe['di_plus'] > dataframe['di_minus']) &
            (dataframe['momentum'] > 0) &
            (dataframe['close'] > dataframe['ema']) &
            (dataframe['rsi'] < 70)
        )

        # 做空：ADX 強 + DI- > DI+ + 負動量 + 價格在 EMA 下方
        short_condition = (
            (dataframe['adx'] > adx_threshold) &
            (dataframe['di_minus'] > dataframe['di_plus']) &
            (dataframe['momentum'] < 0) &
            (dataframe['close'] < dataframe['ema']) &
            (dataframe['rsi'] > 30)
        )

        dataframe['enter_long'] = long_condition.astype(int)
        dataframe['enter_short'] = short_condition.astype(int)

        return dataframe

    def populate_exit_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params
        adx_threshold = params.get('adx_threshold', 25)

        # 做多出場：ADX 減弱或 DI 反轉
        dataframe['exit_long'] = (
            (dataframe['adx'] < adx_threshold * 0.8) |
            (dataframe['di_minus'] > dataframe['di_plus']) |
            (dataframe['rsi'] > 75)
        ).astype(int)

        # 做空出場
        dataframe['exit_short'] = (
            (dataframe['adx'] < adx_threshold * 0.8) |
            (dataframe['di_plus'] > dataframe['di_minus']) |
            (dataframe['rsi'] < 25)
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

            direction = 0
            strength = 0.0
            confidence = 0.0
            reasons = []

            params = self.config.params
            adx_threshold = params.get('adx_threshold', 25)
            adx_strong = params.get('adx_strong', 40)

            adx_val = latest.get('adx', 20)
            di_plus_val = latest.get('di_plus', 20)
            di_minus_val = latest.get('di_minus', 20)
            di_diff = latest.get('di_diff', 0)
            momentum_val = latest.get('momentum', 0)
            momentum_pct = latest.get('momentum_pct', 0)
            rsi_val = latest.get('rsi', 50)
            ema_val = latest.get('ema', current_price)
            volume_ratio = latest['volume'] / (latest.get('volume_sma', latest['volume']) + 1e-10)

            # 做多信號
            if latest.get('enter_long', 0) == 1:
                direction = 1
                strength = 0.5

                # ADX 強度
                if adx_val > adx_strong:
                    strength += 0.2
                    confidence = 0.7
                    reasons.append(f"超強趨勢(ADX:{adx_val:.0f})")
                elif adx_val > adx_threshold:
                    strength += 0.1
                    confidence = 0.6
                    reasons.append(f"強趨勢(ADX:{adx_val:.0f})")

                # DI 差異
                if di_diff > 10:
                    strength += 0.1
                    reasons.append(f"DI+領先(+{di_diff:.0f})")

                # 動量強度
                if momentum_pct > 2:
                    strength += 0.1
                    reasons.append(f"強動量({momentum_pct:.1f}%)")

                # 成交量
                if volume_ratio > 1.5:
                    strength += 0.1
                    confidence += 0.05
                    reasons.append(f"放量({volume_ratio:.1f}x)")

            # 做空信號
            elif latest.get('enter_short', 0) == 1:
                direction = -1
                strength = 0.5

                if adx_val > adx_strong:
                    strength += 0.2
                    confidence = 0.7
                    reasons.append(f"超強趨勢(ADX:{adx_val:.0f})")
                elif adx_val > adx_threshold:
                    strength += 0.1
                    confidence = 0.6
                    reasons.append(f"強趨勢(ADX:{adx_val:.0f})")

                if di_diff < -10:
                    strength += 0.1
                    reasons.append(f"DI-領先({di_diff:.0f})")

                if momentum_pct < -2:
                    strength += 0.1
                    reasons.append(f"強負動量({momentum_pct:.1f}%)")

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
                    reasons.append("趨勢減弱平多")
                elif position < 0 and latest.get('exit_short', 0) == 1:
                    direction = 1
                    strength = 0.6
                    confidence = 0.7
                    reasons.append("趨勢減弱平空")

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

            atr_val = latest.get('atr', current_price * 0.02)
            if direction > 0:
                stop_loss = current_price - atr_val * 2
                take_profit = current_price + atr_val * 3
            else:
                stop_loss = current_price + atr_val * 2
                take_profit = current_price - atr_val * 3

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
                    'adx': float(adx_val),
                    'di_plus': float(di_plus_val),
                    'di_minus': float(di_minus_val),
                    'momentum_pct': float(momentum_pct),
                    'rsi': float(rsi_val),
                },
                reason=" | ".join(reasons)
            )

            self.record_signal(signal)
            logger.info(f"[ADX_Momentum] {symbol}: {signal_type.value} "
                       f"strength={strength:.2f} conf={confidence:.2f} | {signal.reason}")

            return signal

        except Exception as e:
            logger.error(f"ADX_Momentum error: {e}")
            return None

    def update(self, market_data: Dict[str, Any]):
        pass
