"""
黃金交叉策略

經典的 MA50/MA200 黃金交叉策略
- MA50 上穿 MA200 = 做多（黃金交叉）
- MA50 下穿 MA200 = 做空（死亡交叉）

適合長期趨勢交易

v0.4.0 - Enterprise Edition
"""

from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from src.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from src.strategies.indicators import sma, ema, atr, rsi, adx, cross_above, cross_below
from src.strategies.strategy_manager import StrategyMetadata, StrategyCategory
from src.logger import logger


class GoldenCrossStrategy(BaseStrategy):
    """黃金交叉策略"""

    METADATA = StrategyMetadata(
        name='golden_cross',
        display_name='黃金交叉策略',
        category=StrategyCategory.TREND_FOLLOWING,
        description='經典的 MA50/MA200 黃金交叉與死亡交叉策略',
        author='Freqtrade Community',
        version='1.0.0',
        timeframe='1d',
        min_candles=250,
        tags=['sma', 'golden_cross', 'trend', 'long_term'],
        risk_level=1,
        recommended_pairs=['BTCUSDT', 'ETHUSDT'],
        hyperopt_enabled=True
    )

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="golden_cross",
                enabled=True,
                weight=1.0,
                max_position_size=0.2,
                stop_loss_pct=0.05,
                take_profit_pct=0.15,
                min_signal_strength=0.5,
                min_confidence=0.6,
                cooldown_minutes=1440,  # 1 天冷卻
                params={
                    'fast_period': 50,
                    'slow_period': 200,
                    'ma_type': 'sma',  # 'sma' or 'ema'
                    'adx_filter': True,
                    'adx_threshold': 20,
                    'volume_confirm': True,
                }
            )
        super().__init__(config)

    @classmethod
    def get_hyperopt_space(cls) -> Dict[str, Any]:
        return {
            'fast_period': {'type': 'int', 'low': 20, 'high': 100},
            'slow_period': {'type': 'int', 'low': 100, 'high': 300},
            'adx_threshold': {'type': 'int', 'low': 15, 'high': 30},
            'ma_type': {'type': 'categorical', 'choices': ['sma', 'ema']},
        }

    def populate_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params
        fast_period = params.get('fast_period', 50)
        slow_period = params.get('slow_period', 200)
        ma_type = params.get('ma_type', 'sma')

        ma_func = sma if ma_type == 'sma' else ema

        # 移動平均線
        dataframe['ma_fast'] = ma_func(dataframe['close'], fast_period)
        dataframe['ma_slow'] = ma_func(dataframe['close'], slow_period)

        # MA 差距百分比
        dataframe['ma_diff_pct'] = (
            (dataframe['ma_fast'] - dataframe['ma_slow']) / dataframe['ma_slow'] * 100
        )

        # 交叉信號
        dataframe['golden_cross'] = cross_above(dataframe['ma_fast'], dataframe['ma_slow'])
        dataframe['death_cross'] = cross_below(dataframe['ma_fast'], dataframe['ma_slow'])

        # ADX
        dataframe['adx'] = adx(dataframe, 14)

        # ATR
        dataframe['atr'] = atr(dataframe, 14)

        # RSI
        dataframe['rsi'] = rsi(dataframe['close'], 14)

        # 成交量
        dataframe['volume_sma'] = sma(dataframe['volume'], 20)

        return dataframe

    def populate_entry_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params
        adx_filter = params.get('adx_filter', True)
        adx_threshold = params.get('adx_threshold', 20)
        volume_confirm = params.get('volume_confirm', True)

        # 做多條件
        long_condition = dataframe['golden_cross'].copy()

        if adx_filter:
            long_condition = long_condition & (dataframe['adx'] > adx_threshold)

        if volume_confirm:
            long_condition = long_condition & (dataframe['volume'] > dataframe['volume_sma'])

        # 做空條件
        short_condition = dataframe['death_cross'].copy()

        if adx_filter:
            short_condition = short_condition & (dataframe['adx'] > adx_threshold)

        if volume_confirm:
            short_condition = short_condition & (dataframe['volume'] > dataframe['volume_sma'])

        dataframe['enter_long'] = long_condition.astype(int)
        dataframe['enter_short'] = short_condition.astype(int)

        return dataframe

    def populate_exit_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # 出場信號：反向交叉
        dataframe['exit_long'] = dataframe['death_cross'].astype(int)
        dataframe['exit_short'] = dataframe['golden_cross'].astype(int)

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

        if len(features) < 250:
            return None

        try:
            df = self.analyze(features.copy())
            latest = df.iloc[-1]

            direction = 0
            strength = 0.0
            confidence = 0.0
            reasons = []

            ma_fast = latest.get('ma_fast', current_price)
            ma_slow = latest.get('ma_slow', current_price)
            ma_diff_pct = latest.get('ma_diff_pct', 0)
            adx_val = latest.get('adx', 20)
            rsi_val = latest.get('rsi', 50)
            volume_ratio = latest['volume'] / (latest.get('volume_sma', latest['volume']) + 1e-10)

            # 黃金交叉做多
            if latest.get('enter_long', 0) == 1:
                direction = 1
                strength = 0.6
                confidence = 0.65

                reasons.append(f"黃金交叉(MA{self.config.params.get('fast_period', 50)}上穿MA{self.config.params.get('slow_period', 200)})")

                # ADX 確認趨勢強度
                if adx_val > 25:
                    strength += 0.15
                    confidence += 0.1
                    reasons.append(f"強趨勢(ADX:{adx_val:.0f})")

                # 成交量確認
                if volume_ratio > 1.5:
                    strength += 0.1
                    confidence += 0.1
                    reasons.append(f"放量確認({volume_ratio:.1f}x)")

                # RSI 位置
                if 40 < rsi_val < 70:
                    confidence += 0.05
                    reasons.append(f"RSI健康({rsi_val:.0f})")

            # 死亡交叉做空
            elif latest.get('enter_short', 0) == 1:
                direction = -1
                strength = 0.6
                confidence = 0.65

                reasons.append(f"死亡交叉(MA{self.config.params.get('fast_period', 50)}下穿MA{self.config.params.get('slow_period', 200)})")

                if adx_val > 25:
                    strength += 0.15
                    confidence += 0.1
                    reasons.append(f"強趨勢(ADX:{adx_val:.0f})")

                if volume_ratio > 1.5:
                    strength += 0.1
                    confidence += 0.1
                    reasons.append(f"放量確認({volume_ratio:.1f}x)")

                if 30 < rsi_val < 60:
                    confidence += 0.05
                    reasons.append(f"RSI健康({rsi_val:.0f})")

            # 平倉信號
            elif position != 0:
                if position > 0 and latest.get('exit_long', 0) == 1:
                    direction = -1
                    strength = 0.7
                    confidence = 0.8
                    reasons.append("死亡交叉平多")
                elif position < 0 and latest.get('exit_short', 0) == 1:
                    direction = 1
                    strength = 0.7
                    confidence = 0.8
                    reasons.append("黃金交叉平空")

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

            # 長期策略使用較寬的止損
            atr_val = latest.get('atr', current_price * 0.03)
            if direction > 0:
                stop_loss = ma_slow * 0.95  # 停在慢線下方 5%
                take_profit = current_price * 1.15
            else:
                stop_loss = ma_slow * 1.05
                take_profit = current_price * 0.85

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
                    'ma_fast': float(ma_fast),
                    'ma_slow': float(ma_slow),
                    'ma_diff_pct': float(ma_diff_pct),
                    'adx': float(adx_val),
                    'rsi': float(rsi_val),
                },
                reason=" | ".join(reasons)
            )

            self.record_signal(signal)
            logger.info(f"[GoldenCross] {symbol}: {signal_type.value} "
                       f"strength={strength:.2f} conf={confidence:.2f} | {signal.reason}")

            return signal

        except Exception as e:
            logger.error(f"GoldenCross error: {e}")
            return None

    def update(self, market_data: Dict[str, Any]):
        pass
