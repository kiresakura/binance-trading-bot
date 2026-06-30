"""
一目均衡表策略

日本經典技術分析策略
- 價格在雲層上方 + TK 交叉 = 做多
- 價格在雲層下方 + TK 交叉 = 做空

v0.4.0 - Enterprise Edition
"""

from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from src.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from src.strategies.indicators import ichimoku, atr, rsi, sma, cross_above, cross_below
from src.strategies.strategy_manager import StrategyMetadata, StrategyCategory
from src.logger import logger


class IchimokuStrategy(BaseStrategy):
    """一目均衡表策略"""

    METADATA = StrategyMetadata(
        name='ichimoku',
        display_name='一目均衡表策略',
        category=StrategyCategory.TREND_FOLLOWING,
        description='日本經典技術分析策略，使用一目均衡表指標',
        author='Freqtrade Community',
        version='1.0.0',
        timeframe='4h',
        min_candles=100,
        tags=['ichimoku', 'japanese', 'trend', 'cloud'],
        risk_level=2,
        recommended_pairs=['BTCUSDT', 'ETHUSDT'],
        hyperopt_enabled=True
    )

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="ichimoku",
                enabled=True,
                weight=1.0,
                max_position_size=0.15,
                stop_loss_pct=0.03,
                take_profit_pct=0.06,
                min_signal_strength=0.5,
                min_confidence=0.55,
                cooldown_minutes=60,
                params={
                    'tenkan_period': 9,
                    'kijun_period': 26,
                    'senkou_period': 52,
                    'chikou_offset': 26,
                    'cloud_filter': True,
                    'tk_cross_confirm': True,
                }
            )
        super().__init__(config)

    @classmethod
    def get_hyperopt_space(cls) -> Dict[str, Any]:
        return {
            'tenkan_period': {'type': 'int', 'low': 7, 'high': 12},
            'kijun_period': {'type': 'int', 'low': 20, 'high': 35},
            'senkou_period': {'type': 'int', 'low': 40, 'high': 65},
        }

    def populate_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params

        # 一目均衡表
        ichi = ichimoku(
            dataframe,
            params.get('tenkan_period', 9),
            params.get('kijun_period', 26),
            params.get('senkou_period', 52)
        )

        dataframe['tenkan_sen'] = ichi['tenkan_sen']
        dataframe['kijun_sen'] = ichi['kijun_sen']
        dataframe['senkou_span_a'] = ichi['senkou_span_a']
        dataframe['senkou_span_b'] = ichi['senkou_span_b']
        dataframe['chikou_span'] = ichi['chikou_span']

        # 雲層頂部和底部
        dataframe['cloud_top'] = dataframe[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        dataframe['cloud_bottom'] = dataframe[['senkou_span_a', 'senkou_span_b']].min(axis=1)

        # 雲層厚度
        dataframe['cloud_thickness'] = abs(dataframe['senkou_span_a'] - dataframe['senkou_span_b'])

        # 雲層顏色（多/空）
        dataframe['cloud_green'] = dataframe['senkou_span_a'] > dataframe['senkou_span_b']

        # TK 交叉
        dataframe['tk_cross_up'] = cross_above(dataframe['tenkan_sen'], dataframe['kijun_sen'])
        dataframe['tk_cross_down'] = cross_below(dataframe['tenkan_sen'], dataframe['kijun_sen'])

        # 價格位置
        dataframe['above_cloud'] = dataframe['close'] > dataframe['cloud_top']
        dataframe['below_cloud'] = dataframe['close'] < dataframe['cloud_bottom']
        dataframe['in_cloud'] = ~dataframe['above_cloud'] & ~dataframe['below_cloud']

        # RSI
        dataframe['rsi'] = rsi(dataframe['close'], 14)

        # ATR
        dataframe['atr'] = atr(dataframe, 14)

        # 成交量
        dataframe['volume_sma'] = sma(dataframe['volume'], 20)

        return dataframe

    def populate_entry_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params
        cloud_filter = params.get('cloud_filter', True)
        tk_cross_confirm = params.get('tk_cross_confirm', True)

        # 做多條件
        long_condition = dataframe['tk_cross_up'].copy()

        if cloud_filter:
            # 價格在雲層上方
            long_condition = long_condition & dataframe['above_cloud']
            # 雲層為綠色（多頭）
            long_condition = long_condition & dataframe['cloud_green']

        if tk_cross_confirm:
            # TK 交叉發生在雲層上方
            long_condition = long_condition & (dataframe['tenkan_sen'] > dataframe['cloud_top'])

        # 做空條件
        short_condition = dataframe['tk_cross_down'].copy()

        if cloud_filter:
            short_condition = short_condition & dataframe['below_cloud']
            short_condition = short_condition & ~dataframe['cloud_green']

        if tk_cross_confirm:
            short_condition = short_condition & (dataframe['tenkan_sen'] < dataframe['cloud_bottom'])

        dataframe['enter_long'] = long_condition.astype(int)
        dataframe['enter_short'] = short_condition.astype(int)

        return dataframe

    def populate_exit_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # 出場：反向 TK 交叉或進入雲層
        dataframe['exit_long'] = (
            dataframe['tk_cross_down'] |
            dataframe['in_cloud'] |
            dataframe['below_cloud']
        ).astype(int)

        dataframe['exit_short'] = (
            dataframe['tk_cross_up'] |
            dataframe['in_cloud'] |
            dataframe['above_cloud']
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

        if len(features) < 100:
            return None

        try:
            df = self.analyze(features.copy())
            latest = df.iloc[-1]

            direction = 0
            strength = 0.0
            confidence = 0.0
            reasons = []

            tenkan = latest.get('tenkan_sen', current_price)
            kijun = latest.get('kijun_sen', current_price)
            cloud_top = latest.get('cloud_top', current_price)
            cloud_bottom = latest.get('cloud_bottom', current_price)
            cloud_green = latest.get('cloud_green', True)
            rsi_val = latest.get('rsi', 50)
            volume_ratio = latest['volume'] / (latest.get('volume_sma', latest['volume']) + 1e-10)

            # 做多信號
            if latest.get('enter_long', 0) == 1:
                direction = 1
                strength = 0.55
                confidence = 0.6

                reasons.append("TK金叉")

                # 價格位置
                if current_price > cloud_top:
                    strength += 0.15
                    reasons.append("價格在雲上")
                    distance_pct = (current_price - cloud_top) / cloud_top * 100
                    if distance_pct > 2:
                        strength += 0.05
                        reasons.append(f"遠離雲層({distance_pct:.1f}%)")

                # 雲層顏色
                if cloud_green:
                    strength += 0.1
                    confidence += 0.1
                    reasons.append("多頭雲")

                # TK 關係
                if tenkan > kijun:
                    tk_diff_pct = (tenkan - kijun) / kijun * 100
                    if tk_diff_pct > 0.5:
                        strength += 0.05
                        reasons.append(f"TK差距({tk_diff_pct:.2f}%)")

                # RSI 確認
                if 40 < rsi_val < 70:
                    confidence += 0.05
                    reasons.append(f"RSI健康({rsi_val:.0f})")

            # 做空信號
            elif latest.get('enter_short', 0) == 1:
                direction = -1
                strength = 0.55
                confidence = 0.6

                reasons.append("TK死叉")

                if current_price < cloud_bottom:
                    strength += 0.15
                    reasons.append("價格在雲下")
                    distance_pct = (cloud_bottom - current_price) / cloud_bottom * 100
                    if distance_pct > 2:
                        strength += 0.05
                        reasons.append(f"遠離雲層({distance_pct:.1f}%)")

                if not cloud_green:
                    strength += 0.1
                    confidence += 0.1
                    reasons.append("空頭雲")

                if tenkan < kijun:
                    tk_diff_pct = (kijun - tenkan) / kijun * 100
                    if tk_diff_pct > 0.5:
                        strength += 0.05
                        reasons.append(f"TK差距({tk_diff_pct:.2f}%)")

                if 30 < rsi_val < 60:
                    confidence += 0.05
                    reasons.append(f"RSI健康({rsi_val:.0f})")

            # 平倉信號
            elif position != 0:
                if position > 0 and latest.get('exit_long', 0) == 1:
                    direction = -1
                    strength = 0.65
                    confidence = 0.75
                    reasons.append("一目均衡表平多")
                elif position < 0 and latest.get('exit_short', 0) == 1:
                    direction = 1
                    strength = 0.65
                    confidence = 0.75
                    reasons.append("一目均衡表平空")

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

            # 止損設在雲層
            atr_val = latest.get('atr', current_price * 0.02)
            if direction > 0:
                stop_loss = min(kijun, cloud_top) * 0.99
                take_profit = current_price + (current_price - stop_loss) * 2
            else:
                stop_loss = max(kijun, cloud_bottom) * 1.01
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
                    'tenkan_sen': float(tenkan),
                    'kijun_sen': float(kijun),
                    'cloud_top': float(cloud_top),
                    'cloud_bottom': float(cloud_bottom),
                    'rsi': float(rsi_val),
                },
                reason=" | ".join(reasons)
            )

            self.record_signal(signal)
            logger.info(f"[Ichimoku] {symbol}: {signal_type.value} "
                       f"strength={strength:.2f} conf={confidence:.2f} | {signal.reason}")

            return signal

        except Exception as e:
            logger.error(f"Ichimoku error: {e}")
            return None

    def update(self, market_data: Dict[str, Any]):
        pass
