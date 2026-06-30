"""
MACD 交叉策略

基於 MACD 金叉死叉的經典趨勢策略
- MACD 線上穿信號線 = 做多
- MACD 線下穿信號線 = 做空

v0.4.0 - Enterprise Edition
"""

from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from src.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from src.strategies.indicators import macd, ema, atr, rsi, adx, cross_above, cross_below
from src.strategies.strategy_manager import StrategyMetadata, StrategyCategory
from src.logger import logger


class MACDCrossStrategy(BaseStrategy):
    """MACD 交叉策略"""

    METADATA = StrategyMetadata(
        name='macd_cross',
        display_name='MACD 交叉策略',
        category=StrategyCategory.TREND_FOLLOWING,
        description='基於 MACD 金叉死叉的趨勢跟蹤策略',
        author='Freqtrade Community',
        version='1.0.0',
        timeframe='4h',
        min_candles=100,
        tags=['macd', 'crossover', 'trend'],
        risk_level=2,
        recommended_pairs=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
        hyperopt_enabled=True
    )

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="macd_cross",
                enabled=True,
                weight=1.0,
                max_position_size=0.15,
                stop_loss_pct=0.03,
                take_profit_pct=0.06,
                min_signal_strength=0.4,
                min_confidence=0.5,
                cooldown_minutes=60,
                params={
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9,
                    'ema_filter': 200,
                    'adx_threshold': 20,
                    'histogram_confirm': True,
                }
            )
        super().__init__(config)

    @classmethod
    def get_hyperopt_space(cls) -> Dict[str, Any]:
        return {
            'macd_fast': {'type': 'int', 'low': 8, 'high': 16},
            'macd_slow': {'type': 'int', 'low': 20, 'high': 32},
            'macd_signal': {'type': 'int', 'low': 7, 'high': 12},
            'ema_filter': {'type': 'int', 'low': 100, 'high': 250},
            'adx_threshold': {'type': 'int', 'low': 15, 'high': 30},
        }

    def populate_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params

        # MACD
        dataframe['macd'], dataframe['macd_signal'], dataframe['macd_hist'] = macd(
            dataframe['close'],
            params.get('macd_fast', 12),
            params.get('macd_slow', 26),
            params.get('macd_signal', 9)
        )

        # EMA 趨勢過濾
        dataframe['ema_filter'] = ema(dataframe['close'], params.get('ema_filter', 200))

        # ADX 趨勢強度
        dataframe['adx'] = adx(dataframe, 14)

        # ATR
        dataframe['atr'] = atr(dataframe, 14)

        # RSI
        dataframe['rsi'] = rsi(dataframe['close'], 14)

        # 交叉信號
        dataframe['macd_cross_up'] = cross_above(dataframe['macd'], dataframe['macd_signal'])
        dataframe['macd_cross_down'] = cross_below(dataframe['macd'], dataframe['macd_signal'])

        return dataframe

    def populate_entry_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params
        adx_threshold = params.get('adx_threshold', 20)
        histogram_confirm = params.get('histogram_confirm', True)

        # 做多：MACD 金叉 + 價格在 EMA 上方 + ADX 確認
        long_condition = (
            dataframe['macd_cross_up'] &
            (dataframe['close'] > dataframe['ema_filter']) &
            (dataframe['adx'] > adx_threshold)
        )

        if histogram_confirm:
            long_condition = long_condition & (dataframe['macd_hist'] > 0)

        # 做空：MACD 死叉 + 價格在 EMA 下方 + ADX 確認
        short_condition = (
            dataframe['macd_cross_down'] &
            (dataframe['close'] < dataframe['ema_filter']) &
            (dataframe['adx'] > adx_threshold)
        )

        if histogram_confirm:
            short_condition = short_condition & (dataframe['macd_hist'] < 0)

        dataframe['enter_long'] = long_condition.astype(int)
        dataframe['enter_short'] = short_condition.astype(int)

        return dataframe

    def populate_exit_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # 做多出場：MACD 死叉
        dataframe['exit_long'] = dataframe['macd_cross_down'].astype(int)

        # 做空出場：MACD 金叉
        dataframe['exit_short'] = dataframe['macd_cross_up'].astype(int)

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
            prev = df.iloc[-2] if len(df) > 1 else latest

            direction = 0
            strength = 0.0
            confidence = 0.0
            reasons = []

            macd_val = latest.get('macd', 0)
            macd_signal_val = latest.get('macd_signal', 0)
            macd_hist = latest.get('macd_hist', 0)
            prev_macd_hist = prev.get('macd_hist', 0)
            adx_val = latest.get('adx', 20)
            rsi_val = latest.get('rsi', 50)
            ema_filter = latest.get('ema_filter', current_price)

            # 做多信號
            if latest.get('enter_long', 0) == 1:
                direction = 1
                strength = 0.5

                # MACD 柱狀圖強度
                if macd_hist > 0:
                    hist_strength = min(abs(macd_hist) / (abs(macd_val) + 1e-10), 0.3)
                    strength += hist_strength
                    reasons.append(f"MACD金叉(hist:{macd_hist:.4f})")

                # ADX 趨勢強度
                if adx_val > 25:
                    strength += 0.15
                    confidence = 0.7
                    reasons.append(f"強趨勢(ADX:{adx_val:.0f})")
                else:
                    confidence = 0.55

                # 價格相對於 EMA
                distance_pct = (current_price - ema_filter) / ema_filter * 100
                if 0 < distance_pct < 5:
                    strength += 0.1
                    reasons.append(f"EMA支撐({distance_pct:.1f}%)")

                # RSI 確認
                if 40 < rsi_val < 60:
                    confidence += 0.1
                    reasons.append(f"RSI中性({rsi_val:.0f})")

            # 做空信號
            elif latest.get('enter_short', 0) == 1:
                direction = -1
                strength = 0.5

                if macd_hist < 0:
                    hist_strength = min(abs(macd_hist) / (abs(macd_val) + 1e-10), 0.3)
                    strength += hist_strength
                    reasons.append(f"MACD死叉(hist:{macd_hist:.4f})")

                if adx_val > 25:
                    strength += 0.15
                    confidence = 0.7
                    reasons.append(f"強趨勢(ADX:{adx_val:.0f})")
                else:
                    confidence = 0.55

                distance_pct = (ema_filter - current_price) / ema_filter * 100
                if 0 < distance_pct < 5:
                    strength += 0.1
                    reasons.append(f"EMA壓力({distance_pct:.1f}%)")

                if 40 < rsi_val < 60:
                    confidence += 0.1
                    reasons.append(f"RSI中性({rsi_val:.0f})")

            # 平倉信號
            elif position != 0:
                if position > 0 and latest.get('exit_long', 0) == 1:
                    direction = -1
                    strength = 0.6
                    confidence = 0.7
                    reasons.append("MACD死叉平多")
                elif position < 0 and latest.get('exit_short', 0) == 1:
                    direction = 1
                    strength = 0.6
                    confidence = 0.7
                    reasons.append("MACD金叉平空")

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
                stop_loss = current_price - atr_val * 2.5
                take_profit = current_price + atr_val * 4
            else:
                stop_loss = current_price + atr_val * 2.5
                take_profit = current_price - atr_val * 4

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
                    'macd': float(macd_val),
                    'macd_signal': float(macd_signal_val),
                    'macd_hist': float(macd_hist),
                    'adx': float(adx_val),
                    'rsi': float(rsi_val),
                },
                reason=" | ".join(reasons)
            )

            self.record_signal(signal)
            logger.info(f"[MACD_Cross] {symbol}: {signal_type.value} "
                       f"strength={strength:.2f} conf={confidence:.2f} | {signal.reason}")

            return signal

        except Exception as e:
            logger.error(f"MACD_Cross error: {e}")
            return None

    def update(self, market_data: Dict[str, Any]):
        pass
