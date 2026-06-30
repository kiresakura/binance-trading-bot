"""
多時間框架策略

結合多個時間框架分析的策略
- 高時間框架確認趨勢方向
- 低時間框架尋找入場點

v0.4.0 - Enterprise Edition
"""

from datetime import datetime
from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np

from src.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from src.strategies.indicators import ema, sma, rsi, macd, atr, adx, bollinger_bands
from src.strategies.strategy_manager import StrategyMetadata, StrategyCategory
from src.logger import logger


class MultiTimeframeStrategy(BaseStrategy):
    """多時間框架策略"""

    METADATA = StrategyMetadata(
        name='multi_timeframe',
        display_name='多時間框架策略',
        category=StrategyCategory.SWING,
        description='結合多個時間框架分析的波段交易策略',
        author='Freqtrade Community',
        version='1.0.0',
        timeframe='1h',
        min_candles=100,
        tags=['multi_timeframe', 'swing', 'trend'],
        risk_level=2,
        recommended_pairs=['BTCUSDT', 'ETHUSDT'],
        hyperopt_enabled=True
    )

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="multi_timeframe",
                enabled=True,
                weight=1.0,
                max_position_size=0.15,
                stop_loss_pct=0.025,
                take_profit_pct=0.05,
                min_signal_strength=0.5,
                min_confidence=0.55,
                cooldown_minutes=30,
                params={
                    # 高時間框架參數（模擬 4h）
                    'htf_ema_fast': 50,  # 對應 4h 的 EMA12-13
                    'htf_ema_slow': 200,  # 對應 4h 的 EMA50
                    'htf_rsi_period': 56,  # 對應 4h 的 RSI14
                    # 低時間框架參數（1h）
                    'ltf_ema_fast': 12,
                    'ltf_ema_slow': 26,
                    'ltf_rsi_period': 14,
                    'ltf_rsi_oversold': 30,
                    'ltf_rsi_overbought': 70,
                    # 信號確認
                    'require_htf_trend': True,
                    'require_ltf_pullback': True,
                }
            )
        super().__init__(config)

    @classmethod
    def get_hyperopt_space(cls) -> Dict[str, Any]:
        return {
            'htf_ema_fast': {'type': 'int', 'low': 40, 'high': 80},
            'htf_ema_slow': {'type': 'int', 'low': 150, 'high': 250},
            'ltf_ema_fast': {'type': 'int', 'low': 8, 'high': 20},
            'ltf_ema_slow': {'type': 'int', 'low': 20, 'high': 40},
            'ltf_rsi_oversold': {'type': 'int', 'low': 25, 'high': 40},
            'ltf_rsi_overbought': {'type': 'int', 'low': 60, 'high': 75},
        }

    @property
    def informative_timeframes(self) -> List[str]:
        """輔助時間框架"""
        return ['4h']

    def populate_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params

        # === 模擬高時間框架指標（使用更長週期）===
        # 這模擬了 4h 時間框架的行為
        dataframe['htf_ema_fast'] = ema(dataframe['close'], params.get('htf_ema_fast', 50))
        dataframe['htf_ema_slow'] = ema(dataframe['close'], params.get('htf_ema_slow', 200))
        dataframe['htf_rsi'] = rsi(dataframe['close'], params.get('htf_rsi_period', 56))

        # 高時間框架趨勢
        dataframe['htf_trend'] = np.where(
            dataframe['htf_ema_fast'] > dataframe['htf_ema_slow'], 1,
            np.where(dataframe['htf_ema_fast'] < dataframe['htf_ema_slow'], -1, 0)
        )

        # === 低時間框架指標（1h）===
        dataframe['ltf_ema_fast'] = ema(dataframe['close'], params.get('ltf_ema_fast', 12))
        dataframe['ltf_ema_slow'] = ema(dataframe['close'], params.get('ltf_ema_slow', 26))
        dataframe['ltf_rsi'] = rsi(dataframe['close'], params.get('ltf_rsi_period', 14))

        # MACD
        dataframe['macd'], dataframe['macd_signal'], dataframe['macd_hist'] = macd(
            dataframe['close'], 12, 26, 9
        )

        # 布林帶
        dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = bollinger_bands(
            dataframe['close'], 20, 2.0
        )
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lower']) / (
            dataframe['bb_upper'] - dataframe['bb_lower'] + 1e-10
        )

        # ATR
        dataframe['atr'] = atr(dataframe, 14)

        # ADX
        dataframe['adx'] = adx(dataframe, 14)

        # 成交量
        dataframe['volume_sma'] = sma(dataframe['volume'], 20)

        # 回調檢測
        dataframe['pullback_to_ema'] = (
            (dataframe['close'] > dataframe['ltf_ema_slow'] * 0.99) &
            (dataframe['close'] < dataframe['ltf_ema_slow'] * 1.01)
        )

        return dataframe

    def populate_entry_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params
        require_htf_trend = params.get('require_htf_trend', True)
        require_ltf_pullback = params.get('require_ltf_pullback', True)
        ltf_rsi_oversold = params.get('ltf_rsi_oversold', 30)
        ltf_rsi_overbought = params.get('ltf_rsi_overbought', 70)

        # 做多條件
        # 高時間框架上升趨勢 + 低時間框架回調到支撐
        long_condition = pd.Series([True] * len(dataframe), index=dataframe.index)

        if require_htf_trend:
            long_condition = long_condition & (dataframe['htf_trend'] == 1)
            long_condition = long_condition & (dataframe['htf_rsi'] > 40)

        # 低時間框架入場條件
        ltf_long_setup = (
            (dataframe['ltf_rsi'] < 45) &  # RSI 回調
            (dataframe['close'] > dataframe['ltf_ema_slow']) &  # 價格在均線上
            (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1)) &  # MACD 轉向
            (dataframe['bb_percent'] < 0.5)  # 在布林帶下半部
        )

        if require_ltf_pullback:
            ltf_long_setup = ltf_long_setup | dataframe['pullback_to_ema']

        long_condition = long_condition & ltf_long_setup

        # 做空條件
        short_condition = pd.Series([True] * len(dataframe), index=dataframe.index)

        if require_htf_trend:
            short_condition = short_condition & (dataframe['htf_trend'] == -1)
            short_condition = short_condition & (dataframe['htf_rsi'] < 60)

        ltf_short_setup = (
            (dataframe['ltf_rsi'] > 55) &
            (dataframe['close'] < dataframe['ltf_ema_slow']) &
            (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1)) &
            (dataframe['bb_percent'] > 0.5)
        )

        short_condition = short_condition & ltf_short_setup

        dataframe['enter_long'] = long_condition.astype(int)
        dataframe['enter_short'] = short_condition.astype(int)

        return dataframe

    def populate_exit_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        params = self.config.params
        ltf_rsi_overbought = params.get('ltf_rsi_overbought', 70)
        ltf_rsi_oversold = params.get('ltf_rsi_oversold', 30)

        # 出場條件
        dataframe['exit_long'] = (
            (dataframe['htf_trend'] == -1) |  # 高時間框架趨勢反轉
            (dataframe['ltf_rsi'] > ltf_rsi_overbought) |  # 超買
            (dataframe['close'] < dataframe['ltf_ema_slow'] * 0.98)  # 跌破均線
        ).astype(int)

        dataframe['exit_short'] = (
            (dataframe['htf_trend'] == 1) |
            (dataframe['ltf_rsi'] < ltf_rsi_oversold) |
            (dataframe['close'] > dataframe['ltf_ema_slow'] * 1.02)
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

            params = self.config.params
            htf_trend = latest.get('htf_trend', 0)
            htf_rsi = latest.get('htf_rsi', 50)
            ltf_rsi = latest.get('ltf_rsi', 50)
            ltf_ema_fast = latest.get('ltf_ema_fast', current_price)
            ltf_ema_slow = latest.get('ltf_ema_slow', current_price)
            macd_hist = latest.get('macd_hist', 0)
            bb_percent = latest.get('bb_percent', 0.5)
            adx_val = latest.get('adx', 20)
            volume_ratio = latest['volume'] / (latest.get('volume_sma', latest['volume']) + 1e-10)

            # 做多信號
            if latest.get('enter_long', 0) == 1:
                direction = 1
                strength = 0.55
                confidence = 0.6

                # 高時間框架趨勢確認
                if htf_trend == 1:
                    strength += 0.15
                    reasons.append("HTF上升趨勢")
                    if htf_rsi > 50:
                        confidence += 0.1
                        reasons.append(f"HTF_RSI強勢({htf_rsi:.0f})")

                # 低時間框架回調
                if ltf_rsi < 40:
                    strength += 0.1
                    reasons.append(f"LTF回調(RSI:{ltf_rsi:.0f})")

                # MACD 確認
                if macd_hist > 0:
                    strength += 0.05
                    reasons.append("MACD正向")

                # 布林帶位置
                if bb_percent < 0.3:
                    strength += 0.05
                    reasons.append(f"BB低位({bb_percent:.2f})")

                # 成交量
                if volume_ratio > 1.2:
                    confidence += 0.05
                    reasons.append(f"放量({volume_ratio:.1f}x)")

            # 做空信號
            elif latest.get('enter_short', 0) == 1:
                direction = -1
                strength = 0.55
                confidence = 0.6

                if htf_trend == -1:
                    strength += 0.15
                    reasons.append("HTF下降趨勢")
                    if htf_rsi < 50:
                        confidence += 0.1
                        reasons.append(f"HTF_RSI弱勢({htf_rsi:.0f})")

                if ltf_rsi > 60:
                    strength += 0.1
                    reasons.append(f"LTF反彈(RSI:{ltf_rsi:.0f})")

                if macd_hist < 0:
                    strength += 0.05
                    reasons.append("MACD負向")

                if bb_percent > 0.7:
                    strength += 0.05
                    reasons.append(f"BB高位({bb_percent:.2f})")

                if volume_ratio > 1.2:
                    confidence += 0.05
                    reasons.append(f"放量({volume_ratio:.1f}x)")

            # 平倉信號
            elif position != 0:
                if position > 0 and latest.get('exit_long', 0) == 1:
                    direction = -1
                    strength = 0.65
                    confidence = 0.75
                    reasons.append("MTF平多")
                elif position < 0 and latest.get('exit_short', 0) == 1:
                    direction = 1
                    strength = 0.65
                    confidence = 0.75
                    reasons.append("MTF平空")

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

            # 止損止盈
            atr_val = latest.get('atr', current_price * 0.02)
            if direction > 0:
                stop_loss = ltf_ema_slow * 0.98
                take_profit = current_price + (current_price - stop_loss) * 2.5
            else:
                stop_loss = ltf_ema_slow * 1.02
                take_profit = current_price - (stop_loss - current_price) * 2.5

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
                    'htf_trend': int(htf_trend),
                    'htf_rsi': float(htf_rsi),
                    'ltf_rsi': float(ltf_rsi),
                    'bb_percent': float(bb_percent),
                    'adx': float(adx_val),
                },
                reason=" | ".join(reasons)
            )

            self.record_signal(signal)
            logger.info(f"[MultiTF] {symbol}: {signal_type.value} "
                       f"strength={strength:.2f} conf={confidence:.2f} | {signal.reason}")

            return signal

        except Exception as e:
            logger.error(f"MultiTF error: {e}")
            return None

    def update(self, market_data: Dict[str, Any]):
        pass
