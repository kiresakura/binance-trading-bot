"""
趨勢跟蹤策略 (優化版)

使用移動平均線交叉、ADX 和價格動量來識別趨勢
適合趨勢市場

優化內容：
1. 降低信號門檻，更容易觸發
2. 動態參數調整
3. 更好的趨勢確認邏輯
"""

from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from src.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from src.logger import logger


class TrendFollowingStrategy(BaseStrategy):
    """
    趨勢跟蹤策略 (優化版)

    信號邏輯：
    1. 快慢均線交叉判斷趨勢方向
    2. ADX > 20 確認趨勢存在（降低門檻）
    3. 價格動量確認
    4. RSI 過濾極端區域
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="trend_following",
                enabled=True,
                weight=1.0,
                max_position_size=0.2,  # 20%
                stop_loss_pct=0.018,    # 1.8% 止損（優化）
                take_profit_pct=0.045,  # 4.5% 止盈（風報比 1:2.5）
                min_signal_strength=0.45, # 提高門檻減少假信號
                min_confidence=0.55,      # 提高信心門檻
                cooldown_minutes=3,       # 適當冷卻期
                params={
                    'fast_period': 7,     # 更快的均線
                    'slow_period': 21,
                    'adx_threshold': 20,  # 降低 ADX 門檻
                    'rsi_overbought': 75,
                    'rsi_oversold': 25,
                    'use_dynamic_params': True,
                    # 成交量確認參數
                    'volume_threshold': 1.2,       # 基礎成交量確認閾值
                    'volume_surge_threshold': 2.0, # 巨量閾值
                    'volume_dry_threshold': 0.8,   # 縮量閾值
                    'volume_surge_bonus': 0.15,    # 巨量加分
                    'volume_dry_penalty': 0.15,    # 縮量減分
                }
            )
        super().__init__(config)
        self._trend_strength = 0.0
        self._last_adx = 20.0

    def generate_signal(
        self,
        symbol: str,
        features: pd.DataFrame,
        current_price: float,
        position: float = 0.0
    ) -> Optional[Signal]:
        """生成趨勢跟蹤信號"""

        if not self.should_generate_signal():
            return None

        if len(features) < 25:
            return None

        try:
            latest = features.iloc[-1]
            prev = features.iloc[-2] if len(features) > 1 else latest
            prev2 = features.iloc[-3] if len(features) > 2 else prev

            # 獲取指標 - 使用更寬鬆的默認值
            close = latest.get('close', current_price)
            ema_7 = latest.get('ema_7', latest.get('ema_9', close))
            ema_21 = latest.get('ema_21', latest.get('sma_25', close))
            sma_50 = latest.get('sma_50', close)
            adx = latest.get('adx', 25)  # 默認假設有趨勢
            rsi = latest.get('rsi_14', 50)
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            macd_hist = latest.get('macd_hist', macd - macd_signal)

            # 價格動量
            returns_1 = (close - prev.get('close', close)) / prev.get('close', close) if prev.get('close', close) > 0 else 0
            returns_5 = latest.get('returns_5', returns_1 * 5)

            # 成交量數據
            volume = latest.get('volume', 1)
            volume_sma = latest.get('volume_sma_20', volume)
            volume_ratio = volume / volume_sma if volume_sma > 0 else 1.0

            # 前一根的數據
            prev_ema_7 = prev.get('ema_7', prev.get('ema_9', prev.get('close', close)))
            prev_ema_21 = prev.get('ema_21', prev.get('sma_25', prev.get('close', close)))
            prev_macd_hist = prev.get('macd_hist', 0)

            params = self.config.params
            adx_threshold = params.get('adx_threshold', 20)
            rsi_overbought = params.get('rsi_overbought', 75)
            rsi_oversold = params.get('rsi_oversold', 25)
            # 成交量參數
            volume_threshold = params.get('volume_threshold', 1.2)
            volume_surge_threshold = params.get('volume_surge_threshold', 2.0)
            volume_dry_threshold = params.get('volume_dry_threshold', 0.8)
            volume_surge_bonus = params.get('volume_surge_bonus', 0.15)
            volume_dry_penalty = params.get('volume_dry_penalty', 0.15)

            # 成交量分析
            volume_confirmed = volume_ratio >= volume_threshold
            is_volume_surge = volume_ratio >= volume_surge_threshold
            is_volume_dry = volume_ratio < volume_dry_threshold

            direction = 0
            strength = 0.0
            confidence = 0.0
            reasons = []

            # 動態調整 ADX 門檻
            if params.get('use_dynamic_params', True):
                # 根據近期 ADX 調整門檻
                self._last_adx = adx
                if adx > 30:
                    adx_threshold = 18  # 強趨勢時更容易進場
                elif adx < 15:
                    adx_threshold = 25  # 弱趨勢時要求更高

            # 趨勢判斷
            is_trending = adx >= adx_threshold

            # EMA 排列
            ema_bullish = ema_7 > ema_21
            ema_bearish = ema_7 < ema_21

            # EMA 交叉檢測
            ema_cross_up = prev_ema_7 <= prev_ema_21 and ema_7 > ema_21
            ema_cross_down = prev_ema_7 >= prev_ema_21 and ema_7 < ema_21

            # MACD 動量
            macd_bullish = macd_hist > 0
            macd_bearish = macd_hist < 0
            macd_increasing = macd_hist > prev_macd_hist
            macd_decreasing = macd_hist < prev_macd_hist

            # 價格位置
            price_above_ema = current_price > ema_21
            price_below_ema = current_price < ema_21

            # ====== 做多信號 ======
            if ema_bullish and price_above_ema:
                direction = 1
                strength = 0.35  # 基礎強度
                confidence = 0.4
                reasons.append(f"EMA多頭({ema_7:.0f}>{ema_21:.0f})")

                # 趨勢確認加分
                if is_trending:
                    strength += 0.15
                    confidence += 0.1
                    reasons.append(f"ADX確認({adx:.1f})")

                # EMA 金叉加分
                if ema_cross_up:
                    strength += 0.2
                    confidence += 0.15
                    reasons.append("EMA金叉")

                # MACD 確認
                if macd_bullish:
                    strength += 0.1
                    confidence += 0.1
                    reasons.append("MACD正向")

                    if macd_increasing:
                        strength += 0.05
                        reasons.append("動量增強")

                # RSI 確認
                if 40 < rsi < rsi_overbought:
                    confidence += 0.1
                    reasons.append(f"RSI健康({rsi:.0f})")
                elif rsi >= rsi_overbought:
                    strength *= 0.7  # 超買減少強度
                    reasons.append(f"RSI超買警告({rsi:.0f})")
                elif rsi <= rsi_oversold + 10:
                    strength += 0.1  # 超賣反彈機會
                    reasons.append(f"超賣反彈({rsi:.0f})")

                # 動量確認
                if returns_5 > 0.005:  # 0.5% 漲幅
                    confidence += 0.05
                    reasons.append("短期動量正向")

                # 成交量確認
                if volume_confirmed:
                    strength += 0.1
                    confidence += 0.1
                    reasons.append(f"量能確認({volume_ratio:.1f}x)")

                if is_volume_surge:
                    strength += volume_surge_bonus
                    confidence += volume_surge_bonus * 0.8
                    reasons.append(f"巨量突破({volume_ratio:.1f}x)")
                elif is_volume_dry:
                    strength -= volume_dry_penalty
                    confidence -= volume_dry_penalty
                    reasons.append(f"縮量警告({volume_ratio:.1f}x)")

            # ====== 做空信號 ======
            elif ema_bearish and price_below_ema:
                direction = -1
                strength = 0.35
                confidence = 0.4
                reasons.append(f"EMA空頭({ema_7:.0f}<{ema_21:.0f})")

                if is_trending:
                    strength += 0.15
                    confidence += 0.1
                    reasons.append(f"ADX確認({adx:.1f})")

                if ema_cross_down:
                    strength += 0.2
                    confidence += 0.15
                    reasons.append("EMA死叉")

                if macd_bearish:
                    strength += 0.1
                    confidence += 0.1
                    reasons.append("MACD負向")

                    if macd_decreasing:
                        strength += 0.05
                        reasons.append("動量減弱")

                if rsi_oversold < rsi < 60:
                    confidence += 0.1
                    reasons.append(f"RSI健康({rsi:.0f})")
                elif rsi <= rsi_oversold:
                    strength *= 0.7
                    reasons.append(f"RSI超賣警告({rsi:.0f})")
                elif rsi >= rsi_overbought - 10:
                    strength += 0.1
                    reasons.append(f"超買回落({rsi:.0f})")

                if returns_5 < -0.005:
                    confidence += 0.05
                    reasons.append("短期動量負向")

                # 成交量確認
                if volume_confirmed:
                    strength += 0.1
                    confidence += 0.1
                    reasons.append(f"量能確認({volume_ratio:.1f}x)")

                if is_volume_surge:
                    strength += volume_surge_bonus
                    confidence += volume_surge_bonus * 0.8
                    reasons.append(f"巨量下跌({volume_ratio:.1f}x)")
                elif is_volume_dry:
                    strength -= volume_dry_penalty
                    confidence -= volume_dry_penalty
                    reasons.append(f"縮量警告({volume_ratio:.1f}x)")

            # ====== 平倉信號 ======
            elif position != 0:
                if position > 0:  # 持有多倉
                    if ema_cross_down or (rsi > rsi_overbought and macd_decreasing):
                        direction = -1
                        strength = 0.5
                        confidence = 0.65
                        reasons.append("趨勢反轉平多")
                elif position < 0:  # 持有空倉
                    if ema_cross_up or (rsi < rsi_oversold and macd_increasing):
                        direction = 1
                        strength = 0.5
                        confidence = 0.65
                        reasons.append("趨勢反轉平空")

            # 檢查最小要求
            if direction == 0:
                return None

            strength = min(1.0, strength)
            confidence = min(1.0, confidence)

            if strength < self.config.min_signal_strength:
                return None

            # 確定信號類型
            if direction > 0:
                signal_type = SignalType.STRONG_BUY if strength > 0.65 else SignalType.BUY
            else:
                signal_type = SignalType.STRONG_SELL if strength > 0.65 else SignalType.SELL

            # 動態止損止盈
            atr = latest.get('atr', current_price * 0.015)
            if direction > 0:
                stop_loss = current_price - atr * 1.5
                take_profit = current_price + atr * 2.5
            else:
                stop_loss = current_price + atr * 1.5
                take_profit = current_price - atr * 2.5

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
                    'ema_7': float(ema_7),
                    'ema_21': float(ema_21),
                    'adx': float(adx),
                    'rsi': float(rsi),
                    'macd_hist': float(macd_hist),
                    'volume_ratio': float(volume_ratio),
                    'price': float(current_price)
                },
                reason=" | ".join(reasons)
            )

            self.record_signal(signal)
            logger.info(f"[TrendFollowing] {symbol}: {signal_type.value} "
                       f"strength={strength:.2f} conf={confidence:.2f} | {signal.reason}")

            return signal

        except Exception as e:
            logger.error(f"TrendFollowing error: {e}")
            return None

    def update(self, market_data: Dict[str, Any]):
        """更新策略狀態"""
        if 'adx' in market_data:
            self._last_adx = market_data['adx']
