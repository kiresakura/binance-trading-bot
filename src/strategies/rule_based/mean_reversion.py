"""
均值回歸策略 (優化版)

使用布林帶、RSI 和價格偏離來識別超買超賣機會
適合震盪市場

優化內容：
1. 更靈敏的超買超賣判斷
2. 添加隨機指標確認
3. 改進的入場時機
"""

from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from src.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from src.logger import logger


class MeanReversionStrategy(BaseStrategy):
    """
    均值回歸策略 (優化版)

    信號邏輯：
    1. 價格觸及布林帶邊界或 RSI 極端
    2. 等待反轉確認信號
    3. 在低 ADX 環境中效果最好
    4. 使用隨機指標輔助判斷
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="mean_reversion",
                enabled=True,
                weight=1.0,
                max_position_size=0.15,
                stop_loss_pct=0.012,    # 1.2% 止損（更緊）
                take_profit_pct=0.03,   # 3% 止盈（風報比 1:2.5）
                min_signal_strength=0.45, # 提高門檻減少假信號
                min_confidence=0.55,      # 提高信心門檻
                cooldown_minutes=3,
                params={
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30,
                    'rsi_extreme_high': 80,
                    'rsi_extreme_low': 20,
                    'adx_max': 30,  # 提高 ADX 上限
                    'stoch_overbought': 80,
                    'stoch_oversold': 20,
                    # 成交量確認參數
                    'volume_threshold': 1.2,       # 基礎成交量確認閾值
                    'volume_surge_threshold': 2.0, # 巨量閾值
                    'volume_dry_threshold': 0.8,   # 縮量閾值
                    'volume_surge_bonus': 0.12,    # 巨量加分（均值回歸稍低）
                    'volume_dry_penalty': 0.1,     # 縮量減分
                }
            )
        super().__init__(config)

    def generate_signal(
        self,
        symbol: str,
        features: pd.DataFrame,
        current_price: float,
        position: float = 0.0
    ) -> Optional[Signal]:
        """生成均值回歸信號"""

        if not self.should_generate_signal():
            return None

        if len(features) < 20:
            return None

        try:
            latest = features.iloc[-1]
            prev = features.iloc[-2] if len(features) > 1 else latest

            # 獲取指標
            close = latest.get('close', current_price)
            bb_upper = latest.get('bb_upper', current_price * 1.02)
            bb_middle = latest.get('bb_middle', current_price)
            bb_lower = latest.get('bb_lower', current_price * 0.98)
            bb_width = latest.get('bb_width', 0.04)

            # 計算 BB position
            if bb_upper != bb_lower:
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            else:
                bb_position = 0.5

            rsi = latest.get('rsi_14', 50)
            prev_rsi = prev.get('rsi_14', rsi)
            adx = latest.get('adx', 20)

            stoch_k = latest.get('stoch_k', 50)
            stoch_d = latest.get('stoch_d', 50)
            prev_stoch_k = prev.get('stoch_k', stoch_k)

            # MACD 用於確認
            macd_hist = latest.get('macd_hist', 0)
            prev_macd_hist = prev.get('macd_hist', 0)

            # 成交量數據
            volume = latest.get('volume', 1)
            volume_sma = latest.get('volume_sma_20', volume)
            volume_ratio = volume / volume_sma if volume_sma > 0 else 1.0

            params = self.config.params
            rsi_overbought = params.get('rsi_overbought', 70)
            rsi_oversold = params.get('rsi_oversold', 30)
            rsi_extreme_high = params.get('rsi_extreme_high', 80)
            rsi_extreme_low = params.get('rsi_extreme_low', 20)
            adx_max = params.get('adx_max', 30)
            stoch_overbought = params.get('stoch_overbought', 80)
            stoch_oversold = params.get('stoch_oversold', 20)
            # 成交量參數
            volume_threshold = params.get('volume_threshold', 1.2)
            volume_surge_threshold = params.get('volume_surge_threshold', 2.0)
            volume_dry_threshold = params.get('volume_dry_threshold', 0.8)
            volume_surge_bonus = params.get('volume_surge_bonus', 0.12)
            volume_dry_penalty = params.get('volume_dry_penalty', 0.1)

            # 成交量分析
            volume_confirmed = volume_ratio >= volume_threshold
            is_volume_surge = volume_ratio >= volume_surge_threshold
            is_volume_dry = volume_ratio < volume_dry_threshold

            direction = 0
            strength = 0.0
            confidence = 0.0
            reasons = []

            # 在強趨勢市場中減少交易
            trend_penalty = 1.0
            if adx > adx_max:
                trend_penalty = 0.6
            elif adx > 25:
                trend_penalty = 0.8

            # ====== 超賣做多信號 ======
            is_oversold = (
                bb_position < 0.15 or  # 靠近下軌
                rsi < rsi_oversold or  # RSI 超賣
                (stoch_k < stoch_oversold and stoch_d < stoch_oversold)  # 隨機指標超賣
            )

            # 反轉確認
            rsi_turning_up = rsi > prev_rsi and prev_rsi < rsi_oversold + 5
            stoch_turning_up = stoch_k > prev_stoch_k and prev_stoch_k < stoch_oversold + 5
            macd_turning_up = macd_hist > prev_macd_hist

            if is_oversold:
                direction = 1
                strength = 0.3
                confidence = 0.35

                # 根據超賣程度加分
                if bb_position < 0.1:
                    strength += 0.2
                    confidence += 0.1
                    reasons.append(f"強烈超賣(BB:{bb_position:.2f})")
                elif bb_position < 0.2:
                    strength += 0.1
                    confidence += 0.05
                    reasons.append(f"超賣(BB:{bb_position:.2f})")

                if rsi < rsi_extreme_low:
                    strength += 0.2
                    confidence += 0.15
                    reasons.append(f"RSI極度超賣({rsi:.0f})")
                elif rsi < rsi_oversold:
                    strength += 0.1
                    confidence += 0.1
                    reasons.append(f"RSI超賣({rsi:.0f})")

                if stoch_k < stoch_oversold:
                    strength += 0.1
                    confidence += 0.05
                    reasons.append(f"隨機指標超賣({stoch_k:.0f})")

                # 反轉確認加分
                if rsi_turning_up:
                    strength += 0.15
                    confidence += 0.15
                    reasons.append("RSI反轉確認")

                if stoch_turning_up:
                    strength += 0.1
                    confidence += 0.1
                    reasons.append("隨機指標反轉")

                if macd_turning_up:
                    confidence += 0.1
                    reasons.append("MACD動量改善")

                # 低 ADX 加分
                if adx < 20:
                    confidence += 0.1
                    reasons.append(f"震盪環境(ADX:{adx:.0f})")

                # 成交量確認（均值回歸：放量反轉更可靠）
                if volume_confirmed:
                    strength += 0.08
                    confidence += 0.08
                    reasons.append(f"量能確認({volume_ratio:.1f}x)")

                if is_volume_surge:
                    # 放量超賣反彈更有力
                    strength += volume_surge_bonus
                    confidence += volume_surge_bonus * 0.8
                    reasons.append(f"放量反轉({volume_ratio:.1f}x)")
                elif is_volume_dry:
                    strength -= volume_dry_penalty * 0.5  # 均值回歸縮量懲罰較輕
                    reasons.append(f"縮量警告({volume_ratio:.1f}x)")

            # ====== 超買做空信號 ======
            is_overbought = (
                bb_position > 0.85 or
                rsi > rsi_overbought or
                (stoch_k > stoch_overbought and stoch_d > stoch_overbought)
            )

            rsi_turning_down = rsi < prev_rsi and prev_rsi > rsi_overbought - 5
            stoch_turning_down = stoch_k < prev_stoch_k and prev_stoch_k > stoch_overbought - 5
            macd_turning_down = macd_hist < prev_macd_hist

            if is_overbought and direction == 0:
                direction = -1
                strength = 0.3
                confidence = 0.35

                if bb_position > 0.9:
                    strength += 0.2
                    confidence += 0.1
                    reasons.append(f"強烈超買(BB:{bb_position:.2f})")
                elif bb_position > 0.8:
                    strength += 0.1
                    confidence += 0.05
                    reasons.append(f"超買(BB:{bb_position:.2f})")

                if rsi > rsi_extreme_high:
                    strength += 0.2
                    confidence += 0.15
                    reasons.append(f"RSI極度超買({rsi:.0f})")
                elif rsi > rsi_overbought:
                    strength += 0.1
                    confidence += 0.1
                    reasons.append(f"RSI超買({rsi:.0f})")

                if stoch_k > stoch_overbought:
                    strength += 0.1
                    confidence += 0.05
                    reasons.append(f"隨機指標超買({stoch_k:.0f})")

                if rsi_turning_down:
                    strength += 0.15
                    confidence += 0.15
                    reasons.append("RSI反轉確認")

                if stoch_turning_down:
                    strength += 0.1
                    confidence += 0.1
                    reasons.append("隨機指標反轉")

                if macd_turning_down:
                    confidence += 0.1
                    reasons.append("MACD動量減弱")

                if adx < 20:
                    confidence += 0.1
                    reasons.append(f"震盪環境(ADX:{adx:.0f})")

                # 成交量確認（均值回歸：放量反轉更可靠）
                if volume_confirmed:
                    strength += 0.08
                    confidence += 0.08
                    reasons.append(f"量能確認({volume_ratio:.1f}x)")

                if is_volume_surge:
                    # 放量超買回落更有力
                    strength += volume_surge_bonus
                    confidence += volume_surge_bonus * 0.8
                    reasons.append(f"放量反轉({volume_ratio:.1f}x)")
                elif is_volume_dry:
                    strength -= volume_dry_penalty * 0.5
                    reasons.append(f"縮量警告({volume_ratio:.1f}x)")

            # ====== 平倉信號：回歸到均值 ======
            if position != 0 and direction == 0:
                if position > 0:  # 多頭持倉
                    if current_price >= bb_middle or rsi > 55:
                        direction = -1
                        strength = 0.5
                        confidence = 0.7
                        reasons.append("價格回歸中軌，平多")
                    elif rsi > rsi_overbought:
                        direction = -1
                        strength = 0.6
                        confidence = 0.75
                        reasons.append("RSI超買，平多止盈")

                elif position < 0:  # 空頭持倉
                    if current_price <= bb_middle or rsi < 45:
                        direction = 1
                        strength = 0.5
                        confidence = 0.7
                        reasons.append("價格回歸中軌，平空")
                    elif rsi < rsi_oversold:
                        direction = 1
                        strength = 0.6
                        confidence = 0.75
                        reasons.append("RSI超賣，平空止盈")

            # 應用趨勢懲罰
            strength *= trend_penalty
            confidence *= trend_penalty

            # 檢查最小要求
            if direction == 0:
                return None

            strength = min(1.0, strength)
            confidence = min(1.0, confidence)

            if strength < self.config.min_signal_strength:
                return None

            # 確定信號類型
            if direction > 0:
                signal_type = SignalType.STRONG_BUY if strength > 0.6 else SignalType.BUY
            else:
                signal_type = SignalType.STRONG_SELL if strength > 0.6 else SignalType.SELL

            # 止損止盈目標
            if direction > 0:
                stop_loss = bb_lower * 0.995  # 略低於下軌
                take_profit = bb_middle
            else:
                stop_loss = bb_upper * 1.005  # 略高於上軌
                take_profit = bb_middle

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
                    'bb_position': float(bb_position),
                    'bb_upper': float(bb_upper),
                    'bb_middle': float(bb_middle),
                    'bb_lower': float(bb_lower),
                    'rsi': float(rsi),
                    'stoch_k': float(stoch_k),
                    'adx': float(adx),
                    'volume_ratio': float(volume_ratio),
                    'price': float(current_price)
                },
                reason=" | ".join(reasons)
            )

            self.record_signal(signal)
            logger.info(f"[MeanReversion] {symbol}: {signal_type.value} "
                       f"strength={strength:.2f} conf={confidence:.2f} | {signal.reason}")

            return signal

        except Exception as e:
            logger.error(f"MeanReversion error: {e}")
            return None

    def update(self, market_data: Dict[str, Any]):
        """更新策略狀態"""
        pass
