"""
動量策略 (優化版)

使用價格動量、成交量和相對強度來捕捉強勢行情
適合突破和趨勢初期

優化內容：
1. 更靈敏的突破檢測
2. 成交量分析改進
3. 動態倉位管理
"""

from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from src.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from src.logger import logger


class MomentumStrategy(BaseStrategy):
    """
    動量策略 (優化版)

    信號邏輯：
    1. 價格突破近期高點/低點
    2. 成交量放大確認
    3. MACD 動量方向
    4. RSI 動量區間確認
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is None:
            config = StrategyConfig(
                name="momentum",
                enabled=True,
                weight=1.0,
                max_position_size=0.18,
                stop_loss_pct=0.018,     # 1.8% 止損（優化）
                take_profit_pct=0.045,   # 4.5% 止盈（風報比 1:2.5）
                min_signal_strength=0.45, # 提高門檻減少假信號
                min_confidence=0.55,      # 提高信心門檻
                cooldown_minutes=3,
                params={
                    'lookback_period': 15,   # 縮短回看期
                    'volume_threshold': 1.2, # 降低成交量門檻
                    'breakout_pct': 0.003,   # 0.3% 突破確認
                    'momentum_lookback': 5,
                    'use_volume_profile': True,
                    # 成交量異常檢測參數
                    'volume_surge_threshold': 2.0,    # 巨量閾值（>2x 均量）
                    'volume_dry_threshold': 0.8,      # 縮量閾值（<0.8x 均量）
                    'volume_surge_bonus': 0.15,       # 巨量確認加分
                    'volume_dry_penalty': 0.2,        # 無量突破減分
                }
            )
        super().__init__(config)
        self._recent_highs = []
        self._recent_lows = []

    def generate_signal(
        self,
        symbol: str,
        features: pd.DataFrame,
        current_price: float,
        position: float = 0.0
    ) -> Optional[Signal]:
        """生成動量信號"""

        if not self.should_generate_signal():
            return None

        if len(features) < 20:
            return None

        try:
            latest = features.iloc[-1]
            prev = features.iloc[-2] if len(features) > 1 else latest

            # 基礎指標
            close = latest.get('close', current_price)
            high = latest.get('high', close)
            low = latest.get('low', close)
            volume = latest.get('volume', 1)

            rsi = latest.get('rsi_14', 50)
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            macd_hist = latest.get('macd_hist', macd - macd_signal)
            prev_macd_hist = prev.get('macd_hist', 0)

            adx = latest.get('adx', 25)
            atr = latest.get('atr', current_price * 0.015)

            # 成交量分析
            volume_sma = latest.get('volume_sma_20', volume)
            volume_ratio = volume / volume_sma if volume_sma > 0 else 1.0

            # 動量計算
            returns_1 = latest.get('returns_1', 0)
            returns_3 = latest.get('returns_3', returns_1 * 3)
            returns_5 = latest.get('returns_5', returns_1 * 5)

            params = self.config.params
            lookback = params.get('lookback_period', 15)
            volume_threshold = params.get('volume_threshold', 1.2)
            breakout_pct = params.get('breakout_pct', 0.003)
            # 成交量異常檢測參數
            volume_surge_threshold = params.get('volume_surge_threshold', 2.0)
            volume_dry_threshold = params.get('volume_dry_threshold', 0.8)
            volume_surge_bonus = params.get('volume_surge_bonus', 0.15)
            volume_dry_penalty = params.get('volume_dry_penalty', 0.2)

            # 成交量異常分析
            is_volume_surge = volume_ratio >= volume_surge_threshold  # 巨量
            is_volume_dry = volume_ratio < volume_dry_threshold       # 縮量

            # 計算近期高低點
            if 'close' in features.columns:
                close_prices = features['close'].values[-lookback:]
                high_prices = features['high'].values[-lookback:] if 'high' in features.columns else close_prices
                low_prices = features['low'].values[-lookback:] if 'low' in features.columns else close_prices
            else:
                close_prices = [current_price] * lookback
                high_prices = [current_price] * lookback
                low_prices = [current_price] * lookback

            recent_high = np.max(high_prices)
            recent_low = np.min(low_prices)
            price_range = recent_high - recent_low if recent_high > recent_low else current_price * 0.02

            direction = 0
            strength = 0.0
            confidence = 0.0
            reasons = []

            # ====== 計算動量得分 ======
            momentum_score = 0

            # 短期回報得分
            if returns_1 > 0.003:
                momentum_score += 1
            elif returns_1 < -0.003:
                momentum_score -= 1

            # 中期回報得分
            if returns_3 > 0.008:
                momentum_score += 1
            elif returns_3 < -0.008:
                momentum_score -= 1

            if returns_5 > 0.012:
                momentum_score += 1
            elif returns_5 < -0.012:
                momentum_score -= 1

            # MACD 動量得分
            if macd_hist > 0 and macd_hist > prev_macd_hist:
                momentum_score += 1
            elif macd_hist < 0 and macd_hist < prev_macd_hist:
                momentum_score -= 1

            # ====== 突破檢測 ======
            breakout_up_level = recent_high * (1 - breakout_pct)
            breakout_down_level = recent_low * (1 + breakout_pct)

            is_breakout_up = current_price >= breakout_up_level
            is_breakout_down = current_price <= breakout_down_level
            volume_confirmed = volume_ratio >= volume_threshold

            # 價格在範圍中的位置
            if price_range > 0:
                range_position = (current_price - recent_low) / price_range
            else:
                range_position = 0.5

            # ====== 做多信號 ======
            if momentum_score >= 1 or is_breakout_up:
                direction = 1
                strength = 0.3
                confidence = 0.35

                # 突破加分
                if is_breakout_up:
                    strength += 0.2
                    confidence += 0.15
                    reasons.append(f"向上突破({recent_high:.0f})")

                # 動量得分加分
                if momentum_score >= 2:
                    strength += 0.15
                    confidence += 0.1
                    reasons.append(f"強動量({momentum_score})")
                elif momentum_score >= 1:
                    strength += 0.1
                    confidence += 0.05
                    reasons.append(f"正動量({momentum_score})")

                # 成交量確認
                if volume_confirmed:
                    strength += 0.15
                    confidence += 0.15
                    reasons.append(f"量能確認({volume_ratio:.1f}x)")

                # 成交量異常檢測
                if is_volume_surge:
                    # 巨量突破 - 加分
                    strength += volume_surge_bonus
                    confidence += volume_surge_bonus * 0.8
                    reasons.append(f"巨量確認({volume_ratio:.1f}x)")
                elif is_volume_dry and is_breakout_up:
                    # 無量突破 - 減分（可能是假突破）
                    strength -= volume_dry_penalty
                    confidence -= volume_dry_penalty
                    reasons.append(f"無量突破警告({volume_ratio:.1f}x)")

                # MACD 確認
                if macd_hist > 0:
                    confidence += 0.1
                    reasons.append("MACD正向")
                    if macd_hist > prev_macd_hist:
                        strength += 0.05
                        reasons.append("動量增強")

                # RSI 區間
                if 45 < rsi < 70:
                    confidence += 0.1
                    reasons.append(f"RSI健康({rsi:.0f})")
                elif rsi >= 70:
                    strength *= 0.8
                    reasons.append(f"RSI偏高({rsi:.0f})")
                elif rsi <= 35:
                    strength += 0.1  # 超賣反彈
                    confidence += 0.05
                    reasons.append(f"超賣反彈({rsi:.0f})")

                # ADX 趨勢強度
                if adx > 25:
                    confidence += 0.1
                    reasons.append(f"趨勢強勁(ADX:{adx:.0f})")

            # ====== 做空信號 ======
            elif momentum_score <= -1 or is_breakout_down:
                direction = -1
                strength = 0.3
                confidence = 0.35

                if is_breakout_down:
                    strength += 0.2
                    confidence += 0.15
                    reasons.append(f"向下突破({recent_low:.0f})")

                if momentum_score <= -2:
                    strength += 0.15
                    confidence += 0.1
                    reasons.append(f"強空動量({momentum_score})")
                elif momentum_score <= -1:
                    strength += 0.1
                    confidence += 0.05
                    reasons.append(f"負動量({momentum_score})")

                if volume_confirmed:
                    strength += 0.15
                    confidence += 0.15
                    reasons.append(f"量能確認({volume_ratio:.1f}x)")

                # 成交量異常檢測
                if is_volume_surge:
                    # 巨量下跌 - 加分（恐慌賣壓）
                    strength += volume_surge_bonus
                    confidence += volume_surge_bonus * 0.8
                    reasons.append(f"巨量確認({volume_ratio:.1f}x)")
                elif is_volume_dry and is_breakout_down:
                    # 無量跌破 - 減分（可能是假跌破）
                    strength -= volume_dry_penalty
                    confidence -= volume_dry_penalty
                    reasons.append(f"無量跌破警告({volume_ratio:.1f}x)")

                if macd_hist < 0:
                    confidence += 0.1
                    reasons.append("MACD負向")
                    if macd_hist < prev_macd_hist:
                        strength += 0.05
                        reasons.append("動量減弱")

                if 30 < rsi < 55:
                    confidence += 0.1
                    reasons.append(f"RSI健康({rsi:.0f})")
                elif rsi <= 30:
                    strength *= 0.8
                    reasons.append(f"RSI偏低({rsi:.0f})")
                elif rsi >= 65:
                    strength += 0.1
                    confidence += 0.05
                    reasons.append(f"超買回落({rsi:.0f})")

                if adx > 25:
                    confidence += 0.1
                    reasons.append(f"趨勢強勁(ADX:{adx:.0f})")

            # ====== 平倉信號 ======
            elif position != 0:
                if position > 0:  # 多頭
                    if momentum_score <= -1 or rsi > 75:
                        direction = -1
                        strength = 0.5
                        confidence = 0.65
                        reasons.append("動量衰竭平多")
                elif position < 0:  # 空頭
                    if momentum_score >= 1 or rsi < 25:
                        direction = 1
                        strength = 0.5
                        confidence = 0.65
                        reasons.append("動量反轉平空")

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

            # 動態止損止盈（基於 ATR）
            atr_multiplier = 1.8
            if direction > 0:
                stop_loss = current_price - atr * atr_multiplier
                take_profit = current_price + atr * atr_multiplier * 2.2
            else:
                stop_loss = current_price + atr * atr_multiplier
                take_profit = current_price - atr * atr_multiplier * 2.2

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
                    'momentum_score': float(momentum_score),
                    'volume_ratio': float(volume_ratio),
                    'rsi': float(rsi),
                    'macd_hist': float(macd_hist),
                    'adx': float(adx),
                    'recent_high': float(recent_high),
                    'recent_low': float(recent_low),
                    'price': float(current_price)
                },
                reason=" | ".join(reasons)
            )

            self.record_signal(signal)
            logger.info(f"[Momentum] {symbol}: {signal_type.value} "
                       f"strength={strength:.2f} conf={confidence:.2f} | {signal.reason}")

            return signal

        except Exception as e:
            logger.error(f"Momentum error: {e}")
            return None

    def update(self, market_data: Dict[str, Any]):
        """更新策略狀態"""
        if 'high' in market_data:
            self._recent_highs.append(market_data['high'])
            self._recent_highs = self._recent_highs[-20:]
        if 'low' in market_data:
            self._recent_lows.append(market_data['low'])
            self._recent_lows = self._recent_lows[-20:]
