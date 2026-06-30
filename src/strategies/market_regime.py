"""
市場狀態分析模組

判斷市場狀態（趨勢/震盪/高波動）並動態調整策略權重
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from src.logger import logger


class MarketRegime(Enum):
    """市場狀態枚舉"""
    STRONG_TREND_UP = "strong_trend_up"      # 強勢上漲趨勢
    TREND_UP = "trend_up"                     # 上漲趨勢
    WEAK_TREND_UP = "weak_trend_up"           # 弱上漲趨勢
    STRONG_TREND_DOWN = "strong_trend_down"  # 強勢下跌趨勢
    TREND_DOWN = "trend_down"                 # 下跌趨勢
    WEAK_TREND_DOWN = "weak_trend_down"       # 弱下跌趨勢
    RANGING = "ranging"                       # 震盪整理
    HIGH_VOLATILITY = "high_volatility"       # 高波動
    LOW_VOLATILITY = "low_volatility"         # 低波動（收斂）
    BREAKOUT_UP = "breakout_up"               # 向上突破
    BREAKOUT_DOWN = "breakout_down"           # 向下突破


@dataclass
class RegimeAnalysis:
    """市場狀態分析結果"""
    regime: MarketRegime
    confidence: float  # 0-1
    trend_strength: float  # -1 到 1（負為下跌，正為上漲）
    volatility_level: float  # 0-1
    regime_duration: int  # 持續的 K 線數

    # 各狀態概率
    regime_probabilities: Dict[str, float] = field(default_factory=dict)

    # 策略權重建議
    strategy_weights: Dict[str, float] = field(default_factory=dict)

    # 附加信息
    indicators: Dict[str, float] = field(default_factory=dict)
    analysis_time: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'regime': self.regime.value,
            'confidence': self.confidence,
            'trend_strength': self.trend_strength,
            'volatility_level': self.volatility_level,
            'regime_duration': self.regime_duration,
            'regime_probabilities': self.regime_probabilities,
            'strategy_weights': self.strategy_weights,
            'indicators': self.indicators,
            'analysis_time': self.analysis_time.isoformat()
        }


class MarketRegimeAnalyzer:
    """
    市場狀態分析器

    功能：
    1. 分析當前市場狀態（趨勢/震盪/高波動）
    2. 根據狀態動態調整策略權重
    3. 檢測狀態轉換
    """

    # 策略適用的市場狀態映射
    STRATEGY_REGIME_MAPPING = {
        'trend_following': {
            MarketRegime.STRONG_TREND_UP: 1.5,
            MarketRegime.TREND_UP: 1.2,
            MarketRegime.WEAK_TREND_UP: 0.8,
            MarketRegime.STRONG_TREND_DOWN: 1.5,
            MarketRegime.TREND_DOWN: 1.2,
            MarketRegime.WEAK_TREND_DOWN: 0.8,
            MarketRegime.RANGING: 0.3,
            MarketRegime.HIGH_VOLATILITY: 0.5,
            MarketRegime.LOW_VOLATILITY: 0.4,
            MarketRegime.BREAKOUT_UP: 1.0,
            MarketRegime.BREAKOUT_DOWN: 1.0,
        },
        'momentum': {
            MarketRegime.STRONG_TREND_UP: 1.4,
            MarketRegime.TREND_UP: 1.1,
            MarketRegime.WEAK_TREND_UP: 0.7,
            MarketRegime.STRONG_TREND_DOWN: 1.4,
            MarketRegime.TREND_DOWN: 1.1,
            MarketRegime.WEAK_TREND_DOWN: 0.7,
            MarketRegime.RANGING: 0.4,
            MarketRegime.HIGH_VOLATILITY: 0.6,
            MarketRegime.LOW_VOLATILITY: 0.3,
            MarketRegime.BREAKOUT_UP: 1.5,
            MarketRegime.BREAKOUT_DOWN: 1.5,
        },
        'mean_reversion': {
            MarketRegime.STRONG_TREND_UP: 0.2,
            MarketRegime.TREND_UP: 0.4,
            MarketRegime.WEAK_TREND_UP: 0.7,
            MarketRegime.STRONG_TREND_DOWN: 0.2,
            MarketRegime.TREND_DOWN: 0.4,
            MarketRegime.WEAK_TREND_DOWN: 0.7,
            MarketRegime.RANGING: 1.5,
            MarketRegime.HIGH_VOLATILITY: 0.8,
            MarketRegime.LOW_VOLATILITY: 1.2,
            MarketRegime.BREAKOUT_UP: 0.3,
            MarketRegime.BREAKOUT_DOWN: 0.3,
        },
    }

    def __init__(self):
        # 歷史狀態記錄
        self._regime_history: List[Tuple[datetime, MarketRegime]] = []
        self._current_regime = MarketRegime.RANGING
        self._regime_start_time: Optional[datetime] = None
        self._regime_bar_count = 0

        # 配置參數
        self.config = {
            # ADX 閾值
            'adx_strong_trend': 35,
            'adx_trend': 25,
            'adx_weak_trend': 18,
            'adx_no_trend': 15,

            # 波動率閾值（基於 ATR 百分比）
            'volatility_high': 0.025,   # 2.5%
            'volatility_low': 0.008,    # 0.8%

            # 布林帶寬度閾值
            'bb_width_high': 0.05,      # 5%
            'bb_width_low': 0.015,      # 1.5%

            # 突破確認
            'breakout_volume_ratio': 1.5,
            'breakout_price_pct': 0.005,  # 0.5%

            # 狀態確認所需 K 線數
            'regime_confirm_bars': 3,
        }

        logger.info("MarketRegimeAnalyzer initialized")

    def analyze(self, features: pd.DataFrame, current_price: float) -> RegimeAnalysis:
        """
        分析市場狀態

        Args:
            features: 特徵 DataFrame
            current_price: 當前價格

        Returns:
            RegimeAnalysis 結果
        """
        if len(features) < 30:
            return self._create_default_analysis()

        latest = features.iloc[-1]

        # 計算各項指標
        indicators = self._calculate_indicators(features, latest, current_price)

        # 計算各狀態概率
        regime_probs = self._calculate_regime_probabilities(indicators)

        # 確定當前狀態
        regime, confidence = self._determine_regime(regime_probs, indicators)

        # 計算趨勢強度
        trend_strength = self._calculate_trend_strength(indicators)

        # 計算波動率水平
        volatility_level = self._calculate_volatility_level(indicators)

        # 更新狀態歷史
        self._update_regime_history(regime)

        # 計算策略權重
        strategy_weights = self._calculate_strategy_weights(regime, confidence, indicators)

        analysis = RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_level=volatility_level,
            regime_duration=self._regime_bar_count,
            regime_probabilities=regime_probs,
            strategy_weights=strategy_weights,
            indicators=indicators
        )

        logger.debug(f"Market regime: {regime.value}, confidence: {confidence:.2f}, "
                    f"trend_strength: {trend_strength:.2f}")

        return analysis

    def _calculate_indicators(
        self,
        features: pd.DataFrame,
        latest: pd.Series,
        current_price: float
    ) -> Dict[str, float]:
        """計算分析指標"""
        indicators = {}

        # ADX - 趨勢強度
        indicators['adx'] = latest.get('adx', 20)

        # 趨勢方向 EMA
        ema_9 = latest.get('ema_9', latest.get('ema_7', current_price))
        ema_21 = latest.get('ema_21', latest.get('sma_25', current_price))
        sma_50 = latest.get('sma_50', current_price)

        indicators['ema_9'] = ema_9
        indicators['ema_21'] = ema_21
        indicators['sma_50'] = sma_50
        indicators['ema_trend'] = 1 if ema_9 > ema_21 else (-1 if ema_9 < ema_21 else 0)
        indicators['price_vs_sma50'] = (current_price - sma_50) / sma_50 if sma_50 > 0 else 0

        # RSI
        indicators['rsi'] = latest.get('rsi_14', 50)

        # MACD
        indicators['macd_hist'] = latest.get('macd_hist', 0)
        indicators['macd'] = latest.get('macd', 0)

        # 布林帶
        bb_upper = latest.get('bb_upper', current_price * 1.02)
        bb_lower = latest.get('bb_lower', current_price * 0.98)
        bb_middle = latest.get('bb_middle', current_price)
        bb_width = latest.get('bb_width', 0.04)

        indicators['bb_upper'] = bb_upper
        indicators['bb_lower'] = bb_lower
        indicators['bb_middle'] = bb_middle
        indicators['bb_width'] = bb_width

        # BB position
        if bb_upper != bb_lower:
            indicators['bb_position'] = (current_price - bb_lower) / (bb_upper - bb_lower)
        else:
            indicators['bb_position'] = 0.5

        # ATR 波動率
        atr = latest.get('atr', current_price * 0.015)
        indicators['atr'] = atr
        indicators['atr_pct'] = atr / current_price if current_price > 0 else 0

        # 波動率
        indicators['volatility'] = latest.get('volatility_10', indicators['atr_pct'])

        # 成交量
        volume = latest.get('volume', 1)
        volume_sma = latest.get('volume_sma_20', volume)
        indicators['volume_ratio'] = volume / volume_sma if volume_sma > 0 else 1

        # 計算斜率（使用最近 10 根 K 線）
        if 'close' in features.columns and len(features) >= 10:
            closes = features['close'].values[-10:]
            x = np.arange(len(closes))
            slope = np.polyfit(x, closes, 1)[0]
            indicators['price_slope'] = slope / current_price if current_price > 0 else 0
        else:
            indicators['price_slope'] = 0

        # 計算近期高低點
        if 'high' in features.columns and 'low' in features.columns:
            highs = features['high'].values[-20:]
            lows = features['low'].values[-20:]
            recent_high = np.max(highs)
            recent_low = np.min(lows)

            indicators['recent_high'] = recent_high
            indicators['recent_low'] = recent_low
            indicators['price_vs_high'] = (current_price - recent_high) / recent_high if recent_high > 0 else 0
            indicators['price_vs_low'] = (current_price - recent_low) / recent_low if recent_low > 0 else 0

        # 布林帶收斂（squeeze）
        if len(features) >= 20 and 'bb_width' in features.columns:
            bb_widths = features['bb_width'].values[-20:]
            current_bb_width = bb_width
            avg_bb_width = np.mean(bb_widths)
            indicators['bb_squeeze'] = current_bb_width < avg_bb_width * 0.6
        else:
            indicators['bb_squeeze'] = False

        return indicators

    def _calculate_regime_probabilities(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """計算各市場狀態的概率"""
        probs = {regime.value: 0.0 for regime in MarketRegime}

        adx = indicators['adx']
        ema_trend = indicators['ema_trend']
        rsi = indicators['rsi']
        bb_position = indicators['bb_position']
        bb_width = indicators['bb_width']
        volatility = indicators['volatility']
        volume_ratio = indicators['volume_ratio']
        price_slope = indicators['price_slope']
        bb_squeeze = indicators.get('bb_squeeze', False)
        price_vs_high = indicators.get('price_vs_high', 0)
        price_vs_low = indicators.get('price_vs_low', 0)

        config = self.config

        # ====== 趨勢狀態 ======
        # 強勢上漲
        if adx > config['adx_strong_trend'] and ema_trend > 0 and price_slope > 0.001:
            probs[MarketRegime.STRONG_TREND_UP.value] = min(0.9, 0.4 + (adx - 35) * 0.02)
        elif adx > config['adx_trend'] and ema_trend > 0:
            probs[MarketRegime.TREND_UP.value] = min(0.8, 0.3 + (adx - 25) * 0.03)
        elif adx > config['adx_weak_trend'] and ema_trend > 0:
            probs[MarketRegime.WEAK_TREND_UP.value] = min(0.6, 0.2 + (adx - 18) * 0.03)

        # 強勢下跌
        if adx > config['adx_strong_trend'] and ema_trend < 0 and price_slope < -0.001:
            probs[MarketRegime.STRONG_TREND_DOWN.value] = min(0.9, 0.4 + (adx - 35) * 0.02)
        elif adx > config['adx_trend'] and ema_trend < 0:
            probs[MarketRegime.TREND_DOWN.value] = min(0.8, 0.3 + (adx - 25) * 0.03)
        elif adx > config['adx_weak_trend'] and ema_trend < 0:
            probs[MarketRegime.WEAK_TREND_DOWN.value] = min(0.6, 0.2 + (adx - 18) * 0.03)

        # ====== 震盪狀態 ======
        if adx < config['adx_no_trend'] and 0.25 < bb_position < 0.75:
            probs[MarketRegime.RANGING.value] = min(0.8, 0.4 + (config['adx_no_trend'] - adx) * 0.03)
        elif adx < config['adx_weak_trend'] and 0.2 < bb_position < 0.8:
            probs[MarketRegime.RANGING.value] = max(probs[MarketRegime.RANGING.value], 0.3)

        # ====== 波動狀態 ======
        if volatility > config['volatility_high'] or bb_width > config['bb_width_high']:
            probs[MarketRegime.HIGH_VOLATILITY.value] = min(0.85, 0.4 + volatility * 10)

        if volatility < config['volatility_low'] or bb_width < config['bb_width_low']:
            probs[MarketRegime.LOW_VOLATILITY.value] = min(0.8, 0.5 + (config['volatility_low'] - volatility) * 20)

        # ====== 突破狀態 ======
        # 向上突破
        if (price_vs_high > -0.01 and  # 接近或突破近期高點
            volume_ratio > config['breakout_volume_ratio'] and
            (bb_squeeze or bb_position > 0.9)):
            probs[MarketRegime.BREAKOUT_UP.value] = min(0.85, 0.4 + volume_ratio * 0.15)

        # 向下突破
        if (price_vs_low < 0.01 and  # 接近或跌破近期低點
            volume_ratio > config['breakout_volume_ratio'] and
            (bb_squeeze or bb_position < 0.1)):
            probs[MarketRegime.BREAKOUT_DOWN.value] = min(0.85, 0.4 + volume_ratio * 0.15)

        # 正規化概率
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            probs[MarketRegime.RANGING.value] = 1.0

        return probs

    def _determine_regime(
        self,
        regime_probs: Dict[str, float],
        indicators: Dict[str, float]
    ) -> Tuple[MarketRegime, float]:
        """確定當前市場狀態"""
        # 找出概率最高的狀態
        max_regime = max(regime_probs, key=regime_probs.get)
        max_prob = regime_probs[max_regime]

        # 將字符串轉回枚舉
        regime = MarketRegime(max_regime)

        # 計算置信度（考慮與第二高概率的差距）
        sorted_probs = sorted(regime_probs.values(), reverse=True)
        if len(sorted_probs) > 1:
            prob_gap = sorted_probs[0] - sorted_probs[1]
            confidence = min(1.0, max_prob + prob_gap * 0.5)
        else:
            confidence = max_prob

        return regime, confidence

    def _calculate_trend_strength(self, indicators: Dict[str, float]) -> float:
        """計算趨勢強度（-1 到 1）"""
        adx = indicators['adx']
        ema_trend = indicators['ema_trend']
        rsi = indicators['rsi']
        macd_hist = indicators['macd_hist']
        price_slope = indicators['price_slope']

        # ADX 標準化（0-1）
        adx_normalized = min(1.0, adx / 50)

        # RSI 偏離度（-1 到 1）
        rsi_deviation = (rsi - 50) / 50

        # 價格斜率標準化
        slope_normalized = np.clip(price_slope * 100, -1, 1)

        # MACD 方向
        macd_direction = 1 if macd_hist > 0 else (-1 if macd_hist < 0 else 0)

        # 綜合計算
        direction = ema_trend
        if direction == 0:
            direction = 1 if rsi > 50 else -1

        strength = (
            adx_normalized * 0.4 +
            abs(rsi_deviation) * 0.2 +
            abs(slope_normalized) * 0.2 +
            0.2
        ) * direction

        return np.clip(strength, -1, 1)

    def _calculate_volatility_level(self, indicators: Dict[str, float]) -> float:
        """計算波動率水平（0-1）"""
        volatility = indicators['volatility']
        bb_width = indicators['bb_width']
        atr_pct = indicators['atr_pct']

        # 綜合波動率指標
        combined = (volatility * 0.4 + bb_width * 0.3 + atr_pct * 0.3)

        # 標準化到 0-1（假設 5% 為極端波動）
        normalized = min(1.0, combined / 0.05)

        return normalized

    def _update_regime_history(self, regime: MarketRegime):
        """更新狀態歷史"""
        now = datetime.utcnow()

        if regime != self._current_regime:
            # 狀態變化
            self._regime_history.append((now, regime))
            self._current_regime = regime
            self._regime_start_time = now
            self._regime_bar_count = 1

            # 保留最近 100 個狀態
            if len(self._regime_history) > 100:
                self._regime_history = self._regime_history[-100:]

            logger.info(f"Market regime changed to: {regime.value}")
        else:
            self._regime_bar_count += 1

    def _calculate_strategy_weights(
        self,
        regime: MarketRegime,
        confidence: float,
        indicators: Dict[str, float]
    ) -> Dict[str, float]:
        """計算各策略的權重調整"""
        weights = {}

        for strategy_name, regime_weights in self.STRATEGY_REGIME_MAPPING.items():
            base_weight = regime_weights.get(regime, 1.0)

            # 根據置信度調整
            # 低置信度時權重趨向 1.0
            adjusted_weight = 1.0 + (base_weight - 1.0) * confidence

            # 額外調整
            if strategy_name == 'mean_reversion':
                # 均值回歸在高波動時需要更謹慎
                volatility = indicators['volatility']
                if volatility > 0.03:
                    adjusted_weight *= 0.7

            elif strategy_name == 'momentum':
                # 動量策略在成交量放大時更有效
                volume_ratio = indicators['volume_ratio']
                if volume_ratio > 1.5:
                    adjusted_weight *= 1.1

            weights[strategy_name] = round(adjusted_weight, 3)

        return weights

    def _create_default_analysis(self) -> RegimeAnalysis:
        """創建默認分析結果"""
        return RegimeAnalysis(
            regime=MarketRegime.RANGING,
            confidence=0.5,
            trend_strength=0.0,
            volatility_level=0.5,
            regime_duration=0,
            regime_probabilities={r.value: 0.1 for r in MarketRegime},
            strategy_weights={
                'trend_following': 1.0,
                'momentum': 1.0,
                'mean_reversion': 1.0
            },
            indicators={}
        )

    def get_regime_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """獲取最近 n 個狀態變化"""
        history = self._regime_history[-n:] if self._regime_history else []
        return [
            {'timestamp': ts.isoformat(), 'regime': regime.value}
            for ts, regime in history
        ]

    def get_current_regime(self) -> MarketRegime:
        """獲取當前市場狀態"""
        return self._current_regime

    def should_reduce_exposure(self) -> Tuple[bool, str]:
        """
        判斷是否應該減少曝險

        Returns:
            (是否減少, 原因)
        """
        regime = self._current_regime

        # 高波動時減少曝險
        if regime == MarketRegime.HIGH_VOLATILITY:
            return True, "High volatility detected"

        # 狀態頻繁切換時減少曝險
        if len(self._regime_history) >= 5:
            recent = self._regime_history[-5:]
            unique_regimes = len(set(r for _, r in recent))
            if unique_regimes >= 4:
                return True, "Unstable market regime"

        return False, ""


# 全局實例
market_regime_analyzer = MarketRegimeAnalyzer()
