"""
Multi-Timeframe Analysis Service - 多時間框架分析服務

提供：
- 多時間框架趨勢分析
- 時間框架一致性檢查
- 支撐阻力位識別
- 趨勢強度評估
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

from src.logger import logger


class Trend(str, Enum):
    """趨勢方向"""
    STRONG_UP = "strong_up"
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    STRONG_DOWN = "strong_down"


class Timeframe(str, Enum):
    """時間框架"""
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


@dataclass
class TimeframeAnalysis:
    """單一時間框架分析結果"""
    timeframe: str
    trend: Trend
    trend_strength: float  # 0-1
    rsi: float
    macd_signal: str  # bullish, bearish, neutral
    ema_alignment: bool  # EMA9 > EMA21 > EMA50
    support_level: float
    resistance_level: float
    volume_trend: str  # increasing, decreasing, stable
    momentum: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "trend": self.trend.value if isinstance(self.trend, Trend) else self.trend
        }


@dataclass
class MultiTimeframeResult:
    """多時間框架分析結果"""
    symbol: str
    timestamp: str
    timeframes: Dict[str, TimeframeAnalysis]
    # 綜合分析
    overall_trend: Trend
    trend_alignment: float  # 時間框架一致性 0-1
    recommended_action: str  # BUY, SELL, HOLD
    confidence: float
    key_levels: Dict[str, float]
    reasoning: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "timeframes": {k: v.to_dict() for k, v in self.timeframes.items()},
            "overall_trend": self.overall_trend.value,
            "trend_alignment": self.trend_alignment,
            "recommended_action": self.recommended_action,
            "confidence": self.confidence,
            "key_levels": self.key_levels,
            "reasoning": self.reasoning
        }


class MultiTimeframeService:
    """多時間框架分析服務"""

    def __init__(self):
        # 時間框架權重（較長時間框架權重較高）
        self.timeframe_weights = {
            Timeframe.M5: 0.1,
            Timeframe.M15: 0.15,
            Timeframe.H1: 0.25,
            Timeframe.H4: 0.25,
            Timeframe.D1: 0.15,
            Timeframe.W1: 0.1
        }
        logger.info("MultiTimeframeService initialized")

    async def analyze(
        self,
        symbol: str,
        timeframes: Optional[List[Timeframe]] = None
    ) -> MultiTimeframeResult:
        """
        執行多時間框架分析

        Args:
            symbol: 交易對
            timeframes: 要分析的時間框架列表

        Returns:
            MultiTimeframeResult: 分析結果
        """
        if timeframes is None:
            timeframes = [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]

        # 獲取各時間框架數據並分析
        tf_analyses = {}

        for tf in timeframes:
            try:
                analysis = await self._analyze_timeframe(symbol, tf)
                tf_analyses[tf.value] = analysis
            except Exception as e:
                logger.warning(f"Failed to analyze {symbol} on {tf.value}: {e}")

        if not tf_analyses:
            return self._empty_result(symbol)

        # 綜合分析
        overall_trend = self._calculate_overall_trend(tf_analyses)
        trend_alignment = self._calculate_trend_alignment(tf_analyses)
        recommended_action, confidence = self._generate_recommendation(
            tf_analyses, overall_trend, trend_alignment
        )
        key_levels = self._identify_key_levels(tf_analyses)
        reasoning = self._generate_reasoning(tf_analyses, overall_trend, trend_alignment)

        return MultiTimeframeResult(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat(),
            timeframes=tf_analyses,
            overall_trend=overall_trend,
            trend_alignment=trend_alignment,
            recommended_action=recommended_action,
            confidence=confidence,
            key_levels=key_levels,
            reasoning=reasoning
        )

    async def _analyze_timeframe(
        self,
        symbol: str,
        timeframe: Timeframe
    ) -> TimeframeAnalysis:
        """分析單一時間框架"""
        from src.services.binance_client import BinanceClient

        client = BinanceClient()

        # 獲取 K 線數據
        klines = await client.get_klines(
            symbol=symbol,
            interval=timeframe.value,
            limit=100
        )

        if not klines or len(klines) < 50:
            raise ValueError(f"Insufficient data for {timeframe.value}")

        # 提取價格數據
        closes = np.array([float(k[4]) for k in klines])
        highs = np.array([float(k[2]) for k in klines])
        lows = np.array([float(k[3]) for k in klines])
        volumes = np.array([float(k[5]) for k in klines])

        # 計算指標
        rsi = self._calculate_rsi(closes)
        macd, macd_signal_line, macd_hist = self._calculate_macd(closes)
        ema9 = self._calculate_ema(closes, 9)
        ema21 = self._calculate_ema(closes, 21)
        ema50 = self._calculate_ema(closes, 50)

        # 判斷趨勢
        trend, trend_strength = self._determine_trend(closes, ema9, ema21, ema50, rsi)

        # MACD 信號
        if macd_hist[-1] > 0 and macd_hist[-1] > macd_hist[-2]:
            macd_signal = "bullish"
        elif macd_hist[-1] < 0 and macd_hist[-1] < macd_hist[-2]:
            macd_signal = "bearish"
        else:
            macd_signal = "neutral"

        # EMA 排列
        ema_alignment = ema9[-1] > ema21[-1] > ema50[-1]

        # 支撐阻力位
        support = min(lows[-20:])
        resistance = max(highs[-20:])

        # 成交量趨勢
        vol_sma = np.mean(volumes[-20:])
        recent_vol = np.mean(volumes[-5:])
        if recent_vol > vol_sma * 1.2:
            volume_trend = "increasing"
        elif recent_vol < vol_sma * 0.8:
            volume_trend = "decreasing"
        else:
            volume_trend = "stable"

        # 動量
        momentum = (closes[-1] - closes[-10]) / closes[-10] * 100

        return TimeframeAnalysis(
            timeframe=timeframe.value,
            trend=trend,
            trend_strength=trend_strength,
            rsi=rsi[-1],
            macd_signal=macd_signal,
            ema_alignment=ema_alignment,
            support_level=support,
            resistance_level=resistance,
            volume_trend=volume_trend,
            momentum=momentum
        )

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """計算 RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)

        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

        rs = avg_gain / np.where(avg_loss == 0, 0.0001, avg_loss)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(
        self,
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """計算 MACD"""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd, signal)
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """計算 EMA"""
        multiplier = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[period-1] = np.mean(prices[:period])

        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]

        return ema

    def _determine_trend(
        self,
        closes: np.ndarray,
        ema9: np.ndarray,
        ema21: np.ndarray,
        ema50: np.ndarray,
        rsi: np.ndarray
    ) -> Tuple[Trend, float]:
        """判斷趨勢和強度"""
        current_price = closes[-1]
        current_rsi = rsi[-1]

        # 計算趨勢分數
        score = 0

        # EMA 排列
        if ema9[-1] > ema21[-1] > ema50[-1]:
            score += 2
        elif ema9[-1] < ema21[-1] < ema50[-1]:
            score -= 2

        # 價格相對於 EMA
        if current_price > ema50[-1]:
            score += 1
        else:
            score -= 1

        # RSI
        if current_rsi > 60:
            score += 1
        elif current_rsi < 40:
            score -= 1

        # 價格動量
        price_change = (closes[-1] - closes[-20]) / closes[-20] * 100
        if price_change > 5:
            score += 1
        elif price_change < -5:
            score -= 1

        # 判斷趨勢
        if score >= 4:
            trend = Trend.STRONG_UP
        elif score >= 2:
            trend = Trend.UP
        elif score <= -4:
            trend = Trend.STRONG_DOWN
        elif score <= -2:
            trend = Trend.DOWN
        else:
            trend = Trend.NEUTRAL

        # 趨勢強度
        strength = min(1.0, abs(score) / 5)

        return trend, strength

    def _calculate_overall_trend(
        self,
        tf_analyses: Dict[str, TimeframeAnalysis]
    ) -> Trend:
        """計算整體趨勢"""
        trend_values = {
            Trend.STRONG_UP: 2,
            Trend.UP: 1,
            Trend.NEUTRAL: 0,
            Trend.DOWN: -1,
            Trend.STRONG_DOWN: -2
        }

        weighted_score = 0
        total_weight = 0

        for tf_str, analysis in tf_analyses.items():
            tf = Timeframe(tf_str)
            weight = self.timeframe_weights.get(tf, 0.1)
            trend_value = trend_values.get(analysis.trend, 0)
            weighted_score += trend_value * weight
            total_weight += weight

        if total_weight > 0:
            avg_score = weighted_score / total_weight
        else:
            avg_score = 0

        if avg_score >= 1.5:
            return Trend.STRONG_UP
        elif avg_score >= 0.5:
            return Trend.UP
        elif avg_score <= -1.5:
            return Trend.STRONG_DOWN
        elif avg_score <= -0.5:
            return Trend.DOWN
        else:
            return Trend.NEUTRAL

    def _calculate_trend_alignment(
        self,
        tf_analyses: Dict[str, TimeframeAnalysis]
    ) -> float:
        """計算時間框架趨勢一致性"""
        if not tf_analyses:
            return 0

        trends = [a.trend for a in tf_analyses.values()]

        # 計算趨勢方向一致性
        up_trends = sum(1 for t in trends if t in [Trend.UP, Trend.STRONG_UP])
        down_trends = sum(1 for t in trends if t in [Trend.DOWN, Trend.STRONG_DOWN])

        total = len(trends)
        max_aligned = max(up_trends, down_trends)

        return max_aligned / total if total > 0 else 0

    def _generate_recommendation(
        self,
        tf_analyses: Dict[str, TimeframeAnalysis],
        overall_trend: Trend,
        trend_alignment: float
    ) -> Tuple[str, float]:
        """生成交易建議"""
        # 基礎信心度
        base_confidence = trend_alignment

        # 根據整體趨勢調整
        if overall_trend in [Trend.STRONG_UP, Trend.UP]:
            action = "BUY"
            if overall_trend == Trend.STRONG_UP:
                base_confidence += 0.1
        elif overall_trend in [Trend.STRONG_DOWN, Trend.DOWN]:
            action = "SELL"
            if overall_trend == Trend.STRONG_DOWN:
                base_confidence += 0.1
        else:
            action = "HOLD"
            base_confidence = 0.5

        # 只有在趨勢一致性夠高時才建議交易
        if trend_alignment < 0.6:
            action = "HOLD"
            base_confidence *= 0.8

        confidence = min(0.95, base_confidence)

        return action, confidence

    def _identify_key_levels(
        self,
        tf_analyses: Dict[str, TimeframeAnalysis]
    ) -> Dict[str, float]:
        """識別關鍵價位"""
        supports = []
        resistances = []

        for analysis in tf_analyses.values():
            supports.append(analysis.support_level)
            resistances.append(analysis.resistance_level)

        return {
            "strong_support": min(supports) if supports else 0,
            "support": np.mean(supports) if supports else 0,
            "resistance": np.mean(resistances) if resistances else 0,
            "strong_resistance": max(resistances) if resistances else 0
        }

    def _generate_reasoning(
        self,
        tf_analyses: Dict[str, TimeframeAnalysis],
        overall_trend: Trend,
        trend_alignment: float
    ) -> List[str]:
        """生成分析說明"""
        reasons = []

        # 整體趨勢
        trend_descriptions = {
            Trend.STRONG_UP: "多時間框架顯示強勁上升趨勢",
            Trend.UP: "整體趨勢偏多",
            Trend.NEUTRAL: "市場處於整理階段",
            Trend.DOWN: "整體趨勢偏空",
            Trend.STRONG_DOWN: "多時間框架顯示強勁下跌趨勢"
        }
        reasons.append(trend_descriptions.get(overall_trend, "趨勢不明確"))

        # 趨勢一致性
        if trend_alignment >= 0.8:
            reasons.append(f"時間框架高度一致 ({trend_alignment:.0%})")
        elif trend_alignment >= 0.6:
            reasons.append(f"時間框架較為一致 ({trend_alignment:.0%})")
        else:
            reasons.append(f"時間框架分歧較大 ({trend_alignment:.0%})，建議觀望")

        # 各時間框架特殊情況
        for tf_str, analysis in tf_analyses.items():
            if analysis.rsi > 70:
                reasons.append(f"{tf_str} RSI 超買 ({analysis.rsi:.1f})")
            elif analysis.rsi < 30:
                reasons.append(f"{tf_str} RSI 超賣 ({analysis.rsi:.1f})")

            if analysis.volume_trend == "increasing":
                reasons.append(f"{tf_str} 成交量放大")

        return reasons

    def _empty_result(self, symbol: str) -> MultiTimeframeResult:
        """返回空結果"""
        return MultiTimeframeResult(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat(),
            timeframes={},
            overall_trend=Trend.NEUTRAL,
            trend_alignment=0,
            recommended_action="HOLD",
            confidence=0,
            key_levels={},
            reasoning=["無法獲取足夠數據進行分析"]
        )


# 單例實例
multi_timeframe_service = MultiTimeframeService()
