"""
Sentiment Analysis Service - 情緒分析服務

提供：
- Fear & Greed Index
- 社交媒體情緒分析
- 新聞情緒分析
- 市場情緒綜合評估
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp

from src.logger import logger


class SentimentLevel(str, Enum):
    """情緒等級"""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class SentimentData:
    """情緒數據"""
    source: str
    score: float  # 0-100, 50 為中性
    level: SentimentLevel
    timestamp: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "level": self.level.value if isinstance(self.level, SentimentLevel) else self.level
        }


@dataclass
class MarketSentiment:
    """市場情緒綜合結果"""
    symbol: str
    timestamp: str
    # 各來源情緒
    fear_greed_index: Optional[SentimentData] = None
    social_sentiment: Optional[SentimentData] = None
    news_sentiment: Optional[SentimentData] = None
    # 綜合評估
    overall_score: float = 50.0
    overall_level: SentimentLevel = SentimentLevel.NEUTRAL
    trading_signal: str = "HOLD"
    confidence: float = 0.5
    insights: List[str] = None

    def __post_init__(self):
        if self.insights is None:
            self.insights = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "fear_greed_index": self.fear_greed_index.to_dict() if self.fear_greed_index else None,
            "social_sentiment": self.social_sentiment.to_dict() if self.social_sentiment else None,
            "news_sentiment": self.news_sentiment.to_dict() if self.news_sentiment else None,
            "overall_score": self.overall_score,
            "overall_level": self.overall_level.value,
            "trading_signal": self.trading_signal,
            "confidence": self.confidence,
            "insights": self.insights
        }


class SentimentService:
    """情緒分析服務"""

    def __init__(self):
        self._cache: Dict[str, MarketSentiment] = {}
        self._cache_ttl = 300  # 5 分鐘緩存
        self._cache_timestamps: Dict[str, datetime] = {}
        logger.info("SentimentService initialized")

    async def get_market_sentiment(
        self,
        symbol: str = "BTCUSDT",
        include_social: bool = True,
        include_news: bool = True
    ) -> MarketSentiment:
        """
        獲取市場情緒

        Args:
            symbol: 交易對
            include_social: 是否包含社交媒體情緒
            include_news: 是否包含新聞情緒

        Returns:
            MarketSentiment: 市場情緒
        """
        # 檢查緩存
        cache_key = f"{symbol}_{include_social}_{include_news}"
        if cache_key in self._cache:
            cache_time = self._cache_timestamps.get(cache_key)
            if cache_time and (datetime.utcnow() - cache_time).total_seconds() < self._cache_ttl:
                return self._cache[cache_key]

        # 並行獲取各來源數據
        tasks = [self._get_fear_greed_index()]

        if include_social:
            tasks.append(self._get_social_sentiment(symbol))
        if include_news:
            tasks.append(self._get_news_sentiment(symbol))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 解析結果
        fear_greed = results[0] if not isinstance(results[0], Exception) else None
        social = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None
        news = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None

        # 計算綜合情緒
        sentiment = self._calculate_overall_sentiment(
            symbol=symbol,
            fear_greed=fear_greed,
            social=social,
            news=news
        )

        # 緩存結果
        self._cache[cache_key] = sentiment
        self._cache_timestamps[cache_key] = datetime.utcnow()

        return sentiment

    async def _get_fear_greed_index(self) -> Optional[SentimentData]:
        """獲取 Fear & Greed Index"""
        try:
            url = "https://api.alternative.me/fng/?limit=1"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("data"):
                            fng = data["data"][0]
                            score = int(fng["value"])
                            classification = fng["value_classification"]

                            # 映射到情緒等級
                            if score <= 20:
                                level = SentimentLevel.EXTREME_FEAR
                            elif score <= 40:
                                level = SentimentLevel.FEAR
                            elif score <= 60:
                                level = SentimentLevel.NEUTRAL
                            elif score <= 80:
                                level = SentimentLevel.GREED
                            else:
                                level = SentimentLevel.EXTREME_GREED

                            return SentimentData(
                                source="fear_greed_index",
                                score=score,
                                level=level,
                                timestamp=datetime.utcnow().isoformat(),
                                details={
                                    "classification": classification,
                                    "next_update": fng.get("time_until_update")
                                }
                            )
        except Exception as e:
            logger.warning(f"Failed to get Fear & Greed Index: {e}")

        return None

    async def _get_social_sentiment(self, symbol: str) -> Optional[SentimentData]:
        """
        獲取社交媒體情緒

        這裡使用簡化的模擬邏輯，實際應用中可以整合：
        - Twitter/X API
        - Reddit API
        - Telegram 群組分析
        - Discord 社區分析
        """
        try:
            # 模擬社交情緒分析
            # 實際實現時，可以對接 LunarCrush、Santiment 等 API

            # 根據近期價格波動估算社交情緒
            from src.services.binance_client import BinanceClient
            client = BinanceClient()

            klines = await client.get_klines(symbol, "1h", limit=24)
            if not klines:
                return None

            # 計算 24 小時價格變化
            closes = [float(k[4]) for k in klines]
            price_change = (closes[-1] - closes[0]) / closes[0] * 100

            # 映射價格變化到情緒分數
            # 大漲 -> 貪婪，大跌 -> 恐懼
            base_score = 50
            score = base_score + price_change * 2  # 每 1% 變化 = 2 分
            score = max(0, min(100, score))

            if score <= 20:
                level = SentimentLevel.EXTREME_FEAR
            elif score <= 40:
                level = SentimentLevel.FEAR
            elif score <= 60:
                level = SentimentLevel.NEUTRAL
            elif score <= 80:
                level = SentimentLevel.GREED
            else:
                level = SentimentLevel.EXTREME_GREED

            return SentimentData(
                source="social_sentiment",
                score=score,
                level=level,
                timestamp=datetime.utcnow().isoformat(),
                details={
                    "price_change_24h": round(price_change, 2),
                    "data_source": "price_based_estimation"
                }
            )

        except Exception as e:
            logger.warning(f"Failed to get social sentiment: {e}")

        return None

    async def _get_news_sentiment(self, symbol: str) -> Optional[SentimentData]:
        """
        獲取新聞情緒

        簡化版本，實際應用可以整合：
        - CryptoPanic API
        - Messari News
        - CoinDesk, CoinTelegraph RSS
        """
        try:
            # 使用 CryptoPanic 免費 API
            base_currency = symbol.replace("USDT", "").replace("BUSD", "")
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token=DEMO&currencies={base_currency}&public=true&kind=news"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", [])

                        if not results:
                            return None

                        # 分析新聞情緒（簡化版本）
                        positive_count = 0
                        negative_count = 0

                        for news in results[:20]:  # 分析最近 20 條
                            title = news.get("title", "").lower()
                            # 簡單的關鍵詞情緒分析
                            positive_words = ["bull", "surge", "soar", "rally", "gain", "high", "up", "growth"]
                            negative_words = ["bear", "crash", "plunge", "fall", "drop", "low", "down", "loss"]

                            if any(word in title for word in positive_words):
                                positive_count += 1
                            if any(word in title for word in negative_words):
                                negative_count += 1

                        total = positive_count + negative_count
                        if total > 0:
                            score = 50 + (positive_count - negative_count) / total * 50
                        else:
                            score = 50

                        score = max(0, min(100, score))

                        if score <= 20:
                            level = SentimentLevel.EXTREME_FEAR
                        elif score <= 40:
                            level = SentimentLevel.FEAR
                        elif score <= 60:
                            level = SentimentLevel.NEUTRAL
                        elif score <= 80:
                            level = SentimentLevel.GREED
                        else:
                            level = SentimentLevel.EXTREME_GREED

                        return SentimentData(
                            source="news_sentiment",
                            score=score,
                            level=level,
                            timestamp=datetime.utcnow().isoformat(),
                            details={
                                "news_count": len(results),
                                "positive_count": positive_count,
                                "negative_count": negative_count
                            }
                        )

        except Exception as e:
            logger.warning(f"Failed to get news sentiment: {e}")

        # 返回中性預設值
        return SentimentData(
            source="news_sentiment",
            score=50,
            level=SentimentLevel.NEUTRAL,
            timestamp=datetime.utcnow().isoformat(),
            details={"note": "Unable to fetch news data"}
        )

    def _calculate_overall_sentiment(
        self,
        symbol: str,
        fear_greed: Optional[SentimentData],
        social: Optional[SentimentData],
        news: Optional[SentimentData]
    ) -> MarketSentiment:
        """計算綜合情緒"""
        scores = []
        weights = []
        insights = []

        # Fear & Greed Index（權重最高）
        if fear_greed:
            scores.append(fear_greed.score)
            weights.append(0.5)
            insights.append(f"Fear & Greed Index: {fear_greed.score:.0f} ({fear_greed.level.value})")

        # 社交情緒
        if social:
            scores.append(social.score)
            weights.append(0.3)
            if social.details:
                change = social.details.get("price_change_24h", 0)
                insights.append(f"24h 價格變化: {change:+.2f}%")

        # 新聞情緒
        if news:
            scores.append(news.score)
            weights.append(0.2)

        # 計算加權平均
        if scores:
            total_weight = sum(weights)
            overall_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            overall_score = 50

        # 判斷整體情緒等級
        if overall_score <= 20:
            overall_level = SentimentLevel.EXTREME_FEAR
        elif overall_score <= 40:
            overall_level = SentimentLevel.FEAR
        elif overall_score <= 60:
            overall_level = SentimentLevel.NEUTRAL
        elif overall_score <= 80:
            overall_level = SentimentLevel.GREED
        else:
            overall_level = SentimentLevel.EXTREME_GREED

        # 生成交易信號
        # 逆向思維：極度恐懼時買入，極度貪婪時賣出
        if overall_level == SentimentLevel.EXTREME_FEAR:
            trading_signal = "BUY"
            confidence = 0.7
            insights.append("極度恐懼 - 可能是買入機會（逆向投資）")
        elif overall_level == SentimentLevel.FEAR:
            trading_signal = "BUY"
            confidence = 0.6
            insights.append("市場恐懼 - 考慮逢低買入")
        elif overall_level == SentimentLevel.EXTREME_GREED:
            trading_signal = "SELL"
            confidence = 0.7
            insights.append("極度貪婪 - 考慮獲利了結")
        elif overall_level == SentimentLevel.GREED:
            trading_signal = "HOLD"
            confidence = 0.5
            insights.append("市場偏貪婪 - 謹慎追高")
        else:
            trading_signal = "HOLD"
            confidence = 0.5
            insights.append("市場情緒中性")

        return MarketSentiment(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat(),
            fear_greed_index=fear_greed,
            social_sentiment=social,
            news_sentiment=news,
            overall_score=overall_score,
            overall_level=overall_level,
            trading_signal=trading_signal,
            confidence=confidence,
            insights=insights
        )

    def get_contrarian_signal(self, sentiment: MarketSentiment) -> Dict[str, Any]:
        """
        獲取逆向交易信號

        基於「別人恐懼我貪婪，別人貪婪我恐懼」原則
        """
        score = sentiment.overall_score

        if score <= 25:
            signal = "STRONG_BUY"
            reason = "極度恐懼通常是市場底部"
        elif score <= 40:
            signal = "BUY"
            reason = "恐懼情緒可能帶來買入機會"
        elif score >= 75:
            signal = "STRONG_SELL"
            reason = "極度貪婪通常預示回調"
        elif score >= 60:
            signal = "SELL"
            reason = "貪婪情緒可能預示風險"
        else:
            signal = "HOLD"
            reason = "情緒中性，等待更明確信號"

        return {
            "signal": signal,
            "reason": reason,
            "sentiment_score": score,
            "sentiment_level": sentiment.overall_level.value,
            "confidence": abs(score - 50) / 50  # 距離中性越遠，信心越高
        }


# 單例實例
sentiment_service = SentimentService()
