"""實時市場狀態分析服務"""

import asyncio
from typing import Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.services.binance_client import binance_client
from src.logger import logger


class MarketAnalyzer:
    """
    實時市場狀態分析器

    即使交易引擎未運行，也能分析市場狀態
    """

    def __init__(self):
        self.states = ['trending_up', 'trending_down', 'ranging', 'volatile', 'breakout']
        self._current_state = 'unknown'
        self._state_probabilities = {s: 0.2 for s in self.states}
        self._last_update: Optional[datetime] = None
        self._cache_duration = timedelta(seconds=30)  # 快取 30 秒
        self._klines_cache: Dict[str, pd.DataFrame] = {}

    async def get_market_state(self, symbol: str = "BTCUSDT") -> Dict:
        """
        獲取市場狀態

        Args:
            symbol: 交易對

        Returns:
            市場狀態字典
        """
        now = datetime.now()

        # 檢查快取
        if (self._last_update and
            now - self._last_update < self._cache_duration and
            self._current_state != 'unknown'):
            return {
                "state": self._current_state,
                "probabilities": self._state_probabilities.copy(),
                "symbol": symbol,
                "updated_at": self._last_update.isoformat()
            }

        try:
            # 獲取 K 線數據
            klines = await self._get_klines(symbol)

            if klines is None or len(klines) < 30:
                return {
                    "state": "unknown",
                    "probabilities": {s: 0.2 for s in self.states},
                    "symbol": symbol,
                    "error": "Insufficient data"
                }

            # 計算技術指標
            features = self._calculate_features(klines)

            # 分類市場狀態
            state = self._classify(features)

            self._last_update = now

            return {
                "state": state,
                "probabilities": self._state_probabilities.copy(),
                "symbol": symbol,
                "updated_at": now.isoformat(),
                "indicators": {
                    "adx": float(features.get('adx', 0)),
                    "rsi": float(features.get('rsi', 50)),
                    "volatility": float(features.get('volatility', 0)),
                    "trend": "up" if features.get('trend_ema', 0) > 0 else "down"
                }
            }

        except Exception as e:
            logger.warning(f"Market state analysis failed: {e}")
            return {
                "state": self._current_state if self._current_state != 'unknown' else "ranging",
                "probabilities": self._state_probabilities.copy(),
                "symbol": symbol,
                "error": str(e)
            }

    async def _get_klines(self, symbol: str, interval: str = "15m", limit: int = 100) -> Optional[pd.DataFrame]:
        """獲取 K 線數據"""
        try:
            klines = await binance_client.get_klines(symbol, interval, limit)

            if not klines:
                return None

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # 轉換數據類型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            logger.warning(f"Failed to get klines: {e}")
            return None

    def _calculate_features(self, df: pd.DataFrame) -> Dict:
        """計算技術指標"""
        features = {}

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        features['rsi'] = float((100 - (100 / (1 + rs))).iloc[-1]) if not rs.isna().iloc[-1] else 50

        # ADX (簡化版)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        plus_dm = (high - high.shift()).where((high - high.shift()) > (low.shift() - low), 0)
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = (low.shift() - low).where((low.shift() - low) > (high - high.shift()), 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)

        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        features['adx'] = float(dx.rolling(14).mean().iloc[-1]) if not dx.isna().iloc[-1] else 20

        # EMA 趨勢
        ema9 = close.ewm(span=9).mean()
        ema21 = close.ewm(span=21).mean()
        features['trend_ema'] = float(ema9.iloc[-1] - ema21.iloc[-1])

        # SMA 斜率
        sma25 = close.rolling(25).mean()
        features['sma_slope'] = float(sma25.iloc[-1] - sma25.iloc[-5]) if len(sma25) >= 5 else 0

        # 波動率
        returns = close.pct_change()
        features['volatility'] = float(returns.rolling(10).std().iloc[-1]) if len(returns) >= 10 else 0

        # 布林帶
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        features['bb_width'] = float((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / sma20.iloc[-1]) if sma20.iloc[-1] > 0 else 0.02
        features['bb_position'] = float((close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])) if (bb_upper.iloc[-1] - bb_lower.iloc[-1]) > 0 else 0.5

        # 成交量比率
        avg_volume = volume.rolling(20).mean()
        features['volume_ratio'] = float(volume.iloc[-1] / avg_volume.iloc[-1]) if avg_volume.iloc[-1] > 0 else 1

        # 布林帶擠壓
        features['bb_squeeze'] = features['bb_width'] < 0.03

        return features

    def _classify(self, features: Dict) -> str:
        """分類市場狀態"""
        probs = {}

        adx = features.get('adx', 20)
        trend_ema = features.get('trend_ema', 0)
        sma_slope = features.get('sma_slope', 0)
        volatility = features.get('volatility', 0)
        bb_width = features.get('bb_width', 0.02)
        bb_position = features.get('bb_position', 0.5)
        rsi = features.get('rsi', 50)
        bb_squeeze = features.get('bb_squeeze', False)
        volume_ratio = features.get('volume_ratio', 1)

        # 趨勢判斷
        if adx > 25:
            if trend_ema > 0 and sma_slope > 0:
                probs['trending_up'] = 0.6
                probs['trending_down'] = 0.1
            elif trend_ema < 0 and sma_slope < 0:
                probs['trending_up'] = 0.1
                probs['trending_down'] = 0.6
            else:
                probs['trending_up'] = 0.3
                probs['trending_down'] = 0.3
        else:
            probs['trending_up'] = 0.15
            probs['trending_down'] = 0.15

        # 波動判斷
        if volatility > 0.03 or bb_width > 0.06:
            probs['volatile'] = 0.5
        else:
            probs['volatile'] = 0.1

        # 震盪判斷
        if adx < 20 and 0.2 < bb_position < 0.8 and 35 < rsi < 65:
            probs['ranging'] = 0.5
        else:
            probs['ranging'] = 0.15

        # 突破判斷
        if bb_squeeze and volume_ratio > 1.5:
            probs['breakout'] = 0.4
        else:
            probs['breakout'] = 0.1

        # 正規化
        total = sum(probs.values())
        for k in probs:
            probs[k] /= total

        # 更新狀態
        self._state_probabilities = probs
        self._current_state = max(probs, key=probs.get)

        return self._current_state


# 全局實例
market_analyzer = MarketAnalyzer()
