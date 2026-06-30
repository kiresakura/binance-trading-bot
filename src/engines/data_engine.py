"""
實時數據引擎

負責：
- WebSocket 連接管理
- 實時數據接收和處理
- 特徵計算和緩存
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import websockets
import pandas as pd
import numpy as np

from src.config import settings
from src.logger import logger
from src.features.technical import TechnicalFeatures
from src.features.feature_store import feature_store

# 用於 CPU 密集型特徵計算的線程池
_feature_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="feature_")


class ConnectionState(Enum):
    """連接狀態"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class KlineData:
    """K線數據"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = False


@dataclass
class SubscriptionConfig:
    """訂閱配置"""
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    intervals: List[str] = field(default_factory=lambda: ["1m", "5m", "15m"])
    include_depth: bool = False
    depth_levels: int = 5


class DataEngine:
    """
    實時數據引擎

    功能：
    - 管理 Binance WebSocket 連接
    - 接收和處理 K 線數據
    - 計算技術指標特徵
    - 緩存數據到 Redis
    """

    def __init__(self, config: Optional[SubscriptionConfig] = None):
        self.config = config or SubscriptionConfig()
        self.state = ConnectionState.DISCONNECTED

        # WebSocket
        self.ws = None
        self.ws_url = settings.BINANCE_WS_URL

        # 數據緩存（內存）
        self._kline_cache: Dict[str, pd.DataFrame] = {}  # symbol:interval -> DataFrame
        self._latest_prices: Dict[str, float] = {}

        # 特徵計算
        self.feature_calculator = TechnicalFeatures()

        # 回調函數
        self._callbacks: Dict[str, List[Callable]] = {
            'kline': [],
            'signal': [],
            'error': []
        }

        # 控制
        self._running = False
        self._reconnect_delay = 1
        self._max_reconnect_delay = 60
        self._heartbeat_task = None

        logger.info(f"DataEngine initialized with symbols: {self.config.symbols}")

    # ==================== 連接管理 ====================

    async def start(self):
        """啟動數據引擎"""
        if self._running:
            return

        self._running = True
        logger.info("Starting DataEngine...")

        # 初始化特徵存儲
        await feature_store.init()

        # 啟動 WebSocket 連接
        asyncio.create_task(self._connect_loop())

        # 啟動心跳檢測
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self):
        """停止數據引擎"""
        self._running = False
        self.state = ConnectionState.DISCONNECTED

        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        if self.ws:
            await self.ws.close()

        await feature_store.close()
        logger.info("DataEngine stopped")

    async def _connect_loop(self):
        """連接循環，處理重連"""
        while self._running:
            try:
                await self._connect()
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                self.state = ConnectionState.RECONNECTING

                # 指數退避
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay
                )

    async def _connect(self):
        """建立 WebSocket 連接"""
        self.state = ConnectionState.CONNECTING

        # 構建訂閱流
        streams = self._build_streams()
        # 移除 ws_url 結尾的 /ws，因為多流訂閱使用 /stream 格式
        base_ws_url = self.ws_url.rstrip('/ws')
        url = f"{base_ws_url}/stream?streams={'/'.join(streams)}"

        logger.info(f"Connecting to WebSocket: {url[:100]}...")

        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5
        ) as ws:
            self.ws = ws
            self.state = ConnectionState.CONNECTED
            self._reconnect_delay = 1  # 重置延遲
            logger.info("WebSocket connected successfully")

            # 接收消息循環
            async for message in ws:
                await self._handle_message(message)

    def _build_streams(self) -> List[str]:
        """構建 WebSocket 訂閱流列表"""
        streams = []

        for symbol in self.config.symbols:
            symbol_lower = symbol.lower()

            # K 線流
            for interval in self.config.intervals:
                streams.append(f"{symbol_lower}@kline_{interval}")

            # 深度流（可選）
            if self.config.include_depth:
                streams.append(f"{symbol_lower}@depth{self.config.depth_levels}@100ms")

        return streams

    async def _heartbeat_loop(self):
        """心跳檢測循環"""
        while self._running:
            await asyncio.sleep(30)

            if self.state == ConnectionState.CONNECTED:
                # 檢查最後數據時間
                # 如果超過 1 分鐘沒有數據，可能連接有問題
                pass

    # ==================== 消息處理 ====================

    async def _handle_message(self, message: str):
        """處理 WebSocket 消息"""
        try:
            data = json.loads(message)

            # 多流格式
            if 'stream' in data:
                stream = data['stream']
                payload = data['data']

                if '@kline_' in stream:
                    await self._handle_kline(payload)
                elif '@depth' in stream:
                    await self._handle_depth(payload)

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self._emit_callback('error', {'error': str(e)})

    async def _handle_kline(self, data: dict):
        """處理 K 線數據"""
        try:
            kline = data['k']
            symbol = data['s']
            interval = kline['i']

            kline_data = KlineData(
                timestamp=datetime.fromtimestamp(kline['t'] / 1000),
                open=float(kline['o']),
                high=float(kline['h']),
                low=float(kline['l']),
                close=float(kline['c']),
                volume=float(kline['v']),
                is_closed=kline['x']
            )

            # 更新最新價格
            self._latest_prices[symbol] = kline_data.close

            # 更新緩存
            cache_key = f"{symbol}:{interval}"
            await self._update_kline_cache(cache_key, kline_data)

            # 如果 K 線完成，計算特徵
            if kline_data.is_closed:
                await self._process_closed_kline(symbol, interval)

            # 觸發回調
            await self._emit_callback('kline', {
                'symbol': symbol,
                'interval': interval,
                'kline': kline_data,
                'price': kline_data.close
            })

        except Exception as e:
            logger.error(f"Error handling kline: {e}")

    async def _handle_depth(self, data: dict):
        """處理深度數據"""
        # TODO: 實現深度數據處理
        pass

    async def _update_kline_cache(self, cache_key: str, kline: KlineData):
        """更新 K 線緩存"""
        if cache_key not in self._kline_cache:
            self._kline_cache[cache_key] = pd.DataFrame(
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

        df = self._kline_cache[cache_key]

        # 檢查是否是同一根 K 線（更新）或新 K 線
        if len(df) > 0 and df.iloc[-1]['timestamp'] == kline.timestamp:
            # 更新現有 K 線
            df.iloc[-1] = [
                kline.timestamp, kline.open, kline.high,
                kline.low, kline.close, kline.volume
            ]
        else:
            # 添加新 K 線（使用 loc 避免 FutureWarning）
            new_data = {
                'timestamp': kline.timestamp,
                'open': kline.open,
                'high': kline.high,
                'low': kline.low,
                'close': kline.close,
                'volume': kline.volume
            }
            df.loc[len(df)] = new_data

            # 保留最近 1000 根
            if len(df) > 1000:
                df = df.tail(1000).reset_index(drop=True)

        self._kline_cache[cache_key] = df

    async def _process_closed_kline(self, symbol: str, interval: str):
        """處理完成的 K 線，計算特徵"""
        cache_key = f"{symbol}:{interval}"
        df = self._kline_cache.get(cache_key)

        if df is None or len(df) < 50:  # 需要足夠數據計算指標
            return

        try:
            loop = asyncio.get_event_loop()
            df_copy = df.copy()

            # 將 CPU 密集型的特徵計算移到線程池執行，避免阻塞事件循環
            features_df = await loop.run_in_executor(
                _feature_executor,
                self.feature_calculator.calculate_all,
                df_copy
            )

            # 存儲到 Redis
            await feature_store.store_klines(symbol, interval, df)
            await feature_store.store_features(symbol, interval, features_df)

            # 計算信號特徵 - 也移到線程池
            signal_features = await loop.run_in_executor(
                _feature_executor,
                self.feature_calculator.calculate_signal_features,
                df_copy
            )

            # 生成交易信號 (這個是輕量計算，可以同步執行)
            signal = self._generate_signal(symbol, signal_features)

            if signal:
                await feature_store.store_signal(symbol, signal)
                await self._emit_callback('signal', signal)

            logger.debug(f"Processed kline for {symbol}:{interval}, signal: {signal_features.get('overall_signal', 0):.2f}")

        except Exception as e:
            logger.error(f"Error processing closed kline: {e}")

    def _generate_signal(
        self,
        symbol: str,
        features: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        根據特徵生成交易信號

        Returns:
            信號字典或 None（無明確信號）
        """
        overall = features.get('overall_signal', 0)
        trend = features.get('trend_score', 0)
        rsi = features.get('rsi_14', 50)

        # 信號強度閾值
        STRONG_SIGNAL_THRESHOLD = 0.3
        WEAK_SIGNAL_THRESHOLD = 0.15

        # 確定方向
        if overall > STRONG_SIGNAL_THRESHOLD:
            direction = 1
            strength = min(overall, 1.0)
        elif overall < -STRONG_SIGNAL_THRESHOLD:
            direction = -1
            strength = min(abs(overall), 1.0)
        elif abs(overall) > WEAK_SIGNAL_THRESHOLD:
            direction = 1 if overall > 0 else -1
            strength = abs(overall)
        else:
            direction = 0
            strength = abs(overall)

        return {
            'symbol': symbol,
            'direction': direction,  # 1=做多, -1=做空, 0=觀望
            'strength': strength,
            'strategy': 'technical_composite',
            'features': {
                'overall_signal': overall,
                'trend_score': trend,
                'rsi': rsi,
                'volatility': features.get('volatility', 0),
                'volume_ratio': features.get('volume_ratio', 1)
            },
            'recommendation': self._get_recommendation(direction, strength)
        }

    def _get_recommendation(self, direction: int, strength: float) -> str:
        """獲取交易建議文字"""
        if direction == 0:
            return "HOLD - No clear signal"
        elif direction == 1:
            if strength > 0.7:
                return "STRONG BUY - Multiple indicators aligned bullish"
            elif strength > 0.5:
                return "BUY - Bullish momentum detected"
            else:
                return "WEAK BUY - Consider waiting for confirmation"
        else:
            if strength > 0.7:
                return "STRONG SELL - Multiple indicators aligned bearish"
            elif strength > 0.5:
                return "SELL - Bearish momentum detected"
            else:
                return "WEAK SELL - Consider waiting for confirmation"

    # ==================== 回調管理 ====================

    def on_kline(self, callback: Callable):
        """註冊 K 線回調"""
        self._callbacks['kline'].append(callback)

    def on_signal(self, callback: Callable):
        """註冊信號回調"""
        self._callbacks['signal'].append(callback)

    def on_error(self, callback: Callable):
        """註冊錯誤回調"""
        self._callbacks['error'].append(callback)

    async def _emit_callback(self, event: str, data: Any):
        """觸發回調"""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    # ==================== 數據訪問 ====================

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """獲取最新價格"""
        return self._latest_prices.get(symbol)

    def get_all_prices(self) -> Dict[str, float]:
        """獲取所有最新價格"""
        return self._latest_prices.copy()

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """獲取 K 線數據"""
        cache_key = f"{symbol}:{interval}"
        df = self._kline_cache.get(cache_key)
        if df is not None:
            return df.tail(limit).copy()
        return None

    async def get_features(
        self,
        symbol: str,
        interval: str = "1m"
    ) -> Optional[pd.DataFrame]:
        """獲取特徵數據"""
        return await feature_store.get_features(symbol, interval)

    async def get_latest_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """獲取最新信號"""
        return await feature_store.get_latest_signal(symbol)

    # ==================== 狀態信息 ====================

    def get_status(self) -> Dict[str, Any]:
        """獲取引擎狀態"""
        return {
            'state': self.state.value,
            'running': self._running,
            'symbols': self.config.symbols,
            'intervals': self.config.intervals,
            'cached_symbols': list(self._kline_cache.keys()),
            'latest_prices': self._latest_prices
        }


# 全局實例
data_engine = DataEngine()
