import asyncio
from typing import List
from src.services.binance_client import binance_client
from src.api.websocket_manager import ws_manager
from src.logger import logger
from src.config import settings

class MarketPublisher:
    """市場數據廣播服務 (Market Data Publisher)"""
    
    def __init__(self):
        self._is_running = False
        self._task = None
        # 默認關注的交易對
        self.symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
        self.interval = 1.0  # 1 秒更新一次

    async def start(self):
        """啟動廣播服務"""
        if self._is_running:
            return
            
        self._is_running = True
        self._task = asyncio.create_task(self._broadcast_loop())
        logger.info("Market Publisher service started")

    async def stop(self):
        """停止廣播服務"""
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Market Publisher service stopped")

    async def _broadcast_loop(self):
        """廣播循環"""
        while self._is_running:
            # 如果沒有 WebSocket 連接，則降低頻率以節省資源
            if ws_manager.get_connection_count() == 0:
                await asyncio.sleep(5)
                continue

            try:
                # 併發獲取所有交易對的數據
                tasks = [self._process_symbol(symbol) for symbol in self.symbols]
                await asyncio.gather(*tasks)

            except Exception as e:
                logger.error(f"Market Publisher error: {e}")
            
            await asyncio.sleep(self.interval)

    async def _process_symbol(self, symbol: str):
        """處理單個交易對的數據更新"""
        try:
            # 1. 獲取訂單簿 (Depth)
            depth = await binance_client.get_order_book(symbol=symbol, limit=20)
            
            # 廣播深度更新
            await ws_manager.broadcast_order_book(
                symbol=symbol,
                bids=depth.get('bids', []),
                asks=depth.get('asks', [])
            )
            
            # 2. 廣播最新價格 (使用 Depth 的中間價或 ticker)
            # 為了效率，這裡簡單取中間價，或可調用 get_ticker_price
            if depth.get('bids') and depth.get('asks'):
                best_bid = float(depth['bids'][0][0])
                best_ask = float(depth['asks'][0][0])
                mid_price = (best_bid + best_ask) / 2
                
                await ws_manager.broadcast_price(
                    symbol=symbol,
                    price=mid_price
                )

        except Exception as e:
            # logger.debug(f"Failed to fetch data for {symbol}: {e}")
            pass

# 全局實例
market_publisher = MarketPublisher()
