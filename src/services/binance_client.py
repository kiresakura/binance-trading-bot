"""幣安 API 客戶端 - 修復版（含精度處理）"""

import hmac
import hashlib
import time
import aiohttp
from typing import Dict, List, Optional
from decimal import Decimal, ROUND_DOWN
from src.config import settings
from src.logger import logger


class BinanceClient:
    """幣安 API 客戶端"""

    def __init__(self):
        self.api_key = settings.BINANCE_API_KEY
        self.api_secret = settings.BINANCE_API_SECRET

        if settings.BINANCE_TESTNET:
            self.base_url = "https://testnet.binance.vision/api"
        else:
            self.base_url = "https://api.binance.com/api"

        self.session: Optional[aiohttp.ClientSession] = None
        
        # 交易對精度緩存
        self._symbol_info_cache: Dict[str, Dict] = {}
        self._cache_timestamp: float = 0
        self._cache_ttl: float = 3600  # 1 小時

    async def _get_session(self) -> aiohttp.ClientSession:
        """獲取或創建 HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    def _generate_signature(self, params: Dict) -> str:
        """生成請求簽名"""
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        signed: bool = False
    ) -> Dict:
        """發送 API 請求"""
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"

        headers = {
            "X-MBX-APIKEY": self.api_key
        }

        if params is None:
            params = {}

        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._generate_signature(params)

        try:
            if method == "GET":
                async with session.get(url, params=params, headers=headers) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        logger.error(f"Binance API error: {data}")
                    return data
            elif method == "POST":
                async with session.post(url, params=params, headers=headers) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        logger.error(f"Binance API error: {data}")
                    return data
            elif method == "DELETE":
                async with session.delete(url, params=params, headers=headers) as resp:
                    data = await resp.json()
                    return data

        except Exception as e:
            logger.error(f"Binance API request failed: {e}")
            raise

    # ==================== 精度處理（新增）====================

    async def _ensure_symbol_info(self, symbol: str) -> Dict:
        """確保有該交易對的精度信息"""
        now = time.time()
        
        # 檢查緩存是否有效
        if symbol in self._symbol_info_cache and (now - self._cache_timestamp) < self._cache_ttl:
            return self._symbol_info_cache[symbol]
        
        # 重新獲取 exchange info
        try:
            exchange_info = await self.get_exchange_info()
            
            for sym_info in exchange_info.get('symbols', []):
                sym = sym_info['symbol']
                filters = {f['filterType']: f for f in sym_info.get('filters', [])}
                
                # 提取精度信息
                lot_size = filters.get('LOT_SIZE', {})
                price_filter = filters.get('PRICE_FILTER', {})
                notional = filters.get('NOTIONAL', {}) or filters.get('MIN_NOTIONAL', {})
                
                self._symbol_info_cache[sym] = {
                    'baseAsset': sym_info.get('baseAsset'),
                    'quoteAsset': sym_info.get('quoteAsset'),
                    'stepSize': Decimal(lot_size.get('stepSize', '0.00000001')),
                    'minQty': Decimal(lot_size.get('minQty', '0.00000001')),
                    'maxQty': Decimal(lot_size.get('maxQty', '9999999')),
                    'tickSize': Decimal(price_filter.get('tickSize', '0.01')),
                    'minPrice': Decimal(price_filter.get('minPrice', '0.01')),
                    'minNotional': Decimal(notional.get('minNotional', '5')),
                }
            
            self._cache_timestamp = now
            logger.debug(f"Exchange info cached for {len(self._symbol_info_cache)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to get exchange info: {e}")
            # 返回默認值
            return {
                'stepSize': Decimal('0.00001'),
                'minQty': Decimal('0.00001'),
                'maxQty': Decimal('9999999'),
                'tickSize': Decimal('0.01'),
                'minNotional': Decimal('5'),
            }
        
        return self._symbol_info_cache.get(symbol, {
            'stepSize': Decimal('0.00001'),
            'minQty': Decimal('0.00001'),
            'maxQty': Decimal('9999999'),
            'tickSize': Decimal('0.01'),
            'minNotional': Decimal('5'),
        })

    def _format_quantity(self, quantity: Decimal, step_size: Decimal) -> str:
        """
        格式化數量，符合 Binance 要求：
        1. 不能有科學記號
        2. 精度符合 stepSize
        """
        if step_size <= 0:
            step_size = Decimal('0.00000001')
        
        # 計算小數位數
        step_str = str(step_size)
        if '.' in step_str:
            precision = len(step_str.split('.')[1].rstrip('0')) or 0
        else:
            precision = 0
        
        # Round down 到 stepSize 的倍數
        remainder = quantity % step_size
        rounded = quantity - remainder
        
        # 確保不低於 0
        if rounded < 0:
            rounded = Decimal('0')
        
        # 格式化為普通字串（不會有科學記號）
        if precision > 0:
            return f"{rounded:.{precision}f}"
        else:
            return str(int(rounded))

    def _format_price(self, price: Decimal, tick_size: Decimal) -> str:
        """格式化價格"""
        if tick_size <= 0:
            tick_size = Decimal('0.01')
        
        tick_str = str(tick_size)
        if '.' in tick_str:
            precision = len(tick_str.split('.')[1].rstrip('0')) or 0
        else:
            precision = 0
        
        # Round to tick size
        rounded = (price / tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * tick_size
        
        if precision > 0:
            return f"{rounded:.{precision}f}"
        else:
            return str(int(rounded))

    # ==================== 帳戶相關 ====================

    async def get_account_info(self) -> Dict:
        """獲取帳戶信息"""
        return await self._request("GET", "/v3/account", signed=True)

    async def get_balances(self) -> List[Dict]:
        """獲取所有餘額（只返回有餘額的）"""
        account = await self.get_account_info()
        balances = account.get("balances", [])
        
        # 過濾有餘額的資產
        non_zero_balances = [
            {
                "asset": b["asset"],
                "free": b["free"],
                "locked": b["locked"],
                "total": str(Decimal(b["free"]) + Decimal(b["locked"]))
            }
            for b in balances
            if Decimal(b["free"]) > 0 or Decimal(b["locked"]) > 0
        ]
        
        return non_zero_balances

    # ==================== 訂單相關 ====================

    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """獲取當前掛單"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/v3/openOrders", params=params, signed=True)

    async def get_all_orders(self, symbol: str, limit: int = 50) -> List[Dict]:
        """獲取歷史訂單"""
        params = {
            "symbol": symbol,
            "limit": limit
        }
        return await self._request("GET", "/v3/allOrders", params=params, signed=True)

    async def get_my_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """獲取成交歷史"""
        params = {
            "symbol": symbol,
            "limit": limit
        }
        return await self._request("GET", "/v3/myTrades", params=params, signed=True)

    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """獲取訂單簿深度"""
        params = {
            "symbol": symbol,
            "limit": limit
        }
        return await self._request("GET", "/v3/depth", params=params)

    # ==================== 市場數據 ====================

    async def get_ticker_price(self, symbol: str = None) -> Dict:
        """獲取最新價格"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/v3/ticker/price", params=params)

    async def get_ticker_24h(self, symbol: str = None) -> Dict:
        """獲取 24 小時行情"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/v3/ticker/24hr", params=params)

    async def get_exchange_info(self) -> Dict:
        """獲取交易規則"""
        return await self._request("GET", "/v3/exchangeInfo")

    async def get_klines(self, symbol: str, interval: str = "15m", limit: int = 100) -> List:
        """獲取 K 線數據"""
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        return await self._request("GET", "/v3/klines", params=params, signed=False)

    async def get_historical_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 1000,
        start_time: int = None,
        end_time: int = None
    ) -> List:
        """獲取歷史 K 線數據"""
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        return await self._request("GET", "/v3/klines", params=params, signed=False)

    # ==================== 下單相關（修復版）====================

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal = None,
        time_in_force: str = "GTC"
    ) -> Dict:
        """
        創建訂單（自動處理精度）
        
        修復內容：
        1. 自動獲取交易對精度（stepSize, tickSize）
        2. 格式化 quantity 避免科學記號
        3. 檢查最小名義價值
        """
        # 獲取交易對精度信息
        sym_info = await self._ensure_symbol_info(symbol)
        step_size = sym_info.get('stepSize', Decimal('0.00001'))
        tick_size = sym_info.get('tickSize', Decimal('0.01'))
        min_qty = sym_info.get('minQty', Decimal('0.00001'))
        min_notional = sym_info.get('minNotional', Decimal('5'))
        
        # 確保 quantity 是 Decimal
        if not isinstance(quantity, Decimal):
            quantity = Decimal(str(quantity))
        
        # 格式化 quantity
        formatted_qty = self._format_quantity(quantity, step_size)
        
        # 檢查最小數量
        if Decimal(formatted_qty) < min_qty:
            logger.warning(
                f"⚠️ Quantity {formatted_qty} below minimum {min_qty} for {symbol}. "
                f"Order will likely fail."
            )
        
        # 獲取當前價格（用於檢查 minNotional）
        if price is None:
            ticker = await self.get_ticker_price(symbol)
            current_price = Decimal(ticker.get('price', '0'))
        else:
            current_price = price if isinstance(price, Decimal) else Decimal(str(price))
        
        # 檢查最小名義價值
        notional = Decimal(formatted_qty) * current_price
        if notional < min_notional:
            logger.warning(
                f"⚠️ Notional value ${notional:.2f} below minimum ${min_notional} for {symbol}. "
                f"Order will likely fail. (qty={formatted_qty}, price={current_price})"
            )
        
        # 構建參數
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": formatted_qty  # 使用格式化後的數量
        }

        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("Price is required for LIMIT orders")
            formatted_price = self._format_price(
                price if isinstance(price, Decimal) else Decimal(str(price)),
                tick_size
            )
            params["price"] = formatted_price
            params["timeInForce"] = time_in_force
        
        logger.info(f"📤 Submitting order: {symbol} {side} {formatted_qty} @ {params.get('price', 'MARKET')}")

        return await self._request("POST", "/v3/order", params=params, signed=True)

    async def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """取消訂單"""
        params = {
            "symbol": symbol,
            "orderId": order_id
        }
        return await self._request("DELETE", "/v3/order", params=params, signed=True)

    async def cancel_all_orders(self, symbol: str) -> List[Dict]:
        """取消指定交易對的所有掛單"""
        try:
            # 1. 獲取當前掛單
            open_orders = await self.get_open_orders(symbol=symbol)
            if not open_orders:
                return []
            
            # 2. 併發取消
            tasks = []
            for order in open_orders:
                tasks.append(self.cancel_order(symbol=symbol, order_id=order['orderId']))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 過濾成功結果
            cancelled = [r for r in results if isinstance(r, dict) and 'orderId' in r]
            logger.info(f"Cancelled {len(cancelled)} orders for {symbol}")
            return cancelled
            
        except Exception as e:
            logger.error(f"Failed to cancel all orders for {symbol}: {e}")
            return []

    async def close(self):
        """關閉連接"""
        if self.session and not self.session.closed:
            await self.session.close()


# 全局客戶端實例
binance_client = BinanceClient()
