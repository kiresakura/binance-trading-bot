"""智能订单执行系统"""

import asyncio
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from src.logger import logger
from src.services.risk_manager import Order, risk_management_system
from src.services.fault_tolerance import fault_tolerant_engine
from src.services.binance_client import binance_client


@dataclass
class OrderChunk:
    """订单分片"""
    quantity: Decimal
    price: Optional[Decimal]
    delay: float  # 延迟执行时间（秒）
    chunk_id: int


@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    order_id: Optional[str]
    executed_price: Optional[Decimal]
    executed_quantity: Decimal
    total_cost: Decimal
    commission: Decimal
    timestamp: datetime
    error_message: Optional[str] = None


class OrderBookAnalyzer:
    """订单簿分析器"""

    # 快取
    _depth_cache: Dict[str, tuple] = {}  # {symbol: (data, timestamp)}

    async def analyze_depth(
        self,
        symbol: str,
        side: str = "BUY"
    ) -> Dict[str, any]:
        """分析订单簿深度（從 Binance API 獲取實際數據）"""
        try:
            # 檢查快取（5 秒有效）
            cache_key = symbol
            if cache_key in self._depth_cache:
                cached_data, timestamp = self._depth_cache[cache_key]
                if (datetime.utcnow() - timestamp).total_seconds() < 5:
                    return cached_data

            # 從 Binance 獲取訂單簿
            order_book = await binance_client.get_order_book(symbol, limit=20)

            if not order_book:
                logger.warning(f"Failed to get order book for {symbol}, using fallback")
                return self._get_fallback_depth(symbol)

            # 解析訂單簿數據
            bids = []
            asks = []

            for bid in order_book.get('bids', []):
                price = Decimal(str(bid[0]))
                quantity = Decimal(str(bid[1]))
                bids.append((price, quantity))

            for ask in order_book.get('asks', []):
                price = Decimal(str(ask[0]))
                quantity = Decimal(str(ask[1]))
                asks.append((price, quantity))

            # 計算流動性指標
            bid_volume = sum(float(q) for _, q in bids[:10])
            ask_volume = sum(float(q) for _, q in asks[:10])
            total_volume = bid_volume + ask_volume

            # 計算價差
            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0.001
            else:
                spread = 0.001

            # 計算買賣失衡
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0

            # 流動性評分（基於成交量和價差）
            # 高成交量 + 低價差 = 高流動性
            liquidity_score = min(1.0, total_volume / 100) * (1 - min(spread * 100, 1))

            depth_data = {
                'bids': bids,
                'asks': asks,
                'liquidity_score': max(0.1, liquidity_score),
                'spread': spread,
                'imbalance': imbalance,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume
            }

            # 更新快取
            self._depth_cache[cache_key] = (depth_data, datetime.utcnow())

            logger.debug(f"Order book analyzed: {symbol}, liquidity={liquidity_score:.2f}, spread={spread:.4%}")
            return depth_data

        except Exception as e:
            logger.error(f"Failed to analyze order book: {e}")
            return self._get_fallback_depth(symbol)

    def _get_fallback_depth(self, symbol: str) -> Dict[str, any]:
        """獲取備用訂單簿數據（當 API 失敗時）"""
        return {
            'bids': [],
            'asks': [],
            'liquidity_score': 0.5,
            'spread': 0.001,
            'imbalance': 0.0,
            'bid_volume': 0,
            'ask_volume': 0
        }

    def calculate_available_liquidity(
        self,
        depth: List[Tuple[Decimal, Decimal]],
        target_quantity: Decimal
    ) -> Dict:
        """计算可用流动性"""
        cumulative_quantity = Decimal('0')
        levels_needed = 0
        weighted_avg_price = Decimal('0')

        for price, quantity in depth:
            cumulative_quantity += quantity
            weighted_avg_price += price * quantity
            levels_needed += 1

            if cumulative_quantity >= target_quantity:
                weighted_avg_price /= cumulative_quantity
                return {
                    'sufficient_liquidity': True,
                    'levels_needed': levels_needed,
                    'avg_price': weighted_avg_price,
                    'total_available': cumulative_quantity
                }

        return {
            'sufficient_liquidity': False,
            'levels_needed': levels_needed,
            'avg_price': weighted_avg_price,
            'total_available': cumulative_quantity
        }


class SlippagePredictor:
    """滑点预测器"""

    def __init__(self):
        self.historical_slippage = []
        self.model_trained = False

    async def predict(self, order_chunk: OrderChunk) -> Dict:
        """预测滑点"""
        try:
            # TODO: 使用机器学习模型预测滑点
            # 这里使用简单的启发式方法

            # 基于订单大小的滑点估计
            base_slippage = float(order_chunk.quantity) * 0.0001

            # 市场波动率调整
            volatility_factor = await self._get_volatility_factor()
            adjusted_slippage = base_slippage * (1 + volatility_factor)

            # 添加随机成分
            predicted_slippage = adjusted_slippage * np.random.uniform(0.8, 1.2)

            return {
                'predicted_slippage': predicted_slippage,
                'confidence': 0.75,
                'volatility_factor': volatility_factor
            }

        except Exception as e:
            logger.error(f"Slippage prediction failed: {e}")
            return {'predicted_slippage': 0.001, 'confidence': 0.5}

    async def _get_volatility_factor(self) -> float:
        """获取波动率因子（基於實際市場數據）"""
        try:
            # 獲取 BTC 的最近波動率作為市場基準
            klines = await binance_client.get_klines("BTCUSDT", "5m", limit=20)
            if not klines or len(klines) < 10:
                return 0.5

            # 計算收益率標準差
            closes = [float(k[4]) for k in klines]
            returns = np.diff(closes) / closes[:-1]
            volatility = float(np.std(returns))

            # 將波動率標準化到 0-1 範圍
            # 假設正常波動率在 0.001-0.01 之間
            normalized = min(1.0, volatility / 0.01)
            return normalized

        except Exception as e:
            logger.error(f"Failed to get volatility factor: {e}")
            return 0.5

    def update_model(self, actual_slippage: float):
        """更新模型（在线学习）"""
        self.historical_slippage.append(actual_slippage)

        # 保留最近 1000 个数据点
        if len(self.historical_slippage) > 1000:
            self.historical_slippage.pop(0)


class MarketImpactModel:
    """市场影响模型"""

    def __init__(self):
        self.impact_history = []

    def estimate_impact(
        self,
        order_size: Decimal,
        average_volume: Decimal
    ) -> float:
        """估计市场影响"""
        # 简化的市场影响模型
        # 实际应该使用更复杂的模型（如 Almgren-Chriss 模型）

        if average_volume == 0:
            return 0.01  # 默认1%

        volume_ratio = float(order_size) / float(average_volume)

        # 非线性影响函数
        impact = 0.01 * (volume_ratio ** 0.5)

        return min(impact, 0.05)  # 最大5%

    def update_impact_model(self, execution_data: Dict):
        """更新影响模型"""
        self.impact_history.append(execution_data)


class SmartOrderExecutor:
    """智能订单执行器"""

    def __init__(self):
        self.order_book_analyzer = OrderBookAnalyzer()
        self.slippage_predictor = SlippagePredictor()
        self.market_impact_model = MarketImpactModel()
        self.commission_rate = Decimal('0.001')  # 0.1%

    async def execute_order(
        self,
        order: Order,
        strategy: str = "adaptive",
        skip_risk_check: bool = False
    ) -> List[ExecutionResult]:
        """执行订单（智能策略）

        Args:
            order: 訂單物件
            strategy: 執行策略 (adaptive/twap/vwap/immediate)
            skip_risk_check: 是否跳過風險檢查（用於平倉訂單）
        """
        try:
            from src.config import settings
            from src.services.binance_client import binance_client

            # 🆕 0. 最低訂單金額檢查（MARKET 訂單需要獲取當前價格）
            min_order_value = getattr(settings, 'MIN_ORDER_VALUE', 6.0)

            # 如果是 MARKET 訂單且沒有價格，從 Binance 獲取當前價格
            if order.price is None or float(order.price) == 0:
                try:
                    ticker = await binance_client.get_ticker_price(order.symbol)
                    if not isinstance(ticker, dict) or "price" not in ticker:
                        raise ValueError(f"unexpected ticker response for {order.symbol}: {ticker!r}")
                    current_price = float(ticker["price"])
                    if current_price <= 0:
                        raise ValueError(f"non-positive price for {order.symbol}: {current_price}")
                    order_value = float(order.quantity) * current_price
                    logger.info(f"MARKET order: fetched current price {current_price} for {order.symbol}, order value: ${order_value:.2f}")
                except Exception as e:
                    logger.warning(f"Failed to get current price, skipping min value check: {e}")
                    order_value = float('inf')  # 跳過最低金額檢查
            else:
                order_value = float(order.quantity) * float(order.price)

            if order_value < min_order_value:
                logger.warning(f"Order value ${order_value:.2f} below minimum ${min_order_value}. Skipping order.")
                return [ExecutionResult(
                    success=False,
                    order_id=None,
                    executed_price=None,
                    executed_quantity=Decimal('0'),
                    total_cost=Decimal('0'),
                    commission=Decimal('0'),
                    timestamp=datetime.utcnow(),
                    error_message=f"Order value ${order_value:.2f} below minimum ${min_order_value}"
                )]

            # 1. 风险验证（可以跳過，用於從 API 已驗證過的訂單）
            if not skip_risk_check:
                validation = await risk_management_system.validate_order(order)

                if not validation.is_valid:
                    logger.warning(f"Order rejected by risk system: {validation.reason}")
                    return [ExecutionResult(
                        success=False,
                        order_id=None,
                        executed_price=None,
                        executed_quantity=Decimal('0'),
                        total_cost=Decimal('0'),
                        commission=Decimal('0'),
                        timestamp=datetime.utcnow(),
                        error_message=validation.reason
                    )]

            # 2. 分析订单簿
            depth = await self.order_book_analyzer.analyze_depth(
                order.symbol,
                order.side
            )

            # 3. 根据策略选择执行方法
            if strategy == "adaptive":
                results = await self._execute_adaptive(order, depth)
            elif strategy == "twap":
                results = await self._execute_twap(order)
            elif strategy == "vwap":
                results = await self._execute_vwap(order)
            else:
                results = await self._execute_immediate(order)

            # 4. 更新模型
            for result in results:
                if result.success and result.executed_price:
                    actual_slippage = abs(
                        float(result.executed_price - (order.price or result.executed_price))
                        / float(order.price or result.executed_price)
                    )
                    self.slippage_predictor.update_model(actual_slippage)

            return results

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return [ExecutionResult(
                success=False,
                order_id=None,
                executed_price=None,
                executed_quantity=Decimal('0'),
                total_cost=Decimal('0'),
                commission=Decimal('0'),
                timestamp=datetime.utcnow(),
                error_message=str(e)
            )]

    async def _execute_adaptive(
        self,
        order: Order,
        depth: Dict
    ) -> List[ExecutionResult]:
        """自适应执行策略"""
        logger.info(f"Executing order with adaptive strategy: {order.symbol}")

        # 🆕 小額訂單直接執行，不分塊
        from src.config import settings
        min_order_value = getattr(settings, 'MIN_ORDER_VALUE', 6.0)
        order_value = float(order.quantity) * float(order.price or 0)
        
        # 訂單金額 < $50 時，直接執行不分塊
        if order_value < 50.0:
            logger.info(f"Small order (${order_value:.2f}), executing immediately without chunking")
            return await self._execute_immediate(order)

        # 判断是否需要分割订单
        liquidity_score = depth.get('liquidity_score', 0.5)

        if liquidity_score > 0.8:
            # 高流动性，直接执行
            return await self._execute_immediate(order)
        else:
            # 低流动性，分批执行
            return await self.execute_large_order(order)

    async def execute_large_order(
        self,
        order: Order
    ) -> List[ExecutionResult]:
        """大单智能分割执行"""
        logger.info(f"Executing large order: {order.symbol} {order.quantity}")

        # 1. 分析订单簿深度
        depth = await self.order_book_analyzer.analyze_depth(
            order.symbol,
            order.side
        )

        # 2. 计算最优分割策略
        chunks = self._calculate_optimal_chunks(order, depth)

        results = []
        for chunk in chunks:
            # 3. 预测滑点
            slippage_prediction = await self.slippage_predictor.predict(chunk)
            logger.debug(f"Predicted slippage: {slippage_prediction['predicted_slippage']:.4%}")

            # 4. 找到最优执行时机
            await asyncio.sleep(chunk.delay)

            # 5. 执行订单块
            result = await self._execute_chunk(chunk, order)
            results.append(result)

            # 6. 更新市场影响模型
            if result.success:
                self.market_impact_model.update_impact_model({
                    'chunk_size': chunk.quantity,
                    'actual_price': result.executed_price,
                    'slippage': slippage_prediction['predicted_slippage']
                })

        logger.info(f"Large order execution completed: {len(results)} chunks")
        return results

    def _calculate_optimal_chunks(
        self,
        order: Order,
        depth: Dict
    ) -> List[OrderChunk]:
        """计算最优订单分片"""
        total_quantity = order.quantity
        num_chunks = self._determine_chunk_count(total_quantity, depth)

        chunks = []
        remaining = total_quantity

        for i in range(num_chunks):
            # 使用非均匀分割（开始和结束较小）
            if i == 0 or i == num_chunks - 1:
                chunk_ratio = 0.15
            else:
                chunk_ratio = 0.7 / max(1, num_chunks - 2)

            chunk_quantity = min(remaining, total_quantity * Decimal(str(chunk_ratio)))
            remaining -= chunk_quantity

            # 计算延迟（避免市场影响）
            delay = i * np.random.uniform(5, 15)  # 5-15秒间隔

            chunks.append(OrderChunk(
                quantity=chunk_quantity,
                price=order.price,
                delay=delay,
                chunk_id=i
            ))

        # 确保所有数量都分配了
        if remaining > 0:
            chunks[-1].quantity += remaining

        return chunks

    def _determine_chunk_count(
        self,
        quantity: Decimal,
        depth: Dict
    ) -> int:
        """确定分片数量"""
        liquidity_score = depth.get('liquidity_score', 0.5)

        # 基于流动性分数和订单大小确定分片数
        base_chunks = 3

        if liquidity_score < 0.3:
            base_chunks = 10
        elif liquidity_score < 0.6:
            base_chunks = 5

        return base_chunks

    async def _execute_chunk(
        self,
        chunk: OrderChunk,
        original_order: Order
    ) -> ExecutionResult:
        """执行单个订单块"""
        try:
            logger.info(
                f"Executing chunk {chunk.chunk_id}: "
                f"{chunk.quantity} @ {chunk.price or 'MARKET'}"
            )

            # 构建订单
            order_data = {
                'symbol': original_order.symbol,
                'side': original_order.side,
                'quantity': float(chunk.quantity),
                'price': float(chunk.price) if chunk.price else None,
                'type': original_order.order_type
            }

            # 使用容错引擎执行
            result = await fault_tolerant_engine.execute_with_failover(order_data)

            if result and result.get('status') == 'success':
                executed_price = Decimal(str(result.get('price', chunk.price or 0)))
                executed_qty = chunk.quantity
                total_cost = executed_price * executed_qty
                commission = total_cost * self.commission_rate

                return ExecutionResult(
                    success=True,
                    order_id=result.get('order_id'),
                    executed_price=executed_price,
                    executed_quantity=executed_qty,
                    total_cost=total_cost,
                    commission=commission,
                    timestamp=datetime.utcnow()
                )

            return ExecutionResult(
                success=False,
                order_id=None,
                executed_price=None,
                executed_quantity=Decimal('0'),
                total_cost=Decimal('0'),
                commission=Decimal('0'),
                timestamp=datetime.utcnow(),
                error_message="Execution failed"
            )

        except Exception as e:
            logger.error(f"Chunk execution failed: {e}")
            return ExecutionResult(
                success=False,
                order_id=None,
                executed_price=None,
                executed_quantity=Decimal('0'),
                total_cost=Decimal('0'),
                commission=Decimal('0'),
                timestamp=datetime.utcnow(),
                error_message=str(e)
            )

    async def _execute_immediate(self, order: Order) -> List[ExecutionResult]:
        """立即执行（市价单）"""
        logger.info(f"Executing immediate order: {order.symbol} {order.side} {order.quantity}")

        order_data = {
            'symbol': order.symbol,
            'side': order.side,
            'quantity': float(order.quantity),
            'type': 'MARKET'
        }

        result = await fault_tolerant_engine.execute_with_failover(order_data)
        logger.info(f"Execution result: {result}")

        # Binance 返回的狀態是 FILLED, NEW, PARTIALLY_FILLED 等
        # 也支持舊的 'success' 格式
        if result:
            status = result.get('status', '')
            is_success = status in ['FILLED', 'NEW', 'PARTIALLY_FILLED', 'success']

            if is_success:
                # 從 Binance 響應中提取價格
                # MARKET 訂單的成交價格在 fills 數組中
                fills = result.get('fills', [])
                if fills:
                    # 計算加權平均價格
                    total_qty = sum(float(f.get('qty', 0)) for f in fills)
                    total_cost_from_fills = sum(float(f.get('price', 0)) * float(f.get('qty', 0)) for f in fills)
                    executed_price = Decimal(str(total_cost_from_fills / total_qty)) if total_qty > 0 else Decimal('0')
                else:
                    # 備用：使用 price 或 cummulativeQuoteQty
                    executed_price = Decimal(str(result.get('price', 0) or result.get('cummulativeQuoteQty', 0)))

                total_cost = executed_price * order.quantity
                commission = total_cost * self.commission_rate
                order_id = str(result.get('orderId', result.get('order_id', '')))

                logger.info(f"Order executed successfully: {order_id} @ {executed_price}")

                return [ExecutionResult(
                    success=True,
                    order_id=order_id,
                    executed_price=executed_price,
                    executed_quantity=order.quantity,
                    total_cost=total_cost,
                    commission=commission,
                    timestamp=datetime.utcnow()
                )]
            else:
                error_msg = result.get('msg', result.get('message', f'Execution failed with status: {status}'))
                logger.error(f"Order execution failed: {error_msg}")
                return [ExecutionResult(
                    success=False,
                    order_id=None,
                    executed_price=None,
                    executed_quantity=Decimal('0'),
                    total_cost=Decimal('0'),
                    commission=Decimal('0'),
                    timestamp=datetime.utcnow(),
                    error_message=error_msg
                )]

        logger.error("Execution returned None - possible connection issue or emergency stop")
        return [ExecutionResult(
            success=False,
            order_id=None,
            executed_price=None,
            executed_quantity=Decimal('0'),
            total_cost=Decimal('0'),
            commission=Decimal('0'),
            timestamp=datetime.utcnow(),
            error_message="Execution returned None"
        )]

    async def _execute_twap(self, order: Order) -> List[ExecutionResult]:
        """TWAP (Time-Weighted Average Price) 执行"""
        logger.info(f"Executing TWAP order: {order.symbol}")
        # TODO: 实现 TWAP 策略
        return await self._execute_immediate(order)

    async def _execute_vwap(self, order: Order) -> List[ExecutionResult]:
        """VWAP (Volume-Weighted Average Price) 执行"""
        logger.info(f"Executing VWAP order: {order.symbol}")
        # TODO: 实现 VWAP 策略
        return await self._execute_immediate(order)


# 全局智能订单执行器实例
smart_order_executor = SmartOrderExecutor()
