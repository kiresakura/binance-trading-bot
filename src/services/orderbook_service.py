"""
Order Book Depth Analysis Service - 訂單簿深度分析服務

提供：
- 訂單簿深度分析
- 買賣壓力指標
- 大單監控
- 支撐阻力識別
- 市場微結構分析
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

from src.logger import logger


class OrderBookSide(str, Enum):
    """訂單簿方向"""
    BID = "bid"
    ASK = "ask"


class PressureLevel(str, Enum):
    """壓力等級"""
    VERY_LOW = "very_low"
    LOW = "low"
    NEUTRAL = "neutral"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class OrderLevel:
    """訂單層級"""
    price: float
    quantity: float
    total_value: float
    cumulative_quantity: float
    cumulative_value: float
    depth_percentage: float  # 相對於總深度的百分比

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LargeOrder:
    """大單"""
    side: OrderBookSide
    price: float
    quantity: float
    value: float
    relative_size: float  # 相對於平均訂單大小
    price_distance_pct: float  # 距離當前價格的百分比
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "side": self.side.value if isinstance(self.side, OrderBookSide) else self.side
        }


@dataclass
class SupportResistance:
    """支撐阻力位"""
    price: float
    strength: float  # 強度 0-1
    volume: float
    level_type: str  # support, resistance
    distance_pct: float  # 距離當前價格的百分比

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrderBookAnalysis:
    """訂單簿分析結果"""
    symbol: str
    timestamp: str

    # 基本數據
    best_bid: float
    best_ask: float
    spread: float
    spread_pct: float
    mid_price: float

    # 深度數據
    bid_depth: float  # 買方深度（總量）
    ask_depth: float  # 賣方深度（總量）
    bid_value: float  # 買方價值（總額）
    ask_value: float  # 賣方價值（總額）

    # 壓力指標
    imbalance: float  # 買賣不平衡 (-1 到 1)
    pressure_ratio: float  # 買壓/賣壓比率
    pressure_level: PressureLevel

    # 大單信息
    large_bid_orders: List[LargeOrder]
    large_ask_orders: List[LargeOrder]

    # 支撐阻力
    support_levels: List[SupportResistance]
    resistance_levels: List[SupportResistance]

    # 交易信號
    signal: str  # BUY, SELL, HOLD
    signal_strength: float
    reasoning: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "spread_pct": self.spread_pct,
            "mid_price": self.mid_price,
            "bid_depth": self.bid_depth,
            "ask_depth": self.ask_depth,
            "bid_value": self.bid_value,
            "ask_value": self.ask_value,
            "imbalance": self.imbalance,
            "pressure_ratio": self.pressure_ratio,
            "pressure_level": self.pressure_level.value if isinstance(self.pressure_level, PressureLevel) else self.pressure_level,
            "large_bid_orders": [o.to_dict() for o in self.large_bid_orders],
            "large_ask_orders": [o.to_dict() for o in self.large_ask_orders],
            "support_levels": [s.to_dict() for s in self.support_levels],
            "resistance_levels": [r.to_dict() for r in self.resistance_levels],
            "signal": self.signal,
            "signal_strength": self.signal_strength,
            "reasoning": self.reasoning
        }


@dataclass
class OrderBookConfig:
    """訂單簿分析配置"""
    depth_levels: int = 20           # 分析的深度層級數
    large_order_threshold: float = 5.0  # 大單閾值（相對於平均）
    imbalance_threshold: float = 0.3    # 不平衡閾值
    support_resistance_threshold: float = 2.0  # 支撐阻力閾值
    value_weighted: bool = True         # 是否使用價值加權

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OrderBookService:
    """訂單簿服務"""

    def __init__(self):
        self.config = OrderBookConfig()
        self._cache: Dict[str, OrderBookAnalysis] = {}
        self._cache_ttl = 5  # 5 秒緩存
        self._cache_timestamps: Dict[str, datetime] = {}
        logger.info("OrderBookService initialized")

    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    async def analyze(self, symbol: str) -> OrderBookAnalysis:
        """
        分析訂單簿

        Args:
            symbol: 交易對

        Returns:
            OrderBookAnalysis: 分析結果
        """
        # 檢查緩存
        if symbol in self._cache:
            cache_time = self._cache_timestamps.get(symbol)
            if cache_time and (datetime.utcnow() - cache_time).total_seconds() < self._cache_ttl:
                return self._cache[symbol]

        # 獲取訂單簿數據
        orderbook = await self._fetch_orderbook(symbol)

        if not orderbook:
            return self._empty_analysis(symbol)

        # 執行分析
        analysis = self._analyze_orderbook(symbol, orderbook)

        # 緩存結果
        self._cache[symbol] = analysis
        self._cache_timestamps[symbol] = datetime.utcnow()

        return analysis

    async def _fetch_orderbook(
        self,
        symbol: str,
        limit: int = None
    ) -> Optional[Dict[str, Any]]:
        """獲取訂單簿數據"""
        try:
            from src.services.binance_client import BinanceClient
            client = BinanceClient()

            if limit is None:
                limit = self.config.depth_levels * 2

            orderbook = await client.get_order_book(symbol, limit=limit)
            return orderbook

        except Exception as e:
            logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
            return None

    def _analyze_orderbook(
        self,
        symbol: str,
        orderbook: Dict[str, Any]
    ) -> OrderBookAnalysis:
        """分析訂單簿"""
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            return self._empty_analysis(symbol)

        # 基本數據
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        spread_pct = spread / mid_price * 100

        # 計算深度
        bid_levels = self._calculate_levels(bids, OrderBookSide.BID, mid_price)
        ask_levels = self._calculate_levels(asks, OrderBookSide.ASK, mid_price)

        bid_depth = sum(l.quantity for l in bid_levels)
        ask_depth = sum(l.quantity for l in ask_levels)
        bid_value = sum(l.total_value for l in bid_levels)
        ask_value = sum(l.total_value for l in ask_levels)

        # 計算不平衡
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
        pressure_ratio = bid_value / ask_value if ask_value > 0 else 1

        # 判斷壓力等級
        pressure_level = self._determine_pressure_level(imbalance, pressure_ratio)

        # 識別大單
        large_bids = self._find_large_orders(bid_levels, OrderBookSide.BID, mid_price)
        large_asks = self._find_large_orders(ask_levels, OrderBookSide.ASK, mid_price)

        # 識別支撐阻力
        support_levels = self._find_support_resistance(bid_levels, "support", mid_price)
        resistance_levels = self._find_support_resistance(ask_levels, "resistance", mid_price)

        # 生成交易信號
        signal, signal_strength, reasoning = self._generate_signal(
            imbalance=imbalance,
            pressure_ratio=pressure_ratio,
            large_bids=large_bids,
            large_asks=large_asks,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            spread_pct=spread_pct
        )

        return OrderBookAnalysis(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat(),
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            spread_pct=spread_pct,
            mid_price=mid_price,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            bid_value=bid_value,
            ask_value=ask_value,
            imbalance=imbalance,
            pressure_ratio=pressure_ratio,
            pressure_level=pressure_level,
            large_bid_orders=large_bids,
            large_ask_orders=large_asks,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            signal=signal,
            signal_strength=signal_strength,
            reasoning=reasoning
        )

    def _calculate_levels(
        self,
        orders: List[List],
        side: OrderBookSide,
        mid_price: float
    ) -> List[OrderLevel]:
        """計算訂單層級"""
        levels = []
        cumulative_qty = 0
        cumulative_value = 0
        total_value = sum(float(o[0]) * float(o[1]) for o in orders[:self.config.depth_levels])

        for i, order in enumerate(orders[:self.config.depth_levels]):
            price = float(order[0])
            quantity = float(order[1])
            value = price * quantity

            cumulative_qty += quantity
            cumulative_value += value

            depth_pct = value / total_value if total_value > 0 else 0

            levels.append(OrderLevel(
                price=price,
                quantity=quantity,
                total_value=value,
                cumulative_quantity=cumulative_qty,
                cumulative_value=cumulative_value,
                depth_percentage=depth_pct
            ))

        return levels

    def _determine_pressure_level(
        self,
        imbalance: float,
        pressure_ratio: float
    ) -> PressureLevel:
        """判斷壓力等級"""
        if imbalance > 0.3 and pressure_ratio > 1.5:
            return PressureLevel.VERY_HIGH
        elif imbalance > 0.15 and pressure_ratio > 1.2:
            return PressureLevel.HIGH
        elif imbalance < -0.3 and pressure_ratio < 0.67:
            return PressureLevel.VERY_LOW
        elif imbalance < -0.15 and pressure_ratio < 0.83:
            return PressureLevel.LOW
        else:
            return PressureLevel.NEUTRAL

    def _find_large_orders(
        self,
        levels: List[OrderLevel],
        side: OrderBookSide,
        mid_price: float
    ) -> List[LargeOrder]:
        """識別大單"""
        if not levels:
            return []

        avg_quantity = np.mean([l.quantity for l in levels])
        large_orders = []

        for level in levels:
            relative_size = level.quantity / avg_quantity if avg_quantity > 0 else 0

            if relative_size >= self.config.large_order_threshold:
                distance_pct = abs(level.price - mid_price) / mid_price * 100

                large_orders.append(LargeOrder(
                    side=side,
                    price=level.price,
                    quantity=level.quantity,
                    value=level.total_value,
                    relative_size=relative_size,
                    price_distance_pct=distance_pct,
                    timestamp=datetime.utcnow().isoformat()
                ))

        return large_orders

    def _find_support_resistance(
        self,
        levels: List[OrderLevel],
        level_type: str,
        mid_price: float
    ) -> List[SupportResistance]:
        """識別支撐阻力位"""
        if not levels:
            return []

        avg_value = np.mean([l.total_value for l in levels])
        sr_levels = []

        for level in levels:
            strength = level.total_value / avg_value if avg_value > 0 else 0

            if strength >= self.config.support_resistance_threshold:
                distance_pct = abs(level.price - mid_price) / mid_price * 100

                sr_levels.append(SupportResistance(
                    price=level.price,
                    strength=min(1.0, strength / 5),  # 標準化到 0-1
                    volume=level.quantity,
                    level_type=level_type,
                    distance_pct=distance_pct
                ))

        # 按強度排序
        sr_levels.sort(key=lambda x: x.strength, reverse=True)

        return sr_levels[:5]  # 返回前 5 個

    def _generate_signal(
        self,
        imbalance: float,
        pressure_ratio: float,
        large_bids: List[LargeOrder],
        large_asks: List[LargeOrder],
        support_levels: List[SupportResistance],
        resistance_levels: List[SupportResistance],
        spread_pct: float
    ) -> Tuple[str, float, List[str]]:
        """生成交易信號"""
        score = 0
        reasoning = []

        # 不平衡分析
        if imbalance > self.config.imbalance_threshold:
            score += 1
            reasoning.append(f"買方壓力較大（不平衡度：{imbalance:.2f}）")
        elif imbalance < -self.config.imbalance_threshold:
            score -= 1
            reasoning.append(f"賣方壓力較大（不平衡度：{imbalance:.2f}）")

        # 壓力比率
        if pressure_ratio > 1.5:
            score += 1
            reasoning.append(f"買方價值顯著高於賣方（比率：{pressure_ratio:.2f}）")
        elif pressure_ratio < 0.67:
            score -= 1
            reasoning.append(f"賣方價值顯著高於買方（比率：{pressure_ratio:.2f}）")

        # 大單分析
        large_bid_value = sum(o.value for o in large_bids)
        large_ask_value = sum(o.value for o in large_asks)

        if large_bid_value > large_ask_value * 2:
            score += 1
            reasoning.append("大買單明顯多於大賣單")
        elif large_ask_value > large_bid_value * 2:
            score -= 1
            reasoning.append("大賣單明顯多於大買單")

        # 支撐阻力分析
        if support_levels and support_levels[0].strength > 0.5:
            score += 0.5
            reasoning.append(f"強支撐位於 {support_levels[0].price:.4f}")

        if resistance_levels and resistance_levels[0].strength > 0.5:
            score -= 0.5
            reasoning.append(f"強阻力位於 {resistance_levels[0].price:.4f}")

        # 價差分析
        if spread_pct > 0.1:
            reasoning.append(f"價差較大（{spread_pct:.3f}%），注意滑點風險")

        # 確定信號
        if score >= 1.5:
            signal = "BUY"
            signal_strength = min(1.0, score / 3)
        elif score <= -1.5:
            signal = "SELL"
            signal_strength = min(1.0, abs(score) / 3)
        else:
            signal = "HOLD"
            signal_strength = 0.5

        if not reasoning:
            reasoning.append("訂單簿結構平衡")

        return signal, signal_strength, reasoning

    def _empty_analysis(self, symbol: str) -> OrderBookAnalysis:
        """返回空分析"""
        return OrderBookAnalysis(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat(),
            best_bid=0,
            best_ask=0,
            spread=0,
            spread_pct=0,
            mid_price=0,
            bid_depth=0,
            ask_depth=0,
            bid_value=0,
            ask_value=0,
            imbalance=0,
            pressure_ratio=1,
            pressure_level=PressureLevel.NEUTRAL,
            large_bid_orders=[],
            large_ask_orders=[],
            support_levels=[],
            resistance_levels=[],
            signal="HOLD",
            signal_strength=0,
            reasoning=["無法獲取訂單簿數據"]
        )

    async def get_market_depth_chart_data(
        self,
        symbol: str,
        levels: int = 50
    ) -> Dict[str, Any]:
        """獲取市場深度圖表數據"""
        try:
            from src.services.binance_client import BinanceClient
            client = BinanceClient()

            orderbook = await client.get_order_book(symbol, limit=levels)

            if not orderbook:
                return {"bids": [], "asks": [], "mid_price": 0}

            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])

            # 計算累積深度
            bid_data = []
            ask_data = []
            bid_cumulative = 0
            ask_cumulative = 0

            for bid in bids:
                bid_cumulative += float(bid[1])
                bid_data.append({
                    "price": float(bid[0]),
                    "quantity": float(bid[1]),
                    "cumulative": bid_cumulative
                })

            for ask in asks:
                ask_cumulative += float(ask[1])
                ask_data.append({
                    "price": float(ask[0]),
                    "quantity": float(ask[1]),
                    "cumulative": ask_cumulative
                })

            mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2 if bids and asks else 0

            return {
                "bids": bid_data,
                "asks": ask_data,
                "mid_price": mid_price,
                "total_bid_depth": bid_cumulative,
                "total_ask_depth": ask_cumulative
            }

        except Exception as e:
            logger.error(f"Failed to get market depth chart data: {e}")
            return {"bids": [], "asks": [], "mid_price": 0}

    async def monitor_order_flow(
        self,
        symbol: str,
        duration_seconds: int = 60,
        interval_ms: int = 1000
    ) -> Dict[str, Any]:
        """監控訂單流"""
        snapshots = []
        start_time = datetime.utcnow()

        iterations = int(duration_seconds * 1000 / interval_ms)

        for _ in range(iterations):
            analysis = await self.analyze(symbol)
            snapshots.append({
                "timestamp": analysis.timestamp,
                "imbalance": analysis.imbalance,
                "bid_depth": analysis.bid_depth,
                "ask_depth": analysis.ask_depth,
                "spread_pct": analysis.spread_pct
            })

            await asyncio.sleep(interval_ms / 1000)

        # 分析訂單流
        if not snapshots:
            return {}

        avg_imbalance = np.mean([s["imbalance"] for s in snapshots])
        imbalance_trend = snapshots[-1]["imbalance"] - snapshots[0]["imbalance"]

        avg_spread = np.mean([s["spread_pct"] for s in snapshots])

        return {
            "symbol": symbol,
            "duration_seconds": duration_seconds,
            "samples": len(snapshots),
            "avg_imbalance": avg_imbalance,
            "imbalance_trend": imbalance_trend,
            "avg_spread_pct": avg_spread,
            "flow_direction": "BUY" if avg_imbalance > 0.1 else ("SELL" if avg_imbalance < -0.1 else "NEUTRAL"),
            "snapshots": snapshots
        }


# 單例實例
orderbook_service = OrderBookService()
