"""
Cross-Exchange Arbitrage Service - 跨市場套利服務

提供：
- 價差監控
- 套利機會識別
- 風險評估
- 套利執行策略
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp

from src.logger import logger


class ArbitrageType(str, Enum):
    """套利類型"""
    SPOT_SPOT = "spot_spot"          # 現貨-現貨
    SPOT_FUTURES = "spot_futures"    # 現貨-期貨
    TRIANGULAR = "triangular"        # 三角套利
    CROSS_EXCHANGE = "cross_exchange"  # 跨交易所


class Exchange(str, Enum):
    """交易所"""
    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"
    HUOBI = "huobi"
    KUCOIN = "kucoin"


@dataclass
class PriceData:
    """價格數據"""
    exchange: Exchange
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "exchange": self.exchange.value if isinstance(self.exchange, Exchange) else self.exchange
        }


@dataclass
class ArbitrageOpportunity:
    """套利機會"""
    opportunity_id: str
    arbitrage_type: ArbitrageType
    symbol: str
    buy_exchange: Exchange
    sell_exchange: Exchange
    buy_price: float
    sell_price: float
    spread: float           # 價差（絕對值）
    spread_pct: float       # 價差百分比
    potential_profit_pct: float  # 潛在利潤（扣除費用後）
    max_size: float         # 最大可執行數量
    risk_score: float       # 風險評分 0-1
    timestamp: str
    expires_at: str         # 過期時間
    is_executable: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "arbitrage_type": self.arbitrage_type.value if isinstance(self.arbitrage_type, ArbitrageType) else self.arbitrage_type,
            "symbol": self.symbol,
            "buy_exchange": self.buy_exchange.value if isinstance(self.buy_exchange, Exchange) else self.buy_exchange,
            "sell_exchange": self.sell_exchange.value if isinstance(self.sell_exchange, Exchange) else self.sell_exchange,
            "buy_price": self.buy_price,
            "sell_price": self.sell_price,
            "spread": self.spread,
            "spread_pct": self.spread_pct,
            "potential_profit_pct": self.potential_profit_pct,
            "max_size": self.max_size,
            "risk_score": self.risk_score,
            "timestamp": self.timestamp,
            "expires_at": self.expires_at,
            "is_executable": self.is_executable
        }


@dataclass
class ArbitrageConfig:
    """套利配置"""
    # 最小利潤閾值
    min_profit_pct: float = 0.1       # 最小利潤 0.1%
    min_spread_pct: float = 0.05      # 最小價差 0.05%

    # 費用
    trading_fee_pct: float = 0.1      # 交易費 0.1%
    transfer_fee_pct: float = 0.05    # 轉帳費 0.05%

    # 風險控制
    max_position_value: float = 1000  # 最大倉位價值
    max_execution_time_ms: int = 5000 # 最大執行時間
    max_risk_score: float = 0.7       # 最大風險評分

    # 監控
    scan_interval_seconds: int = 5    # 掃描間隔
    opportunity_ttl_seconds: int = 30 # 機會有效期

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArbitrageTrade:
    """套利交易記錄"""
    trade_id: str
    opportunity_id: str
    symbol: str
    buy_exchange: Exchange
    sell_exchange: Exchange
    buy_price: float
    sell_price: float
    quantity: float
    expected_profit_pct: float
    actual_profit_pct: Optional[float]
    status: str  # pending, executed, failed, cancelled
    executed_at: Optional[str]
    error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "buy_exchange": self.buy_exchange.value if isinstance(self.buy_exchange, Exchange) else self.buy_exchange,
            "sell_exchange": self.sell_exchange.value if isinstance(self.sell_exchange, Exchange) else self.sell_exchange
        }


class ArbitrageService:
    """套利服務"""

    def __init__(self):
        self.config = ArbitrageConfig()
        self._price_cache: Dict[str, Dict[str, PriceData]] = {}
        self._opportunities: Dict[str, ArbitrageOpportunity] = {}
        self._trades: List[ArbitrageTrade] = []
        self._opportunity_counter = 0
        self._trade_counter = 0
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False

        # 交易所 API 端點
        self._exchange_endpoints = {
            Exchange.BINANCE: "https://api.binance.com/api/v3/ticker/bookTicker",
            Exchange.BYBIT: "https://api.bybit.com/v5/market/tickers",
            Exchange.OKX: "https://www.okx.com/api/v5/market/ticker",
        }

        logger.info("ArbitrageService initialized")

    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Arbitrage config updated: {key} = {value}")

    async def start_monitoring(self, symbols: List[str] = None):
        """啟動價格監控"""
        if self._is_monitoring:
            logger.warning("Monitoring already running")
            return

        if symbols is None:
            symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(symbols)
        )
        logger.info(f"Arbitrage monitoring started for {symbols}")

    async def stop_monitoring(self):
        """停止價格監控"""
        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Arbitrage monitoring stopped")

    async def _monitoring_loop(self, symbols: List[str]):
        """監控循環"""
        while self._is_monitoring:
            try:
                # 獲取各交易所價格
                await self._fetch_all_prices(symbols)

                # 識別套利機會
                await self._scan_opportunities(symbols)

                # 清理過期機會
                self._cleanup_expired_opportunities()

                await asyncio.sleep(self.config.scan_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)

    async def _fetch_all_prices(self, symbols: List[str]):
        """獲取所有交易所價格"""
        tasks = []
        for exchange in [Exchange.BINANCE]:  # 目前只支持 Binance
            for symbol in symbols:
                tasks.append(self._fetch_price(exchange, symbol))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_price(self, exchange: Exchange, symbol: str):
        """獲取單一交易所價格"""
        try:
            if exchange == Exchange.BINANCE:
                url = f"https://api.binance.com/api/v3/ticker/bookTicker?symbol={symbol}"

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            data = await response.json()
                            price_data = PriceData(
                                exchange=exchange,
                                symbol=symbol,
                                bid_price=float(data["bidPrice"]),
                                ask_price=float(data["askPrice"]),
                                bid_size=float(data["bidQty"]),
                                ask_size=float(data["askQty"]),
                                timestamp=datetime.utcnow().isoformat()
                            )

                            if symbol not in self._price_cache:
                                self._price_cache[symbol] = {}
                            self._price_cache[symbol][exchange.value] = price_data

        except Exception as e:
            logger.warning(f"Failed to fetch price from {exchange.value} for {symbol}: {e}")

    async def _scan_opportunities(self, symbols: List[str]):
        """掃描套利機會"""
        for symbol in symbols:
            prices = self._price_cache.get(symbol, {})

            if len(prices) < 2:
                continue

            # 比較所有交易所組合
            exchanges = list(prices.keys())
            for i in range(len(exchanges)):
                for j in range(i + 1, len(exchanges)):
                    await self._check_opportunity(
                        symbol,
                        prices[exchanges[i]],
                        prices[exchanges[j]]
                    )

    async def _check_opportunity(
        self,
        symbol: str,
        price1: PriceData,
        price2: PriceData
    ):
        """檢查套利機會"""
        # 計算兩個方向的價差
        # 方向1：在交易所1買入，在交易所2賣出
        spread1 = price2.bid_price - price1.ask_price
        spread1_pct = spread1 / price1.ask_price * 100

        # 方向2：在交易所2買入，在交易所1賣出
        spread2 = price1.bid_price - price2.ask_price
        spread2_pct = spread2 / price2.ask_price * 100

        # 檢查方向1
        if spread1_pct > self.config.min_spread_pct:
            profit1 = spread1_pct - 2 * self.config.trading_fee_pct - self.config.transfer_fee_pct
            if profit1 > self.config.min_profit_pct:
                self._create_opportunity(
                    symbol=symbol,
                    buy_exchange=price1.exchange,
                    sell_exchange=price2.exchange,
                    buy_price=price1.ask_price,
                    sell_price=price2.bid_price,
                    spread=spread1,
                    spread_pct=spread1_pct,
                    profit_pct=profit1,
                    max_size=min(price1.ask_size, price2.bid_size)
                )

        # 檢查方向2
        if spread2_pct > self.config.min_spread_pct:
            profit2 = spread2_pct - 2 * self.config.trading_fee_pct - self.config.transfer_fee_pct
            if profit2 > self.config.min_profit_pct:
                self._create_opportunity(
                    symbol=symbol,
                    buy_exchange=price2.exchange,
                    sell_exchange=price1.exchange,
                    buy_price=price2.ask_price,
                    sell_price=price1.bid_price,
                    spread=spread2,
                    spread_pct=spread2_pct,
                    profit_pct=profit2,
                    max_size=min(price2.ask_size, price1.bid_size)
                )

    def _create_opportunity(
        self,
        symbol: str,
        buy_exchange: Exchange,
        sell_exchange: Exchange,
        buy_price: float,
        sell_price: float,
        spread: float,
        spread_pct: float,
        profit_pct: float,
        max_size: float
    ):
        """創建套利機會"""
        self._opportunity_counter += 1
        opportunity_id = f"arb_{self._opportunity_counter}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        # 計算風險評分
        risk_score = self._calculate_risk_score(spread_pct, max_size)

        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=self.config.opportunity_ttl_seconds)

        opportunity = ArbitrageOpportunity(
            opportunity_id=opportunity_id,
            arbitrage_type=ArbitrageType.CROSS_EXCHANGE,
            symbol=symbol,
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            buy_price=buy_price,
            sell_price=sell_price,
            spread=spread,
            spread_pct=spread_pct,
            potential_profit_pct=profit_pct,
            max_size=max_size,
            risk_score=risk_score,
            timestamp=now.isoformat(),
            expires_at=expires_at.isoformat(),
            is_executable=risk_score < self.config.max_risk_score
        )

        self._opportunities[opportunity_id] = opportunity

        logger.info(
            f"Arbitrage opportunity: {symbol} "
            f"Buy@{buy_exchange.value}:{buy_price:.4f} "
            f"Sell@{sell_exchange.value}:{sell_price:.4f} "
            f"Profit: {profit_pct:.3f}%"
        )

    def _calculate_risk_score(self, spread_pct: float, max_size: float) -> float:
        """計算風險評分"""
        # 風險因素：
        # 1. 價差太小 -> 滑點風險高
        # 2. 深度太淺 -> 執行風險高

        spread_risk = max(0, 1 - spread_pct / 1.0)  # 價差越大，風險越低
        depth_risk = max(0, 1 - max_size / 10)       # 深度越大，風險越低

        return (spread_risk + depth_risk) / 2

    def _cleanup_expired_opportunities(self):
        """清理過期機會"""
        now = datetime.utcnow()
        expired = []

        for opp_id, opp in self._opportunities.items():
            expires = datetime.fromisoformat(opp.expires_at)
            if now > expires:
                expired.append(opp_id)

        for opp_id in expired:
            del self._opportunities[opp_id]

    async def execute_arbitrage(
        self,
        opportunity_id: str,
        quantity: Optional[float] = None
    ) -> ArbitrageTrade:
        """執行套利"""
        opportunity = self._opportunities.get(opportunity_id)
        if not opportunity:
            raise ValueError(f"Opportunity {opportunity_id} not found")

        if not opportunity.is_executable:
            raise ValueError(f"Opportunity {opportunity_id} is not executable")

        # 檢查是否過期
        if datetime.utcnow() > datetime.fromisoformat(opportunity.expires_at):
            raise ValueError(f"Opportunity {opportunity_id} has expired")

        # 確定交易數量
        if quantity is None:
            quantity = min(
                opportunity.max_size,
                self.config.max_position_value / opportunity.buy_price
            )

        self._trade_counter += 1
        trade_id = f"arb_trade_{self._trade_counter}"

        trade = ArbitrageTrade(
            trade_id=trade_id,
            opportunity_id=opportunity_id,
            symbol=opportunity.symbol,
            buy_exchange=opportunity.buy_exchange,
            sell_exchange=opportunity.sell_exchange,
            buy_price=opportunity.buy_price,
            sell_price=opportunity.sell_price,
            quantity=quantity,
            expected_profit_pct=opportunity.potential_profit_pct,
            actual_profit_pct=None,
            status="pending",
            executed_at=None,
            error=None
        )

        try:
            # 這裡應該實際執行交易
            # 目前為模擬執行
            await asyncio.sleep(0.1)

            trade.status = "executed"
            trade.executed_at = datetime.utcnow().isoformat()
            trade.actual_profit_pct = opportunity.potential_profit_pct * 0.9  # 模擬滑點

            logger.info(f"Arbitrage executed: {trade_id}, Profit: {trade.actual_profit_pct:.3f}%")

        except Exception as e:
            trade.status = "failed"
            trade.error = str(e)
            logger.error(f"Arbitrage execution failed: {e}")

        self._trades.append(trade)
        return trade

    def get_opportunities(
        self,
        symbol: Optional[str] = None,
        min_profit: Optional[float] = None,
        executable_only: bool = True
    ) -> List[Dict[str, Any]]:
        """獲取套利機會"""
        opportunities = list(self._opportunities.values())

        if symbol:
            opportunities = [o for o in opportunities if o.symbol == symbol]

        if min_profit is not None:
            opportunities = [o for o in opportunities if o.potential_profit_pct >= min_profit]

        if executable_only:
            opportunities = [o for o in opportunities if o.is_executable]

        # 按利潤排序
        opportunities.sort(key=lambda o: o.potential_profit_pct, reverse=True)

        return [o.to_dict() for o in opportunities]

    def get_trade_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """獲取交易歷史"""
        return [t.to_dict() for t in self._trades[-limit:]]

    def get_statistics(self) -> Dict[str, Any]:
        """獲取統計數據"""
        executed_trades = [t for t in self._trades if t.status == "executed"]
        failed_trades = [t for t in self._trades if t.status == "failed"]

        total_profit = sum(t.actual_profit_pct or 0 for t in executed_trades)
        avg_profit = total_profit / len(executed_trades) if executed_trades else 0

        return {
            "total_opportunities": self._opportunity_counter,
            "active_opportunities": len(self._opportunities),
            "total_trades": len(self._trades),
            "executed_trades": len(executed_trades),
            "failed_trades": len(failed_trades),
            "total_profit_pct": total_profit,
            "avg_profit_pct": avg_profit,
            "is_monitoring": self._is_monitoring
        }

    async def analyze_triangular_arbitrage(
        self,
        base_currency: str = "BTC"
    ) -> List[Dict[str, Any]]:
        """分析三角套利機會"""
        # 三角套利路徑示例：USDT -> BTC -> ETH -> USDT

        opportunities = []

        try:
            from src.services.binance_client import BinanceClient
            client = BinanceClient()

            # 獲取相關價格
            btc_usdt = await client.get_price("BTCUSDT")
            eth_usdt = await client.get_price("ETHUSDT")
            eth_btc = await client.get_price("ETHBTC")

            if not all([btc_usdt, eth_usdt, eth_btc]):
                return []

            # 計算三角套利
            # 路徑1：USDT -> BTC -> ETH -> USDT
            # 買入 BTC，用 BTC 買入 ETH，賣出 ETH 獲得 USDT
            btc_amount = 1.0 / btc_usdt
            eth_amount = btc_amount / eth_btc
            final_usdt = eth_amount * eth_usdt

            profit_pct_1 = (final_usdt - 1.0) * 100 - 3 * self.config.trading_fee_pct

            if profit_pct_1 > self.config.min_profit_pct:
                opportunities.append({
                    "path": ["USDT", "BTC", "ETH", "USDT"],
                    "profit_pct": profit_pct_1,
                    "prices": {
                        "BTCUSDT": btc_usdt,
                        "ETHBTC": eth_btc,
                        "ETHUSDT": eth_usdt
                    }
                })

            # 路徑2：USDT -> ETH -> BTC -> USDT
            eth_amount_2 = 1.0 / eth_usdt
            btc_amount_2 = eth_amount_2 * eth_btc
            final_usdt_2 = btc_amount_2 * btc_usdt

            profit_pct_2 = (final_usdt_2 - 1.0) * 100 - 3 * self.config.trading_fee_pct

            if profit_pct_2 > self.config.min_profit_pct:
                opportunities.append({
                    "path": ["USDT", "ETH", "BTC", "USDT"],
                    "profit_pct": profit_pct_2,
                    "prices": {
                        "ETHUSDT": eth_usdt,
                        "ETHBTC": eth_btc,
                        "BTCUSDT": btc_usdt
                    }
                })

        except Exception as e:
            logger.error(f"Failed to analyze triangular arbitrage: {e}")

        return opportunities


# 單例實例
arbitrage_service = ArbitrageService()
