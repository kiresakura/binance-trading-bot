"""多层风险管理系统"""

from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np
from src.logger import logger
from src.config import settings
from src.config import settings
from src.services.binance_client import binance_client
from src.services.cache_service import cache_service


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    reason: Optional[str] = None
    risk_score: float = 0.0
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class Order:
    """订单模型"""
    symbol: str
    side: str  # BUY or SELL
    quantity: Decimal
    price: Optional[Decimal] = None
    order_type: str = "MARKET"
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class RiskLayer:
    """风险层基类"""

    async def validate(self, order: Order) -> ValidationResult:
        """验证订单"""
        raise NotImplementedError


class PositionSizeController(RiskLayer):
    """第1层：仓位控制"""

    def __init__(
        self,
        max_position_size: float = 0.1,  # 单笔最大10%
        max_total_exposure: float = 0.5   # 总暴露最大50%
    ):
        self.max_position_size = max_position_size
        self.max_total_exposure = max_total_exposure

    async def validate(self, order: Order) -> ValidationResult:
        """验证仓位大小"""
        try:
            # 获取当前投资组合价值
            portfolio_value = await self._get_portfolio_value()

            # 计算订单价值
            order_value = float(order.quantity) * float(order.price or 0)
            position_ratio = order_value / portfolio_value if portfolio_value > 0 else 0

            # 检查单笔仓位限制
            if position_ratio > self.max_position_size:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Position size {position_ratio:.2%} exceeds max {self.max_position_size:.2%}",
                    risk_score=1.0
                )

            # 检查总暴露
            current_exposure = await self._get_total_exposure()
            new_exposure = current_exposure + position_ratio

            if new_exposure > self.max_total_exposure:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Total exposure {new_exposure:.2%} exceeds max {self.max_total_exposure:.2%}",
                    risk_score=0.9
                )

            # 计算风险分数
            risk_score = position_ratio / self.max_position_size

            warnings = []
            if position_ratio > self.max_position_size * 0.7:
                warnings.append(f"Position size approaching limit ({position_ratio:.2%})")

            return ValidationResult(
                is_valid=True,
                risk_score=risk_score,
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"Position size validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                reason=f"Validation error: {str(e)}",
                risk_score=1.0
            )

    async def _get_portfolio_value(self) -> float:
        """获取投资组合价值（從 Binance API 獲取實際餘額）"""
        try:
            # 獲取帳戶餘額
            balances = await binance_client.get_balances()

            if not balances:
                logger.warning("Unable to fetch balances, using default value")
                return 10000.0

            total_usdt_value = 0.0

            for balance in balances:
                asset = balance.get("asset", "")
                total = float(balance.get("total", 0))

                if total <= 0:
                    continue

                # 處理活期理財代幣 (LD 開頭)
                base_asset = asset[2:] if asset.startswith("LD") else asset

                # 計算 USDT 價值
                if base_asset == "USDT":
                    total_usdt_value += total
                else:
                    # 嘗試獲取該資產的 USDT 價格
                    try:
                        ticker = await binance_client.get_ticker_price(f"{base_asset}USDT")
                        if isinstance(ticker, dict) and "price" in ticker:
                            price = float(ticker["price"])
                            total_usdt_value += total * price
                    except Exception:
                        # 如果無法獲取價格，嘗試 BTC 計價
                        try:
                            ticker = await binance_client.get_ticker_price(f"{base_asset}BTC")
                            if isinstance(ticker, dict) and "price" in ticker:
                                btc_price = float(ticker["price"])
                                btc_usdt = await binance_client.get_ticker_price("BTCUSDT")
                                if isinstance(btc_usdt, dict) and "price" in btc_usdt:
                                    total_usdt_value += total * btc_price * float(btc_usdt["price"])
                        except Exception:
                            pass

            # 確保返回一個合理的最小值
            return max(total_usdt_value, 100.0)

        except Exception as e:
            logger.error(f"Failed to get portfolio value: {e}")
            return 10000.0  # 發生錯誤時返回默認值

    async def _get_total_exposure(self) -> float:
        """获取当前总暴露（從 Binance 獲取實際持倉）"""
        try:
            # 獲取投資組合總價值
            portfolio_value = await self._get_portfolio_value()
            if portfolio_value <= 0:
                return 0.0

            # 獲取帳戶餘額
            balances = await binance_client.get_balances()
            if not balances:
                return 0.0

            # 計算非穩定幣的暴露
            exposed_value = 0.0

            for balance in balances:
                asset = balance.get("asset", "")
                total = float(balance.get("total", 0))

                if total <= 0:
                    continue

                # 處理活期理財代幣
                base_asset = asset[2:] if asset.startswith("LD") else asset

                # 跳過穩定幣（USDT, USDC, BUSD 等）
                if base_asset in ["USDT", "USDC", "BUSD", "DAI", "TUSD"]:
                    continue

                # 計算該資產的 USDT 價值
                try:
                    ticker = await binance_client.get_ticker_price(f"{base_asset}USDT")
                    if isinstance(ticker, dict) and "price" in ticker:
                        price = float(ticker["price"])
                        exposed_value += total * price
                except Exception:
                    pass

            # 計算暴露比例
            exposure = exposed_value / portfolio_value if portfolio_value > 0 else 0.0
            return min(exposure, 1.0)

        except Exception as e:
            logger.error(f"Failed to get total exposure: {e}")
            return 0.2  # 發生錯誤時返回保守估計值


class DrawdownProtection(RiskLayer):
    """第2层：回撤保护"""

    def __init__(
        self,
        max_daily_loss: float = 0.02,      # 日内最大亏损2%
        max_weekly_loss: float = 0.05,     # 周内最大亏损5%
        max_monthly_loss: float = 0.10     # 月内最大亏损10%
    ):
        self.max_daily_loss = max_daily_loss
        self.max_weekly_loss = max_weekly_loss
        self.max_monthly_loss = max_monthly_loss
        # 回撤追蹤狀態（實例級別，避免跨實例共享造成狀態洩漏）
        self._pnl_history: Dict[str, list] = {"day": [], "week": [], "month": []}
        self._peak_values: Dict[str, float] = {"day": 0.0, "week": 0.0, "month": 0.0}

    async def validate(self, order: Order) -> ValidationResult:
        """验证回撤限制"""
        try:
            # 获取不同时间范围的回撤
            daily_drawdown = await self._calculate_drawdown(period='day')
            weekly_drawdown = await self._calculate_drawdown(period='week')
            monthly_drawdown = await self._calculate_drawdown(period='month')

            # 检查日内回撤
            if abs(daily_drawdown) > self.max_daily_loss:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Daily drawdown {daily_drawdown:.2%} exceeds limit {self.max_daily_loss:.2%}",
                    risk_score=1.0
                )

            # 检查周内回撤
            if abs(weekly_drawdown) > self.max_weekly_loss:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Weekly drawdown {weekly_drawdown:.2%} exceeds limit {self.max_weekly_loss:.2%}",
                    risk_score=0.95
                )

            # 检查月内回撤
            if abs(monthly_drawdown) > self.max_monthly_loss:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Monthly drawdown {monthly_drawdown:.2%} exceeds limit {self.max_monthly_loss:.2%}",
                    risk_score=0.9
                )

            # 计算风险分数
            risk_score = max(
                abs(daily_drawdown) / self.max_daily_loss,
                abs(weekly_drawdown) / self.max_weekly_loss,
                abs(monthly_drawdown) / self.max_monthly_loss
            )

            warnings = []
            if abs(daily_drawdown) > self.max_daily_loss * 0.7:
                warnings.append(f"Daily drawdown approaching limit ({daily_drawdown:.2%})")

            return ValidationResult(
                is_valid=True,
                risk_score=risk_score,
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"Drawdown protection validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                reason=f"Validation error: {str(e)}",
                risk_score=1.0
            )

    async def _calculate_drawdown(self, period: str = 'day') -> float:
        """计算指定时期的回撤（基於實際交易記錄）"""
        try:
            # 獲取當前投資組合價值
            from src.services.risk_manager import risk_management_system
            position_controller = risk_management_system.risk_layers[0]
            current_value = await position_controller._get_portfolio_value()

            # 更新峰值
            if current_value > self._peak_values.get(period, 0):
                self._peak_values[period] = current_value

            peak = self._peak_values.get(period, current_value)
            if peak <= 0:
                return 0.0

            # 計算回撤
            drawdown = (current_value - peak) / peak
            return drawdown  # 回撤為負值

        except Exception as e:
            logger.error(f"Failed to calculate drawdown: {e}")
            return 0.0


class CorrelationAnalyzer(RiskLayer):
    """第3层：相关性风险分析"""

    def __init__(self, max_correlation: float = 0.8):
        self.max_correlation = max_correlation
        # self.correlation_cache = {}  # Removed in favor of Redis

    async def validate(self, order: Order) -> ValidationResult:
        """验证相关性风险"""
        try:
            # 获取当前持仓
            current_positions = await self._get_current_positions()

            if not current_positions:
                return ValidationResult(is_valid=True, risk_score=0.0)

            # 计算与现有持仓的相关性
            max_corr = 0.0
            correlated_assets = []

            for position in current_positions:
                correlation = await self._get_correlation(
                    order.symbol,
                    position['symbol']
                )

                if abs(correlation) > self.max_correlation:
                    correlated_assets.append({
                        'symbol': position['symbol'],
                        'correlation': correlation
                    })

                max_corr = max(max_corr, abs(correlation))

            # 如果发现高度相关的资产
            if correlated_assets:
                return ValidationResult(
                    is_valid=False,
                    reason=f"High correlation with existing positions: {correlated_assets}",
                    risk_score=max_corr
                )

            # 计算风险分数
            risk_score = max_corr / self.max_correlation if self.max_correlation > 0 else 0

            warnings = []
            if max_corr > self.max_correlation * 0.7:
                warnings.append(f"Correlation approaching limit ({max_corr:.2f})")

            return ValidationResult(
                is_valid=True,
                risk_score=risk_score,
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return ValidationResult(
                is_valid=False,
                reason=f"Validation error: {str(e)}",
                risk_score=1.0
            )

    async def _get_current_positions(self) -> List[Dict]:
        """获取当前持仓（從 Binance 獲取實際持倉）"""
        try:
            balances = await binance_client.get_balances()
            positions = []

            for balance in balances:
                asset = balance.get("asset", "")
                total = float(balance.get("total", 0))

                # 跳過零餘額和穩定幣
                base_asset = asset[2:] if asset.startswith("LD") else asset
                if total <= 0 or base_asset in ["USDT", "USDC", "BUSD", "DAI", "TUSD"]:
                    continue

                # 計算 USDT 價值
                try:
                    ticker = await binance_client.get_ticker_price(f"{base_asset}USDT")
                    if isinstance(ticker, dict) and "price" in ticker:
                        value = total * float(ticker["price"])
                        if value > 10:  # 只包含價值超過 $10 的持倉
                            positions.append({
                                "symbol": f"{base_asset}USDT",
                                "quantity": total,
                                "value": value
                            })
                except Exception:
                    pass

            return positions

        except Exception as e:
            logger.error(f"Failed to get current positions: {e}")
            return []

    async def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """获取两个资产的相关系数（使用歷史價格數據）"""
        cache_key = f"corr:{symbol1}:{symbol2}"
        reverse_key = f"corr:{symbol2}:{symbol1}"

        # Check Redis cache
        cached_val = await cache_service.get(cache_key)
        if cached_val is not None:
            return float(cached_val)
            
        cached_reverse = await cache_service.get(reverse_key)
        if cached_reverse is not None:
            return float(cached_reverse)

        try:
            # Get historical K-lines (recent 50 1h candles)
            klines1 = await binance_client.get_klines(symbol1, "1h", limit=50)
            klines2 = await binance_client.get_klines(symbol2, "1h", limit=50)

            if not klines1 or not klines2:
                return 0.0

            # Extract close prices
            prices1 = [float(k[4]) for k in klines1] 
            prices2 = [float(k[4]) for k in klines2]

            # Ensure same length
            min_len = min(len(prices1), len(prices2))
            prices1 = prices1[:min_len]
            prices2 = prices2[:min_len]

            if min_len < 10:
                return 0.0

            # Calculate returns
            returns1 = np.diff(prices1) / prices1[:-1]
            returns2 = np.diff(prices2) / prices2[:-1]

            # Calculate correlation
            correlation = float(np.corrcoef(returns1, returns2)[0, 1])

            # Handle NaN
            if np.isnan(correlation):
                correlation = 0.0

            # Cache result in Redis (1 hour expiration)
            await cache_service.set(cache_key, correlation, expire=3600)

            return correlation

        except Exception as e:
            logger.error(f"Failed to calculate correlation: {e}")
            return 0.0  # Return 0 on error


class BlackSwanProtection(RiskLayer):
    """第4层：黑天鹅事件保护"""

    def __init__(
        self,
        volatility_threshold: float = 3.0,  # 3倍标准差
        volume_spike_threshold: float = 5.0  # 5倍平均成交量
    ):
        self.volatility_threshold = volatility_threshold
        self.volume_spike_threshold = volume_spike_threshold

    async def validate(self, order: Order) -> ValidationResult:
        """检测极端市场条件"""
        try:
            # 检测波动率异常
            current_volatility = await self._get_current_volatility(order.symbol)
            historical_volatility = await self._get_historical_volatility(order.symbol)

            volatility_ratio = current_volatility / historical_volatility if historical_volatility > 0 else 1

            if volatility_ratio > self.volatility_threshold:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Extreme volatility detected: {volatility_ratio:.2f}x normal",
                    risk_score=1.0
                )

            # 检测成交量异常
            current_volume = await self._get_current_volume(order.symbol)
            avg_volume = await self._get_average_volume(order.symbol)

            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            if volume_ratio > self.volume_spike_threshold:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Abnormal volume spike detected: {volume_ratio:.2f}x average",
                    risk_score=0.9
                )

            # 计算风险分数
            risk_score = max(
                volatility_ratio / self.volatility_threshold,
                volume_ratio / self.volume_spike_threshold
            )

            warnings = []
            if volatility_ratio > self.volatility_threshold * 0.7:
                warnings.append(f"Elevated volatility ({volatility_ratio:.2f}x)")
            if volume_ratio > self.volume_spike_threshold * 0.7:
                warnings.append(f"Elevated volume ({volume_ratio:.2f}x)")

            return ValidationResult(
                is_valid=True,
                risk_score=min(risk_score, 1.0),
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"Black swan protection failed: {e}")
            return ValidationResult(
                is_valid=False,
                reason=f"Validation error: {str(e)}",
                risk_score=1.0
            )

    # Removed in-memory caches in favor of Redis
    # _volatility_cache: Dict[str, tuple] = {} 
    # _volume_cache: Dict[str, tuple] = {}

    async def _get_current_volatility(self, symbol: str) -> float:
        """获取当前波动率（使用最近 K 線計算）"""
        try:
            # 獲取最近 20 根 5 分鐘 K 線
            klines = await binance_client.get_klines(symbol, "5m", limit=20)
            if not klines or len(klines) < 10:
                return 0.02

            # 計算收益率
            closes = [float(k[4]) for k in klines]
            returns = np.diff(closes) / closes[:-1]

            # 當前波動率 = 收益率標準差
            volatility = float(np.std(returns))
            return volatility if volatility > 0 else 0.01

        except Exception as e:
            logger.error(f"Failed to get current volatility: {e}")
            return 0.02

    async def _get_historical_volatility(self, symbol: str) -> float:
        """获取历史平均波动率（使用較長時間範圍）"""
        cache_key = f"hist_vol:{symbol}"

        # Check Redis cache
        cached_val = await cache_service.get(cache_key)
        if cached_val is not None:
             return float(cached_val)

        try:
            # Get recent 200 1h candles (~8 days)
            klines = await binance_client.get_klines(symbol, "1h", limit=200)
            if not klines or len(klines) < 50:
                return 0.02

            # Calculate returns
            closes = [float(k[4]) for k in klines]
            returns = np.diff(closes) / closes[:-1]

            # Historical Volatility = Std Dev of Returns
            volatility = float(np.std(returns))

            # Cache result in Redis (1 hour expiration)
            await cache_service.set(cache_key, volatility, expire=3600)

            return volatility if volatility > 0 else 0.02

        except Exception as e:
            logger.error(f"Failed to get historical volatility: {e}")
            return 0.02

    async def _get_current_volume(self, symbol: str) -> float:
        """获取当前成交量（最近 1 小時）"""
        try:
            # 獲取最近 1 根 1 小時 K 線
            klines = await binance_client.get_klines(symbol, "1h", limit=1)
            if not klines:
                return 1000.0

            # 成交量在索引 5
            volume = float(klines[-1][5])
            return volume if volume > 0 else 1000.0

        except Exception as e:
            logger.error(f"Failed to get current volume: {e}")
            return 1000.0

    async def _get_average_volume(self, symbol: str) -> float:
        """获取平均成交量（過去 24 小時平均）"""
        cache_key = f"avg_vol:{symbol}"

        # Check Redis cache
        cached_val = await cache_service.get(cache_key)
        if cached_val is not None:
             return float(cached_val)

        try:
            # Get recent 24 1h candles
            klines = await binance_client.get_klines(symbol, "1h", limit=24)
            if not klines:
                return 1000.0

            # Calculate average volume
            volumes = [float(k[5]) for k in klines]
            avg_volume = float(np.mean(volumes))

            # Cache result in Redis (1 hour expiration)
            await cache_service.set(cache_key, avg_volume, expire=3600)

            return avg_volume if avg_volume > 0 else 1000.0

        except Exception as e:
            logger.error(f"Failed to get average volume: {e}")
            return 1000.0


class CircuitBreaker(RiskLayer):
    """第5层：熔断机制"""

    def __init__(
        self,
        max_trades_per_minute: int = 10,
        max_trades_per_hour: int = 100,
        cooldown_period: int = 300  # 5分钟冷却期
    ):
        self.max_trades_per_minute = max_trades_per_minute
        self.max_trades_per_hour = max_trades_per_hour
        self.cooldown_period = cooldown_period
        self.is_circuit_broken = False
        self.circuit_break_time = None
        # 交易時戳（實例級別，避免跨實例共享造成狀態洩漏）
        self._trade_timestamps: list = []

    async def validate(self, order: Order) -> ValidationResult:
        """检查熔断状态"""
        try:
            # 检查是否在冷却期
            if self.is_circuit_broken:
                if self.circuit_break_time:
                    elapsed = (datetime.utcnow() - self.circuit_break_time).total_seconds()
                    if elapsed < self.cooldown_period:
                        return ValidationResult(
                            is_valid=False,
                            reason=f"Circuit breaker active. Cooldown remaining: {self.cooldown_period - elapsed:.0f}s",
                            risk_score=1.0
                        )
                    else:
                        # 重置熔断器
                        self.is_circuit_broken = False
                        self.circuit_break_time = None
                        logger.info("Circuit breaker reset")

            # 检查交易频率
            trades_last_minute = await self._get_trade_count(minutes=1)
            trades_last_hour = await self._get_trade_count(minutes=60)

            if trades_last_minute >= self.max_trades_per_minute:
                await self._trigger_circuit_breaker(
                    f"Exceeded max trades per minute: {trades_last_minute}"
                )
                return ValidationResult(
                    is_valid=False,
                    reason="Circuit breaker triggered: Too many trades per minute",
                    risk_score=1.0
                )

            if trades_last_hour >= self.max_trades_per_hour:
                await self._trigger_circuit_breaker(
                    f"Exceeded max trades per hour: {trades_last_hour}"
                )
                return ValidationResult(
                    is_valid=False,
                    reason="Circuit breaker triggered: Too many trades per hour",
                    risk_score=1.0
                )

            # 计算风险分数
            risk_score = max(
                trades_last_minute / self.max_trades_per_minute,
                trades_last_hour / self.max_trades_per_hour
            )

            warnings = []
            if trades_last_minute > self.max_trades_per_minute * 0.7:
                warnings.append(f"Trading frequency high ({trades_last_minute}/min)")

            return ValidationResult(
                is_valid=True,
                risk_score=risk_score,
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"Circuit breaker validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                reason=f"Validation error: {str(e)}",
                risk_score=1.0
            )

    async def _get_trade_count(self, minutes: int) -> int:
        """获取指定时间内的交易次数"""
        try:
            cutoff = datetime.utcnow() - timedelta(minutes=minutes)

            # 清理過期的記錄
            self._trade_timestamps = [
                ts for ts in self._trade_timestamps
                if ts > cutoff
            ]

            return len(self._trade_timestamps)

        except Exception as e:
            logger.error(f"Failed to get trade count: {e}")
            return 0

    def record_trade(self):
        """記錄一筆交易（供外部調用）"""
        self._trade_timestamps.append(datetime.utcnow())

        # 限制列表大小，只保留最近 1000 筆
        if len(self._trade_timestamps) > 1000:
            self._trade_timestamps = self._trade_timestamps[-1000:]

    async def _trigger_circuit_breaker(self, reason: str):
        """触发熔断器"""
        self.is_circuit_broken = True
        self.circuit_break_time = datetime.utcnow()
        logger.critical(f"CIRCUIT BREAKER TRIGGERED: {reason}")

        # TODO: 发送紧急通知


class RiskManagementSystem:
    """多层风险管理系统"""

    def __init__(self):
        self.risk_layers: List[RiskLayer] = [
            PositionSizeController(),      # 第1层：仓位控制
            DrawdownProtection(),           # 第2层：回撤保护
            CorrelationAnalyzer(),          # 第3层：相关性风险
            BlackSwanProtection(),          # 第4层：极端事件
            CircuitBreaker()                # 第5层：熔断机制
        ]

    async def validate_order(self, order: Order) -> ValidationResult:
        """多层风险验证"""
        all_warnings = []
        max_risk_score = 0.0

        for risk_layer in self.risk_layers:
            try:
                result = await risk_layer.validate(order)

                # 收集警告
                if result.warnings:
                    all_warnings.extend(result.warnings)

                # 更新最大风险分数
                max_risk_score = max(max_risk_score, result.risk_score)

                # 如果任何一层拒绝，立即返回
                if not result.is_valid:
                    logger.warning(
                        f"Order rejected by {risk_layer.__class__.__name__}: "
                        f"{result.reason}"
                    )
                    await self._send_alert(order, result)
                    return result

            except Exception as e:
                logger.error(f"Risk layer {risk_layer.__class__.__name__} failed: {e}")
                return ValidationResult(
                    is_valid=False,
                    reason=f"Risk validation error: {str(e)}",
                    risk_score=1.0
                )

        # 所有层都通过
        return ValidationResult(
            is_valid=True,
            risk_score=max_risk_score,
            warnings=all_warnings
        )

    async def _send_alert(self, order: Order, result: ValidationResult):
        """发送风险警报"""
        alert_message = f"""
        ⚠️ 风险警报

        订单被拒绝:
        - 交易对: {order.symbol}
        - 方向: {order.side}
        - 数量: {order.quantity}
        - 拒绝原因: {result.reason}
        - 风险分数: {result.risk_score:.2f}
        """

        logger.warning(alert_message)
        # TODO: 发送 Telegram 通知


# 全局风险管理系统实例
risk_management_system = RiskManagementSystem()
