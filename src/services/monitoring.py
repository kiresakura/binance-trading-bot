"""Prometheus 监控指标收集"""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from datetime import datetime
from typing import Dict, Optional
import psutil
import time
from src.logger import logger


class MetricsCollector:
    """Prometheus 指标收集器"""

    def __init__(self):
        # 创建独立的注册表
        self.registry = CollectorRegistry()

        # === 交易指标 ===
        self.trades_total = Counter(
            'trading_bot_trades_total',
            'Total number of trades executed',
            ['symbol', 'side', 'status'],
            registry=self.registry
        )

        self.trade_volume = Counter(
            'trading_bot_trade_volume_total',
            'Total trading volume in USDT',
            ['symbol'],
            registry=self.registry
        )

        self.trade_latency = Histogram(
            'trading_bot_trade_latency_seconds',
            'Trade execution latency in seconds',
            ['symbol', 'order_type'],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
            registry=self.registry
        )

        self.slippage = Histogram(
            'trading_bot_slippage_percent',
            'Trade slippage in percent',
            ['symbol'],
            buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
            registry=self.registry
        )

        # === 投资组合指标 ===
        self.current_balance = Gauge(
            'trading_bot_balance_usdt',
            'Current balance in USDT',
            registry=self.registry
        )

        self.unrealized_pnl = Gauge(
            'trading_bot_unrealized_pnl_usdt',
            'Unrealized profit/loss in USDT',
            registry=self.registry
        )

        # 已實現盈虧可正可負（虧損為負），必須用 Gauge；
        # Counter 只接受非負增量，虧損時 inc(負值) 會丟 ValueError。
        self.realized_pnl = Gauge(
            'trading_bot_realized_pnl_usdt',
            'Realized profit/loss in USDT (cumulative, can be negative)',
            registry=self.registry
        )

        self.position_count = Gauge(
            'trading_bot_open_positions',
            'Number of open positions',
            registry=self.registry
        )

        self.position_value = Gauge(
            'trading_bot_position_value_usdt',
            'Total position value in USDT',
            ['symbol'],
            registry=self.registry
        )

        # === 风险指标 ===
        self.current_drawdown = Gauge(
            'trading_bot_drawdown_percent',
            'Current drawdown in percent',
            registry=self.registry
        )

        self.sharpe_ratio = Gauge(
            'trading_bot_sharpe_ratio',
            'Current Sharpe ratio',
            registry=self.registry
        )

        self.risk_score = Gauge(
            'trading_bot_risk_score',
            'Current risk score (0-1)',
            registry=self.registry
        )

        self.risk_rejections = Counter(
            'trading_bot_risk_rejections_total',
            'Orders rejected by risk system',
            ['reason'],
            registry=self.registry
        )

        # === 系统指标 ===
        self.websocket_connected = Gauge(
            'trading_bot_websocket_connected',
            'WebSocket connection status (1=connected, 0=disconnected)',
            ['endpoint'],
            registry=self.registry
        )

        self.websocket_latency = Histogram(
            'trading_bot_websocket_latency_ms',
            'WebSocket message latency in milliseconds',
            ['endpoint'],
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
            registry=self.registry
        )

        self.api_requests_total = Counter(
            'trading_bot_api_requests_total',
            'Total API requests',
            ['endpoint', 'method', 'status_code'],
            registry=self.registry
        )

        self.api_latency = Histogram(
            'trading_bot_api_latency_seconds',
            'API request latency',
            ['endpoint', 'method'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )

        self.errors_total = Counter(
            'trading_bot_errors_total',
            'Total errors',
            ['component', 'error_type'],
            registry=self.registry
        )

        # === CPU 和内存指标 ===
        self.cpu_usage = Gauge(
            'trading_bot_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )

        self.memory_usage = Gauge(
            'trading_bot_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )

        self.memory_percent = Gauge(
            'trading_bot_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )

        # === 策略指标 ===
        self.strategy_signals = Counter(
            'trading_bot_strategy_signals_total',
            'Strategy signals generated',
            ['strategy', 'signal_type'],
            registry=self.registry
        )

        self.strategy_performance = Gauge(
            'trading_bot_strategy_return_percent',
            'Strategy return in percent',
            ['strategy'],
            registry=self.registry
        )

        # === 应用信息 ===
        self.app_info = Info(
            'trading_bot_app',
            'Trading bot application info',
            registry=self.registry
        )

        self.app_info.info({
            'version': '0.2.0',
            'environment': 'production'
        })

    # === 交易指标记录方法 ===

    def record_trade(
        self,
        symbol: str,
        side: str,
        status: str,
        volume: float,
        latency: float,
        slippage: float
    ):
        """记录交易"""
        self.trades_total.labels(symbol=symbol, side=side, status=status).inc()
        self.trade_volume.labels(symbol=symbol).inc(volume)
        self.trade_latency.labels(symbol=symbol, order_type='MARKET').observe(latency)

        if slippage > 0:
            self.slippage.labels(symbol=symbol).observe(slippage)

        logger.debug(f"Trade recorded: {symbol} {side} {volume}")

    def update_portfolio_metrics(
        self,
        balance: float,
        unrealized_pnl: float,
        position_count: int,
        drawdown: float,
        sharpe_ratio: float
    ):
        """更新投资组合指标"""
        self.current_balance.set(balance)
        self.unrealized_pnl.set(unrealized_pnl)
        self.position_count.set(position_count)
        self.current_drawdown.set(drawdown)
        self.sharpe_ratio.set(sharpe_ratio)

    def record_realized_pnl(self, pnl: float):
        """记录已实现盈亏"""
        self.realized_pnl.inc(pnl)

    def update_position_value(self, symbol: str, value: float):
        """更新持仓价值"""
        self.position_value.labels(symbol=symbol).set(value)

    # === 风险指标记录方法 ===

    def record_risk_rejection(self, reason: str):
        """记录风险拒绝"""
        self.risk_rejections.labels(reason=reason).inc()
        logger.warning(f"Risk rejection: {reason}")

    def update_risk_score(self, score: float):
        """更新风险分数"""
        self.risk_score.set(score)

    # === 系统指标记录方法 ===

    def record_websocket_status(self, endpoint: str, connected: bool):
        """记录 WebSocket 状态"""
        self.websocket_connected.labels(endpoint=endpoint).set(1 if connected else 0)

    def record_websocket_latency(self, endpoint: str, latency_ms: float):
        """记录 WebSocket 延迟"""
        self.websocket_latency.labels(endpoint=endpoint).observe(latency_ms)

    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency: float
    ):
        """记录 API 请求"""
        self.api_requests_total.labels(
            endpoint=endpoint,
            method=method,
            status_code=str(status_code)
        ).inc()

        self.api_latency.labels(
            endpoint=endpoint,
            method=method
        ).observe(latency)

    def record_error(self, component: str, error_type: str):
        """记录错误"""
        self.errors_total.labels(
            component=component,
            error_type=error_type
        ).inc()

    # === 系统资源监控 ===

    def update_system_metrics(self):
        """更新系统资源指标"""
        try:
            # CPU 使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_usage.set(cpu_percent)

            # 内存使用
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.used)
            self.memory_percent.set(memory.percent)

        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")

    # === 策略指标记录方法 ===

    def record_strategy_signal(self, strategy: str, signal_type: str):
        """记录策略信号"""
        self.strategy_signals.labels(
            strategy=strategy,
            signal_type=signal_type
        ).inc()

    def update_strategy_performance(self, strategy: str, return_pct: float):
        """更新策略性能"""
        self.strategy_performance.labels(strategy=strategy).set(return_pct)

    # === 导出指标 ===

    def get_metrics(self) -> bytes:
        """获取 Prometheus 格式的指标"""
        return generate_latest(self.registry)


class PerformanceTracker:
    """性能追踪器（用于延迟测量）"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.start_time = None

    def __enter__(self):
        """开始计时"""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """结束计时并记录"""
        if self.start_time:
            latency = time.time() - self.start_time
            # 延迟会在外部具体记录到相应的指标

    @staticmethod
    def measure_latency(func):
        """装饰器：测量函数执行时间"""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            latency = time.time() - start_time

            # 记录到日志
            logger.debug(f"{func.__name__} latency: {latency:.4f}s")

            return result, latency

        return wrapper


# 全局指标收集器实例
metrics_collector = MetricsCollector()
