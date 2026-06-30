"""
交易引擎

整合所有模組的主交易引擎
負責 24/7 自動交易
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import threading

from src.config import settings
from src.logger import logger
from src.services.runtime_settings import runtime_settings
from src.services.binance_client import binance_client
from src.services.risk_manager import risk_management_system, Order
from src.services.order_executor import smart_order_executor
from src.services.fault_tolerance import fault_tolerant_engine
from src.services.audit_logger import audit_logger
from src.services.monitoring import metrics_collector

from .data_engine import DataEngine, SubscriptionConfig
from .strategy_engine import StrategyEngine, AggregatedSignal, strategy_engine
from .learning_engine import LearningEngine, learning_engine

# WebSocket 廣播（延遲導入避免循環依賴）
def get_ws_manager():
    """獲取 WebSocket 管理器（延遲導入）"""
    try:
        from src.api.websocket_manager import ws_manager
        return ws_manager
    except ImportError:
        return None


# Telegram 通知（延遲導入避免循環依賴）
def get_telegram_notifier():
    """獲取 Telegram 通知器（延遲導入）"""
    try:
        from src.services.telegram_notifier import telegram_notifier
        return telegram_notifier
    except ImportError:
        return None


# 虛擬交易服務（延遲導入避免循環依賴）
def get_virtual_trading_service():
    """獲取虛擬交易服務（延遲導入）"""
    try:
        from src.services.virtual_trading_service import virtual_trading_service
        return virtual_trading_service
    except ImportError:
        return None


# Arena 服務（延遲導入避免循環依賴）
def get_arena_service():
    """獲取 Arena 服務（延遲導入）"""
    try:
        from src.services.arena_service import arena_service
        return arena_service
    except ImportError:
        return None


class TradingState(Enum):
    """交易狀態"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class TradingConfig:
    """交易配置"""
    # 交易對（與 config.py TRADING_PAIRS 保持一致）
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOGEUSDT"])
    intervals: List[str] = field(default_factory=lambda: ["1m", "5m", "15m"])

    # 交易參數
    min_signal_strength: float = 0.3
    min_confidence: float = 0.3
    max_position_per_symbol: float = 0.2  # 每個交易對最大倉位

    # 風險控制
    max_daily_trades: int = 50
    max_daily_loss: float = 0.05  # 5%
    cooldown_minutes: int = 2

    # 執行（從全局設定讀取）
    use_testnet: bool = field(default_factory=lambda: settings.BINANCE_TESTNET)
    paper_trading: bool = field(default_factory=lambda: settings.DRY_RUN)  # DRY_RUN=True 時為模擬交易


@dataclass
class Position:
    """持倉信息"""
    symbol: str
    side: str  # "LONG" or "SHORT"
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    entry_time: datetime
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None


class TradingEngine:
    """
    主交易引擎

    整合：
    - 數據引擎：實時市場數據
    - 策略引擎：信號生成
    - 學習引擎：模型訓練
    - 風險管理：訂單驗證
    - 訂單執行：智能執行
    - 容錯系統：故障恢復

    提供 24/7 自動交易能力
    """

    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config or TradingConfig()
        self.state = TradingState.STOPPED

        # 子引擎 - 使用全局實例以保持狀態一致性
        self.data_engine = DataEngine(
            SubscriptionConfig(
                symbols=self.config.symbols,
                intervals=self.config.intervals
            )
        )
        self.strategy_engine = strategy_engine  # 使用全局實例
        self.learning_engine = learning_engine  # 使用全局實例

        # 交易狀態
        self._positions: Dict[str, Position] = {}
        self._pending_orders: Dict[str, Dict] = {}
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._last_trade_time: Dict[str, datetime] = {}

        # 訂單冪等性：記錄最近處理過的訂單，防止重複下單
        self._recent_orders: Dict[str, datetime] = {}  # symbol -> last_order_time
        self._order_cooldown_seconds: int = 30  # 同一 symbol 最少間隔秒數

        # 賬戶信息
        self._account_balance: Decimal = Decimal("0")
        self._available_balance: Decimal = Decimal("0")

        # 控制
        self._running = False
        self._main_loop_task = None

        # 線程安全鎖
        self._position_lock = asyncio.Lock()
        self._order_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._balance_lock = asyncio.Lock()

        logger.info(f"TradingEngine initialized with symbols: {self.config.symbols}")

    # ==================== 生命週期 ====================

    async def start(self):
        """啟動交易引擎"""
        if self._running:
            logger.warning("TradingEngine already running")
            return

        # 檢查全局交易開關（使用運行時設定）
        if not runtime_settings.trading_enabled:
            logger.warning("Trading is disabled (trading_enabled=False). Engine will start in monitoring mode only.")

        self.state = TradingState.STARTING
        logger.info(f"Starting TradingEngine... (trading_enabled={runtime_settings.trading_enabled}, dry_run={runtime_settings.dry_run})")

        try:
            # 初始化子系統
            await self._initialize_subsystems()

            # 獲取賬戶信息
            await self._update_account_info()

            # 🆕 同步 Binance 持倉（防止孤兒持倉）
            await self._sync_positions_from_binance()

            # 註冊數據回調
            self.data_engine.on_kline(self._on_kline)
            self.data_engine.on_signal(self._on_signal)
            self.data_engine.on_error(self._on_error)

            # 啟動數據引擎
            await self.data_engine.start()

            # 啟動學習引擎
            await self.learning_engine.start()

            # 啟動主循環
            self._running = True
            self._main_loop_task = asyncio.create_task(self._main_loop())

            self.state = TradingState.RUNNING
            logger.info("TradingEngine started successfully")

            # 記錄審計日誌
            await audit_logger.log_system_event(
                event_type="SYSTEM",
                action="TRADING_ENGINE_START",
                details={
                    'symbols': self.config.symbols,
                    'paper_trading': self.config.paper_trading
                }
            )

            # 🆕 發送 Telegram 啟動通知
            await self._send_startup_notification()

        except Exception as e:
            self.state = TradingState.ERROR
            logger.error(f"Failed to start TradingEngine: {e}")
            raise

    async def stop(self):
        """停止交易引擎"""
        logger.info("Stopping TradingEngine...")

        self._running = False
        self.state = TradingState.STOPPED

        # 取消主循環
        if self._main_loop_task:
            self._main_loop_task.cancel()

        # 停止數據引擎
        await self.data_engine.stop()

        # 關閉所有持倉（如果不是模擬交易）
        if not self.config.paper_trading:
            await self._close_all_positions("Engine stopped")

        # 記錄審計日誌
        await audit_logger.log_system_event(
            event_type="SYSTEM",
            action="TRADING_ENGINE_STOP",
            details={'daily_pnl': self._daily_pnl}
        )

        logger.info("TradingEngine stopped")

    async def pause(self):
        """暫停交易（保持數據流）"""
        self.state = TradingState.PAUSED
        logger.info("TradingEngine paused")

    async def resume(self):
        """恢復交易"""
        if self.state == TradingState.PAUSED:
            self.state = TradingState.RUNNING
            logger.info("TradingEngine resumed")

    async def _send_startup_notification(self):
        """發送啟動通知到 Telegram"""
        try:
            telegram = get_telegram_notifier()
            if not telegram or not telegram.enabled:
                logger.info("Telegram notifier not available, skipping startup notification")
                return

            # 收集策略信息
            base_strategies = ["trend_following", "mean_reversion", "momentum"]
            community_strategies = [
                "bb_rsi", "macd_cross", "golden_cross", "scalping",
                "breakout", "adx_momentum", "ichimoku", "supertrend",
                "range_filter", "multi_timeframe"
            ]

            # 獲取餘額
            balance = float(self._available_balance) if self._available_balance else 0.0

            # 發送策略載入通知
            await telegram.notify_strategies_loaded(
                base_strategies=base_strategies,
                community_strategies=community_strategies,
                symbols=self.config.symbols,
                balance=balance
            )

            logger.info("Startup notification sent to Telegram")

        except Exception as e:
            logger.warning(f"Failed to send startup notification: {e}")

    async def _initialize_subsystems(self):
        """初始化子系統"""
        # 初始化容錯引擎
        try:
            await fault_tolerant_engine.init()
        except Exception as e:
            logger.warning(f"Fault tolerance not available: {e}")

        # 審計日誌已在模組導入時初始化，無需 start()
        logger.info("Audit logger ready")

        # 更新監控指標
        try:
            metrics_collector.update_system_metrics()
        except Exception as e:
            logger.warning(f"Metrics collector error: {e}")

        # 註冊交易策略
        self._register_strategies()

    def _register_strategies(self):
        """註冊交易策略"""
        from src.strategies import (
            TrendFollowingStrategy,
            MeanReversionStrategy,
            MomentumStrategy
        )
        from src.strategies.community import COMMUNITY_STRATEGIES

        # ==================== 基礎策略（rule_based） ====================
        
        # 趨勢跟蹤策略
        trend_strategy = TrendFollowingStrategy()
        self.strategy_engine.register_strategy(
            trend_strategy,
            weight=1.0,
            market_states=['trending_up', 'trending_down']
        )

        # 均值回歸策略
        mean_rev_strategy = MeanReversionStrategy()
        self.strategy_engine.register_strategy(
            mean_rev_strategy,
            weight=0.8,
            market_states=['ranging']
        )

        # 動量策略
        momentum_strategy = MomentumStrategy()
        self.strategy_engine.register_strategy(
            momentum_strategy,
            weight=0.9,
            market_states=['trending_up', 'trending_down', 'breakout']
        )

        # ==================== 社區策略（community） ====================
        
        # 策略分類對應的市場狀態
        category_market_states = {
            'trend_following': ['trending_up', 'trending_down'],
            'mean_reversion': ['ranging'],
            'momentum': ['trending_up', 'trending_down', 'breakout'],
            'breakout': ['breakout', 'trending_up'],
            'scalping': ['ranging', 'trending_up', 'trending_down'],
            'volatility': ['high_volatility', 'ranging'],
            'swing': ['trending_up', 'trending_down', 'ranging'],
        }
        
        # 風險等級對應的權重
        risk_to_weight = {
            1: 1.0,   # 極低風險 -> 高權重
            2: 0.9,   # 低風險
            3: 0.7,   # 中等風險
            4: 0.5,   # 高風險 -> 低權重
            5: 0.3,   # 極高風險
        }
        
        for strategy_meta in COMMUNITY_STRATEGIES:
            try:
                strategy_class = strategy_meta['class']
                strategy_name = strategy_meta['name']
                category = strategy_meta.get('category', 'trend_following')
                risk_level = strategy_meta.get('risk_level', 3)
                
                # 實例化策略
                strategy_instance = strategy_class()
                
                # 根據風險等級決定權重
                weight = risk_to_weight.get(risk_level, 0.7)
                
                # 根據分類決定適用的市場狀態
                market_states = category_market_states.get(category, ['ranging'])
                
                # 註冊策略
                self.strategy_engine.register_strategy(
                    strategy_instance,
                    weight=weight,
                    market_states=market_states
                )
                
                logger.info(f"✅ Registered community strategy: {strategy_name} "
                           f"(weight={weight}, category={category})")
                
            except Exception as e:
                logger.error(f"❌ Failed to register strategy {strategy_meta.get('name', 'unknown')}: {e}")

        logger.info(f"🎯 Total registered strategies: {len(self.strategy_engine._strategies)} - "
                    f"{list(self.strategy_engine._strategies.keys())}")

    # ==================== 主循環 ====================

    async def _main_loop(self):
        """主交易循環"""
        logger.info("Main trading loop started")

        while self._running:
            try:
                if self.state != TradingState.RUNNING:
                    await asyncio.sleep(1)
                    continue

                # 定期任務
                await self._periodic_tasks()

                # 短暫休眠
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(5)

        logger.info("Main trading loop ended")

    async def _periodic_tasks(self):
        """定期任務"""
        now = datetime.utcnow()

        # 每分鐘更新賬戶信息
        if now.second == 0:
            await self._update_account_info()

        # 每分鐘檢查持倉
        if now.second == 30:
            await self._check_positions()
        
        # 每天 UTC 00:00 發送績效報告
        if now.hour == 0 and now.minute == 0 and now.second == 0:
            await self._send_daily_performance()

        # 每小時重置計數器
        if now.minute == 0 and now.second == 0:
            self._daily_trades = 0  # 簡化：每小時重置

    # ==================== 數據回調 ====================

    async def _on_kline(self, data: Dict[str, Any]):
        """K線數據回調"""
        symbol = data.get('symbol')
        price = data.get('price')

        # 更新持倉當前價格
        if symbol in self._positions:
            self._positions[symbol].current_price = Decimal(str(price))
            self._update_position_pnl(symbol)

    async def _on_signal(self, signal: Dict[str, Any]):
        """信號回調"""
        symbol = signal.get('symbol')
        direction = signal.get('direction', 0)
        strength = signal.get('strength', 0)
        
        # Debug: 記錄所有收到的信號
        logger.info(f"📊 Signal received: {symbol} dir={direction} str={strength:.3f}")
        
        # 獲取當前價格用於虛擬交易
        current_price = signal.get('price', 0)
        if current_price == 0 and symbol in self._positions:
            current_price = float(self._positions[symbol].current_price)
        
        # 為所有 SHADOW_RUNNING 模型執行虛擬交易（無論信號強度）
        if abs(strength) >= 0.3:  # 降低虛擬交易門檻
            asyncio.create_task(
                self._execute_virtual_trades_for_all_models(
                    symbol, direction, strength, current_price, signal
                )
            )
        
        if self.state != TradingState.RUNNING:
            logger.debug(f"Signal ignored: state={self.state}")
            return

        # 檢查信號強度
        if abs(strength) < self.config.min_signal_strength:
            logger.debug(f"Signal too weak: {strength:.3f} < {self.config.min_signal_strength}")
            return

        # 處理信號
        await self._process_signal(symbol, direction, strength, signal)

    async def _on_error(self, error: Dict[str, Any]):
        """錯誤回調"""
        logger.error(f"Data engine error: {error}")

    async def _execute_virtual_trades_for_all_models(
        self,
        symbol: str,
        direction: int,
        strength: float,
        current_price: float,
        signal: Dict[str, Any]
    ):
        """為所有 SHADOW_RUNNING 模型執行虛擬交易"""
        try:
            arena_service = get_arena_service()
            virtual_service = get_virtual_trading_service()
            
            if not arena_service or not virtual_service:
                return
            
            # 初始化服務
            await arena_service.initialize()
            await virtual_service.initialize()
            
            # 獲取所有 SHADOW_RUNNING 模型
            models = await arena_service.list_models(status="SHADOW_RUNNING")
            
            if not models:
                return
            
            price = Decimal(str(current_price)) if current_price > 0 else None
            if not price:
                # 嘗試從 Binance 獲取價格
                try:
                    ticker = await binance_client.get_ticker_price(symbol)
                    price = Decimal(str(ticker.get('price', 0)))
                except:
                    return
            
            if price <= 0:
                return
            
            # 為每個模型執行虛擬交易
            for model in models:
                try:
                    model_version = model.version
                    
                    # 獲取或創建虛擬帳戶
                    account = await virtual_service.get_or_create_account(model_version)
                    if not account:
                        continue
                    
                    # 獲取當前虛擬持倉
                    position = await virtual_service.get_position(model_version, symbol)
                    
                    # 根據信號方向決定操作
                    if direction > 0:  # 做多信號
                        if not position:
                            # 沒有持倉，買入
                            result = await virtual_service.execute_virtual_buy(
                                model_version=model_version,
                                symbol=symbol,
                                price=price,
                                signal_strength=abs(strength),
                                signal_confidence=signal.get('confidence', 0.5),
                                position_ratio=0.15  # 使用 15% 資金
                            )
                            if result.get('success'):
                                logger.debug(f"[Virtual] {model_version} BUY {symbol}")
                                
                    elif direction < 0:  # 做空信號（平倉）
                        if position:
                            # 有持倉，賣出
                            result = await virtual_service.execute_virtual_sell(
                                model_version=model_version,
                                symbol=symbol,
                                price=price,
                                signal_strength=abs(strength),
                                signal_confidence=signal.get('confidence', 0.5)
                            )
                            if result.get('success'):
                                pnl = result.get('realized_pnl', 0)
                                emoji = "🟢" if pnl > 0 else "🔴"
                                logger.debug(f"[Virtual] {model_version} SELL {symbol} | {emoji} PnL: {pnl:.4f}")
                    
                    else:  # 中性信號，也可以考慮平倉
                        pass
                        
                except Exception as e:
                    logger.warning(f"Virtual trade failed for {model_version}: {e}")
                    continue
            
            # 更新所有持倉的價格
            await virtual_service.update_positions_price(symbol, price)
            
        except Exception as e:
            logger.warning(f"Failed to execute virtual trades: {e}")

    # ==================== 信號處理 ====================

    async def _process_signal(
        self,
        symbol: str,
        direction: int,
        strength: float,
        signal: Dict[str, Any]
    ):
        """處理交易信號"""
        # 發送信號通知 (可選，強度 > 0.5 才發)
        if abs(strength) >= 0.5:
            telegram = get_telegram_notifier()
            if telegram and telegram.enabled:
                try:
                    await telegram.notify_signal(
                        symbol=symbol,
                        direction=direction,
                        strength=abs(strength),
                        confidence=signal.get('confidence', 0.5),
                        strategy=signal.get('strategy', 'ensemble'),
                        reason=signal.get('reason', '')
                    )
                except Exception as e:
                    logger.warning(f"Failed to send signal notification: {e}")
        
        # 檢查全局交易開關（使用運行時設定）
        if not runtime_settings.trading_enabled:
            logger.debug(f"Signal ignored (trading_enabled=False): {symbol} direction={direction}")
            return

        # 檢查模擬模式
        if runtime_settings.dry_run:
            logger.info(f"[DRY_RUN] Would process signal: {symbol} direction={direction} strength={strength:.2f}")
            # 記錄但不執行，繼續下面的邏輯以記錄模擬結果

        # 檢查冷卻期
        if not self._check_cooldown(symbol):
            return

        # 檢查每日限制
        if self._daily_trades >= self.config.max_daily_trades:
            logger.warning("Daily trade limit reached")
            return

        # 檢查每日虧損
        if self._daily_pnl <= -self.config.max_daily_loss:
            logger.warning("Daily loss limit reached")
            return

        # 獲取當前持倉
        current_position = self._positions.get(symbol)

        # 決定操作
        if direction > 0:  # 做多信號
            if current_position is None:
                await self._open_position(symbol, "LONG", strength, signal)
            elif current_position.side == "SHORT":
                await self._close_position(symbol, "Signal reversal")
                await self._open_position(symbol, "LONG", strength, signal)

        elif direction < 0:  # 做空信號
            if current_position is None:
                await self._open_position(symbol, "SHORT", strength, signal)
            elif current_position.side == "LONG":
                await self._close_position(symbol, "Signal reversal")
                await self._open_position(symbol, "SHORT", strength, signal)

        else:  # 平倉信號
            if current_position is not None:
                await self._close_position(symbol, "Neutral signal")

    def _check_cooldown(self, symbol: str) -> bool:
        """檢查冷卻期"""
        last_trade = self._last_trade_time.get(symbol)
        if last_trade is None:
            return True

        elapsed = (datetime.utcnow() - last_trade).total_seconds()
        return elapsed >= self.config.cooldown_minutes * 60

    # ==================== 交易執行 ====================

    async def _open_position(
        self,
        symbol: str,
        side: str,
        strength: float,
        signal: Dict[str, Any]
    ):
        """開倉（線程安全）"""
        async with self._position_lock:
            try:
                # 🆕 冪等性檢查：防止短時間內重複下單
                now = datetime.utcnow()
                last_order_time = self._recent_orders.get(symbol)
                if last_order_time:
                    seconds_since_last = (now - last_order_time).total_seconds()
                    if seconds_since_last < self._order_cooldown_seconds:
                        logger.warning(
                            f"Order cooldown active for {symbol}: "
                            f"{seconds_since_last:.1f}s < {self._order_cooldown_seconds}s, skipping"
                        )
                        return

                # 再次檢查是否已有持倉（防止競爭條件）
                if symbol in self._positions:
                    logger.warning(f"Position for {symbol} already exists, skipping open")
                    return

                # 🆕 檢查最大持倉數量
                max_positions = getattr(settings, 'MAX_CONCURRENT_POSITIONS', 2)
                if len(self._positions) >= max_positions:
                    logger.warning(f"Max concurrent positions ({max_positions}) reached, skipping {symbol}")
                    return

                # 計算倉位大小
                async with self._balance_lock:
                    available = float(self._available_balance)

                # 🆕 最低訂單金額檢查
                min_order_value = getattr(settings, 'MIN_ORDER_VALUE', 6.0)
                position_value = available * self.config.max_position_per_symbol * strength
                
                if position_value < min_order_value:
                    # 嘗試使用最低金額
                    if available >= min_order_value * 1.2:
                        position_value = min_order_value
                        logger.info(f"Position value adjusted to minimum ${min_order_value} for {symbol}")
                    else:
                        logger.warning(f"Insufficient balance for {symbol}: ${available:.2f} < ${min_order_value * 1.2:.2f} required")
                        return

                current_price = self.data_engine.get_latest_price(symbol)

                if current_price is None or current_price <= 0:
                    logger.warning(f"No price data for {symbol}")
                    return

                quantity = Decimal(str(position_value / current_price))

                # 創建訂單
                order = Order(
                    symbol=symbol,
                    side="BUY" if side == "LONG" else "SELL",
                    quantity=quantity,
                    price=Decimal(str(current_price)),
                    order_type="MARKET"
                )

                # 風險驗證
                validation = await risk_management_system.validate_order(order)

                if not validation.is_valid:
                    logger.warning(f"Order rejected by risk manager: {validation.reason}")
                    await audit_logger.log_risk_rejection(order, validation.reason, validation.risk_score)
                    return

                # 執行訂單
                if self.config.paper_trading:
                    # 模擬交易
                    result = await self._paper_trade(order)
                    order_success = result.get('success', False)
                    order_id = result.get('order_id', 'paper')
                else:
                    # 真實交易
                    results = await smart_order_executor.execute_order(order)
                    # execute_order 回傳 List[ExecutionResult]
                    result = results[0] if results else None
                    order_success = result.success if result else False
                    order_id = result.order_id if result else None

                if order_success:
                    # 🆕 記錄訂單時間（冪等性）
                    self._recent_orders[symbol] = datetime.utcnow()

                    # 記錄持倉
                    self._positions[symbol] = Position(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        entry_price=Decimal(str(current_price)),
                        current_price=Decimal(str(current_price)),
                        unrealized_pnl=Decimal("0"),
                        entry_time=datetime.utcnow(),
                        stop_loss=Decimal(str(current_price * (0.98 if side == "LONG" else 1.02))),
                        take_profit=Decimal(str(current_price * (1.04 if side == "LONG" else 0.96)))
                    )

                    async with self._state_lock:
                        self._daily_trades += 1
                    self._last_trade_time[symbol] = datetime.utcnow()

                    logger.info(f"Opened {side} position for {symbol}: {quantity} @ {current_price}")

                    # 記錄審計
                    await audit_logger.log_order_created(
                        order_id=order_id or 'paper',
                        order_data={'symbol': symbol, 'side': side, 'quantity': str(quantity)},
                        risk_score=validation.risk_score
                    )

                    # 廣播 WebSocket 通知
                    ws = get_ws_manager()
                    if ws:
                        from src.api.websocket_manager import MessageType
                        await ws.broadcast_order_event(
                            event_type=MessageType.ORDER_FILLED,
                            symbol=symbol,
                            side="BUY" if side == "LONG" else "SELL",
                            quantity=float(quantity),
                            price=float(current_price),
                            order_id=order_id or 'paper',
                            status="FILLED"
                        )
                        await ws.broadcast_position_update(
                            symbol=symbol,
                            side=side.lower(),
                            size=float(quantity),
                            entry_price=float(current_price),
                            current_price=float(current_price),
                            unrealized_pnl=0.0,
                            pnl_percent=0.0
                        )

                    # 發送 Telegram 通知
                    telegram = get_telegram_notifier()
                    if telegram and telegram.enabled:
                        await telegram.notify_order_placed(
                            symbol=symbol,
                            side="BUY" if side == "LONG" else "SELL",
                            quantity=float(quantity),
                            price=float(current_price),
                            order_id=order_id or 'paper'
                        )

            except Exception as e:
                logger.error(f"Failed to open position: {e}")

    async def _close_position(self, symbol: str, reason: str):
        """平倉（線程安全）"""
        async with self._position_lock:
            position = self._positions.get(symbol)
            if position is None:
                return

            try:
                current_price = self.data_engine.get_latest_price(symbol) or float(position.current_price)

                # 創建平倉訂單
                order = Order(
                    symbol=symbol,
                    side="SELL" if position.side == "LONG" else "BUY",
                    quantity=position.quantity,
                    price=Decimal(str(current_price)),
                    order_type="MARKET"
                )

                # 執行
                if self.config.paper_trading:
                    # 模擬交易
                    result = await self._paper_trade(order)
                    order_success = result.get('success', False)
                    order_id = result.get('order_id', 'paper')
                    executed_price = result.get('executed_price', current_price)
                else:
                    # 真實交易
                    results = await smart_order_executor.execute_order(order)
                    # execute_order 回傳 List[ExecutionResult]
                    result = results[0] if results else None
                    order_success = result.success if result else False
                    order_id = result.order_id if result else None
                    executed_price = result.executed_price if result else current_price

                if order_success:
                    # 計算 PnL
                    pnl_pct = self._calculate_pnl_pct(position, current_price)

                    async with self._state_lock:
                        self._daily_pnl += pnl_pct

                    # 記錄到學習引擎
                    self.learning_engine.record_trade_result(pnl_pct)

                    # 移除持倉
                    del self._positions[symbol]

                    logger.info(f"Closed position for {symbol}: PnL {pnl_pct:.2%}, Reason: {reason}")

                    # 記錄審計
                    await audit_logger.log_trade_execution(
                        trade_id=f"close_{symbol}_{datetime.utcnow().timestamp()}",
                        trade_data={
                            'symbol': symbol,
                            'pnl_pct': pnl_pct,
                            'reason': reason
                        }
                    )

                    # WebSocket 廣播平倉事件
                    ws = get_ws_manager()
                    if ws:
                        from src.api.websocket_manager import MessageType
                        await ws.broadcast_order_event(
                            event_type=MessageType.ORDER_FILLED,
                            symbol=symbol,
                            side=order.side,
                            quantity=float(order.quantity),
                            price=float(executed_price or current_price),
                            order_id=order_id or '',
                            pnl_pct=pnl_pct,
                            reason=reason
                        )
                        # 廣播持倉更新（已平倉）
                        await ws.broadcast_position_update(
                            symbol=symbol,
                            side="CLOSED",
                            quantity=0,
                            entry_price=0,
                            current_price=current_price,
                            unrealized_pnl=0,
                            pnl_pct=pnl_pct
                        )

                    # 發送 Telegram 通知
                    telegram = get_telegram_notifier()
                    if telegram and telegram.enabled:
                        # 計算實際盈虧金額
                        entry = float(position.entry_price)
                        pnl_amount = (current_price - entry) * float(position.quantity)
                        if position.side == "SHORT":
                            pnl_amount = -pnl_amount

                        await telegram.notify_order_filled(
                            symbol=symbol,
                            side=order.side,
                            quantity=float(order.quantity),
                            price=float(result.get('executed_price', current_price)),
                            order_id=result.get('order_id', ''),
                            pnl=pnl_amount,
                            pnl_pct=pnl_pct
                        )

            except Exception as e:
                logger.error(f"Failed to close position: {e}")


    async def _send_daily_performance(self):
        """發送每日績效報告"""
        try:
            telegram = get_telegram_notifier()
            if not telegram or not telegram.enabled:
                return
            
            # 收集績效數據
            total_pnl = self._daily_pnl
            trade_count = self._daily_trades
            
            # 計算勝率（從學習引擎獲取）
            win_rate = 0.0
            if hasattr(self, 'learning_engine') and self.learning_engine:
                metrics = self.learning_engine.get_performance_metrics()
                win_rate = metrics.get('win_rate', 0.0)
            
            # 獲取活躍策略
            active_strategies = list(self.strategy_engine.strategies.keys()) if self.strategy_engine else []
            
            await telegram.notify_daily_performance(
                total_pnl=total_pnl * 100,  # 轉為百分比顯示用的金額
                total_pnl_pct=total_pnl,
                today_pnl=total_pnl * 100,
                trade_count=trade_count,
                win_rate=win_rate,
                max_drawdown=0.0,  # TODO: 實現回撤追蹤
                sharpe_ratio=0.0,  # TODO: 實現夏普比率
                active_strategies=active_strategies
            )
            logger.info("Daily performance report sent")
        except Exception as e:
            logger.error(f"Failed to send daily performance: {e}")

    async def _close_all_positions(self, reason: str):
        """關閉所有持倉"""
        for symbol in list(self._positions.keys()):
            await self._close_position(symbol, reason)

    async def _paper_trade(self, order: Order) -> Dict[str, Any]:
        """模擬交易"""
        # 模擬延遲
        await asyncio.sleep(0.1)

        return {
            'success': True,
            'order_id': f"paper_{datetime.utcnow().timestamp()}",
            'executed_price': float(order.price),
            'executed_quantity': float(order.quantity)
        }

    # ==================== 持倉管理 ====================

    async def _check_positions(self):
        """檢查持倉（止損止盈）"""
        telegram = get_telegram_notifier()
        
        for symbol, position in list(self._positions.items()):
            current_price = self.data_engine.get_latest_price(symbol)
            if current_price is None:
                continue

            # 更新當前價格
            position.current_price = Decimal(str(current_price))
            self._update_position_pnl(symbol)
            
            # 計算盈虧百分比（用於止損止盈判斷）
            pnl_pct = self._calculate_pnl_pct(position, current_price)

            # 🔕 已停用持倉更新通知（太吵了）
            # 只在買入和賣出時發送通知，不再發送持倉更新
            # if telegram and telegram.enabled and abs(pnl_pct) >= 0.01:
            #     try:
            #         await telegram.notify_position_update(...)
            #     except Exception as e:
            #         logger.warning(f"Failed to send position update: {e}")

            # 檢查止損
            if position.stop_loss:
                if position.side == "LONG" and current_price <= float(position.stop_loss):
                    await self._close_position(symbol, "Stop loss triggered")
                    continue
                elif position.side == "SHORT" and current_price >= float(position.stop_loss):
                    await self._close_position(symbol, "Stop loss triggered")
                    continue

            # 檢查止盈
            if position.take_profit:
                if position.side == "LONG" and current_price >= float(position.take_profit):
                    await self._close_position(symbol, "Take profit triggered")
                    continue
                elif position.side == "SHORT" and current_price <= float(position.take_profit):
                    await self._close_position(symbol, "Take profit triggered")
                    continue

    def _update_position_pnl(self, symbol: str):
        """更新持倉盈虧"""
        position = self._positions.get(symbol)
        if position is None:
            return

        if position.side == "LONG":
            pnl = (position.current_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - position.current_price) * position.quantity

        position.unrealized_pnl = pnl

    def _calculate_pnl_pct(self, position: Position, exit_price: float) -> float:
        """計算盈虧百分比"""
        entry = float(position.entry_price)
        if position.side == "LONG":
            return (exit_price - entry) / entry
        else:
            return (entry - exit_price) / entry

    # ==================== 賬戶信息 ====================

    async def _update_account_info(self):
        """更新賬戶信息"""
        try:
            if self.config.paper_trading:
                # 模擬賬戶
                if self._account_balance == 0:
                    self._account_balance = Decimal("10000")
                    self._available_balance = Decimal("10000")
            else:
                # 真實賬戶
                account = await binance_client.get_account_info()
                if account:
                    # 找到 USDT 餘額
                    for balance in account.get('balances', []):
                        if balance['asset'] == 'USDT':
                            self._account_balance = Decimal(balance['free']) + Decimal(balance['locked'])
                            self._available_balance = Decimal(balance['free'])
                            break

        except Exception as e:
            logger.error(f"Failed to update account info: {e}")

    async def _sync_positions_from_binance(self):
        """
        🆕 從 Binance 同步實際持倉

        啟動時檢查 Binance 帳戶餘額，找出「孤兒持倉」
        （Binance 有幣但 bot 沒追蹤的情況）
        """
        if self.config.paper_trading:
            logger.info("Paper trading mode - skipping Binance position sync")
            return

        try:
            logger.info("Syncing positions from Binance...")

            # 獲取所有非零餘額
            balances = await binance_client.get_balances()

            # 我們追蹤的交易對基礎資產
            tracked_base_assets = set()
            for symbol in self.config.symbols:
                # BTCUSDT -> BTC, ETHUSDT -> ETH
                if symbol.endswith('USDT'):
                    tracked_base_assets.add(symbol[:-4])

            orphan_positions = []
            synced_positions = []

            for balance in balances:
                asset = balance['asset']
                total = Decimal(balance['total'])

                # 跳過穩定幣和小額餘額
                if asset in ['USDT', 'BUSD', 'USDC', 'BNB'] and asset not in tracked_base_assets:
                    continue
                if total < Decimal('0.0001'):
                    continue

                # 檢查是否是我們追蹤的資產
                symbol = f"{asset}USDT"

                if asset in tracked_base_assets:
                    # 這是我們追蹤的交易對
                    if symbol not in self._positions:
                        # Binance 有但 bot 沒追蹤 -> 嘗試同步
                        try:
                            # 獲取當前價格
                            ticker = await binance_client.get_ticker_price(symbol)
                            current_price = Decimal(ticker.get('price', '0'))

                            if current_price > 0:
                                # 創建持倉記錄（假設是 LONG）
                                self._positions[symbol] = Position(
                                    symbol=symbol,
                                    side="LONG",  # 假設是做多
                                    quantity=total,
                                    entry_price=current_price,  # 不知道真正入場價
                                    current_price=current_price,
                                    unrealized_pnl=Decimal("0"),
                                    entry_time=datetime.utcnow(),
                                    stop_loss=current_price * Decimal("0.95"),
                                    take_profit=current_price * Decimal("1.05")
                                )
                                synced_positions.append({
                                    'symbol': symbol,
                                    'quantity': str(total),
                                    'price': str(current_price)
                                })
                                logger.warning(
                                    f"⚠️ Synced orphan position: {symbol} = {total} @ {current_price}"
                                )
                        except Exception as e:
                            logger.error(f"Failed to sync position for {symbol}: {e}")
                else:
                    # 不在追蹤列表中的資產
                    orphan_positions.append({
                        'asset': asset,
                        'quantity': str(total)
                    })

            # 報告結果
            if synced_positions:
                logger.warning(f"🔄 Synced {len(synced_positions)} orphan positions: {synced_positions}")

                # 發送 Telegram 通知
                telegram = get_telegram_notifier()
                if telegram and telegram.enabled:
                    await telegram.notify_system_event(
                        event="Orphan Positions Synced",
                        status="warning",
                        details=f"Synced {len(synced_positions)} positions from Binance: {synced_positions}"
                    )

            if orphan_positions:
                logger.warning(
                    f"⚠️ Found {len(orphan_positions)} untracked assets (not in trading pairs): {orphan_positions}"
                )
                logger.warning(
                    "Consider selling these manually or adding them to TRADING_PAIRS"
                )

            if not synced_positions and not orphan_positions:
                logger.info("✅ No orphan positions found - Binance and bot are in sync")

        except Exception as e:
            logger.error(f"Failed to sync positions from Binance: {e}")
            # 不阻止引擎啟動

    async def sync_positions_manual(self) -> Dict[str, Any]:
        """
        🆕 手動同步持倉（雙向同步）

        比較 Binance 實際餘額與 bot 追蹤的持倉：
        1. 添加 Binance 有但 bot 沒追蹤的持倉
        2. 移除 bot 有但 Binance 已沒有的持倉
        3. 更新數量不匹配的持倉

        Returns:
            Dict with synced, removed, updated lists
        """
        if self.config.paper_trading:
            return {
                "synced": [],
                "removed": [],
                "updated": [],
                "message": "Paper trading mode - no sync needed"
            }

        synced = []
        removed = []
        updated = []

        try:
            logger.info("🔄 Starting manual position sync...")

            # 獲取 Binance 所有非零餘額
            balances = await binance_client.get_balances()

            # 建立資產餘額映射 {asset: total_quantity}
            binance_balances = {}
            for balance in balances:
                asset = balance['asset']
                total = Decimal(balance['total'])
                if total > Decimal('0.00001'):  # 忽略灰塵餘額
                    binance_balances[asset] = total

            # 我們追蹤的交易對基礎資產
            tracked_base_assets = set()
            for symbol in self.config.symbols:
                if symbol.endswith('USDT'):
                    tracked_base_assets.add(symbol[:-4])

            async with self._position_lock:
                # 1. 檢查 bot 持倉是否還存在於 Binance
                positions_to_remove = []
                for symbol, position in self._positions.items():
                    base_asset = symbol[:-4] if symbol.endswith('USDT') else symbol

                    binance_qty = binance_balances.get(base_asset, Decimal('0'))

                    if binance_qty < Decimal('0.00001'):
                        # Binance 沒有這個資產了，標記移除
                        positions_to_remove.append(symbol)
                        removed.append({
                            'symbol': symbol,
                            'bot_quantity': str(position.quantity),
                            'reason': 'No balance on Binance'
                        })
                        logger.warning(f"⚠️ Position {symbol} no longer exists on Binance, removing...")
                    elif abs(binance_qty - position.quantity) > Decimal('0.00001'):
                        # 數量不匹配，更新
                        old_qty = position.quantity
                        position.quantity = binance_qty
                        updated.append({
                            'symbol': symbol,
                            'old_quantity': str(old_qty),
                            'new_quantity': str(binance_qty)
                        })
                        logger.warning(f"⚠️ Updated {symbol} quantity: {old_qty} -> {binance_qty}")

                # 移除已經不存在的持倉
                for symbol in positions_to_remove:
                    del self._positions[symbol]

                # 2. 檢查 Binance 有但 bot 沒追蹤的資產
                for asset, qty in binance_balances.items():
                    if asset in ['USDT', 'BUSD', 'USDC']:
                        continue  # 跳過穩定幣

                    symbol = f"{asset}USDT"

                    if asset in tracked_base_assets and symbol not in self._positions:
                        # 這是我們追蹤的交易對，但 bot 沒有記錄
                        try:
                            ticker = await binance_client.get_ticker_price(symbol)
                            current_price = Decimal(ticker.get('price', '0'))

                            if current_price > 0:
                                self._positions[symbol] = Position(
                                    symbol=symbol,
                                    side="LONG",
                                    quantity=qty,
                                    entry_price=current_price,  # 不知道真正入場價
                                    current_price=current_price,
                                    unrealized_pnl=Decimal("0"),
                                    entry_time=datetime.utcnow(),
                                    stop_loss=current_price * Decimal("0.95"),
                                    take_profit=current_price * Decimal("1.05")
                                )
                                synced.append({
                                    'symbol': symbol,
                                    'quantity': str(qty),
                                    'price': str(current_price)
                                })
                                logger.warning(f"⚠️ Synced orphan position: {symbol} = {qty} @ {current_price}")
                        except Exception as e:
                            logger.error(f"Failed to sync {symbol}: {e}")

            # 報告結果
            result = {
                "synced": synced,
                "removed": removed,
                "updated": updated
            }

            if synced or removed or updated:
                logger.info(f"🔄 Sync completed: synced={len(synced)}, removed={len(removed)}, updated={len(updated)}")

                # 發送 Telegram 通知
                telegram = get_telegram_notifier()
                if telegram and telegram.enabled:
                    await telegram.notify_system_event(
                        event="Manual Position Sync",
                        status="info",
                        details=f"Synced: {synced}, Removed: {removed}, Updated: {updated}"
                    )
            else:
                logger.info("✅ Positions already in sync with Binance")

            return result

        except Exception as e:
            logger.error(f"Manual position sync failed: {e}", exc_info=True)
            raise

    async def remove_position_record(self, symbol: str) -> Dict[str, Any]:
        """
        強制移除持倉記錄（不實際賣出）

        用於清理無法賣出的灰塵持倉或同步問題導致的虛假記錄。

        Args:
            symbol: 交易對符號，如 "BNBUSDT"

        Returns:
            Dict with success status and details
        """
        async with self._position_lock:
            if symbol not in self._positions:
                return {
                    "success": False,
                    "message": f"Position {symbol} not found"
                }

            position = self._positions[symbol]
            position_info = {
                "symbol": symbol,
                "quantity": str(position.quantity),
                "entry_price": str(position.entry_price),
                "current_price": str(position.current_price)
            }

            # 移除持倉記錄
            del self._positions[symbol]

            logger.warning(f"🗑️ Manually removed position record: {position_info}")

            # 發送通知
            telegram = get_telegram_notifier()
            if telegram and telegram.enabled:
                await telegram.notify_system_event(
                    event="Position Record Removed",
                    status="warning",
                    details=f"Manually removed: {position_info}"
                )

            return {
                "success": True,
                "message": f"Position {symbol} removed from tracking",
                "removed_position": position_info
            }

    # ==================== 狀態查詢 ====================

    def get_status(self) -> Dict[str, Any]:
        """獲取引擎狀態"""
        return {
            'state': self.state.value,
            'running': self._running,
            'config': {
                'symbols': self.config.symbols,
                'trading_enabled': runtime_settings.trading_enabled,
                'dry_run': runtime_settings.dry_run,
                'paper_trading': runtime_settings.dry_run  # 與 dry_run 同義
            },
            'account': {
                'balance': str(self._account_balance),
                'available': str(self._available_balance)
            },
            'trading': {
                'daily_trades': self._daily_trades,
                'daily_pnl': self._daily_pnl,
                'positions': len(self._positions)
            },
            'data_engine': self.data_engine.get_status(),
            'strategy_engine': self.strategy_engine.get_status(),
            'learning_engine': self.learning_engine.get_status()
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        """獲取所有持倉（iOS App 兼容格式）"""
        positions = []
        for p in self._positions.values():
            # 計算盈虧百分比
            pnl_pct = self._calculate_pnl_pct(p, float(p.current_price))

            positions.append({
                'symbol': p.symbol,
                'side': p.side,
                'quantity': float(p.quantity),  # iOS 期待 Double
                'entry_price': float(p.entry_price),  # iOS 期待 Double
                'current_price': float(p.current_price),  # iOS 期待 Double
                'pnl': float(p.unrealized_pnl),  # iOS 用 pnl 不是 unrealized_pnl
                'pnl_percentage': pnl_pct * 100,  # iOS 期待百分比數值
                'unrealized_pnl': float(p.unrealized_pnl),  # 保留向後兼容
                'entry_time': p.entry_time.isoformat(),
                'stop_loss': float(p.stop_loss) if p.stop_loss else None,
                'take_profit': float(p.take_profit) if p.take_profit else None
            })
        return positions

    def get_performance(self) -> Dict[str, Any]:
        """獲取績效摘要"""
        return {
            'daily_pnl': self._daily_pnl,
            'daily_trades': self._daily_trades,
            'open_positions': len(self._positions),
            'learning_performance': self.learning_engine.get_recent_performance()
        }


# 全局實例
trading_engine = TradingEngine()
