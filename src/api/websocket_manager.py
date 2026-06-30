"""
WebSocket 連接管理器

負責：
- 管理所有 WebSocket 連接
- 廣播即時更新到所有客戶端
- 處理交易信號、訂單成交、狀態變更等事件
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass, asdict
from enum import Enum

from src.logger import logger


class MessageType(str, Enum):
    """WebSocket 消息類型"""
    # 連接相關
    CONNECTED = "connected"
    HEARTBEAT = "heartbeat"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    SUBSCRIBED = "subscribed"

    # 交易相關
    SIGNAL_UPDATE = "signal_update"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_UPDATE = "position_update"
    TRADE_EXECUTED = "trade_executed"

    # 狀態相關
    ENGINE_STATUS = "engine_status"
    MARKET_STATE = "market_state"
    STRATEGY_UPDATE = "strategy_update"

    # 數據相關
    PRICE_UPDATE = "price_update"
    PERFORMANCE_UPDATE = "performance_update"
    ORDER_BOOK_UPDATE = "order_book_update"
    MARKET_DATA = "market_data"

    # Arena 競技場相關
    ARENA_STATUS = "arena_status"
    ARENA_TRADE = "arena_trade"
    ARENA_MODEL_REGISTERED = "arena_model_registered"
    ARENA_CHAMPION_CHANGED = "arena_champion_changed"
    ARENA_LEADERBOARD = "arena_leaderboard"
    ARENA_UPDATE = "arena_update"
    MODEL_SIGNAL = "model_signal"

    # 風險與通知
    RISK_ALERT = "risk_alert"
    NOTIFICATION = "notification"
    SYSTEM_STATUS = "system_status"
    MODEL_UPDATE = "model_update"

    # 多時間框架與情緒
    MULTI_TIMEFRAME = "multi_timeframe"
    SENTIMENT_UPDATE = "sentiment_update"

    # 策略信號視覺化 (v0.9.0)
    SIGNALS_COMBINED = "signals_combined"       # 綜合信號更新
    MTF_ANALYSIS = "mtf_analysis"               # MTF 分析更新
    MARKET_REGIME_UPDATE = "market_regime_update"  # 市場狀態更新
    VOLUME_ANALYSIS = "volume_analysis"         # 成交量分析更新
    KELLY_POSITION = "kelly_position"           # Kelly 倉位建議更新
    STRATEGY_SIGNALS = "strategy_signals"       # 所有策略信號更新


@dataclass
class WSMessage:
    """WebSocket 消息結構"""
    type: str
    data: Dict[str, Any]
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class ConnectionManager:
    """WebSocket 連接管理器"""

    # 可用的訂閱頻道
    AVAILABLE_CHANNELS = {
        "all",           # 接收所有消息
        "arena",         # Arena 競技場更新
        "trades",        # 交易執行通知
        "prices",        # 價格更新
        "risks",         # 風險警報
        "system",        # 系統狀態
        "models",        # 模型更新
        "sentiment",     # 情緒分析
        "orderbook",     # 訂單簿深度
        "signals",       # 策略信號視覺化 (v0.9.0)
    }

    def __init__(self):
        # 活躍連接
        self.active_connections: Set[WebSocket] = set()
        # 連接元數據（用於追蹤）
        self._connection_info: Dict[WebSocket, Dict] = {}
        # 心跳任務
        self._heartbeat_task: Optional[asyncio.Task] = None
        # 廣播鎖
        self._broadcast_lock = asyncio.Lock()
        # 頻道訂閱 (channel -> websockets)
        self._channel_subscriptions: Dict[str, Set[WebSocket]] = {
            ch: set() for ch in self.AVAILABLE_CHANNELS
        }
        # 統計數據
        self._stats = {
            "total_connections": 0,
            "total_messages_sent": 0,
            "total_messages_received": 0,
        }

        logger.info("WebSocket ConnectionManager initialized")

    async def connect(self, websocket: WebSocket, client_id: str = None):
        """接受新的 WebSocket 連接"""
        await websocket.accept()
        self.active_connections.add(websocket)

        # 記錄連接信息
        cid = client_id or f"client_{len(self.active_connections)}"
        self._connection_info[websocket] = {
            "client_id": cid,
            "connected_at": datetime.utcnow().isoformat(),
            "last_heartbeat": datetime.utcnow().isoformat(),
            "subscriptions": {"all"}  # 默認訂閱 all 頻道
        }

        # 默認加入 all 頻道
        self._channel_subscriptions["all"].add(websocket)
        self._stats["total_connections"] += 1

        logger.info(f"WebSocket connected: {cid} (Total: {len(self.active_connections)})")

        # 發送連接成功消息
        await self.send_personal(websocket, WSMessage(
            type=MessageType.CONNECTED,
            data={
                "client_id": cid,
                "message": "Connected to trading system",
                "server_time": datetime.utcnow().isoformat(),
                "available_channels": list(self.AVAILABLE_CHANNELS)
            }
        ))

        # 啟動心跳（如果尚未啟動）
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    def disconnect(self, websocket: WebSocket):
        """斷開 WebSocket 連接"""
        if websocket in self.active_connections:
            client_info = self._connection_info.pop(websocket, {})
            self.active_connections.discard(websocket)

            # 從所有頻道移除
            for channel in self._channel_subscriptions.values():
                channel.discard(websocket)

            logger.info(f"WebSocket disconnected: {client_info.get('client_id', 'unknown')} "
                       f"(Remaining: {len(self.active_connections)})")

    async def subscribe(self, websocket: WebSocket, channels: List[str]) -> bool:
        """訂閱頻道"""
        if websocket not in self._connection_info:
            return False

        subscribed = []
        for channel in channels:
            if channel in self._channel_subscriptions:
                self._channel_subscriptions[channel].add(websocket)
                self._connection_info[websocket]["subscriptions"].add(channel)
                subscribed.append(channel)

        if subscribed:
            await self.send_personal(websocket, WSMessage(
                type=MessageType.SUBSCRIBED,
                data={
                    "channels": subscribed,
                    "all_subscriptions": list(self._connection_info[websocket]["subscriptions"])
                }
            ))
            logger.debug(f"Client subscribed to channels: {subscribed}")

        return bool(subscribed)

    async def unsubscribe(self, websocket: WebSocket, channels: List[str]) -> bool:
        """取消訂閱頻道"""
        if websocket not in self._connection_info:
            return False

        for channel in channels:
            if channel in self._channel_subscriptions:
                self._channel_subscriptions[channel].discard(websocket)
                self._connection_info[websocket]["subscriptions"].discard(channel)

        return True

    async def handle_client_message(self, websocket: WebSocket, raw_message: str):
        """處理來自客戶端的消息"""
        self._stats["total_messages_received"] += 1

        try:
            message = json.loads(raw_message)
            msg_type = message.get("type", "")

            if msg_type == "ping":
                # 心跳響應
                if websocket in self._connection_info:
                    self._connection_info[websocket]["last_heartbeat"] = datetime.utcnow().isoformat()
                await self.send_personal(websocket, WSMessage(
                    type=MessageType.PONG,
                    data={"server_time": datetime.utcnow().isoformat()}
                ))

            elif msg_type == "subscribe":
                # 訂閱頻道
                channels = message.get("channels", [])
                await self.subscribe(websocket, channels)

            elif msg_type == "unsubscribe":
                # 取消訂閱
                channels = message.get("channels", [])
                await self.unsubscribe(websocket, channels)

            else:
                logger.debug(f"Unknown message type: {msg_type}")

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON received: {raw_message[:100]}")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")

    async def send_personal(self, websocket: WebSocket, message: WSMessage):
        """發送消息到特定連接"""
        try:
            await websocket.send_text(message.to_json())
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: WSMessage):
        """廣播消息到所有連接"""
        if not self.active_connections:
            return

        async with self._broadcast_lock:
            disconnected = set()
            message_json = message.to_json()

            for connection in self.active_connections:
                try:
                    await connection.send_text(message_json)
                except Exception as e:
                    logger.warning(f"Broadcast failed for a connection: {e}")
                    disconnected.add(connection)

            # 清理斷開的連接
            for conn in disconnected:
                self.disconnect(conn)

    async def broadcast_signal(
        self,
        symbol: str,
        direction: int,
        strength: float,
        confidence: float,
        strategy_name: str = None,
        strategy: str = None,
        reason: str = None,
        recommendation: str = None
    ):
        """廣播交易信號"""
        # 支持 strategy 或 strategy_name 參數
        strat = strategy_name or strategy or ""
        await self.broadcast(WSMessage(
            type=MessageType.SIGNAL_UPDATE,
            data={
                "symbol": symbol,
                "direction": direction,
                "direction_text": "做多" if direction > 0 else ("做空" if direction < 0 else "觀望"),
                "strength": strength,
                "confidence": confidence,
                "strategy_name": strat,
                "strategy": strat,
                "reason": reason or recommendation or "",
                "recommendation": recommendation or reason or ""
            }
        ))

    async def broadcast_order_event(
        self,
        event_type: MessageType,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_id: str = None,
        status: str = None,
        pnl_pct: float = None,
        reason: str = None
    ):
        """廣播訂單事件"""
        await self.broadcast(WSMessage(
            type=event_type,
            data={
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "status": status,
                "pnl_pct": pnl_pct,
                "reason": reason
            }
        ))

    async def broadcast_position_update(
        self,
        symbol: str,
        side: str,
        size: float = None,
        quantity: float = None,
        entry_price: float = 0,
        current_price: float = 0,
        unrealized_pnl: float = 0,
        pnl_percent: float = None,
        pnl_pct: float = None
    ):
        """廣播持倉更新"""
        # 支持 size 或 quantity 參數
        pos_size = size if size is not None else (quantity if quantity is not None else 0)
        # 支持 pnl_percent 或 pnl_pct 參數
        pnl_p = pnl_percent if pnl_percent is not None else (pnl_pct if pnl_pct is not None else 0)

        await self.broadcast(WSMessage(
            type=MessageType.POSITION_UPDATE,
            data={
                "symbol": symbol,
                "side": side,
                "size": pos_size,
                "quantity": pos_size,
                "entry_price": entry_price,
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl,
                "pnl_percent": pnl_p,
                "pnl_pct": pnl_p
            }
        ))

    async def broadcast_engine_status(
        self,
        is_running: bool,
        mode: str,
        symbol: str,
        market_state: str,
        total_trades: int,
        today_pnl: float
    ):
        """廣播引擎狀態"""
        await self.broadcast(WSMessage(
            type=MessageType.ENGINE_STATUS,
            data={
                "is_running": is_running,
                "mode": mode,
                "symbol": symbol,
                "market_state": market_state,
                "total_trades": total_trades,
                "today_pnl": today_pnl
            }
        ))

    async def broadcast_order_book(self, symbol: str, bids: List, asks: List):
        """廣播訂單簿更新"""
        await self.broadcast(WSMessage(
            type=MessageType.ORDER_BOOK_UPDATE,
            data={
                "symbol": symbol,
                "bids": bids,  # lists of [price, qty]
                "asks": asks,
                "timestamp": datetime.utcnow().isoformat()
            }
        ))

    async def broadcast_price(self, symbol: str, price: float, change_24h: float = None):
        """廣播價格更新"""
        await self.broadcast(WSMessage(
            type=MessageType.PRICE_UPDATE,
            data={
                "symbol": symbol,
                "price": price,
                "change_24h": change_24h
            }
        ))

    async def broadcast_performance(
        self,
        total_return: float,
        today_return: float,
        win_rate: float,
        total_trades: int
    ):
        """廣播績效更新"""
        await self.broadcast(WSMessage(
            type=MessageType.PERFORMANCE_UPDATE,
            data={
                "total_return": total_return,
                "today_return": today_return,
                "win_rate": win_rate,
                "total_trades": total_trades
            }
        ))

    # ==================== Arena 競技場相關 ====================

    async def broadcast_arena_status(
        self,
        champion: Dict[str, Any],
        challengers: List[Dict[str, Any]],
        is_comparing: bool = False,
        comparison_days: int = 7
    ):
        """廣播 Arena 競技場狀態"""
        await self.broadcast(WSMessage(
            type=MessageType.ARENA_STATUS,
            data={
                "champion": champion,
                "challengers": challengers,
                "is_comparing": is_comparing,
                "comparison_days": comparison_days,
                "total_models": len(challengers) + (1 if champion else 0)
            }
        ))

    async def broadcast_arena_trade(
        self,
        model_version: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        is_winner: bool = None,
        pnl_pct: float = None
    ):
        """廣播 Arena 虛擬交易事件"""
        await self.broadcast(WSMessage(
            type=MessageType.ARENA_TRADE,
            data={
                "model_version": model_version,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "is_winner": is_winner,
                "pnl_pct": pnl_pct
            }
        ))

    async def broadcast_arena_model_registered(
        self,
        model_id: str,
        model_version: str,
        model_type: str,
        accuracy: float,
        symbol: str = None,
        source: str = "ios_local_training"
    ):
        """廣播新模型註冊到 Arena"""
        await self.broadcast(WSMessage(
            type=MessageType.ARENA_MODEL_REGISTERED,
            data={
                "model_id": model_id,
                "model_version": model_version,
                "model_type": model_type,
                "accuracy": accuracy,
                "symbol": symbol,
                "source": source
            }
        ))

    async def broadcast_arena_champion_changed(
        self,
        old_champion: str,
        new_champion: str,
        new_champion_stats: Dict[str, Any]
    ):
        """廣播冠軍變更"""
        await self.broadcast(WSMessage(
            type=MessageType.ARENA_CHAMPION_CHANGED,
            data={
                "old_champion": old_champion,
                "new_champion": new_champion,
                "new_champion_stats": new_champion_stats,
                "changed_at": datetime.utcnow().isoformat()
            }
        ))

    async def broadcast_arena_leaderboard(
        self,
        leaderboard: List[Dict[str, Any]],
        total_models: int = None
    ):
        """廣播 Arena 排行榜"""
        await self.broadcast_to_channel("arena", WSMessage(
            type=MessageType.ARENA_LEADERBOARD,
            data={
                "leaderboard": leaderboard,
                "total_models": total_models or len(leaderboard),
                "updated_at": datetime.utcnow().isoformat()
            }
        ))

    # ==================== 頻道廣播方法 ====================

    async def broadcast_to_channel(self, channel: str, message: WSMessage):
        """廣播消息到指定頻道"""
        if channel not in self._channel_subscriptions:
            return

        # 獲取訂閱該頻道的所有連接
        subscribers = self._channel_subscriptions[channel].copy()

        # 也包括訂閱 "all" 頻道的連接
        if channel != "all":
            subscribers = subscribers.union(self._channel_subscriptions.get("all", set()))

        disconnected = set()
        message_json = message.to_json()

        for websocket in subscribers:
            try:
                await websocket.send_text(message_json)
                self._stats["total_messages_sent"] += 1
            except Exception:
                disconnected.add(websocket)

        for conn in disconnected:
            self.disconnect(conn)

    # ==================== 風險與通知 ====================

    async def broadcast_risk_alert(
        self,
        alert_type: str,
        model_version: str,
        message: str,
        severity: str = "warning",
        symbol: Optional[str] = None,
        action_taken: Optional[str] = None
    ):
        """廣播風險警報"""
        await self.broadcast_to_channel("risks", WSMessage(
            type=MessageType.RISK_ALERT,
            data={
                "alert_type": alert_type,
                "model_version": model_version,
                "message": message,
                "severity": severity,
                "symbol": symbol,
                "action_taken": action_taken
            }
        ))

    async def broadcast_notification(
        self,
        title: str,
        message: str,
        notification_type: str,
        priority: str = "normal",
        data: Optional[Dict[str, Any]] = None
    ):
        """廣播通知"""
        await self.broadcast_to_channel("all", WSMessage(
            type=MessageType.NOTIFICATION,
            data={
                "title": title,
                "message": message,
                "notification_type": notification_type,
                "priority": priority,
                "extra_data": data or {}
            }
        ))

    # ==================== 模型與信號 ====================

    async def broadcast_model_signal(
        self,
        model_version: str,
        symbol: str,
        signal: str,
        confidence: float,
        indicators: Optional[Dict[str, Any]] = None
    ):
        """廣播模型信號"""
        await self.broadcast_to_channel("arena", WSMessage(
            type=MessageType.MODEL_SIGNAL,
            data={
                "model_version": model_version,
                "symbol": symbol,
                "signal": signal,
                "confidence": confidence,
                "indicators": indicators or {}
            }
        ))

    async def broadcast_model_update(
        self,
        model_version: str,
        update_type: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """廣播模型更新"""
        await self.broadcast_to_channel("models", WSMessage(
            type=MessageType.MODEL_UPDATE,
            data={
                "model_version": model_version,
                "update_type": update_type,
                "details": details or {}
            }
        ))

    # ==================== 市場數據 ====================

    async def broadcast_sentiment_update(
        self,
        symbol: str,
        overall_score: float,
        level: str,
        trading_signal: str,
        fear_greed_index: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """廣播情緒更新"""
        await self.broadcast_to_channel("sentiment", WSMessage(
            type=MessageType.SENTIMENT_UPDATE,
            data={
                "symbol": symbol,
                "overall_score": overall_score,
                "level": level,
                "trading_signal": trading_signal,
                "fear_greed_index": fear_greed_index,
                "details": details or {}
            }
        ))

    async def broadcast_multi_timeframe(
        self,
        symbol: str,
        overall_trend: str,
        trend_alignment: float,
        recommended_action: str,
        confidence: float,
        timeframes: Optional[Dict[str, Any]] = None
    ):
        """廣播多時間框架分析結果"""
        await self.broadcast_to_channel("arena", WSMessage(
            type=MessageType.MULTI_TIMEFRAME,
            data={
                "symbol": symbol,
                "overall_trend": overall_trend,
                "trend_alignment": trend_alignment,
                "recommended_action": recommended_action,
                "confidence": confidence,
                "timeframes": timeframes or {}
            }
        ))

    async def broadcast_orderbook_depth(
        self,
        symbol: str,
        bid_depth: float,
        ask_depth: float,
        imbalance: float,
        spread: float,
        top_bids: List[List[float]],
        top_asks: List[List[float]]
    ):
        """廣播訂單簿深度分析"""
        await self.broadcast_to_channel("orderbook", WSMessage(
            type=MessageType.ORDER_BOOK_UPDATE,
            data={
                "symbol": symbol,
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "imbalance": imbalance,
                "spread": spread,
                "top_bids": top_bids,
                "top_asks": top_asks
            }
        ))

    # ==================== 策略信號視覺化 (v0.9.0) ====================

    async def broadcast_signals_combined(
        self,
        symbol: str,
        signals: List[Dict[str, Any]],
        mtf_analysis: Optional[Dict[str, Any]] = None,
        market_regime: Optional[Dict[str, Any]] = None,
        volume_analysis: Optional[Dict[str, Any]] = None,
        kelly_position: Optional[Dict[str, Any]] = None,
        aggregated_signal: Optional[Dict[str, Any]] = None
    ):
        """廣播綜合策略信號更新 - 一次性推送所有信號數據"""
        await self.broadcast_to_channel("signals", WSMessage(
            type=MessageType.SIGNALS_COMBINED,
            data={
                "symbol": symbol,
                "signals": signals,
                "mtf_analysis": mtf_analysis,
                "market_regime": market_regime,
                "volume_analysis": volume_analysis,
                "kelly_position": kelly_position,
                "aggregated_signal": aggregated_signal,
                "timestamp": datetime.utcnow().isoformat()
            }
        ))

    async def broadcast_mtf_analysis(
        self,
        symbol: str,
        overall_trend: str,
        trend_alignment: float,
        primary_trend: str,
        timeframes: List[Dict[str, Any]]
    ):
        """廣播多時間框架分析更新"""
        await self.broadcast_to_channel("signals", WSMessage(
            type=MessageType.MTF_ANALYSIS,
            data={
                "symbol": symbol,
                "overall_trend": overall_trend,
                "trend_alignment": trend_alignment,
                "primary_trend": primary_trend,
                "timeframes": timeframes,
                "timestamp": datetime.utcnow().isoformat()
            }
        ))

    async def broadcast_market_regime_update(
        self,
        symbol: str,
        regime: str,
        confidence: float,
        volatility: float = 0,
        trend_strength: float = 0,
        strategy_weights: Dict[str, float] = None,
        adx: float = 0,
        bb_width: float = 0,
        description: str = ""
    ):
        """廣播市場狀態更新"""
        await self.broadcast_to_channel("signals", WSMessage(
            type=MessageType.MARKET_REGIME_UPDATE,
            data={
                "symbol": symbol,
                "regime": regime,
                "confidence": confidence,
                "volatility": volatility,
                "trend_strength": trend_strength,
                "strategy_weights": strategy_weights or {},
                "adx": adx,
                "bb_width": bb_width,
                "description": description,
                "timestamp": datetime.utcnow().isoformat()
            }
        ))

    async def broadcast_volume_analysis(
        self,
        symbol: str,
        volume_ratio: float,
        volume_status: str = "normal",
        is_surge: bool = False,
        is_dry: bool = False,
        volume_trend: str = "neutral",
        volume_sma: float = 0,
        current_volume: float = 0
    ):
        """廣播成交量分析更新"""
        await self.broadcast_to_channel("signals", WSMessage(
            type=MessageType.VOLUME_ANALYSIS,
            data={
                "symbol": symbol,
                "volume_ratio": volume_ratio,
                "volume_status": volume_status,
                "is_surge": is_surge,
                "is_dry": is_dry,
                "volume_trend": volume_trend,
                "volume_sma": volume_sma,
                "current_volume": current_volume,
                "timestamp": datetime.utcnow().isoformat()
            }
        ))

    async def broadcast_kelly_position(
        self,
        symbol: str,
        kelly_fraction: float,
        adjusted_fraction: float = 0,
        win_rate: float = 0.5,
        avg_win: float = 0.02,
        avg_loss: float = 0.01,
        suggested_size: float = 0,
        risk_level: str = "medium"
    ):
        """廣播 Kelly 倉位建議更新"""
        await self.broadcast_to_channel("signals", WSMessage(
            type=MessageType.KELLY_POSITION,
            data={
                "symbol": symbol,
                "kelly_fraction": kelly_fraction,
                "adjusted_fraction": adjusted_fraction,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "suggested_size": suggested_size,
                "risk_level": risk_level,
                "timestamp": datetime.utcnow().isoformat()
            }
        ))

    async def broadcast_strategy_signals(
        self,
        symbol: str,
        signals: List[Dict[str, Any]]
    ):
        """廣播所有策略信號更新"""
        await self.broadcast_to_channel("signals", WSMessage(
            type=MessageType.STRATEGY_SIGNALS,
            data={
                "symbol": symbol,
                "signals": signals,
                "count": len(signals),
                "timestamp": datetime.utcnow().isoformat()
            }
        ))

    # ==================== 統計與管理 ====================

    def get_stats(self) -> Dict[str, Any]:
        """獲取統計數據"""
        return {
            **self._stats,
            "active_connections": len(self.active_connections),
            "channels": {k: len(v) for k, v in self._channel_subscriptions.items()}
        }

    async def _heartbeat_loop(self):
        """心跳循環"""
        while self.active_connections:
            try:
                await asyncio.sleep(10)  # 每 10 秒發送心跳（從 30 秒改為 10 秒）
                await self.broadcast(WSMessage(
                    type=MessageType.HEARTBEAT,
                    data={"server_time": datetime.utcnow().isoformat()}
                ))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    def get_connection_count(self) -> int:
        """獲取當前連接數"""
        return len(self.active_connections)

    def get_connection_info(self) -> List[Dict]:
        """獲取所有連接信息"""
        return [
            {
                "client_id": info["client_id"],
                "connected_at": info["connected_at"]
            }
            for info in self._connection_info.values()
        ]


# 全局實例
ws_manager = ConnectionManager()
