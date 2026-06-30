"""
Notification Service - 通知服務

提供：
- Telegram 通知
- WebSocket 推送
- 風險警報
- 交易通知
- 模型狀態更新
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json
import aiohttp

from src.logger import logger


class NotificationType(str, Enum):
    """通知類型"""
    TRADE = "trade"              # 交易通知
    RISK_ALERT = "risk_alert"    # 風險警報
    MODEL_UPDATE = "model_update"  # 模型更新
    SYSTEM = "system"            # 系統通知
    PRICE_ALERT = "price_alert"  # 價格警報
    PERFORMANCE = "performance"  # 績效報告


class NotificationPriority(str, Enum):
    """通知優先級"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Notification:
    """通知數據"""
    type: NotificationType
    priority: NotificationPriority
    title: str
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value if isinstance(self.type, NotificationType) else self.type,
            "priority": self.priority.value if isinstance(self.priority, NotificationPriority) else self.priority,
            "title": self.title,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp
        }


@dataclass
class PriceAlert:
    """價格警報配置"""
    symbol: str
    condition: str  # "above", "below", "change_pct"
    target_price: Optional[float] = None
    change_pct: Optional[float] = None
    triggered: bool = False
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


class NotificationService:
    """通知服務"""

    def __init__(self):
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.telegram_token and self.telegram_chat_id)

        # 通知歷史
        self._notification_history: List[Notification] = []
        self._max_history = 500

        # 價格警報
        self._price_alerts: Dict[str, List[PriceAlert]] = {}

        # 通知頻率限制（避免頻繁通知）
        self._last_notifications: Dict[str, datetime] = {}
        self._min_interval_seconds = 60  # 同類型通知最小間隔

        # 通知偏好設定
        self.preferences = {
            "trade_notifications": True,
            "risk_alerts": True,
            "model_updates": True,
            "performance_reports": True,
            "price_alerts": True,
            "quiet_hours_start": None,  # e.g., "22:00"
            "quiet_hours_end": None,    # e.g., "08:00"
        }

        if self.enabled:
            logger.info("NotificationService initialized with Telegram")
        else:
            logger.warning("NotificationService: Telegram not configured")

    async def send_notification(
        self,
        notification: Notification,
        force: bool = False
    ) -> bool:
        """
        發送通知

        Args:
            notification: 通知對象
            force: 是否強制發送（忽略頻率限制）

        Returns:
            是否成功發送
        """
        # 檢查通知類型是否啟用
        pref_key = f"{notification.type.value}_notifications" if isinstance(notification.type, NotificationType) else f"{notification.type}_notifications"
        if pref_key in self.preferences and not self.preferences.get(pref_key, True):
            return False

        # 檢查安靜時間
        if self._is_quiet_hours() and notification.priority != NotificationPriority.CRITICAL:
            return False

        # 檢查頻率限制
        if not force and not self._check_rate_limit(notification):
            return False

        # 記錄歷史
        self._notification_history.append(notification)
        if len(self._notification_history) > self._max_history:
            self._notification_history.pop(0)

        # 發送到各個渠道
        success = True

        # Telegram
        if self.enabled:
            try:
                await self._send_telegram(notification)
            except Exception as e:
                logger.error(f"Failed to send Telegram notification: {e}")
                success = False

        # WebSocket (如果可用)
        try:
            await self._send_websocket(notification)
        except Exception as e:
            logger.warning(f"Failed to send WebSocket notification: {e}")

        return success

    async def _send_telegram(self, notification: Notification):
        """發送 Telegram 通知"""
        if not self.enabled:
            return

        # 構建消息
        emoji_map = {
            NotificationType.TRADE: "💹",
            NotificationType.RISK_ALERT: "⚠️",
            NotificationType.MODEL_UPDATE: "🤖",
            NotificationType.SYSTEM: "🔔",
            NotificationType.PRICE_ALERT: "📊",
            NotificationType.PERFORMANCE: "📈",
        }

        priority_emoji = {
            NotificationPriority.LOW: "",
            NotificationPriority.NORMAL: "",
            NotificationPriority.HIGH: "❗",
            NotificationPriority.CRITICAL: "🚨",
        }

        ntype = notification.type if isinstance(notification.type, NotificationType) else NotificationType(notification.type)
        npriority = notification.priority if isinstance(notification.priority, NotificationPriority) else NotificationPriority(notification.priority)

        emoji = emoji_map.get(ntype, "📬")
        priority = priority_emoji.get(npriority, "")

        message = f"{emoji} {priority} *{notification.title}*\n\n{notification.message}"

        if notification.data:
            # 添加額外數據
            data_str = "\n".join([f"• {k}: {v}" for k, v in notification.data.items() if v is not None])
            if data_str:
                message += f"\n\n{data_str}"

        # 發送到 Telegram
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_notification": npriority == NotificationPriority.LOW
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.warning(f"Telegram API error: {error}")

    async def _send_websocket(self, notification: Notification):
        """發送 WebSocket 通知"""
        try:
            from src.api.websocket_manager import ws_manager
            await ws_manager.broadcast(
                message_type="notification",
                data=notification.to_dict()
            )
        except Exception as e:
            pass  # WebSocket 可能未初始化

    def _check_rate_limit(self, notification: Notification) -> bool:
        """檢查通知頻率限制"""
        ntype = notification.type.value if isinstance(notification.type, NotificationType) else notification.type
        key = f"{ntype}_{notification.title}"

        last_time = self._last_notifications.get(key)
        now = datetime.utcnow()

        if last_time:
            elapsed = (now - last_time).total_seconds()
            if elapsed < self._min_interval_seconds:
                return False

        self._last_notifications[key] = now
        return True

    def _is_quiet_hours(self) -> bool:
        """檢查是否在安靜時間"""
        start = self.preferences.get("quiet_hours_start")
        end = self.preferences.get("quiet_hours_end")

        if not start or not end:
            return False

        now = datetime.utcnow().time()
        start_time = datetime.strptime(start, "%H:%M").time()
        end_time = datetime.strptime(end, "%H:%M").time()

        if start_time <= end_time:
            return start_time <= now <= end_time
        else:
            # 跨午夜
            return now >= start_time or now <= end_time

    # ==================== 便捷方法 ====================

    async def notify_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        model_version: Optional[str] = None
    ):
        """發送交易通知"""
        if side.upper() == "BUY":
            title = f"買入 {symbol}"
            emoji = "🟢"
        else:
            title = f"賣出 {symbol}"
            emoji = "🔴" if pnl and pnl < 0 else "🟢"

        message = f"{emoji} {side.upper()} {quantity:.6f} @ ${price:.4f}"

        data = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price
        }

        if pnl is not None:
            data["pnl"] = f"${pnl:.2f}"
            data["pnl_pct"] = f"{pnl_pct:.2f}%" if pnl_pct else None
            message += f"\nPnL: {emoji} ${pnl:.2f} ({pnl_pct:.2f}%)"

        if model_version:
            data["model"] = model_version

        await self.send_notification(Notification(
            type=NotificationType.TRADE,
            priority=NotificationPriority.NORMAL,
            title=title,
            message=message,
            data=data
        ))

    async def notify_risk_alert(
        self,
        alert_type: str,
        model_version: str,
        message: str,
        severity: str = "warning",
        symbol: Optional[str] = None
    ):
        """發送風險警報"""
        priority_map = {
            "info": NotificationPriority.LOW,
            "warning": NotificationPriority.HIGH,
            "critical": NotificationPriority.CRITICAL
        }

        title = f"風險警報: {alert_type}"

        await self.send_notification(Notification(
            type=NotificationType.RISK_ALERT,
            priority=priority_map.get(severity, NotificationPriority.HIGH),
            title=title,
            message=message,
            data={
                "alert_type": alert_type,
                "model": model_version,
                "symbol": symbol,
                "severity": severity
            }
        ))

    async def notify_model_update(
        self,
        model_id: str,
        update_type: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """發送模型更新通知"""
        title_map = {
            "trained": "模型訓練完成",
            "deployed": "模型已部署",
            "promoted": "模型晉升為冠軍",
            "expired": "模型即將過期",
            "retired": "模型已退役"
        }

        title = title_map.get(update_type, f"模型更新: {update_type}")
        message = f"模型 {model_id} {update_type}"

        if details:
            for key, value in details.items():
                message += f"\n• {key}: {value}"

        await self.send_notification(Notification(
            type=NotificationType.MODEL_UPDATE,
            priority=NotificationPriority.NORMAL,
            title=title,
            message=message,
            data={"model_id": model_id, "update_type": update_type, **(details or {})}
        ))

    async def notify_performance_report(
        self,
        period: str,
        total_return_pct: float,
        win_rate: float,
        total_trades: int,
        best_trade: Optional[float] = None,
        worst_trade: Optional[float] = None
    ):
        """發送績效報告"""
        emoji = "📈" if total_return_pct >= 0 else "📉"
        title = f"{period} 績效報告"

        message = f"""
{emoji} 總回報: {total_return_pct:+.2f}%
🎯 勝率: {win_rate:.1f}%
📊 交易次數: {total_trades}
"""

        if best_trade is not None:
            message += f"🏆 最佳交易: +{best_trade:.2f}%\n"
        if worst_trade is not None:
            message += f"💀 最差交易: {worst_trade:.2f}%\n"

        await self.send_notification(Notification(
            type=NotificationType.PERFORMANCE,
            priority=NotificationPriority.NORMAL,
            title=title,
            message=message,
            data={
                "period": period,
                "total_return_pct": total_return_pct,
                "win_rate": win_rate,
                "total_trades": total_trades
            }
        ))

    # ==================== 價格警報 ====================

    def add_price_alert(
        self,
        symbol: str,
        condition: str,
        target_price: Optional[float] = None,
        change_pct: Optional[float] = None
    ) -> PriceAlert:
        """添加價格警報"""
        alert = PriceAlert(
            symbol=symbol,
            condition=condition,
            target_price=target_price,
            change_pct=change_pct
        )

        if symbol not in self._price_alerts:
            self._price_alerts[symbol] = []

        self._price_alerts[symbol].append(alert)
        logger.info(f"Price alert added: {symbol} {condition} {target_price or change_pct}")

        return alert

    def remove_price_alert(self, symbol: str, index: int = 0) -> bool:
        """移除價格警報"""
        if symbol in self._price_alerts and len(self._price_alerts[symbol]) > index:
            self._price_alerts[symbol].pop(index)
            return True
        return False

    async def check_price_alerts(self, prices: Dict[str, float]):
        """檢查價格警報"""
        for symbol, current_price in prices.items():
            if symbol not in self._price_alerts:
                continue

            for alert in self._price_alerts[symbol]:
                if alert.triggered:
                    continue

                triggered = False
                message = ""

                if alert.condition == "above" and alert.target_price:
                    if current_price >= alert.target_price:
                        triggered = True
                        message = f"{symbol} 已突破 ${alert.target_price:.4f}，當前價格: ${current_price:.4f}"

                elif alert.condition == "below" and alert.target_price:
                    if current_price <= alert.target_price:
                        triggered = True
                        message = f"{symbol} 已跌破 ${alert.target_price:.4f}，當前價格: ${current_price:.4f}"

                if triggered:
                    alert.triggered = True
                    await self.send_notification(Notification(
                        type=NotificationType.PRICE_ALERT,
                        priority=NotificationPriority.HIGH,
                        title=f"價格警報: {symbol}",
                        message=message,
                        data={
                            "symbol": symbol,
                            "condition": alert.condition,
                            "target_price": alert.target_price,
                            "current_price": current_price
                        }
                    ))

    # ==================== 歷史查詢 ====================

    def get_notification_history(
        self,
        notification_type: Optional[NotificationType] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """獲取通知歷史"""
        history = self._notification_history

        if notification_type:
            history = [n for n in history if n.type == notification_type]

        return [n.to_dict() for n in history[-limit:]]

    def get_price_alerts(self, symbol: Optional[str] = None) -> Dict[str, List[Dict]]:
        """獲取價格警報"""
        if symbol:
            alerts = self._price_alerts.get(symbol, [])
            return {symbol: [asdict(a) for a in alerts]}

        return {s: [asdict(a) for a in alerts] for s, alerts in self._price_alerts.items()}

    def update_preferences(self, **kwargs):
        """更新通知偏好"""
        for key, value in kwargs.items():
            if key in self.preferences:
                self.preferences[key] = value
                logger.info(f"Notification preference updated: {key} = {value}")


# 單例實例
notification_service = NotificationService()
