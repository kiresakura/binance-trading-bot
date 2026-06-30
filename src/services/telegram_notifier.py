"""
Telegram Bot 服務 (升級版)
使用 python-telegram-bot 實現雙向互動
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

from telegram import Bot
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler
from telegram.error import TelegramError

from src.config import settings
from src.logger import logger
from src.services.msg_handler import message_handler

# 定義通知類型常量
class NotificationType:
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    RISK = "RISK"
    SYSTEM = "SYSTEM"
    PERFORMANCE = "PERFORMANCE"
    ERROR = "ERROR"
    STRATEGY = "STRATEGY"

class TelegramBotService:
    # 台灣時區 UTC+8
    TW_TZ = timezone(timedelta(hours=8))
    
    # 消息模板 (相容舊版)
    TEMPLATES = {
        NotificationType.SIGNAL: (
            "🎯 *交易信號偵測*\n"
            "—————————————\n"
            "幣種: `{symbol}`\n"
            "方向: {direction_emoji} *{direction}*\n"
            "價格: `${price}`\n"
            "強度: {strength_bar} ({strength})\n"
            "信心: {confidence_percent}\n"
            "策略: `{strategy}`\n"
            "—————————————\n"
            "原因: {reason}\n"
            "⏰ {time}"
        ),
        NotificationType.ORDER: (
            "🔔 *訂單執行通知*\n"
            "—————————————\n"
            "幣種: `{symbol}`\n"
            "動作: {side} {type}\n"
            "數量: `{quantity}`\n"
            "價格: `${price}`\n"
            "—————————————\n"
            "狀態: *{status}* {status_emoji}\n"
            "訂單號: `{order_id}`\n"
            "⏰ {time}"
        ),
        NotificationType.RISK: (
            "⚠️ *風險監控警報*\n"
            "—————————————\n"
            "類型: *{risk_type}*\n"
            "等級: {level_emoji} *{level}*\n"
            "當前值: `{current_value}`\n"
            "閾值: `{threshold}`\n"
            "—————————————\n"
            "建議: {action}\n"
            "⏰ {time}"
        ),
        NotificationType.SYSTEM: (
            "🖥 *系統狀態通知*\n"
            "—————————————\n"
            "事件: *{event}*\n"
            "級別: {level}\n"
            "—————————————\n"
            "詳情: {details}\n"
            "⏰ {time}"
        ),
        NotificationType.PERFORMANCE: (
            "📈 *交易績效報告*\n"
            "—————————————\n"
            "週期: *{period}*\n"
            "總盈虧: {pnl_emoji} `${total_pnl}`\n"
            "收益率: {roi_percent}\n"
            "交易數: {trade_count} (勝 {win_count}/負 {loss_count})\n"
            "勝率: {win_rate}\n"
            "—————————————\n"
            "最大回撤: {max_drawdown}\n"
            "Sharpe: {sharpe_ratio}\n"
            "⏰ {time}"
        ),
        NotificationType.ERROR: (
            "❌ *系統錯誤警報*\n"
            "—————————————\n"
            "錯誤: `{error_type}`\n"
            "位置: `{location}`\n"
            "—————————————\n"
            "訊息: {message}\n"
            "⏰ {time}"
        )
    }

    def __init__(self):
        self.enabled = settings.TELEGRAM_ENABLED
        self.token = settings.TELEGRAM_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID
        
        self.application: Optional[Application] = None
        self.bot: Optional[Bot] = None
        self._initialized = False

        if self.enabled:
            if not self.token or not self.chat_id:
                logger.warning("Telegram enabled but token/chat_id not configured")
                self.enabled = False

    async def init(self):
        """初始化 Telegram Bot (Application)"""
        if not self.enabled or self._initialized:
            return

        try:
            # 建構 Application
            builder = Application.builder().token(self.token)
            self.application = builder.build()
            self.bot = self.application.bot # 方便發送主動通知

            # 註冊指令處理器
            self.application.add_handler(CommandHandler("start", message_handler.start_command))
            self.application.add_handler(CommandHandler("help", message_handler.help_command))
            self.application.add_handler(CommandHandler("status", message_handler.status_command))
            self.application.add_handler(CommandHandler("balance", message_handler.balance_command))
            self.application.add_handler(CommandHandler("pnl", message_handler.pnl_command))
            self.application.add_handler(CommandHandler("stop", message_handler.stop_command))

            # 初始化並啟動
            await self.application.initialize()
            await self.application.start()
            
            # 支援 Polling 模式 (適合開發/非 Webhook 環境)
            # 注意: 在生產環境如果是 Webhook 模式，這裡需要調整邏輯
            # 由於這是異步啟動，我們使用 create_task 讓 updater 在後台運行
            await self.application.updater.start_polling()

            self._initialized = True
            logger.info("✅ Telegram bot service initialized (Polling mode)")
            
            # 發送啟動通知
            await self.send_message("🚀 *系統啟動成功*\nTelegram 互動服務已上線")

        except Exception as e:
            logger.error(f"❌ Telegram bot initialization failed: {e}")
            self.enabled = False

    async def stop(self):
        """停止 Telegram Bot"""
        if self.application:
            try:
                # 發送關閉通知
                await self.send_message("💤 *系統正在關閉*")
                
                await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
                logger.info("Telegram bot stopped")
            except Exception as e:
                logger.error(f"Error stopping Telegram bot: {e}")

    async def send_message(self, text: str, parse_mode=ParseMode.MARKDOWN):
        """發送普通消息 (主動推播)"""
        if not self.enabled or not self.bot:
            return

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode
            )
        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")

    async def _send_formatted_notification(self, notif_type: str, data: Dict):
        """發送格式化通知 (內部使用)"""
        if not self.enabled:
            return

        template = self.TEMPLATES.get(notif_type)
        if not template:
            return

        # 加入時間
        data["time"] = datetime.now(self.TW_TZ).strftime("%Y-%m-%d %H:%M:%S")

        try:
            message = template.format(**data)
            await self.send_message(message)
        except KeyError as e:
            logger.error(f"Missing key for notification template: {e}")
        except Exception as e:
            logger.error(f"Error formatting/sending notification: {e}")

    # ==================== 相容舊版介面 ====================

    async def notify_signal(self, symbol: str, direction: int, strength: float, confidence: float, strategy: str, reason: str):
        direction_str = "做多 (LONG)" if direction > 0 else "做空 (SHORT)"
        direction_emoji = "🟢" if direction > 0 else "🔴"
        strength_bar = "▮" * int(strength * 5) + "▯" * (5 - int(strength * 5))
        
        await self._send_formatted_notification(NotificationType.SIGNAL, {
            "symbol": symbol,
            "direction": direction_str,
            "direction_emoji": direction_emoji,
            "price": "MARKET", # 信號通常是市價
            "strength": f"{strength:.2f}",
            "strength_bar": strength_bar,
            "confidence_percent": f"{confidence:.1%}",
            "strategy": strategy,
            "reason": message_handler._escape_markdown(reason) if hasattr(message_handler, '_escape_markdown') else reason
        })

    async def notify_order_filled(self, order):
        # 簡單適配 Order 物件
        side_emoji = "🟢" if order.side == "BUY" else "🔴"
        status_emoji = "✅" if order.status == "FILLED" else "⏳"
        
        await self._send_formatted_notification(NotificationType.ORDER, {
            "symbol": order.symbol,
            "side": order.side,
            "type": order.type,
            "quantity": order.quantity,
            "price": order.price if order.price else "MARKET",
            "status": order.status,
            "status_emoji": status_emoji,
            "order_id": order.order_id
        })

    async def notify_system_event(self, event: str, level: str, details: str):
        await self._send_formatted_notification(NotificationType.SYSTEM, {
            "event": event,
            "level": level,
            "details": details
        })

    async def notify_error(self, error_type: str, message: str, location: str = "Unknown"):
        await self._send_formatted_notification(NotificationType.ERROR, {
            "error_type": error_type,
            "location": location,
            "message": message
        })

# 全局實例 - 替代舊版 telegram_notifier
telegram_notifier = TelegramBotService()
# telegram_service = telegram_notifier # Alias if needed
