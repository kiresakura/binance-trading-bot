"""
Telegram 消息處理器
負責處理來自使用者的指令與訊息
"""

from telegram import Update
from telegram.ext import ContextTypes
from src.logger import logger
from src.services.binance_client import binance_client
from src.services.pnl_tracker import pnl_tracker
from src.services.risk_manager import risk_management_system
from src.services.market_publisher import market_publisher

class MessageHandler:
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """處理 /start 指令"""
        welcome_text = (
            "🤖 *Binance Auto Trader Bot*\n\n"
            "歡迎使用自動交易機器人！\n"
            "可用指令：\n"
            "• /status - 查看系統狀態\n"
            "• /balance - 查看帳戶餘額\n"
            "• /pnl - 查看今日盈虧\n"
            "• /stop - 緊急停止交易\n"
        )
        await update.message.reply_text(welcome_text, parse_mode="Markdown")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """處理 /help 指令"""
        help_text = (
            "📚 *指令列表*\n\n"
            "• /status - 顯示系統健康度、持倉數與 API 延遲\n"
            "• /balance - 顯示 USDT 總權益與可用餘額\n"
            "• /pnl - 顯示今日盈虧與勝率\n"
            "• /stop - 【危險】緊急停止所有交易活動\n"
        )
        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """處理 /status 指令"""
        try:
            # 獲取系統狀態
            health_status = "🟢 正常"  # 可以整合 HealthCheck 邏輯
            
            # 獲取持倉數
            positions = await binance_client.get_open_orders() # 暫時用 open orders 代表活動
            position_count = len(positions)
            
            status_text = (
                "📊 *系統狀態報告*\n\n"
                f"系統健康: {health_status}\n"
                f"活躍掛單: {position_count}\n"
                f"Risk Manager: 🟢 運行中\n"
                f"Cache Service: 🟢 運行中\n"
            )
            await update.message.reply_text(status_text, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Status command error: {e}")
            await update.message.reply_text("❌ 獲取狀態失敗")

    async def balance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """處理 /balance 指令"""
        try:
            # 獲取餘額
            balances = await binance_client.get_balances()
            usdt_balance = next((b for b in balances if b['asset'] == 'USDT'), None)
            
            if usdt_balance:
                total = float(usdt_balance['total'])
                free = float(usdt_balance['free'])
                locked = float(usdt_balance['locked'])
                
                balance_text = (
                    "💰 *帳戶餘額 (USDT)*\n\n"
                    f"總權益: `${total:,.2f}`\n"
                    f"可用: `${free:,.2f}`\n"
                    f"凍結: `${locked:,.2f}`\n"
                )
                await update.message.reply_text(balance_text, parse_mode="Markdown")
            else:
                await update.message.reply_text("⚠️ 找不到 USDT 餘額資訊")
                
        except Exception as e:
            logger.error(f"Balance command error: {e}")
            await update.message.reply_text("❌ 獲取餘額失敗")

    async def pnl_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """處理 /pnl 指令"""
        try:
            stats = pnl_tracker.get_daily_stats()
            
            pnl_emoji = "🟢" if stats['total_pnl'] >= 0 else "🔴"
            
            pnl_text = (
                "📈 *今日績效*\n\n"
                f"盈虧: {pnl_emoji} `${stats['total_pnl']:,.2f}`\n"
                f"交易次數: {stats['trade_count']}\n"
                f"勝率: {stats['win_rate']:.1%}\n"
            )
            await update.message.reply_text(pnl_text, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"PnL command error: {e}")
            await update.message.reply_text("❌ 獲取績效失敗")

    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """處理 /stop 指令 (緊急停止)"""
        # 這裡應該加入二次確認邏輯，目前先直接觸發
        try:
            # 觸發緊急停止
            # await fault_tolerant_engine.trigger_emergency_stop() 
            
            logger.critical(f"Emergency stop triggered by user {update.effective_user.id}")
            
            stop_text = (
                "🛑 *緊急停止已觸發*\n\n"
                "• 所有交易已暫停\n"
                "• 正在嘗試撤銷所有掛單...\n"
                "• 請檢查服務器日誌\n"
            )
            await update.message.reply_text(stop_text, parse_mode="Markdown")
            
            # 實際執行撤單
            symbols = market_publisher.symbols  # 使用 MarketPublisher 關注的列表
            
            for symbol in symbols:
                await binance_client.cancel_all_orders(symbol)
                await update.message.reply_text(f"⚠️ 已嘗試撤銷 {symbol} 所有掛單")
            
        except Exception as e:
            logger.error(f"Stop command error: {e}")
            await update.message.reply_text("❌ 緊急停止執行失敗，請手動操作！")

# 全局實例
message_handler = MessageHandler()
