"""
AI 助手服務 - Claude API 整合

提供：
- 即時對話：回答交易相關問題
- 每日報告：自動生成並推送到 Telegram
- 數據上下文：整合 PnL、持倉、Arena 數據
"""

import asyncio
from datetime import datetime, date, timedelta, timezone
from typing import Optional, Dict, Any, List
import json

from src.config import settings
from src.logger import logger

# Anthropic SDK - 動態檢查
def check_anthropic_available():
    """動態檢查 anthropic 套件是否可用"""
    try:
        import anthropic
        return True, anthropic
    except ImportError as e:
        logger.warning(f"anthropic package not installed: {e}")
        return False, None

# 初始嘗試導入
ANTHROPIC_AVAILABLE, anthropic = check_anthropic_available()


# 台灣時區 (UTC+8)
TW_TZ = timezone(timedelta(hours=8))


# 系統提示模板
SYSTEM_PROMPT = """你是一位專業的加密貨幣交易助手，為 Binance 自動交易機器人提供分析和報告。

## 你的能力：
- 分析交易績效和盈虧數據
- 解釋策略決策和市場狀況
- 提供持倉和風險管理的洞察
- 生成每日/週交易摘要報告

## 數據上下文：
你可以訪問以下實時數據：
- 交易歷史和盈虧記錄
- 當前持倉及其表現
- Arena 模型競技場狀態
- 系統健康和配置狀態

## 回應指南：
1. 簡潔且可執行
2. 討論績效時使用數字和百分比
3. 突出風險和機會
4. 貨幣值使用 USD，保留適當精度
5. 預設使用繁體中文回應
6. 適度使用表情符號增加可讀性

## 重要提醒：
- 不提供具體的交易建議或投資建議
- 總是提醒用戶：過去績效不保證未來結果
- 如果數據缺失，清楚說明哪些信息不可用
"""

DAILY_REPORT_PROMPT = """根據提供的交易數據，生成一份簡潔的每日交易摘要報告。

請包含以下部分：
1. 📊 今日摘要 - 總盈虧、勝率、交易次數
2. 💰 持倉狀態 - 當前持倉及未實現盈虧
3. 🏆 Arena 競技場 - 冠軍模型表現、影子交易狀態
4. ⚠️ 風險提示 - 任何需要關注的指標
5. 📈 建議觀察 - 明日觀察重點

報告控制在 500 字以內。使用表情符號標記各部分。
"""


class AIService:
    """AI 助手服務"""

    def __init__(self):
        self._client: Optional[anthropic.Anthropic] = None
        self._initialized = False
        self._enabled = False
        self._scheduler_running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._tokens_used_today = 0
        self._last_token_reset: Optional[date] = None

    async def init(self):
        """初始化 AI 服務"""
        global ANTHROPIC_AVAILABLE, anthropic

        if self._initialized:
            return

        # 動態重新檢查 anthropic 套件
        ANTHROPIC_AVAILABLE, anthropic = check_anthropic_available()

        if not ANTHROPIC_AVAILABLE:
            logger.warning("AI Service disabled: anthropic package not available")
            self._enabled = False
            return

        if not settings.ANTHROPIC_API_KEY:
            logger.warning("AI Service disabled: ANTHROPIC_API_KEY not set")
            self._enabled = False
            return

        if not settings.AI_ENABLED:
            logger.info("AI Service disabled by configuration (AI_ENABLED=False)")
            self._enabled = False
            return

        try:
            self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            self._enabled = True
            self._initialized = True
            self._last_token_reset = date.today()
            logger.info(f"✅ AI Service initialized with model: {settings.AI_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize AI Service: {e}")
            self._enabled = False

    async def start_scheduler(self):
        """啟動每日報告定時任務"""
        if not self._enabled:
            logger.info("AI scheduler not started: service disabled")
            return

        if self._scheduler_running:
            logger.info("AI scheduler already running")
            return

        self._scheduler_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("✅ AI daily report scheduler started")

    async def stop_scheduler(self):
        """停止定時任務"""
        self._scheduler_running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        self._scheduler_task = None
        logger.info("✅ AI scheduler stopped")

    async def _scheduler_loop(self):
        """每日報告排程循環"""
        while self._scheduler_running:
            try:
                now = datetime.now(TW_TZ)
                # 目標時間：台灣時間 08:00
                target_hour = settings.AI_DAILY_REPORT_HOUR
                next_run = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)

                if now >= next_run:
                    next_run += timedelta(days=1)

                wait_seconds = (next_run - now).total_seconds()
                logger.info(f"Next AI daily report at {next_run.strftime('%Y-%m-%d %H:%M')} TW")

                await asyncio.sleep(wait_seconds)

                if self._scheduler_running:
                    await self._generate_and_send_daily_report()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"AI scheduler error: {e}")
                await asyncio.sleep(3600)  # 錯誤後等待 1 小時重試

    async def _generate_and_send_daily_report(self):
        """生成並發送每日報告"""
        try:
            logger.info("Generating daily AI report...")

            # 生成報告
            report = await self.generate_daily_report()

            if not report:
                logger.warning("Failed to generate daily report")
                return

            # 發送到 Telegram
            try:
                from src.services.telegram_notifier import telegram_notifier
                if telegram_notifier.enabled:
                    await telegram_notifier.send_message(report)
                    logger.info("✅ Daily report sent to Telegram")
            except Exception as e:
                logger.error(f"Failed to send report to Telegram: {e}")

            # 緩存報告到數據庫
            await self._cache_daily_report(date.today(), report)

        except Exception as e:
            logger.error(f"Failed to generate/send daily report: {e}")

    async def chat(
        self,
        message: str,
        include_positions: bool = True,
        include_pnl: bool = True,
        include_arena: bool = True
    ) -> Dict[str, Any]:
        """
        處理用戶對話

        Args:
            message: 用戶消息
            include_positions: 是否包含持倉數據
            include_pnl: 是否包含盈虧數據
            include_arena: 是否包含 Arena 數據

        Returns:
            {
                "success": bool,
                "response": str,
                "tokens_used": int,
                "context_sources": List[str]
            }
        """
        if not self._enabled:
            return {
                "success": False,
                "response": "AI 助手未啟用。請在環境變數中設置 ANTHROPIC_API_KEY 和 AI_ENABLED=true",
                "tokens_used": 0,
                "context_sources": []
            }

        # 檢查預算
        if not await self._check_budget(1000):
            return {
                "success": False,
                "response": "已達到今日 token 預算上限，請明天再試",
                "tokens_used": 0,
                "context_sources": []
            }

        try:
            # 構建上下文
            context, sources = await self._build_context(
                include_positions=include_positions,
                include_pnl=include_pnl,
                include_arena=include_arena
            )

            # 構建用戶消息（包含上下文）
            user_message = f"""## 當前數據上下文
{json.dumps(context, indent=2, ensure_ascii=False, default=str)}

## 用戶問題
{message}"""

            # 調用 Claude API
            response = self._client.messages.create(
                model=settings.AI_MODEL,
                max_tokens=settings.AI_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            # 提取回應
            reply = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            # 更新 token 計數
            await self._update_token_usage(tokens_used)

            # 記錄使用
            await self._log_usage("chat", response.usage.input_tokens, response.usage.output_tokens)

            return {
                "success": True,
                "response": reply,
                "tokens_used": tokens_used,
                "context_sources": sources
            }

        except anthropic.RateLimitError:
            logger.warning("Claude API rate limit reached")
            return {
                "success": False,
                "response": "API 請求過於頻繁，請稍後再試",
                "tokens_used": 0,
                "context_sources": []
            }
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return {
                "success": False,
                "response": f"處理請求時發生錯誤: {str(e)}",
                "tokens_used": 0,
                "context_sources": []
            }

    async def generate_daily_report(self, target_date: Optional[date] = None) -> Optional[str]:
        """
        生成每日報告

        Args:
            target_date: 目標日期，預設為昨天

        Returns:
            報告文本
        """
        if not self._enabled:
            return None

        if not await self._check_budget(2000):
            logger.warning("Insufficient token budget for daily report")
            return None

        try:
            # 構建完整上下文
            context, _ = await self._build_context(
                include_positions=True,
                include_pnl=True,
                include_arena=True
            )

            # 添加日期信息
            report_date = target_date or (date.today() - timedelta(days=1))
            context["report_date"] = report_date.isoformat()
            context["report_generated_at"] = datetime.now(TW_TZ).isoformat()

            user_message = f"""## 交易數據
{json.dumps(context, indent=2, ensure_ascii=False, default=str)}

## 任務
{DAILY_REPORT_PROMPT}"""

            # 調用 Claude API
            response = self._client.messages.create(
                model=settings.AI_MODEL,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            report = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            # 更新計數
            await self._update_token_usage(tokens_used)
            await self._log_usage("daily_report", response.usage.input_tokens, response.usage.output_tokens)

            # 添加標題和時間戳
            header = f"📈 **每日交易報告 - {report_date.strftime('%Y/%m/%d')}**\n\n"
            footer = f"\n\n---\n_由 AI 助手自動生成於 {datetime.now(TW_TZ).strftime('%H:%M')}_"

            return header + report + footer

        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}")
            return None

    async def get_status(self) -> Dict[str, Any]:
        """獲取 AI 服務狀態"""
        # 重置每日計數
        await self._reset_daily_count_if_needed()

        next_report_at = None
        if self._scheduler_running:
            now = datetime.now(TW_TZ)
            next_run = now.replace(hour=settings.AI_DAILY_REPORT_HOUR, minute=0, second=0, microsecond=0)
            if now >= next_run:
                next_run += timedelta(days=1)
            next_report_at = next_run.isoformat()

        return {
            "enabled": self._enabled,
            "initialized": self._initialized,
            "model": settings.AI_MODEL if self._enabled else None,
            "scheduler_running": self._scheduler_running,
            "next_report_at": next_report_at,
            "tokens_used_today": self._tokens_used_today,
            "daily_token_budget": settings.AI_DAILY_TOKEN_BUDGET,
            "budget_remaining": settings.AI_DAILY_TOKEN_BUDGET - self._tokens_used_today
        }

    async def _build_context(
        self,
        include_positions: bool = True,
        include_pnl: bool = True,
        include_arena: bool = True
    ) -> tuple[Dict[str, Any], List[str]]:
        """構建數據上下文"""
        context = {
            "timestamp": datetime.now(TW_TZ).isoformat(),
            "trading_enabled": settings.TRADING_ENABLED,
            "dry_run": settings.DRY_RUN,
        }
        sources = []

        # PnL 數據
        if include_pnl:
            try:
                from src.services.pnl_tracker import pnl_tracker
                pnl_summary = await pnl_tracker.get_total_pnl()
                context["pnl"] = {
                    "total_realized": pnl_summary.get("total_realized_pnl", 0),
                    "total_unrealized": pnl_summary.get("total_unrealized_pnl", 0),
                    "today_pnl": pnl_summary.get("today_pnl", 0),
                    "win_rate": pnl_summary.get("win_rate", 0),
                    "trade_count": pnl_summary.get("trade_count", 0),
                }
                sources.append("pnl")
            except Exception as e:
                logger.warning(f"Failed to get PnL data: {e}")
                context["pnl"] = {"error": str(e)}

        # 持倉數據
        if include_positions:
            try:
                from src.services.pnl_tracker import pnl_tracker
                positions_data = await pnl_tracker.get_unrealized_pnl()
                context["positions"] = positions_data.get("positions", [])[:10]  # 最多 10 個
                sources.append("positions")
            except Exception as e:
                logger.warning(f"Failed to get positions data: {e}")
                context["positions"] = []

        # Arena 數據
        if include_arena:
            try:
                from src.services.arena_service import arena_service
                arena_status = await arena_service.get_arena_status()
                context["arena"] = arena_status.to_dict() if arena_status else {}
                sources.append("arena")
            except Exception as e:
                logger.warning(f"Failed to get arena data: {e}")
                context["arena"] = {"error": str(e)}

        return context, sources

    async def _check_budget(self, estimated_tokens: int) -> bool:
        """檢查是否有足夠預算"""
        await self._reset_daily_count_if_needed()
        return (self._tokens_used_today + estimated_tokens) <= settings.AI_DAILY_TOKEN_BUDGET

    async def _update_token_usage(self, tokens: int):
        """更新 token 使用量"""
        await self._reset_daily_count_if_needed()
        self._tokens_used_today += tokens

    async def _reset_daily_count_if_needed(self):
        """如需要則重置每日計數"""
        today = date.today()
        if self._last_token_reset != today:
            self._tokens_used_today = 0
            self._last_token_reset = today

    async def _log_usage(self, endpoint: str, input_tokens: int, output_tokens: int):
        """記錄 API 使用"""
        try:
            from src.utils.database import async_db_manager

            await async_db_manager.execute(
                """
                INSERT INTO ai_usage_log (endpoint, tokens_input, tokens_output, model)
                VALUES ($1, $2, $3, $4)
                """,
                endpoint, input_tokens, output_tokens, settings.AI_MODEL
            )
        except Exception as e:
            # 表可能不存在，忽略錯誤
            logger.debug(f"Failed to log AI usage: {e}")

    async def _cache_daily_report(self, report_date: date, report: str):
        """緩存每日報告"""
        try:
            from src.utils.database import async_db_manager

            await async_db_manager.execute(
                """
                INSERT INTO ai_daily_reports (date, report, tokens_used, generated_at)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                ON CONFLICT (date) DO UPDATE SET
                    report = EXCLUDED.report,
                    tokens_used = EXCLUDED.tokens_used,
                    generated_at = CURRENT_TIMESTAMP
                """,
                report_date, report, self._tokens_used_today
            )
        except Exception as e:
            # 表可能不存在，忽略錯誤
            logger.debug(f"Failed to cache daily report: {e}")

    async def get_cached_report(self, report_date: date) -> Optional[str]:
        """獲取緩存的報告"""
        try:
            from src.utils.database import async_db_manager

            row = await async_db_manager.fetch_one(
                "SELECT report FROM ai_daily_reports WHERE date = $1",
                report_date
            )
            return row["report"] if row else None
        except Exception as e:
            logger.debug(f"Failed to get cached report: {e}")
            return None


# 全局單例
ai_service = AIService()
