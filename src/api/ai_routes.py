"""
AI 助手 API 路由

端點：
- POST /api/v1/ai/chat      - 即時對話
- GET  /api/v1/ai/report/daily - 獲取每日報告
- GET  /api/v1/ai/status    - 服務狀態
"""

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date, datetime

from src.logger import logger
from src.services.ai_service import ai_service


router = APIRouter(prefix="/ai", tags=["AI Assistant"])


# ==================== Request/Response Models ====================

class ChatRequest(BaseModel):
    """聊天請求"""
    message: str = Field(..., min_length=1, max_length=2000, description="用戶消息")
    include_positions: bool = Field(default=True, description="是否包含持倉數據")
    include_pnl: bool = Field(default=True, description="是否包含盈虧數據")
    include_arena: bool = Field(default=True, description="是否包含 Arena 數據")


class ChatResponse(BaseModel):
    """聊天響應"""
    success: bool
    response: str
    tokens_used: int
    context_sources: List[str]


class DailyReportResponse(BaseModel):
    """每日報告響應"""
    success: bool
    date: str
    report: Optional[str] = None
    generated_at: Optional[str] = None
    cached: bool = False
    message: Optional[str] = None


class AIStatusResponse(BaseModel):
    """AI 服務狀態響應"""
    enabled: bool
    initialized: bool
    model: Optional[str] = None
    scheduler_running: bool
    next_report_at: Optional[str] = None
    tokens_used_today: int
    daily_token_budget: int
    budget_remaining: int
    # 調試信息
    config_ai_enabled: Optional[bool] = None
    config_has_api_key: Optional[bool] = None


# ==================== Endpoints ====================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    與 AI 助手對話

    發送消息給 AI 助手，獲取交易相關的分析和建議。
    AI 會自動整合當前的持倉、盈虧和 Arena 數據作為上下文。

    **Token 預算**：每日有使用上限，超過後需等待次日重置。

    **示例問題**：
    - 今天交易狀況如何？
    - 分析一下我的 BTC 持倉
    - Arena 競技場目前是什麼狀態？
    - 最近的勝率是多少？
    """
    try:
        result = await ai_service.chat(
            message=request.message,
            include_positions=request.include_positions,
            include_pnl=request.include_pnl,
            include_arena=request.include_arena
        )

        return ChatResponse(
            success=result["success"],
            response=result["response"],
            tokens_used=result["tokens_used"],
            context_sources=result["context_sources"]
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}"
        )


@router.get("/report/daily", response_model=DailyReportResponse)
async def get_daily_report(
    date_str: Optional[str] = Query(
        default=None,
        alias="date",
        description="報告日期 (YYYY-MM-DD)，預設為今天"
    ),
    regenerate: bool = Query(
        default=False,
        description="是否強制重新生成（忽略緩存）"
    )
):
    """
    獲取每日交易報告

    返回指定日期的交易報告。如果已有緩存會直接返回，
    除非設置 regenerate=true 強制重新生成。

    **注意**：生成報告會消耗 token 預算。
    """
    try:
        # 解析日期
        if date_str:
            try:
                report_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid date format. Use YYYY-MM-DD"
                )
        else:
            report_date = date.today()

        # 檢查緩存
        if not regenerate:
            cached_report = await ai_service.get_cached_report(report_date)
            if cached_report:
                return DailyReportResponse(
                    success=True,
                    date=report_date.isoformat(),
                    report=cached_report,
                    cached=True,
                    message="Report retrieved from cache"
                )

        # 生成新報告
        report = await ai_service.generate_daily_report(report_date)

        if not report:
            status_info = await ai_service.get_status()
            if not status_info["enabled"]:
                return DailyReportResponse(
                    success=False,
                    date=report_date.isoformat(),
                    message="AI service is not enabled"
                )
            elif status_info["budget_remaining"] <= 0:
                return DailyReportResponse(
                    success=False,
                    date=report_date.isoformat(),
                    message="Daily token budget exhausted"
                )
            else:
                return DailyReportResponse(
                    success=False,
                    date=report_date.isoformat(),
                    message="Failed to generate report"
                )

        return DailyReportResponse(
            success=True,
            date=report_date.isoformat(),
            report=report,
            generated_at=datetime.now().isoformat(),
            cached=False,
            message="Report generated successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Daily report endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get daily report: {str(e)}"
        )


@router.get("/status", response_model=AIStatusResponse)
async def get_ai_status():
    """
    獲取 AI 服務狀態

    返回 AI 服務的當前狀態，包括：
    - 是否啟用
    - 使用的模型
    - 定時任務狀態
    - Token 使用情況
    """
    try:
        from src.config import settings

        status_info = await ai_service.get_status()

        return AIStatusResponse(
            enabled=status_info["enabled"],
            initialized=status_info["initialized"],
            model=status_info["model"],
            scheduler_running=status_info["scheduler_running"],
            next_report_at=status_info["next_report_at"],
            tokens_used_today=status_info["tokens_used_today"],
            daily_token_budget=status_info["daily_token_budget"],
            budget_remaining=status_info["budget_remaining"],
            # 調試信息：顯示配置文件中的實際值
            config_ai_enabled=settings.AI_ENABLED,
            config_has_api_key=bool(settings.ANTHROPIC_API_KEY)
        )

    except Exception as e:
        logger.error(f"AI status endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get AI status: {str(e)}"
        )


@router.post("/reinit")
async def reinitialize_ai_service():
    """
    重新初始化 AI 服務

    當環境變數更新後，可以調用此端點重新初始化 AI 服務，
    無需重新部署整個應用程式。
    """
    try:
        from src.config import settings

        # 檢查 anthropic 套件是否可用
        anthropic_available = False
        anthropic_error = None
        try:
            import anthropic
            anthropic_available = True
        except ImportError as e:
            anthropic_error = str(e)

        # 強制重新初始化
        ai_service._initialized = False
        ai_service._enabled = False
        await ai_service.init()

        # 如果啟用成功且配置允許，啟動調度器
        if ai_service._enabled and settings.AI_ENABLED:
            await ai_service.start_scheduler()

        status_info = await ai_service.get_status()

        return {
            "success": True,
            "message": "AI service reinitialized",
            "enabled": status_info["enabled"],
            "initialized": status_info["initialized"],
            "config_ai_enabled": settings.AI_ENABLED,
            "config_has_api_key": bool(settings.ANTHROPIC_API_KEY),
            "anthropic_available": anthropic_available,
            "anthropic_error": anthropic_error
        }

    except Exception as e:
        logger.error(f"AI reinit endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reinitialize AI service: {str(e)}"
        )


@router.post("/report/send")
async def send_daily_report_now():
    """
    立即發送每日報告到 Telegram

    手動觸發每日報告的生成和發送。
    通常用於測試或需要立即獲取報告的情況。
    """
    try:
        status_info = await ai_service.get_status()

        if not status_info["enabled"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="AI service is not enabled"
            )

        # 生成報告
        report = await ai_service.generate_daily_report()

        if not report:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate report"
            )

        # 發送到 Telegram
        telegram_sent = False
        try:
            from src.services.telegram_notifier import telegram_notifier
            if telegram_notifier.enabled:
                await telegram_notifier.send_message(report)
                telegram_sent = True
        except Exception as e:
            logger.warning(f"Failed to send to Telegram: {e}")

        return {
            "success": True,
            "message": "Report generated and sent",
            "telegram_sent": telegram_sent,
            "report_preview": report[:500] + "..." if len(report) > 500 else report
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Send report endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send report: {str(e)}"
        )
