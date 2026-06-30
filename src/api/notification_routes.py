"""
Notification API Routes - 通知系統 API 路由
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from src.services.notification_service import notification_service, NotificationType
from src.logger import logger


router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])


class NotificationPreferences(BaseModel):
    """通知偏好設定"""
    trade_notifications: Optional[bool] = None
    risk_alerts: Optional[bool] = None
    model_updates: Optional[bool] = None
    performance_reports: Optional[bool] = None
    price_alerts: Optional[bool] = None
    quiet_hours_start: Optional[str] = None
    quiet_hours_end: Optional[str] = None


class PriceAlertRequest(BaseModel):
    """價格警報請求"""
    symbol: str
    condition: str  # "above", "below"
    target_price: float


class TestNotificationRequest(BaseModel):
    """測試通知請求"""
    title: str = "測試通知"
    message: str = "這是一條測試通知"


@router.get("/history")
async def get_notification_history(
    notification_type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(50, ge=1, le=200)
) -> Dict[str, Any]:
    """獲取通知歷史"""
    try:
        ntype = None
        if notification_type:
            try:
                ntype = NotificationType(notification_type)
            except ValueError:
                pass

        history = notification_service.get_notification_history(
            notification_type=ntype,
            limit=limit
        )

        return {
            "success": True,
            "count": len(history),
            "notifications": history
        }
    except Exception as e:
        logger.error(f"Failed to get notification history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preferences")
async def get_notification_preferences() -> Dict[str, Any]:
    """獲取通知偏好設定"""
    return {
        "success": True,
        "preferences": notification_service.preferences
    }


@router.post("/preferences")
async def update_notification_preferences(
    prefs: NotificationPreferences
) -> Dict[str, Any]:
    """更新通知偏好設定"""
    try:
        updates = prefs.dict(exclude_none=True)
        notification_service.update_preferences(**updates)

        return {
            "success": True,
            "message": "Preferences updated",
            "preferences": notification_service.preferences
        }
    except Exception as e:
        logger.error(f"Failed to update preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/price-alerts")
async def get_price_alerts(
    symbol: Optional[str] = Query(None, description="Filter by symbol")
) -> Dict[str, Any]:
    """獲取價格警報列表"""
    try:
        alerts = notification_service.get_price_alerts(symbol)

        total_count = sum(len(v) for v in alerts.values())

        return {
            "success": True,
            "count": total_count,
            "alerts": alerts
        }
    except Exception as e:
        logger.error(f"Failed to get price alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/price-alerts")
async def add_price_alert(request: PriceAlertRequest) -> Dict[str, Any]:
    """添加價格警報"""
    try:
        if request.condition not in ["above", "below"]:
            raise HTTPException(
                status_code=400,
                detail="Condition must be 'above' or 'below'"
            )

        alert = notification_service.add_price_alert(
            symbol=request.symbol,
            condition=request.condition,
            target_price=request.target_price
        )

        return {
            "success": True,
            "message": f"Price alert added: {request.symbol} {request.condition} {request.target_price}",
            "alert": {
                "symbol": alert.symbol,
                "condition": alert.condition,
                "target_price": alert.target_price,
                "created_at": alert.created_at
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add price alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/price-alerts/{symbol}")
async def remove_price_alert(
    symbol: str,
    index: int = Query(0, ge=0, description="Alert index to remove")
) -> Dict[str, Any]:
    """移除價格警報"""
    try:
        success = notification_service.remove_price_alert(symbol, index)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"No price alert found for {symbol} at index {index}"
            )

        return {
            "success": True,
            "message": f"Price alert removed for {symbol}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove price alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test")
async def send_test_notification(request: TestNotificationRequest) -> Dict[str, Any]:
    """發送測試通知"""
    try:
        from src.services.notification_service import Notification, NotificationPriority

        notification = Notification(
            type=NotificationType.SYSTEM,
            priority=NotificationPriority.NORMAL,
            title=request.title,
            message=request.message,
            data={"test": True}
        )

        success = await notification_service.send_notification(notification, force=True)

        return {
            "success": success,
            "message": "Test notification sent" if success else "Failed to send notification",
            "telegram_enabled": notification_service.enabled
        }
    except Exception as e:
        logger.error(f"Failed to send test notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_notification_status() -> Dict[str, Any]:
    """獲取通知系統狀態"""
    return {
        "success": True,
        "status": {
            "telegram_enabled": notification_service.enabled,
            "preferences": notification_service.preferences,
            "history_count": len(notification_service._notification_history),
            "active_price_alerts": sum(
                len(alerts) for alerts in notification_service._price_alerts.values()
            )
        }
    }
