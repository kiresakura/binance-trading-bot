"""FastAPI 应用主文件"""

import asyncio
import os
from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from decimal import Decimal
from datetime import datetime
import uuid

from src.config import settings
from src.logger import logger
from src.services.cache_service import cache_service
from src.services.monitoring import metrics_collector
from src.services.runtime_settings import runtime_settings
from src.services.risk_manager import Order, risk_management_system
from src.services.order_executor import smart_order_executor
from src.services.audit_logger import audit_logger
from src.api import history_routes, market_data_routes
from src.services.fault_tolerance import fault_tolerant_engine
from src.services.binance_client import binance_client
from src.api.websocket_manager import ws_manager, MessageType, WSMessage
from src.services.market_publisher import market_publisher
from src.api.arena_routes import router as arena_router
from src.api.market_regime_routes import router as market_regime_router
from src.api.backtest_routes import router as backtest_router
from src.api.adaptive_routes import router as adaptive_router
from src.api.strategy_routes import router as strategy_router
from src.api.model_routes import router as model_router, ml_router
from src.api.pnl_routes import router as pnl_router
from src.api.virtual_trading_routes import router as virtual_trading_router
from src.api.signal_routes import router as signal_router
from src.services.pnl_tracker import pnl_tracker
from src.api.auth import (
    get_current_user, get_optional_user, require_scope,
    TokenData, RequireTrading, RequireAdmin, RequireAuth, OptionalAuth,
    create_access_token
)
from src.api.rate_limiter import RateLimitMiddleware, rate_limiter
from src.api.request_logger import RequestLoggingMiddleware, request_logger
from src.services.telegram_notifier import telegram_notifier


# ==================== Pydantic 模型 ====================

class OrderRequest(BaseModel):
    """订单请求模型"""
    symbol: str = Field(..., example="BTCUSDT")
    side: str = Field(..., example="BUY", description="BUY or SELL")
    quantity: Decimal = Field(..., example="0.5", gt=0)
    price: Optional[Decimal] = Field(None, example="42000")
    order_type: str = Field("MARKET", example="MARKET")
    strategy: str = Field("adaptive", example="adaptive")


class OrderResponse(BaseModel):
    """订单响应模型（iOS 兼容格式）"""
    # iOS App 期待的欄位
    orderId: Optional[str] = None
    symbol: str = ""
    side: str = ""
    orderType: str = "MARKET"
    quantity: float = 0.0
    price: Optional[float] = None
    status: str = ""
    createdAt: Optional[str] = None

    # 原有欄位（保持向後兼容）
    success: bool
    order_id: Optional[str] = None  # 別名，與 orderId 相同
    message: str
    risk_score: Optional[float] = None
    executed_results: Optional[List[dict]] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    environment: str
    components: dict


class RiskValidationResponse(BaseModel):
    """风险验证响应"""
    is_valid: bool
    risk_score: float
    reason: Optional[str]
    warnings: List[str]


class TradingSettingsRequest(BaseModel):
    """交易設定請求"""
    trading_enabled: Optional[bool] = Field(None, description="是否啟用交易")
    dry_run: Optional[bool] = Field(None, description="是否為模擬模式")


class TradingSettingsResponse(BaseModel):
    """交易設定響應"""
    trading_enabled: bool
    dry_run: bool
    testnet: bool
    trading_pairs: List[str]
    default_trading_pair: str
    max_position_size: float
    max_daily_loss: float


# ==================== FastAPI 应用 ====================

# 正式環境關閉互動式 API 文件與 OpenAPI schema（縮小攻擊面）；
# 非 production 才開放 /docs、/redoc。
_IS_PRODUCTION = settings.ENVIRONMENT == "production"
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="企业级加密货币自动交易系统",
    docs_url=None if _IS_PRODUCTION else "/docs",
    redoc_url=None if _IS_PRODUCTION else "/redoc",
    openapi_url=None if _IS_PRODUCTION else "/openapi.json",
)

# CORS 配置 - 允許的來源由環境變數 CORS_ALLOWED_ORIGINS 提供（逗號分隔）
# 預設僅本地開發來源；正式環境請明確設定自己的前端網域，切勿把網域寫死在原始碼中。
_DEFAULT_CORS_ORIGINS = (
    "http://localhost:8080,http://localhost:3000,"
    "http://127.0.0.1:8080,http://127.0.0.1:3000"
)
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# 添加 Rate Limiting 中間件（必須在 CORS 之後）
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

# 添加 Request Logging 中間件
app.add_middleware(RequestLoggingMiddleware, request_logger=request_logger)

# 掛載 Arena 系統路由
app.include_router(arena_router, prefix="/api/v1")

# 掛載 Market Regime 路由
app.include_router(market_regime_router, prefix="/api/v1")

# 掛載 Backtest 和 ML 路由
app.include_router(backtest_router, prefix="/api/v1")

# 掛載 Adaptive Training 和 Hyperopt 路由 (v0.3.0)
app.include_router(adaptive_router)

# 掛載 History 和 Market Data 路由
app.include_router(history_routes.router)
app.include_router(market_data_routes.router)

# Prometheus 指標路由
app.include_router(model_router)

# iOS 本地訓練模型上傳路由
app.include_router(ml_router)

# 掛載 PnL Tracker 路由 (v0.2.1 - 盈虧追蹤系統)
app.include_router(pnl_router, prefix="/api/v1")

# 掛載虛擬交易路由 (競技場模擬交易)
app.include_router(virtual_trading_router, prefix="/api/v1")

# 掛載策略超市路由 (v0.4.0 - Enterprise Edition)
app.include_router(strategy_router)

# 掛載 AI 助手路由 (Claude API 整合)
from src.api.ai_routes import router as ai_router
app.include_router(ai_router, prefix="/api/v1")

# 掛載通知系統路由
from src.api.notification_routes import router as notification_router
app.include_router(notification_router)

# 掛載策略信號路由 (v0.5.0 - iOS 視覺化支持)
app.include_router(signal_router, prefix="/api/v1")


# ==================== 启动和关闭事件 ====================

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    # 安全檢查：production 環境缺少必要設定（如 SECRET_KEY）時，拒絕啟動。
    if settings.ENVIRONMENT == "production":
        from src.config import validate_settings
        validate_settings()

    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")

    # 啟動 Rate Limiter 清理任務
    try:
        await rate_limiter.start()
        logger.info("✅ Rate limiter initialized")
    except Exception as e:
        logger.warning(f"⚠️ Rate limiter startup error: {e}")

    # 初始化 Telegram 通知服務
    try:
        await telegram_notifier.init()
        if telegram_notifier.enabled:
            logger.info("✅ Telegram notifier initialized")
            # 發送啟動通知
            await telegram_notifier.notify_system_event(
                event="Trading Bot Started",
                level="info",
                details=f"Version: {settings.VERSION}, Environment: {settings.ENVIRONMENT}"
            )
        else:
            logger.info("ℹ️ Telegram notifier is disabled")
    except Exception as e:
        logger.warning(f"⚠️ Telegram notifier startup error: {e}")

    # 初始化容错引擎（可選，失敗不阻止啟動）
    try:
        await fault_tolerant_engine.init()
        logger.info("✅ Fault-tolerant engine initialized")
    except Exception as e:
        logger.warning(f"⚠️ Fault-tolerant engine not available: {e}")

    # 更新系统监控指标
    try:
        metrics_collector.update_system_metrics()
        logger.info("✅ Metrics collector initialized")
    except Exception as e:
        logger.warning(f"⚠️ Metrics collector error: {e}")

    # 初始化 Arena 服務
    try:
        from src.services.arena_service import arena_service
        await arena_service.initialize()
        logger.info("✅ Arena service initialized")
    except Exception as e:
        logger.warning(f"⚠️ Arena service not available: {e}")

    # 🔥 啟動 Data Engine（收集市場數據）
    try:
        from src.engines import data_engine
        await data_engine.start()
        logger.info("✅ Data engine started - collecting market data")
    except Exception as e:
        logger.warning(f"⚠️ Data engine start failed: {e}")

    # 自動啟動 Trading Engine（根本性解決方案）
    try:
        from src.engines import trading_engine
        await trading_engine.start()
        logger.info("✅ Trading engine auto-started")
    except Exception as e:
        logger.warning(f"⚠️ Trading engine auto-start failed: {e}")

    # 🔥 啟動 PnL Tracker 定時任務調度器
    try:
        await pnl_tracker.init()
        await pnl_tracker.start_scheduler()
        logger.info("✅ PnL tracker scheduler started")
    except Exception as e:
        logger.warning(f"⚠️ PnL tracker scheduler failed: {e}")

    # 初始化 Cache Service (Redis)
    try:
        await cache_service.init()
    except Exception as e:
        logger.warning(f"⚠️ Cache service init failed: {e}")

    # 🤖 初始化 AI 助手服務 (Claude API)
    try:
        from src.services.ai_service import ai_service
        await ai_service.init()
        if settings.AI_ENABLED:
            await ai_service.start_scheduler()
            logger.info("✅ AI assistant service started with daily report scheduler")
        else:
            logger.info("ℹ️ AI assistant service initialized (scheduler disabled)")
    except Exception as e:
        logger.warning(f"⚠️ AI assistant service init failed: {e}")

    # 🎯 啟動影子交易執行器
    try:
        from src.services.shadow_trading_executor import shadow_trading_executor
        await shadow_trading_executor.start()
        logger.info("✅ Shadow trading executor started")
    except Exception as e:
        logger.warning(f"⚠️ Shadow trading executor start failed: {e}")

    logger.info("🚀 Application startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("Shutting down application...")

    try:
        # 發送關閉通知
        if telegram_notifier.enabled:
            await telegram_notifier.notify_system_event(
                event="Trading Bot Stopped",
                level="info",
                details="Application shutting down"
            )
            # 等待消息發送
            await asyncio.sleep(1)

        # 停止 Telegram 通知服務
        await telegram_notifier.stop()
        logger.info("✅ Telegram notifier stopped")

        # 停止 Rate Limiter
        await rate_limiter.stop()
        logger.info("✅ Rate limiter stopped")

        # 停止 PnL Tracker 調度器
        await pnl_tracker.stop_scheduler()
        logger.info("✅ PnL tracker scheduler stopped")

        # 停止 AI 助手調度器
        try:
            from src.services.ai_service import ai_service
            await ai_service.stop_scheduler()
            logger.info("✅ AI assistant scheduler stopped")
        except Exception as e:
            logger.warning(f"⚠️ AI assistant stop failed: {e}")

        # 停止影子交易執行器
        try:
            from src.services.shadow_trading_executor import shadow_trading_executor
            await shadow_trading_executor.stop()
            logger.info("✅ Shadow trading executor stopped")
        except Exception as e:
            logger.warning(f"⚠️ Shadow trading executor stop failed: {e}")

        # 關閉市場數據廣播
        await market_publisher.stop()
        logger.info("✅ Market publisher stopped")

        await cache_service.close()
        logger.info("✅ Cache service closed")

        await fault_tolerant_engine.close()
        logger.info("✅ Fault-tolerant engine closed")

        logger.info("👋 Application shutdown complete")

    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")


# ==================== 健康检查端点 ====================

@app.get("/", response_model=dict)
async def root():
    """根端点"""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查 - Railway 兼容版本，即使部分組件不可用也返回 healthy"""
    components = {
        "api": "healthy",
        "database": "not_configured",
        "cache": "not_configured",
        "fault_tolerance": "not_configured",
        "prometheus": "healthy"
    }

    # 检查数据库（可選）
    db_error = None
    try:
        from src.utils.database import async_db_manager
        # 確保資料庫已初始化
        if not async_db_manager._initialized:
            await async_db_manager.init()
        db_healthy = await async_db_manager.health_check()
        components["database"] = "healthy" if db_healthy else "degraded"
    except Exception as e:
        db_error = str(e)
        logger.warning(f"Database health check failed: {e}")
        components["database"] = "not_configured"

    # 检查容错引擎（可選）- 與資料庫狀態連動
    try:
        # 如果資料庫 healthy，容錯也設為 healthy（因為容錯現在依賴資料庫）
        if components["database"] == "healthy":
            if fault_tolerant_engine.is_emergency_stopped:
                components["fault_tolerance"] = "emergency_stop"
            else:
                components["fault_tolerance"] = "healthy"
        elif components["database"] == "degraded":
            components["fault_tolerance"] = "degraded"
        else:
            # 檢查是否有 Redis 作為備用
            has_redis = fault_tolerant_engine._state_manager.redis_client is not None
            if has_redis:
                components["fault_tolerance"] = "degraded"
            else:
                components["fault_tolerance"] = "not_configured"
    except Exception as e:
        logger.debug(f"Fault tolerance check failed: {e}")
        components["fault_tolerance"] = "not_configured"

    # 檢查 Cache Service
    if cache_service._initialized:
        components["cache"] = "healthy"
    else:
        components["cache"] = "not_configured"

    # API 本身運行即為 healthy（Railway healthcheck 需要）
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        components=components
    )


@app.get("/debug/db")
async def debug_db(user: TokenData = RequireAdmin):
    """Debug database connection（需 admin 權限；正式環境避免洩漏設定）"""
    from src.config import settings
    result = {
        "database_url_set": bool(settings.DATABASE_URL),
        "database_url_preview": settings.DATABASE_URL[:50] + "..." if settings.DATABASE_URL else None,
        "should_use_sqlite": settings.should_use_sqlite,
        "use_sqlite_flag": settings.USE_SQLITE,
        "db_host": settings.DB_HOST,
        "db_port": settings.DB_PORT,
        "db_name": settings.DB_NAME,
        "db_user": settings.DB_USER,
        "db_password_set": bool(settings.DB_PASSWORD),
        "redis_url_set": bool(settings.REDIS_URL),
    }
    try:
        from src.utils.database import async_db_manager
        if not async_db_manager._initialized:
            await async_db_manager.init()
        db_healthy = await async_db_manager.health_check()
        result["connection"] = "success" if db_healthy else "failed"
    except Exception as e:
        result["connection"] = "error"
        result["error"] = str(e)
    return result


# ==================== Prometheus 指标导出 ====================

@app.get("/metrics")
async def metrics():
    """Prometheus 指标导出"""
    # 更新系统指标
    metrics_collector.update_system_metrics()

    # 生成 Prometheus 格式的指标
    metrics_data = metrics_collector.get_metrics()

    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )


# ==================== 交易端点 ====================

@app.get("/api/v1/portfolio/summary")
async def get_portfolio_summary(user: TokenData = RequireAuth):
    """獲取投資組合摘要（需認證；含帳戶餘額）"""
    try:
        # 獲取帳戶餘額
        balances = await binance_client.get_balances()
        
        # 計算總價值（嘗試獲取 USDT 價值）
        total_usdt_value = Decimal('0')
        portfolio_items = []
        
        for balance in balances:
            asset = balance["asset"]
            total = Decimal(balance["total"])
            free = Decimal(balance["free"])
            locked = Decimal(balance["locked"])

            # 獲取價格
            usdt_value = Decimal('0')
            price = Decimal('0')

            # 處理活期理財代幣 (LD 開頭的是 Flexible Savings)
            display_asset = asset
            base_asset = asset
            asset_type = "spot"

            if asset.startswith("LD"):
                # LDUSDT -> USDT, LDBTC -> BTC, LDRVN -> RVN
                base_asset = asset[2:]  # 移除 "LD" 前綴
                asset_type = "flexible_savings"

            # 計算 USDT 價值
            if base_asset == "USDT":
                usdt_value = total
                price = Decimal('1')
            else:
                # 嘗試獲取價格
                try:
                    ticker = await binance_client.get_ticker_price(f"{base_asset}USDT")
                    if isinstance(ticker, dict) and "price" in ticker:
                        price = Decimal(ticker["price"])
                        usdt_value = total * price
                except:
                    # 如果無法獲取價格，嘗試其他交易對
                    try:
                        # 嘗試 BTC 計價
                        ticker = await binance_client.get_ticker_price(f"{base_asset}BTC")
                        if isinstance(ticker, dict) and "price" in ticker:
                            btc_price = Decimal(ticker["price"])
                            btc_usdt = await binance_client.get_ticker_price("BTCUSDT")
                            if isinstance(btc_usdt, dict) and "price" in btc_usdt:
                                price = btc_price * Decimal(btc_usdt["price"])
                                usdt_value = total * price
                    except:
                        pass

            total_usdt_value += usdt_value

            portfolio_items.append({
                "asset": display_asset,
                "base_asset": base_asset,
                "asset_type": asset_type,
                "free": str(free),
                "locked": str(locked),
                "total": str(total),
                "price_usdt": str(price),
                "value_usdt": str(usdt_value)
            })
        
        return {
            "success": True,
            "total_value_usdt": str(total_usdt_value),
            "assets": portfolio_items,
            "asset_count": len(portfolio_items)
        }
        
    except Exception as e:
        logger.error(f"Failed to get portfolio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get portfolio: {str(e)}"
        )


@app.get("/api/v1/orders")
async def get_orders(symbol: str = "BTCUSDT", limit: int = 20, user: TokenData = RequireAuth):
    """獲取訂單列表（需認證）"""
    try:
        # 獲取歷史訂單
        orders = await binance_client.get_all_orders(symbol=symbol, limit=limit)
        
        # 格式化訂單數據
        formatted_orders = []
        for order in orders:
            formatted_orders.append({
                "order_id": str(order.get("orderId", "")),
                "symbol": order.get("symbol", ""),
                "side": order.get("side", ""),
                "type": order.get("type", ""),
                "quantity": order.get("origQty", "0"),
                "price": order.get("price", "0"),
                "executed_qty": order.get("executedQty", "0"),
                "status": order.get("status", ""),
                "time": order.get("time", 0),
                "update_time": order.get("updateTime", 0)
            })
        
        return {
            "success": True,
            "symbol": symbol,
            "count": len(formatted_orders),
            "orders": formatted_orders
        }
        
    except Exception as e:
        logger.error(f"Failed to get orders: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get orders: {str(e)}"
        )


@app.get("/api/v1/orders/open")
async def get_open_orders(symbol: str = None, user: TokenData = RequireAuth):
    """獲取當前掛單（需認證）"""
    try:
        orders = await binance_client.get_open_orders(symbol=symbol)
        
        return {
            "success": True,
            "count": len(orders),
            "orders": orders
        }
        
    except Exception as e:
        logger.error(f"Failed to get open orders: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get open orders: {str(e)}"
        )


@app.get("/api/v1/market/price")
async def get_market_price(symbol: str = None):
    """獲取市場價格"""
    try:
        price = await binance_client.get_ticker_price(symbol=symbol)
        
        return {
            "success": True,
            "data": price
        }
        
    except Exception as e:
        logger.error(f"Failed to get market price: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get market price: {str(e)}"
        )


@app.get("/api/v1/market/ticker24h")
async def get_ticker_24h(symbol: str = None):
    """獲取 24 小時行情"""
    try:
        ticker = await binance_client.get_ticker_24h(symbol=symbol)
        
        return {
            "success": True,
            "data": ticker
        }
        
    except Exception as e:
        logger.error(f"Failed to get 24h ticker: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get 24h ticker: {str(e)}"
        )


@app.post("/api/v1/orders", response_model=OrderResponse)
async def create_order(
    order_request: OrderRequest,
    user: TokenData = RequireTrading
):
    """创建订单（需 trading 權限）

    执行流程：
    1. 风险验证（5层防御）
    2. 智能执行（自动分割+容错）
    3. 审计日志记录
    4. 监控指标更新

    認證：需以 Bearer JWT 或 X-API-Key 提供具 trading 權限的憑證。
    """
    from datetime import datetime

    try:
        logger.info(f"Received order request: {order_request.symbol} {order_request.side} {order_request.quantity}")

        # 1. 创建订单对象
        order = Order(
            symbol=order_request.symbol,
            side=order_request.side,
            quantity=order_request.quantity,
            price=order_request.price,
            order_type=order_request.order_type
        )

        # 2. 风险验证（SELL 訂單 = 平倉，繞過風險檢查）
        is_close_position = order_request.side.upper() == "SELL"

        if is_close_position:
            # 平倉操作不需要風險驗證
            logger.info(f"Close position order - bypassing risk check for {order_request.symbol}")
            validation = type('ValidationResult', (), {
                'is_valid': True,
                'reason': None,
                'risk_score': 0.0,
                'warnings': []
            })()
        else:
            # 開倉需要風險驗證
            validation = await risk_management_system.validate_order(order)

        if not validation.is_valid:
            # 记录风险拒绝
            await audit_logger.log_risk_rejection(
                symbol=order.symbol,
                reason=validation.reason,
                risk_score=validation.risk_score,
                attempted_quantity=order.quantity,
                attempted_side=order.side
            )

            # 更新监控指标
            metrics_collector.record_risk_rejection(validation.reason)

            logger.warning(f"Order rejected by risk system: {validation.reason}")

            return OrderResponse(
                # iOS 欄位
                orderId=None,
                symbol=order_request.symbol,
                side=order_request.side,
                orderType=order_request.order_type,
                quantity=float(order_request.quantity),
                price=float(order_request.price) if order_request.price else None,
                status="REJECTED",
                createdAt=datetime.utcnow().isoformat(),
                # 原有欄位
                success=False,
                order_id=None,
                message=f"Order rejected: {validation.reason}",
                risk_score=validation.risk_score,
                executed_results=None
            )

        # 3. 智能执行订单（SELL 訂單已在上面驗證過，跳過重複風險檢查）
        execution_results = await smart_order_executor.execute_order(
            order,
            strategy=order_request.strategy,
            skip_risk_check=is_close_position  # SELL 訂單跳過風險檢查
        )

        # 4. 记录审计日志和指标
        successful_results = []
        for result in execution_results:
            if result.success:
                await audit_logger.log_trade(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=result.executed_quantity,
                    price=result.executed_price,
                    order_id=result.order_id,
                    strategy=order_request.strategy,
                    risk_score=validation.risk_score,
                    commission=result.commission
                )

                metrics_collector.record_trade(
                    symbol=order.symbol,
                    side=order.side,
                    status="FILLED",
                    volume=float(result.total_cost),
                    latency=0.5,  # TODO: 实际测量
                    slippage=0.002  # TODO: 实际计算
                )

                successful_results.append({
                    "order_id": result.order_id,
                    "executed_price": float(result.executed_price),
                    "executed_quantity": float(result.executed_quantity),
                    "total_cost": float(result.total_cost),
                    "commission": float(result.commission)
                })

        if successful_results:
            first_result = successful_results[0]
            return OrderResponse(
                # iOS 欄位
                orderId=first_result["order_id"],
                symbol=order_request.symbol,
                side=order_request.side,
                orderType=order_request.order_type,
                quantity=first_result["executed_quantity"],
                price=first_result["executed_price"],
                status="FILLED",
                createdAt=datetime.utcnow().isoformat(),
                # 原有欄位
                success=True,
                order_id=first_result["order_id"],
                message="Order executed successfully",
                risk_score=validation.risk_score,
                executed_results=successful_results
            )
        else:
            # 從執行結果中提取錯誤訊息
            error_message = "Order execution failed"
            if execution_results:
                first_result = execution_results[0]
                if hasattr(first_result, 'error_message') and first_result.error_message:
                    error_message = first_result.error_message

            return OrderResponse(
                # iOS 欄位
                orderId=None,
                symbol=order_request.symbol,
                side=order_request.side,
                orderType=order_request.order_type,
                quantity=float(order_request.quantity),
                price=float(order_request.price) if order_request.price else None,
                status="FAILED",
                createdAt=datetime.utcnow().isoformat(),
                # 原有欄位
                success=False,
                order_id=None,
                message=error_message,
                risk_score=validation.risk_score,
                executed_results=None
            )

    except Exception as e:
        logger.error(f"Order creation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Order creation failed: {str(e)}"
        )


@app.post("/api/v1/orders/validate", response_model=RiskValidationResponse)
async def validate_order(order_request: OrderRequest):
    """仅验证订单风险（不执行）"""
    try:
        order = Order(
            symbol=order_request.symbol,
            side=order_request.side,
            quantity=order_request.quantity,
            price=order_request.price,
            order_type=order_request.order_type
        )

        validation = await risk_management_system.validate_order(order)

        return RiskValidationResponse(
            is_valid=validation.is_valid,
            risk_score=validation.risk_score,
            reason=validation.reason,
            warnings=validation.warnings or []
        )

    except Exception as e:
        logger.error(f"Order validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}"
        )


# ==================== 审计日志端点 ====================

@app.get("/api/v1/audit/verify")
async def verify_audit_chain(user: TokenData = RequireAuth):
    """验证审计日志链完整性（需認證）"""
    try:
        is_valid = await audit_logger.verify_chain_integrity()

        return {
            "is_valid": is_valid,
            "message": "Audit log chain is valid" if is_valid else "Audit log chain integrity violated"
        }

    except Exception as e:
        logger.error(f"Audit chain verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {str(e)}"
        )


@app.get("/api/v1/audit/trail")
async def get_audit_trail(
    resource: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 50,
    user: TokenData = RequireAuth
):
    """获取审计跟踪（需認證；含交易歷史）"""
    try:
        trail = await audit_logger.get_audit_trail(
            resource=resource,
            event_type=event_type,
            limit=limit
        )

        return {
            "count": len(trail),
            "trail": trail
        }

    except Exception as e:
        logger.error(f"Failed to get audit trail: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get audit trail: {str(e)}"
        )


# ==================== 系统控制端点 ====================

@app.post("/api/v1/system/emergency-stop")
async def emergency_stop(user: TokenData = Depends(require_scope("trading"))):
    """触发紧急停止"""
    try:
        await fault_tolerant_engine.emergency_shutdown("Manual emergency stop triggered via API")

        return {
            "success": True,
            "message": "Emergency stop activated"
        }

    except Exception as e:
        logger.error(f"Emergency stop failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Emergency stop failed: {str(e)}"
        )


@app.post("/api/v1/system/resume")
async def resume_trading(user: TokenData = RequireTrading):
    """恢复交易（需 trading 權限）"""
    try:
        await fault_tolerant_engine.resume_trading()

        return {
            "success": True,
            "message": "Trading resumed"
        }

    except Exception as e:
        logger.error(f"Resume trading failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume failed: {str(e)}"
        )


# ==================== 交易設定 API ====================

@app.get("/api/v1/settings/trading", response_model=TradingSettingsResponse)
async def get_trading_settings():
    """獲取交易設定"""
    try:
        all_settings = runtime_settings.get_all_settings()
        return TradingSettingsResponse(
            trading_enabled=all_settings["trading_enabled"],
            dry_run=all_settings["dry_run"],
            testnet=all_settings["testnet"],
            trading_pairs=all_settings["trading_pairs"],
            default_trading_pair=all_settings["default_trading_pair"],
            max_position_size=all_settings["max_position_size"],
            max_daily_loss=all_settings["max_daily_loss"]
        )
    except Exception as e:
        logger.error(f"Get trading settings failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get settings: {str(e)}"
        )


@app.put("/api/v1/settings/trading", response_model=TradingSettingsResponse)
async def update_trading_settings(
    request: TradingSettingsRequest,
    user: TokenData = Depends(require_scope("trading"))
):
    """更新交易設定（前端可控制 trading_enabled 和 dry_run）"""
    try:
        # 更新 trading_enabled
        if request.trading_enabled is not None:
            await runtime_settings.set_trading_enabled(
                request.trading_enabled,
                changed_by="api"
            )
            logger.info(f"trading_enabled updated to {request.trading_enabled} via API")

        # 更新 dry_run
        if request.dry_run is not None:
            await runtime_settings.set_dry_run(
                request.dry_run,
                changed_by="api"
            )
            logger.info(f"dry_run updated to {request.dry_run} via API")

        # 返回更新後的設定
        all_settings = runtime_settings.get_all_settings()
        return TradingSettingsResponse(
            trading_enabled=all_settings["trading_enabled"],
            dry_run=all_settings["dry_run"],
            testnet=all_settings["testnet"],
            trading_pairs=all_settings["trading_pairs"],
            default_trading_pair=all_settings["default_trading_pair"],
            max_position_size=all_settings["max_position_size"],
            max_daily_loss=all_settings["max_daily_loss"]
        )
    except Exception as e:
        logger.error(f"Update trading settings failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update settings: {str(e)}"
        )


@app.get("/api/v1/settings/trading/history")
async def get_settings_history(limit: int = 20):
    """獲取設定變更歷史"""
    try:
        history = runtime_settings.get_change_history(limit=limit)
        return {
            "count": len(history),
            "history": history
        }
    except Exception as e:
        logger.error(f"Get settings history failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get history: {str(e)}"
        )


# ==================== 自動交易引擎 API ====================

@app.get("/api/v1/trading/status")
async def get_trading_status():
    """獲取交易引擎狀態"""
    try:
        from src.engines import trading_engine
        from src.services.market_analyzer import market_analyzer

        status = trading_engine.get_status()
        is_running = status.get('running', False)

        # 獲取市場狀態
        # 如果交易引擎運行中，使用其狀態；否則使用實時分析
        if is_running:
            market_state = status.get('strategy_engine', {}).get('current_market_state', 'unknown')
        else:
            # 實時分析市場狀態
            current_symbol = status.get('config', {}).get('symbols', ['BTCUSDT'])[0] if status.get('config', {}).get('symbols') else "BTCUSDT"
            market_data = await market_analyzer.get_market_state(current_symbol)
            market_state = market_data.get('state', 'unknown')

        # 轉換為 iOS App 期望的格式
        return {
            "is_running": is_running,
            "mode": "paper" if status.get('config', {}).get('paper_trading', True) else "live",
            "current_symbol": status.get('config', {}).get('symbols', ['BTCUSDT'])[0] if status.get('config', {}).get('symbols') else "BTCUSDT",
            "position": status.get('trading', {}).get('positions', 0),
            "started_at": None,
            "total_trades": status.get('trading', {}).get('daily_trades', 0),
            "today_pnl": status.get('trading', {}).get('daily_pnl', 0),
            "total_pnl": status.get('trading', {}).get('daily_pnl', 0),
            "active_strategies": status.get('strategy_engine', {}).get('enabled_strategies', []),
            "market_state": market_state,
            "last_signal": None
        }
    except Exception as e:
        logger.warning(f"Get trading status error: {e}")
        return {
            "is_running": False,
            "mode": "paper",
            "current_symbol": "BTCUSDT",
            "position": 0,
            "started_at": None,
            "total_trades": 0,
            "today_pnl": 0,
            "total_pnl": 0,
            "active_strategies": [],
            "market_state": "ranging",
            "last_signal": None
        }


@app.post("/api/v1/trading/start")
async def start_trading(user: TokenData = Depends(require_scope("trading"))):
    """啟動自動交易"""
    try:
        from src.engines import trading_engine
        await trading_engine.start()
        # 返回啟動後的狀態
        status = trading_engine.get_status()
        return {
            "success": True,
            "message": "Trading engine started",
            "is_running": status.get('running', False),
            "state": status.get('state', 'unknown')
        }
    except Exception as e:
        logger.error(f"Start trading failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/trading/stop")
async def stop_trading(user: TokenData = Depends(require_scope("trading"))):
    """停止自動交易"""
    try:
        from src.engines import trading_engine
        await trading_engine.stop()
        return {"success": True, "message": "Trading engine stopped"}
    except Exception as e:
        logger.error(f"Stop trading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/trading/positions")
async def get_positions(user: TokenData = RequireAuth):
    """獲取當前持倉（需認證）"""
    try:
        from src.engines import trading_engine
        return {"positions": trading_engine.get_positions()}
    except Exception as e:
        return {"positions": [], "error": str(e)}


@app.post("/api/v1/trading/sync")
async def sync_positions(user: TokenData = RequireTrading):
    """
    手動同步持倉（需 trading 權限）

    從 Binance 帳戶同步實際餘額到 bot 的持倉記錄：
    1. 添加 Binance 有但 bot 沒追蹤的持倉
    2. 移除 bot 有但 Binance 已沒有的持倉
    3. 更新數量不匹配的持倉
    """
    try:
        from src.engines import trading_engine
        result = await trading_engine.sync_positions_manual()
        return {
            "success": True,
            "message": "Position sync completed",
            **result
        }
    except Exception as e:
        logger.error(f"Position sync failed: {e}", exc_info=True)
        return {
            "success": False,
            "message": str(e),
            "synced": [],
            "removed": [],
            "updated": []
        }


@app.delete("/api/v1/trading/positions/{symbol}")
async def remove_position(symbol: str, user: TokenData = RequireTrading):
    """
    強制移除持倉記錄（不實際賣出）（需 trading 權限）

    用於清理 "灰塵" 持倉（低於最小訂單價值無法賣出的持倉）
    或同步問題導致的虛假持倉記錄。

    注意：這只移除 bot 的持倉記錄，不會影響 Binance 帳戶餘額。
    """
    try:
        from src.engines import trading_engine
        result = await trading_engine.remove_position_record(symbol.upper())
        return result
    except Exception as e:
        logger.error(f"Remove position failed: {e}", exc_info=True)
        return {
            "success": False,
            "message": str(e)
        }


@app.get("/api/v1/trading/performance")
async def get_trading_performance():
    """獲取交易績效"""
    try:
        from src.engines import trading_engine
        perf = trading_engine.get_performance()
        # 轉換為 iOS App 期望的格式
        return {
            "total_return": perf.get('total_return', 0),
            "today_return": perf.get('today_return', 0),
            "win_rate": perf.get('win_rate', 0),
            "sharpe_ratio": perf.get('sharpe_ratio', 0),
            "max_drawdown": perf.get('max_drawdown', 0),
            "total_trades": perf.get('total_trades', 0),
            "profitable_trades": perf.get('profitable_trades', 0),
            "avg_win": perf.get('avg_win', 0),
            "avg_loss": perf.get('avg_loss', 0),
            "profit_factor": perf.get('profit_factor', 0),
            "start_date": perf.get('start_date'),
            "last_update": perf.get('last_update')
        }
    except Exception as e:
        return {
            "total_return": 0,
            "today_return": 0,
            "win_rate": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "total_trades": 0,
            "profitable_trades": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "start_date": None,
            "last_update": None,
            "error": str(e)
        }


# ==================== 學習引擎 API ====================

# 訓練任務狀態存儲（背景任務用）
training_jobs: Dict[str, Dict[str, Any]] = {}

async def run_training_background(job_id: str):
    """背景執行訓練任務"""
    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["started_at"] = datetime.now().isoformat()
        logger.info(f"Background training started: {job_id}")
        
        from src.engines import learning_engine
        result = await learning_engine.train_models(force=True)
        
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        training_jobs[job_id]["result"] = result
        logger.info(f"Background training completed: {job_id}")
        
    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)
        training_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        logger.error(f"Background training failed: {job_id} - {e}")


@app.get("/api/v1/learning/status")
async def get_learning_status():
    """獲取學習引擎狀態"""
    try:
        from src.engines import learning_engine
        status = learning_engine.get_status()
        # 轉換為 iOS App 期望的格式
        return {
            "is_training": status.get('is_training', False),
            "last_training": status.get('last_training'),
            "trade_count": status.get('trade_count', 0),
            "performance_buffer_size": status.get('performance_buffer_size', 0),
            "active_versions": status.get('active_versions', {}),
            "training_history_length": status.get('training_history_length', 0)
        }
    except Exception as e:
        return {
            "is_training": False,
            "last_training": None,
            "trade_count": 0,
            "performance_buffer_size": 0,
            "active_versions": {},
            "training_history_length": 0,
            "error": str(e)
        }


@app.post("/api/v1/learning/train")
async def trigger_training(user: TokenData = RequireTrading):
    """觸發模型訓練（背景執行）（需 trading 權限）"""
    # 檢查是否有正在執行的訓練
    for job_id, job in training_jobs.items():
        if job.get("status") == "running":
            return {
                "success": False,
                "message": "訓練已在執行中",
                "job_id": job_id,
                "status": "running"
            }
    
    # 創建新任務
    job_id = str(uuid.uuid4())[:8]
    training_jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None
    }
    
    # 背景執行訓練
    asyncio.create_task(run_training_background(job_id))
    logger.info(f"Training job created: {job_id}")
    
    return {
        "success": True,
        "message": "訓練已開始（背景執行）",
        "job_id": job_id,
        "status": "pending"
    }


@app.get("/api/v1/learning/train/{job_id}")
async def get_training_status(job_id: str):
    """查詢訓練任務狀態"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="訓練任務不存在")
    return {"job_id": job_id, **training_jobs[job_id]}


@app.get("/api/v1/learning/status")
async def get_current_training():
    """查詢當前訓練狀態"""
    for job_id, job in training_jobs.items():
        if job.get("status") == "running":
            return {"training": True, "job_id": job_id, **job}
    return {"training": False, "status": "idle", "message": "沒有正在執行的訓練"}


@app.get("/api/v1/learning/history")
async def get_training_history():
    """獲取訓練歷史"""
    try:
        from src.engines import learning_engine
        return {"history": learning_engine.get_training_history()}
    except Exception as e:
        return {"history": [], "error": str(e)}


# ==================== 策略調試 API ====================
# 注意：主要策略 API 已移至 strategy_routes.py (策略超市功能)

@app.get("/api/v1/strategies/debug")
async def get_strategies_debug(user: TokenData = RequireAdmin):
    """除錯：獲取策略引擎完整狀態（需 admin 權限）"""
    try:
        from src.engines import strategy_engine, trading_engine
        return {
            "strategy_engine_status": strategy_engine.get_status(),
            "strategy_engine_id": id(strategy_engine),
            "trading_engine_strategy_id": id(trading_engine.strategy_engine),
            "same_instance": id(strategy_engine) == id(trading_engine.strategy_engine),
            "trading_running": trading_engine._running,
            "trading_state": trading_engine.state.value
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# ==================== 数据引擎 API ====================

@app.get("/api/v1/data/status")
async def get_data_status():
    """獲取數據引擎狀態"""
    try:
        from src.engines import data_engine
        return data_engine.get_status()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/v1/data/prices")
async def get_latest_prices():
    """獲取最新價格"""
    try:
        from src.engines import data_engine
        return {"prices": data_engine.get_all_prices()}
    except Exception as e:
        return {"prices": {}, "error": str(e)}


@app.get("/api/v1/data/signal/{symbol}")
async def get_latest_signal(symbol: str):
    """獲取最新交易信號"""
    try:
        from src.engines import data_engine
        signal = await data_engine.get_latest_signal(symbol.upper())
        return {"signal": signal}
    except Exception as e:
        return {"signal": None, "error": str(e)}


# ==================== WebSocket API ====================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str = Query(default=None)
):
    """
    WebSocket 端點

    連接後會自動接收：
    - signal_update: 交易信號更新
    - order_placed: 訂單已下單
    - order_filled: 訂單已成交
    - position_update: 持倉更新
    - engine_status: 引擎狀態變化
    - price_update: 價格更新
    - heartbeat: 心跳（每30秒）
    """
    await ws_manager.connect(websocket, client_id)

    try:
        while True:
            # 接收客戶端消息
            data = await websocket.receive_text()

            try:
                import json
                message = json.loads(data)
                msg_type = message.get("type", "")

                # 處理客戶端請求
                if msg_type == "ping":
                    await ws_manager.send_personal(websocket, WSMessage(
                        type="pong",
                        data={"server_time": datetime.utcnow().isoformat()}
                    ))

                elif msg_type == "subscribe":
                    # 可以在這裡處理訂閱特定頻道
                    channel = message.get("channel", "all")
                    await ws_manager.send_personal(websocket, WSMessage(
                        type="subscribed",
                        data={"channel": channel}
                    ))

                elif msg_type == "get_status":
                    # 返回當前狀態
                    from src.engines import trading_engine
                    status = trading_engine.get_status()
                    await ws_manager.send_personal(websocket, WSMessage(
                        type=MessageType.ENGINE_STATUS,
                        data={
                            "is_running": status.get('running', False),
                            "mode": "paper" if status.get('config', {}).get('paper_trading', True) else "live",
                            "symbol": status.get('config', {}).get('symbols', ['BTCUSDT'])[0],
                            "market_state": status.get('strategy_engine', {}).get('current_market_state', 'unknown'),
                            "total_trades": status.get('trading', {}).get('daily_trades', 0),
                            "today_pnl": status.get('trading', {}).get('daily_pnl', 0)
                        }
                    ))

            except json.JSONDecodeError:
                await ws_manager.send_personal(websocket, WSMessage(
                    type=MessageType.ERROR,
                    data={"message": "Invalid JSON format"}
                ))

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


@app.get("/api/v1/ws/status")
async def get_websocket_status():
    """獲取 WebSocket 連接狀態"""
    return {
        "active_connections": ws_manager.get_connection_count(),
        "connections": ws_manager.get_connection_info()
    }


# ==================== 运行应用 ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        workers=settings.API_WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )
