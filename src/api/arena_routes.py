"""
Arena API Routes - 影子交易競技場 API 端點

實現：
- GET  /api/v1/arena/status          - 獲取競技場狀態
- POST /api/v1/arena/promote         - 晉升模型
- POST /api/v1/arena/shadow/start    - 啟動影子交易
- POST /api/v1/arena/shadow/stop     - 停止影子交易
- GET  /api/v1/arena/models          - 列出所有模型
- POST /api/v1/arena/models          - 註冊新模型
- GET  /api/v1/arena/models/{version} - 獲取模型詳情
- GET  /api/v1/arena/history         - 晉升歷史
- GET  /api/v1/strategies/sensitivity - 獲取靈敏度配置
- POST /api/v1/strategies/sensitivity - 更新靈敏度配置
"""

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.logger import logger
from src.services.arena_service import arena_service
from src.utils.database import async_db_manager


router = APIRouter(tags=["Arena System"])


# ==================== Request/Response Models ====================

class ModelRegistrationRequest(BaseModel):
    """模型註冊請求"""
    model_config = {"protected_namespaces": ()}

    version: str = Field(..., example="v1.1.0", description="模型版本號")
    model_type: str = Field(..., example="Transformer-RL", description="模型類型")
    config: Optional[Dict[str, Any]] = Field(default=None, description="模型配置")


class PromoteModelRequest(BaseModel):
    """模型晉升請求"""
    challenger_version: str = Field(..., example="v1.1.0", description="要晉升的挑戰者模型版本")
    reason: Optional[str] = Field(default=None, description="晉升原因")


class StartShadowRequest(BaseModel):
    """啟動影子交易請求"""
    challenger_version: str = Field(..., example="v1.1.0", description="挑戰者模型版本")


class SensitivityRequest(BaseModel):
    """策略靈敏度配置請求"""
    sensitivity: str = Field(..., example="moderate", description="靈敏度等級: conservative, moderate, aggressive")
    confidence_threshold: float = Field(..., ge=0, le=1, example=0.70, description="信心閾值 (0-1)")
    strength_threshold: float = Field(..., ge=0, le=1, example=0.60, description="強度閾值 (0-1)")
    position_ratio: float = Field(..., ge=0, le=1, example=0.6, description="倉位比例 (0-1)")


class ModelStatsResponse(BaseModel):
    """模型統計響應"""
    model_config = {"protected_namespaces": ()}

    version: str
    type: str
    win_rate: float
    total_return: float
    trade_count: int
    draw_down: float
    status: str
    sharpe_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_hold_time: Optional[str] = None
    last_update: Optional[str] = None


class ArenaStatusResponse(BaseModel):
    """競技場狀態響應"""
    champion: Optional[ModelStatsResponse] = None
    challenger: Optional[ModelStatsResponse] = None
    comparison_days: int
    promotion_threshold: float
    is_comparing: bool
    started_at: Optional[str] = None


class SensitivityResponse(BaseModel):
    """策略靈敏度配置響應"""
    sensitivity: str
    confidence_threshold: float
    strength_threshold: float
    position_ratio: float


# ==================== Arena Status Endpoints ====================

@router.post("/arena/init")
async def init_arena_state():
    """
    初始化 Arena 狀態表

    確保 arena_state 表有初始記錄，用於修復首次部署後的問題
    """
    try:
        from src.utils.database import async_db_manager

        # 使用 INSERT ... ON CONFLICT 確保記錄存在
        # 注意：PostgreSQL 需要用 0/1 而不是 FALSE/TRUE
        await async_db_manager.execute(
            """
            INSERT INTO arena_state (id, is_comparing, comparison_days, promotion_threshold)
            VALUES (1, 0, 7, 0.05)
            ON CONFLICT(id) DO NOTHING
            """
        )

        logger.info("Arena state initialized successfully")
        return {"success": True, "message": "Arena state initialized"}
    except Exception as e:
        logger.error(f"Failed to init arena state: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to init arena state: {str(e)}"
        )


@router.get("/arena/status", response_model=ArenaStatusResponse)
async def get_arena_status():
    """
    獲取競技場狀態

    返回當前冠軍模型、挑戰者模型及比較配置
    """
    try:
        arena_status = await arena_service.get_arena_status()
        return arena_status.to_dict()
    except Exception as e:
        logger.error(f"Failed to get arena status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get arena status: {str(e)}"
        )


@router.post("/arena/promote")
async def promote_model(request: PromoteModelRequest):
    """
    晉升挑戰者模型為實盤冠軍

    這將：
    1. 將當前冠軍降級為 ARCHIVED
    2. 將挑戰者晉升為 LIVE 狀態
    3. 記錄晉升歷史
    """
    try:
        result = await arena_service.promote_model(
            challenger_version=request.challenger_version,
            reason=request.reason
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Promotion failed")
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to promote model: {str(e)}"
        )


# ==================== Shadow Trading Endpoints ====================

@router.post("/arena/shadow/start")
async def start_shadow_trading(request: StartShadowRequest):
    """
    啟動影子交易模式

    挑戰者模型將開始接收與實盤相同的市場數據，
    但只記錄模擬交易，不會實際下單
    """
    try:
        success = await arena_service.start_shadow_trading(request.challenger_version)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to start shadow trading for {request.challenger_version}"
            )

        return {
            "success": True,
            "message": f"Shadow trading started for {request.challenger_version}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start shadow trading: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start shadow trading: {str(e)}"
        )


@router.post("/arena/shadow/stop")
async def stop_shadow_trading():
    """
    停止影子交易模式
    """
    try:
        success = await arena_service.stop_shadow_trading()

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to stop shadow trading"
            )

        return {
            "success": True,
            "message": "Shadow trading stopped"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop shadow trading: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop shadow trading: {str(e)}"
        )


# ==================== Model Management Endpoints ====================

@router.get("/arena/models")
async def list_models(
    status_filter: Optional[str] = Query(
        default=None,
        alias="status",
        description="過濾模型狀態: TRAINING, SHADOW_RUNNING, LIVE, ARCHIVED"
    )
):
    """
    列出所有模型

    可選按狀態過濾
    """
    try:
        models = await arena_service.list_models(status=status_filter)
        return {
            "success": True,
            "count": len(models),
            "models": [m.to_dict() for m in models]
        }
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@router.post("/arena/models")
async def register_model(request: ModelRegistrationRequest):
    """
    註冊新模型

    新模型將以 TRAINING 狀態創建
    """
    try:
        success = await arena_service.register_model(
            version=request.version,
            model_type=request.model_type,
            config=request.config
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to register model {request.version}"
            )

        return {
            "success": True,
            "message": f"Model {request.version} registered successfully",
            "version": request.version
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register model: {str(e)}"
        )


@router.get("/arena/models/{version}")
async def get_model(version: str):
    """
    獲取模型詳情
    """
    try:
        model = await arena_service.get_model(version)

        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {version} not found"
            )

        return {
            "success": True,
            "model": model.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model {version}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model: {str(e)}"
        )


@router.get("/arena/models/{version}/trades")
async def get_model_shadow_trades(
    version: str,
    limit: int = Query(default=100, le=1000, description="最大返回數量")
):
    """
    獲取模型的虛擬交易記錄
    """
    try:
        trades = await arena_service.get_virtual_trades(version, limit)
        return {
            "success": True,
            "version": version,
            "count": len(trades),
            "trades": trades
        }
    except Exception as e:
        logger.error(f"Failed to get virtual trades for {version}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get virtual trades: {str(e)}"
        )


# ==================== Promotion History ====================

@router.get("/arena/history")
async def get_promotion_history(
    limit: int = Query(default=20, le=100, description="最大返回數量")
):
    """
    獲取模型晉升歷史
    """
    try:
        history = await arena_service.get_promotion_history(limit)
        return {
            "success": True,
            "count": len(history),
            "history": history
        }
    except Exception as e:
        logger.error(f"Failed to get promotion history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get promotion history: {str(e)}"
        )


# ==================== Strategy Sensitivity ====================

@router.get("/strategies/sensitivity", response_model=SensitivityResponse)
async def get_sensitivity():
    """
    獲取策略靈敏度配置
    """
    try:
        config = await arena_service.get_sensitivity_config()
        return config.to_dict()
    except Exception as e:
        logger.error(f"Failed to get sensitivity config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sensitivity config: {str(e)}"
        )


@router.post("/strategies/sensitivity")
async def update_sensitivity(request: SensitivityRequest):
    """
    更新策略靈敏度配置

    三個預設等級：
    - conservative: 信心 > 85%, 強度 > 80%, 倉位 100%
    - moderate: 信心 > 70%, 強度 > 60%, 倉位 60%
    - aggressive: 信心 > 55%, 強度 > 45%, 倉位 30%

    也可以自訂任意閾值
    """
    try:
        # 驗證 sensitivity 值
        valid_sensitivities = ["conservative", "moderate", "aggressive"]
        if request.sensitivity not in valid_sensitivities:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid sensitivity. Must be one of: {valid_sensitivities}"
            )

        success = await arena_service.update_sensitivity_config(
            sensitivity=request.sensitivity,
            confidence_threshold=request.confidence_threshold,
            strength_threshold=request.strength_threshold,
            position_ratio=request.position_ratio
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update sensitivity config"
            )

        return {
            "success": True,
            "message": f"Sensitivity config updated to: {request.sensitivity}",
            "config": {
                "sensitivity": request.sensitivity,
                "confidence_threshold": request.confidence_threshold,
                "strength_threshold": request.strength_threshold,
                "position_ratio": request.position_ratio
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update sensitivity config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update sensitivity config: {str(e)}"
        )


# ==================== 影子交易執行器 ====================

@router.get("/shadow/executor/status")
async def get_shadow_executor_status():
    """
    獲取影子交易執行器狀態

    返回執行器的運行狀態、緩存的模型數量等信息。
    """
    try:
        from src.services.shadow_trading_executor import shadow_trading_executor
        return shadow_trading_executor.get_status()
    except Exception as e:
        logger.error(f"Failed to get shadow executor status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get shadow executor status: {str(e)}"
        )


@router.post("/shadow/executor/start")
async def start_shadow_executor():
    """
    啟動影子交易執行器

    開始定時檢測 SHADOW_RUNNING 狀態的模型，產生並記錄影子交易信號。
    """
    try:
        from src.services.shadow_trading_executor import shadow_trading_executor

        if shadow_trading_executor.is_running:
            return {"success": True, "message": "Shadow executor already running"}

        await shadow_trading_executor.start()
        return {"success": True, "message": "Shadow executor started"}
    except Exception as e:
        logger.error(f"Failed to start shadow executor: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start shadow executor: {str(e)}"
        )


@router.post("/shadow/executor/stop")
async def stop_shadow_executor():
    """
    停止影子交易執行器
    """
    try:
        from src.services.shadow_trading_executor import shadow_trading_executor

        if not shadow_trading_executor.is_running:
            return {"success": True, "message": "Shadow executor already stopped"}

        await shadow_trading_executor.stop()
        return {"success": True, "message": "Shadow executor stopped"}
    except Exception as e:
        logger.error(f"Failed to stop shadow executor: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop shadow executor: {str(e)}"
        )



# ==================== 虛擬帳戶 API ====================

@router.get("/arena/leaderboard")
async def get_leaderboard(
    limit: int = Query(default=10, le=50, description="最大返回數量")
):
    """
    獲取模型排行榜（按總回報排序）
    """
    try:
        leaderboard = await arena_service.get_leaderboard(limit)
        return {
            "success": True,
            "count": len(leaderboard),
            "leaderboard": leaderboard
        }
    except Exception as e:
        logger.error(f"Failed to get leaderboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get leaderboard: {str(e)}"
        )


@router.get("/arena/models/{version}/account")
async def get_virtual_account(version: str):
    """
    獲取模型的虛擬帳戶狀態
    """
    try:
        account = await arena_service.get_virtual_account(version)
        if not account:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Virtual account for {version} not found"
            )
        return {
            "success": True,
            "model_version": version,
            "account": account
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get virtual account for {version}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get virtual account: {str(e)}"
        )


@router.get("/arena/models/{version}/positions")
async def get_virtual_positions(version: str):
    """
    獲取模型的虛擬持倉
    """
    try:
        positions = await arena_service.get_virtual_positions(version)
        return {
            "success": True,
            "model_version": version,
            "count": len(positions),
            "positions": [p.to_dict() for p in positions]
        }
    except Exception as e:
        logger.error(f"Failed to get virtual positions for {version}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get virtual positions: {str(e)}"
        )


@router.get("/arena/models/{version}/virtual-trades")
async def get_model_virtual_trades(
    version: str,
    limit: int = Query(default=50, le=200, description="最大返回數量")
):
    """
    獲取模型的虛擬交易歷史（帶盈虧）
    """
    try:
        trades = await arena_service.get_virtual_trades(version, limit)
        return {
            "success": True,
            "model_version": version,
            "count": len(trades),
            "trades": trades
        }
    except Exception as e:
        logger.error(f"Failed to get virtual trades for {version}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get virtual trades: {str(e)}"
        )


@router.post("/arena/models/{version}/reset")
async def reset_virtual_account(version: str):
    """
    重置模型的虛擬帳戶（回到 10000 USDT）
    """
    try:
        # 刪除現有持倉
        await async_db_manager.execute(
            "DELETE FROM virtual_positions WHERE model_version = $1",
            version
        )
        
        # 重置帳戶
        await async_db_manager.execute(
            """
            UPDATE virtual_accounts 
            SET balance = 10000, total_value = 10000, realized_pnl = 0, unrealized_pnl = 0,
                total_pnl = 0, total_pnl_pct = 0, max_drawdown = 0, peak_value = 10000,
                trade_count = 0, win_count = 0, loss_count = 0, win_rate = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE model_version = $1
            """,
            version
        )
        
        # 重置 arena_models 統計
        await async_db_manager.execute(
            """
            UPDATE arena_models 
            SET win_rate = 0, total_return = 0, trade_count = 0, draw_down = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE version = $1
            """,
            version
        )
        
        logger.info(f"Virtual account reset for {version}")
        
        return {
            "success": True,
            "message": f"Virtual account for {version} reset to 10000 USDT"
        }
    except Exception as e:
        logger.error(f"Failed to reset virtual account for {version}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset virtual account: {str(e)}"
        )


@router.post("/arena/init-all-accounts")
async def init_all_virtual_accounts():
    """
    為所有現有模型初始化虛擬帳戶
    """
    try:
        models = await arena_service.list_models()
        initialized = 0
        
        for model in models:
            success = await arena_service.init_virtual_account(model.version)
            if success:
                initialized += 1
        
        return {
            "success": True,
            "message": f"Initialized {initialized} virtual accounts",
            "total_models": len(models)
        }
    except Exception as e:
        logger.error(f"Failed to init all virtual accounts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to init virtual accounts: {str(e)}"
        )


@router.post("/arena/run-migration")
async def run_virtual_accounts_migration():
    """
    執行虛擬帳戶表 migration
    創建 virtual_accounts, virtual_positions, virtual_trades 表
    """
    try:
        # 創建虛擬帳戶表
        await async_db_manager.execute("""
            CREATE TABLE IF NOT EXISTS virtual_accounts (
                id BIGSERIAL PRIMARY KEY,
                model_version VARCHAR(50) NOT NULL UNIQUE,
                initial_balance DECIMAL(20, 8) NOT NULL DEFAULT 10000,
                balance DECIMAL(20, 8) NOT NULL DEFAULT 10000,
                total_value DECIMAL(20, 8) NOT NULL DEFAULT 10000,
                realized_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
                unrealized_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
                total_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
                total_pnl_pct DECIMAL(10, 4) NOT NULL DEFAULT 0,
                max_drawdown DECIMAL(10, 4) NOT NULL DEFAULT 0,
                peak_value DECIMAL(20, 8) NOT NULL DEFAULT 10000,
                trade_count INTEGER NOT NULL DEFAULT 0,
                win_count INTEGER NOT NULL DEFAULT 0,
                loss_count INTEGER NOT NULL DEFAULT 0,
                win_rate DECIMAL(10, 4) NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 創建虛擬持倉表
        await async_db_manager.execute("""
            CREATE TABLE IF NOT EXISTS virtual_positions (
                id BIGSERIAL PRIMARY KEY,
                model_version VARCHAR(50) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL DEFAULT 'LONG',
                quantity DECIMAL(20, 8) NOT NULL DEFAULT 0,
                entry_price DECIMAL(20, 8) NOT NULL,
                current_price DECIMAL(20, 8),
                unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
                unrealized_pnl_pct DECIMAL(10, 4) DEFAULT 0,
                opened_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model_version, symbol)
            )
        """)
        
        # 創建虛擬交易表
        await async_db_manager.execute("""
            CREATE TABLE IF NOT EXISTS virtual_trades (
                timestamp TIMESTAMPTZ NOT NULL,
                model_version VARCHAR(50) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                quantity DECIMAL(20, 8) NOT NULL,
                price DECIMAL(20, 8) NOT NULL,
                quote_value DECIMAL(20, 8) NOT NULL,
                commission DECIMAL(20, 8) DEFAULT 0,
                realized_pnl DECIMAL(20, 8),
                balance_after DECIMAL(20, 8),
                signal_strength DECIMAL(10, 6),
                signal_confidence DECIMAL(10, 6),
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (timestamp, model_version, symbol, side)
            )
        """)
        
        # 創建索引
        await async_db_manager.execute("""
            CREATE INDEX IF NOT EXISTS idx_virtual_accounts_model ON virtual_accounts (model_version)
        """)
        await async_db_manager.execute("""
            CREATE INDEX IF NOT EXISTS idx_virtual_positions_model ON virtual_positions (model_version)
        """)
        await async_db_manager.execute("""
            CREATE INDEX IF NOT EXISTS idx_virtual_trades_model ON virtual_trades (model_version, timestamp DESC)
        """)
        
        # 為現有模型初始化虛擬帳戶
        await async_db_manager.execute("""
            INSERT INTO virtual_accounts (model_version, initial_balance, balance, total_value, peak_value)
            SELECT version, 10000, 10000, 10000, 10000
            FROM arena_models
            WHERE version NOT IN (SELECT model_version FROM virtual_accounts)
            ON CONFLICT (model_version) DO NOTHING
        """)
        
        logger.info("Virtual accounts migration completed")
        
        return {
            "success": True,
            "message": "Migration completed: virtual_accounts, virtual_positions, virtual_trades tables created"
        }
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Migration failed: {str(e)}"
        )
