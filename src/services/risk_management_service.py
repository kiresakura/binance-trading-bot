"""
Risk Management Service - 風險管理服務

提供：
- 自動停損/止盈
- 模型過期檢查
- 交易頻率限制
- 風險平價倉位管理
- 最大回撤保護
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from src.logger import logger
from src.utils.database import async_db_manager


class RiskAction(str, Enum):
    """風險動作類型"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MAX_DRAWDOWN = "max_drawdown"
    TRAILING_STOP = "trailing_stop"
    TIME_STOP = "time_stop"
    FREQUENCY_LIMIT = "frequency_limit"


@dataclass
class RiskConfig:
    """風險配置"""
    # 停損/止盈設定
    stop_loss_pct: float = 5.0      # 停損百分比 (5%)
    take_profit_pct: float = 10.0   # 止盈百分比 (10%)
    trailing_stop_pct: float = 3.0  # 追蹤止損百分比 (3%)

    # 最大回撤保護
    max_drawdown_pct: float = 15.0  # 最大回撤百分比 (15%)

    # 時間止損
    max_hold_hours: int = 168       # 最大持倉時間 (7天)

    # 交易頻率限制
    min_trade_interval_minutes: int = 30  # 最小交易間隔（分鐘）
    max_trades_per_day: int = 20          # 每日最大交易次數

    # 模型過期設定
    model_expiry_days: int = 30     # 模型過期天數
    model_warning_days: int = 7     # 模型即將過期警告天數

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskAlert:
    """風險警報"""
    alert_type: str
    model_version: str
    symbol: str
    message: str
    severity: str  # low, medium, high, critical
    action_taken: Optional[str]
    details: Dict[str, Any]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RiskManagementService:
    """風險管理服務"""

    def __init__(self):
        self.config = RiskConfig()
        self._alerts: List[RiskAlert] = []
        self._trade_counts: Dict[str, Dict[str, int]] = {}  # model -> {date: count}
        self._last_trade_time: Dict[str, datetime] = {}  # model -> last trade time
        self._peak_values: Dict[str, Decimal] = {}  # model -> peak value

        logger.info("RiskManagementService initialized")

    def update_config(self, **kwargs):
        """更新風險配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Risk config updated: {key} = {value}")

    async def check_position_risk(
        self,
        model_version: str,
        symbol: str,
        entry_price: Decimal,
        current_price: Decimal,
        quantity: Decimal,
        entry_time: datetime,
        peak_price: Optional[Decimal] = None
    ) -> Tuple[bool, Optional[RiskAction], str]:
        """
        檢查持倉風險

        Returns:
            (should_close, action_type, reason)
        """
        if quantity <= 0:
            return False, None, ""

        # 計算盈虧百分比
        pnl_pct = float((current_price - entry_price) / entry_price * 100)

        # 1. 檢查停損
        if pnl_pct <= -self.config.stop_loss_pct:
            reason = f"觸發停損: 虧損 {abs(pnl_pct):.2f}% 超過 {self.config.stop_loss_pct}%"
            await self._create_alert(
                RiskAction.STOP_LOSS, model_version, symbol, reason, "high",
                "auto_close", {"pnl_pct": pnl_pct, "threshold": -self.config.stop_loss_pct}
            )
            return True, RiskAction.STOP_LOSS, reason

        # 2. 檢查止盈
        if pnl_pct >= self.config.take_profit_pct:
            reason = f"觸發止盈: 盈利 {pnl_pct:.2f}% 達到 {self.config.take_profit_pct}%"
            await self._create_alert(
                RiskAction.TAKE_PROFIT, model_version, symbol, reason, "medium",
                "auto_close", {"pnl_pct": pnl_pct, "threshold": self.config.take_profit_pct}
            )
            return True, RiskAction.TAKE_PROFIT, reason

        # 3. 檢查追蹤止損
        if peak_price and peak_price > entry_price:
            # 從最高點回撤
            drawdown_from_peak = float((peak_price - current_price) / peak_price * 100)
            if drawdown_from_peak >= self.config.trailing_stop_pct and pnl_pct > 0:
                reason = f"觸發追蹤止損: 從高點回撤 {drawdown_from_peak:.2f}%"
                await self._create_alert(
                    RiskAction.TRAILING_STOP, model_version, symbol, reason, "medium",
                    "auto_close", {"drawdown_from_peak": drawdown_from_peak, "peak_price": float(peak_price)}
                )
                return True, RiskAction.TRAILING_STOP, reason

        # 4. 檢查時間止損
        hold_time = datetime.utcnow() - entry_time
        if hold_time.total_seconds() / 3600 > self.config.max_hold_hours:
            reason = f"觸發時間止損: 持倉超過 {self.config.max_hold_hours} 小時"
            await self._create_alert(
                RiskAction.TIME_STOP, model_version, symbol, reason, "low",
                "auto_close", {"hold_hours": hold_time.total_seconds() / 3600}
            )
            return True, RiskAction.TIME_STOP, reason

        return False, None, ""

    async def check_account_risk(
        self,
        model_version: str,
        current_value: Decimal,
        initial_value: Decimal = Decimal("10000")
    ) -> Tuple[bool, Optional[RiskAction], str]:
        """
        檢查帳戶級別風險

        Returns:
            (should_stop_trading, action_type, reason)
        """
        # 更新峰值
        if model_version not in self._peak_values:
            self._peak_values[model_version] = current_value
        elif current_value > self._peak_values[model_version]:
            self._peak_values[model_version] = current_value

        peak = self._peak_values[model_version]

        # 計算從峰值的回撤
        if peak > 0:
            drawdown = float((peak - current_value) / peak * 100)
            if drawdown >= self.config.max_drawdown_pct:
                reason = f"觸發最大回撤保護: 回撤 {drawdown:.2f}% 超過 {self.config.max_drawdown_pct}%"
                await self._create_alert(
                    RiskAction.MAX_DRAWDOWN, model_version, "", reason, "critical",
                    "pause_trading", {"drawdown": drawdown, "peak_value": float(peak)}
                )
                return True, RiskAction.MAX_DRAWDOWN, reason

        return False, None, ""

    async def can_trade(
        self,
        model_version: str,
        symbol: str
    ) -> Tuple[bool, str]:
        """
        檢查是否允許交易（頻率限制）

        Returns:
            (can_trade, reason)
        """
        now = datetime.utcnow()
        today = now.strftime("%Y-%m-%d")

        # 檢查最小交易間隔
        if model_version in self._last_trade_time:
            time_since_last = (now - self._last_trade_time[model_version]).total_seconds() / 60
            if time_since_last < self.config.min_trade_interval_minutes:
                remaining = self.config.min_trade_interval_minutes - time_since_last
                return False, f"交易過於頻繁，請等待 {remaining:.1f} 分鐘"

        # 檢查每日交易次數
        if model_version not in self._trade_counts:
            self._trade_counts[model_version] = {}

        daily_count = self._trade_counts[model_version].get(today, 0)
        if daily_count >= self.config.max_trades_per_day:
            return False, f"已達每日最大交易次數 ({self.config.max_trades_per_day})"

        return True, ""

    def record_trade(self, model_version: str):
        """記錄交易（用於頻率限制）"""
        now = datetime.utcnow()
        today = now.strftime("%Y-%m-%d")

        self._last_trade_time[model_version] = now

        if model_version not in self._trade_counts:
            self._trade_counts[model_version] = {}

        self._trade_counts[model_version][today] = \
            self._trade_counts[model_version].get(today, 0) + 1

    async def check_model_expiry(
        self,
        model_version: str,
        trained_at: datetime
    ) -> Tuple[str, Optional[str]]:
        """
        檢查模型是否過期

        Returns:
            (status, warning_message)
            status: "valid", "warning", "expired"
        """
        now = datetime.utcnow()
        age_days = (now - trained_at).days

        if age_days >= self.config.model_expiry_days:
            msg = f"模型已過期: 訓練於 {age_days} 天前，建議重新訓練"
            await self._create_alert(
                "model_expired", model_version, "", msg, "high",
                None, {"age_days": age_days, "expiry_days": self.config.model_expiry_days}
            )
            return "expired", msg

        remaining_days = self.config.model_expiry_days - age_days
        if remaining_days <= self.config.model_warning_days:
            msg = f"模型即將過期: 還有 {remaining_days} 天過期"
            await self._create_alert(
                "model_expiring", model_version, "", msg, "medium",
                None, {"remaining_days": remaining_days}
            )
            return "warning", msg

        return "valid", None

    async def _create_alert(
        self,
        alert_type: str,
        model_version: str,
        symbol: str,
        message: str,
        severity: str,
        action_taken: Optional[str],
        details: Dict[str, Any]
    ):
        """創建風險警報"""
        alert = RiskAlert(
            alert_type=str(alert_type),
            model_version=model_version,
            symbol=symbol,
            message=message,
            severity=severity,
            action_taken=action_taken,
            details=details,
            timestamp=datetime.utcnow().isoformat()
        )

        self._alerts.append(alert)

        # 只保留最近 1000 條警報
        if len(self._alerts) > 1000:
            self._alerts = self._alerts[-1000:]

        # 記錄到日誌
        log_method = logger.warning if severity in ["high", "critical"] else logger.info
        log_method(f"Risk Alert [{severity}]: {message}")

        # 保存到數據庫
        try:
            await async_db_manager.execute(
                """
                INSERT INTO risk_alerts
                (alert_type, model_version, symbol, message, severity, action_taken, details)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                str(alert_type), model_version, symbol, message, severity,
                action_taken, str(details)
            )
        except Exception as e:
            logger.warning(f"Failed to save risk alert to database: {e}")

    def get_recent_alerts(
        self,
        model_version: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """獲取最近的警報"""
        alerts = self._alerts

        if model_version:
            alerts = [a for a in alerts if a.model_version == model_version]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return [a.to_dict() for a in alerts[-limit:]]

    async def auto_manage_positions(self):
        """
        自動管理所有虛擬持倉的風險

        這個方法應該被定期調用（例如每分鐘一次）
        """
        try:
            # 獲取所有虛擬持倉
            positions = await async_db_manager.execute(
                """
                SELECT vp.model_version, vp.symbol, vp.quantity, vp.entry_price,
                       vp.current_price, vp.created_at, vp.peak_price,
                       va.total_value
                FROM virtual_positions vp
                JOIN virtual_accounts va ON vp.model_version = va.model_version
                WHERE vp.quantity > 0
                """
            )

            actions_taken = []

            for pos in positions:
                model_version = pos['model_version']
                symbol = pos['symbol']
                entry_price = Decimal(str(pos['entry_price']))
                current_price = Decimal(str(pos['current_price']))
                quantity = Decimal(str(pos['quantity']))
                entry_time = pos['created_at'] if isinstance(pos['created_at'], datetime) else datetime.fromisoformat(str(pos['created_at']))
                peak_price = Decimal(str(pos['peak_price'])) if pos.get('peak_price') else None
                total_value = Decimal(str(pos['total_value']))

                # 檢查持倉風險
                should_close, action, reason = await self.check_position_risk(
                    model_version, symbol, entry_price, current_price,
                    quantity, entry_time, peak_price
                )

                if should_close:
                    # 執行自動平倉
                    from src.services.arena_service import arena_service
                    result = await arena_service.execute_virtual_trade(
                        model_version=model_version,
                        symbol=symbol,
                        side="SELL",
                        price=current_price,
                        quantity=quantity
                    )

                    if result.get("success"):
                        actions_taken.append({
                            "model_version": model_version,
                            "symbol": symbol,
                            "action": str(action),
                            "reason": reason,
                            "result": result
                        })
                        logger.info(f"Auto risk management: {action} for {model_version} {symbol}")

                # 檢查帳戶風險
                should_stop, account_action, account_reason = await self.check_account_risk(
                    model_version, total_value
                )

                if should_stop:
                    # 標記模型為暫停狀態
                    await async_db_manager.execute(
                        """
                        UPDATE arena_models
                        SET status = 'PAUSED',
                            pause_reason = $1,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE version = $2
                        """,
                        account_reason, model_version
                    )
                    logger.warning(f"Model {model_version} paused due to {account_action}")

            return actions_taken

        except Exception as e:
            logger.error(f"Error in auto_manage_positions: {e}")
            return []

    def calculate_risk_parity_position(
        self,
        symbol: str,
        volatility: float,
        target_risk: float = 0.02,
        account_value: float = 10000
    ) -> float:
        """
        計算風險平價倉位大小

        Args:
            symbol: 交易對
            volatility: 年化波動率
            target_risk: 目標風險（每日）
            account_value: 帳戶價值

        Returns:
            建議倉位大小（USDT）
        """
        if volatility <= 0:
            return 0

        # 日波動率 = 年波動率 / sqrt(365)
        daily_vol = volatility / (365 ** 0.5)

        # 風險平價: position_size = (target_risk * account_value) / daily_vol
        position_size = (target_risk * account_value) / daily_vol

        # 限制最大倉位為帳戶的 30%
        max_position = account_value * 0.3
        position_size = min(position_size, max_position)

        return round(position_size, 2)


# 單例實例
risk_service = RiskManagementService()
