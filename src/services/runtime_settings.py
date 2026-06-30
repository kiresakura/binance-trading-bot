"""運行時設定管理器 - 允許動態修改交易設定"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import logging

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SettingChange:
    """設定變更記錄"""
    key: str
    old_value: Any
    new_value: Any
    changed_at: datetime = field(default_factory=datetime.utcnow)
    changed_by: str = "api"


class RuntimeSettings:
    """
    運行時設定管理器

    允許在不重啟服務的情況下動態修改某些設定。
    這些設定優先於 config.py 中的靜態設定。
    """

    def __init__(self):
        # 可動態修改的設定（從 config.py 初始化）
        self._trading_enabled: bool = settings.TRADING_ENABLED
        self._dry_run: bool = settings.DRY_RUN

        # 變更歷史記錄
        self._change_history: List[SettingChange] = []
        self._max_history: int = 100

        # 變更回調（用於通知其他組件）
        self._callbacks: Dict[str, List[Callable]] = {
            "trading_enabled": [],
            "dry_run": [],
        }

        # 鎖（防止並發修改）
        self._lock = asyncio.Lock()

        logger.info(
            f"RuntimeSettings initialized: "
            f"TRADING_ENABLED={self._trading_enabled}, "
            f"DRY_RUN={self._dry_run}"
        )

    @property
    def trading_enabled(self) -> bool:
        """是否啟用交易功能"""
        return self._trading_enabled

    @property
    def dry_run(self) -> bool:
        """是否為模擬模式（只記錄不下單）"""
        return self._dry_run

    async def set_trading_enabled(self, value: bool, changed_by: str = "api") -> bool:
        """
        設定交易啟用狀態

        Args:
            value: 新的值
            changed_by: 變更來源（例如 "api", "admin", "system"）

        Returns:
            是否成功修改
        """
        async with self._lock:
            if self._trading_enabled == value:
                logger.debug(f"trading_enabled already set to {value}, no change needed")
                return True

            old_value = self._trading_enabled
            self._trading_enabled = value

            # 記錄變更
            self._record_change("trading_enabled", old_value, value, changed_by)

            logger.warning(
                f"TRADING_ENABLED changed: {old_value} -> {value} "
                f"(by {changed_by})"
            )

            # 觸發回調
            await self._trigger_callbacks("trading_enabled", value)

            return True

    async def set_dry_run(self, value: bool, changed_by: str = "api") -> bool:
        """
        設定模擬模式

        Args:
            value: 新的值
            changed_by: 變更來源

        Returns:
            是否成功修改
        """
        async with self._lock:
            if self._dry_run == value:
                logger.debug(f"dry_run already set to {value}, no change needed")
                return True

            old_value = self._dry_run
            self._dry_run = value

            # 記錄變更
            self._record_change("dry_run", old_value, value, changed_by)

            logger.warning(
                f"DRY_RUN changed: {old_value} -> {value} "
                f"(by {changed_by})"
            )

            # 觸發回調
            await self._trigger_callbacks("dry_run", value)

            return True

    def _record_change(self, key: str, old_value: Any, new_value: Any, changed_by: str):
        """記錄設定變更"""
        change = SettingChange(
            key=key,
            old_value=old_value,
            new_value=new_value,
            changed_by=changed_by
        )
        self._change_history.append(change)

        # 限制歷史記錄數量
        if len(self._change_history) > self._max_history:
            self._change_history = self._change_history[-self._max_history:]

    async def _trigger_callbacks(self, key: str, value: Any):
        """觸發設定變更回調"""
        if key in self._callbacks:
            for callback in self._callbacks[key]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(value)
                    else:
                        callback(value)
                except Exception as e:
                    logger.error(f"Callback error for {key}: {e}")

    def register_callback(self, key: str, callback: Callable):
        """註冊設定變更回調"""
        if key in self._callbacks:
            self._callbacks[key].append(callback)
            logger.debug(f"Registered callback for {key}")

    def unregister_callback(self, key: str, callback: Callable):
        """取消註冊設定變更回調"""
        if key in self._callbacks and callback in self._callbacks[key]:
            self._callbacks[key].remove(callback)

    def get_all_settings(self) -> Dict[str, Any]:
        """獲取所有運行時設定"""
        return {
            "trading_enabled": self._trading_enabled,
            "dry_run": self._dry_run,
            # 以下為只讀設定（來自 config.py）
            "testnet": settings.BINANCE_TESTNET,
            "trading_pairs": settings.TRADING_PAIRS,
            "default_trading_pair": settings.DEFAULT_TRADING_PAIR,
            "max_position_size": settings.MAX_POSITION_SIZE,
            "max_daily_loss": settings.MAX_DAILY_LOSS,
        }

    def get_change_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """獲取設定變更歷史"""
        history = self._change_history[-limit:]
        return [
            {
                "key": c.key,
                "old_value": c.old_value,
                "new_value": c.new_value,
                "changed_at": c.changed_at.isoformat(),
                "changed_by": c.changed_by,
            }
            for c in reversed(history)
        ]

    def reset_to_defaults(self):
        """重置為 config.py 中的預設值"""
        self._trading_enabled = settings.TRADING_ENABLED
        self._dry_run = settings.DRY_RUN
        logger.info("RuntimeSettings reset to defaults from config.py")


# 全局單例
runtime_settings = RuntimeSettings()
