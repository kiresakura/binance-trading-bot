"""不可变审计日志系统"""

import json
import hashlib
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from decimal import Decimal
from src.logger import logger
from src.utils.database import async_db_manager


@dataclass
class AuditEntry:
    """审计日志条目"""
    timestamp: datetime
    event_type: str  # TRADE, ORDER, RISK_REJECTION, etc.
    user_id: Optional[str]
    action: str
    resource: str  # symbol, order_id, etc.
    details: Dict
    risk_score: float
    prev_hash: str
    current_hash: str

    def to_dict(self) -> dict:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()

        # 处理 Decimal 类型
        def convert_decimals(obj):
            if isinstance(obj, dict):
                return {k: convert_decimals(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_decimals(v) for v in obj]
            elif isinstance(obj, Decimal):
                return float(obj)
            return obj

        data['details'] = convert_decimals(data['details'])
        return data


class AuditLogger:
    """审计日志记录器"""

    def __init__(self):
        self.last_hash = self._get_genesis_hash()
        self.chain_verified = True

    def _get_genesis_hash(self) -> str:
        """获取创世哈希"""
        return hashlib.sha256(b"genesis_block").hexdigest()

    @staticmethod
    def _canonical_timestamp(ts) -> str:
        """將 timestamp 正規化為固定 ISO 字串，確保「寫入」與「驗證」對同一筆
        資料算出相同雜湊。

        寫入時為 datetime；驗證時從 DB 讀回可能是 datetime（PostgreSQL）或以
        空格分隔的字串（部分 SQLite 介面）。一律轉成 isoformat（T 分隔）。
        """
        if isinstance(ts, datetime):
            return ts.isoformat()
        if isinstance(ts, str):
            return ts.replace(' ', 'T', 1)
        return str(ts)

    def _calculate_hash(self, entry: Dict) -> str:
        """计算条目哈希"""
        # 排除 current_hash 字段
        entry_copy = {k: v for k, v in entry.items() if k != 'current_hash'}

        # 正規化 timestamp：避免 datetime/字串、空格/T 等序列化差異
        # 導致同一筆資料在寫入與驗證時算出不同雜湊（鏈驗證永遠失敗的根因）。
        if 'timestamp' in entry_copy:
            entry_copy['timestamp'] = self._canonical_timestamp(entry_copy['timestamp'])

        # 序列化为 JSON（排序键以保证一致性）
        json_str = json.dumps(entry_copy, sort_keys=True, default=str)

        # 计算 SHA-256 哈希
        hash_obj = hashlib.sha256(json_str.encode('utf-8'))
        return hash_obj.hexdigest()

    async def log_trade(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        order_id: str,
        strategy: str,
        risk_score: float,
        commission: Decimal
    ):
        """记录交易"""
        details = {
            'symbol': symbol,
            'side': side,
            'quantity': float(quantity),
            'price': float(price),
            'order_id': order_id,
            'strategy': strategy,
            'commission': float(commission),
            'total_cost': float(quantity * price)
        }

        await self._log_entry(
            event_type='TRADE',
            action=side,
            resource=symbol,
            details=details,
            risk_score=risk_score
        )

    async def log_order(
        self,
        order_id: str,
        symbol: str,
        order_type: str,
        side: str,
        quantity: Decimal,
        price: Optional[Decimal],
        status: str
    ):
        """记录订单"""
        details = {
            'order_id': order_id,
            'symbol': symbol,
            'type': order_type,
            'side': side,
            'quantity': float(quantity),
            'price': float(price) if price else None,
            'status': status
        }

        await self._log_entry(
            event_type='ORDER',
            action=status,
            resource=order_id,
            details=details,
            risk_score=0.0
        )

    async def log_risk_rejection(
        self,
        symbol: str,
        reason: str,
        risk_score: float,
        attempted_quantity: Decimal,
        attempted_side: str
    ):
        """记录风险拒绝"""
        details = {
            'symbol': symbol,
            'reason': reason,
            'attempted_quantity': float(attempted_quantity),
            'attempted_side': attempted_side,
            'rejected_at': datetime.utcnow().isoformat()
        }

        await self._log_entry(
            event_type='RISK_REJECTION',
            action='REJECTED',
            resource=symbol,
            details=details,
            risk_score=risk_score
        )

    async def log_system_event(
        self,
        event_type: str,
        action: str,
        details: Dict
    ):
        """记录系统事件"""
        await self._log_entry(
            event_type=event_type,
            action=action,
            resource='SYSTEM',
            details=details,
            risk_score=0.0
        )

    async def _log_entry(
        self,
        event_type: str,
        action: str,
        resource: str,
        details: Dict,
        risk_score: float,
        user_id: Optional[str] = None
    ):
        """记录审计条目"""
        try:
            # 创建条目（不包含哈希）
            # 關鍵：時間戳只取一次，且「被雜湊的值」與「存入 DB 的值」必須相同，
            # 否則驗證時重算的雜湊永遠對不上（鏈驗證失效的根因之一）。
            ts = datetime.utcnow()
            entry_dict = {
                'timestamp': ts,
                'event_type': event_type,
                'user_id': user_id or 'system',
                'action': action,
                'resource': resource,
                'details': details,
                'risk_score': risk_score,
                'prev_hash': self.last_hash
            }

            # 计算当前哈希
            current_hash = self._calculate_hash(entry_dict)
            entry_dict['current_hash'] = current_hash

            # 创建审计条目对象
            entry = AuditEntry(
                timestamp=ts,
                event_type=event_type,
                user_id=user_id or 'system',
                action=action,
                resource=resource,
                details=details,
                risk_score=risk_score,
                prev_hash=self.last_hash,
                current_hash=current_hash
            )

            # 写入不可变日志
            await self._write_to_immutable_log(entry)

            # 更新最后一个哈希
            self.last_hash = current_hash

            logger.debug(f"Audit log entry created: {event_type} - {action}")

        except Exception as e:
            logger.error(f"Failed to create audit log entry: {e}")

    async def _write_to_immutable_log(self, entry: AuditEntry):
        """写入不可变日志（数据库）"""
        try:
            query = """
                INSERT INTO audit_log (
                    timestamp, event_type, user_id, action,
                    resource, details, risk_score, prev_hash, current_hash
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """

            await async_db_manager.execute(
                query,
                entry.timestamp,  # 直接傳遞 datetime 對象，不要用 isoformat()
                entry.event_type,
                entry.user_id,
                entry.action,
                entry.resource,
                json.dumps(entry.details),
                entry.risk_score,
                entry.prev_hash,
                entry.current_hash
            )

        except Exception as e:
            logger.error(f"Failed to write audit log to database: {e}")

            # 备份写入文件系统
            await self._write_to_file_backup(entry)

    async def _write_to_file_backup(self, entry: AuditEntry):
        """备份写入文件系统"""
        try:
            import aiofiles
            from pathlib import Path

            # 创建审计日志目录
            audit_dir = Path("logs/audit")
            audit_dir.mkdir(parents=True, exist_ok=True)

            # 按日期分文件
            date_str = entry.timestamp.strftime('%Y%m%d')
            log_file = audit_dir / f"audit_{date_str}.jsonl"

            # 追加写入
            async with aiofiles.open(log_file, 'a') as f:
                await f.write(json.dumps(entry.to_dict()) + '\n')

        except Exception as e:
            logger.critical(f"Failed to write audit log backup: {e}")

    async def verify_chain_integrity(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> bool:
        """验证审计日志链完整性"""
        try:
            logger.info("Verifying audit log chain integrity...")

            # 从数据库获取审计日志
            query = """
                SELECT timestamp, event_type, user_id, action, resource,
                       details, risk_score, prev_hash, current_hash
                FROM audit_log
                WHERE ($1 IS NULL OR timestamp >= $1)
                  AND ($2 IS NULL OR timestamp <= $2)
                ORDER BY timestamp ASC
            """

            results = await async_db_manager.execute(
                query,
                start_time if start_time else None,  # 直接傳遞 datetime 對象
                end_time if end_time else None  # 直接傳遞 datetime 對象
            )

            if not results:
                logger.warning("No audit logs found in specified range")
                return True

            # 验证链
            prev_hash = None
            for row in results:
                # 重建条目字典
                entry_dict = {
                    'timestamp': row['timestamp'],
                    'event_type': row['event_type'],
                    'user_id': row['user_id'],
                    'action': row['action'],
                    'resource': row['resource'],
                    'details': json.loads(row['details']),
                    'risk_score': row['risk_score'],
                    'prev_hash': row['prev_hash']
                }

                # 验证前一个哈希
                if prev_hash and entry_dict['prev_hash'] != prev_hash:
                    logger.error(
                        f"Chain integrity violation at {row['timestamp']}: "
                        f"expected prev_hash {prev_hash}, got {entry_dict['prev_hash']}"
                    )
                    return False

                # 验证当前哈希
                calculated_hash = self._calculate_hash(entry_dict)
                if calculated_hash != row['current_hash']:
                    logger.error(
                        f"Hash mismatch at {row['timestamp']}: "
                        f"expected {row['current_hash']}, got {calculated_hash}"
                    )
                    return False

                prev_hash = row['current_hash']

            logger.info(f"Audit log chain verified: {len(results)} entries OK")
            return True

        except Exception as e:
            logger.error(f"Failed to verify audit log chain: {e}")
            return False

    async def get_audit_trail(
        self,
        resource: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """获取审计跟踪"""
        try:
            conditions = []
            params = []
            param_idx = 1

            if resource:
                conditions.append(f"resource = ${param_idx}")
                params.append(resource)
                param_idx += 1

            if event_type:
                conditions.append(f"event_type = ${param_idx}")
                params.append(event_type)
                param_idx += 1

            if start_time:
                conditions.append(f"timestamp >= ${param_idx}")
                params.append(start_time)  # 直接傳遞 datetime 對象
                param_idx += 1

            if end_time:
                conditions.append(f"timestamp <= ${param_idx}")
                params.append(end_time)  # 直接傳遞 datetime 對象
                param_idx += 1

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            query = f"""
                SELECT timestamp, event_type, user_id, action, resource,
                       details, risk_score, current_hash
                FROM audit_log
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ${param_idx}
            """
            params.append(limit)

            results = await async_db_manager.execute(query, *params)

            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get audit trail: {e}")
            return []


# 全局审计日志记录器实例
audit_logger = AuditLogger()
