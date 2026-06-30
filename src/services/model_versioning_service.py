"""
Model Versioning Service - 模型版本控制服務

提供：
- 模型版本追蹤
- 版本回滾
- 模型比較
- 部署歷史
"""

import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum

from src.logger import logger
from src.utils.database import async_db_manager


class ModelStatus(str, Enum):
    """模型狀態"""
    DRAFT = "draft"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


@dataclass
class ModelVersion:
    """模型版本"""
    version_id: str
    model_id: str
    version_number: str  # e.g., "1.0.0"
    status: ModelStatus
    created_at: str
    # 性能指標
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    # 訓練資訊
    training_data_hash: str = ""
    training_samples: int = 0
    training_duration_seconds: float = 0.0
    # 元數據
    config: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    parent_version: Optional[str] = None
    file_path: str = ""
    file_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "status": self.status.value if isinstance(self.status, ModelStatus) else self.status
        }


@dataclass
class DeploymentRecord:
    """部署記錄"""
    deployment_id: str
    version_id: str
    deployed_at: str
    deployed_by: str
    environment: str  # staging, production
    status: str  # success, failed, rolled_back
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelVersioningService:
    """模型版本控制服務"""

    def __init__(self):
        self.model_dir = Path("models")
        self.versions_dir = self.model_dir / "versions"
        self.archive_dir = self.model_dir / "archive"

        # 確保目錄存在
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # 版本緩存
        self._versions: Dict[str, ModelVersion] = {}
        self._deployments: List[DeploymentRecord] = []

        logger.info("ModelVersioningService initialized")

    async def create_version(
        self,
        model_id: str,
        model_data: Dict[str, Any],
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
        parent_version: Optional[str] = None,
        notes: str = ""
    ) -> ModelVersion:
        """
        創建新模型版本

        Args:
            model_id: 模型 ID
            model_data: 模型數據
            metrics: 性能指標
            config: 訓練配置
            parent_version: 父版本 ID
            notes: 版本說明

        Returns:
            ModelVersion: 新版本
        """
        # 生成版本號
        existing_versions = await self.list_versions(model_id)
        if existing_versions:
            # 遞增版本號
            latest = existing_versions[0]
            major, minor, patch = map(int, latest.version_number.split('.'))
            version_number = f"{major}.{minor}.{patch + 1}"
        else:
            version_number = "1.0.0"

        # 生成版本 ID
        timestamp = datetime.utcnow()
        version_id = f"{model_id}-v{version_number}-{timestamp.strftime('%Y%m%d%H%M%S')}"

        # 計算模型數據哈希
        model_json = json.dumps(model_data, sort_keys=True)
        file_hash = hashlib.sha256(model_json.encode()).hexdigest()[:16]

        # 保存模型文件
        version_dir = self.versions_dir / model_id / version_number
        version_dir.mkdir(parents=True, exist_ok=True)

        model_file = version_dir / "model.json"
        with open(model_file, 'w') as f:
            json.dump(model_data, f, indent=2)

        # 保存版本元數據
        version = ModelVersion(
            version_id=version_id,
            model_id=model_id,
            version_number=version_number,
            status=ModelStatus.DRAFT,
            created_at=timestamp.isoformat(),
            accuracy=metrics.get("accuracy", 0.0),
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            f1_score=metrics.get("f1_score", 0.0),
            training_samples=metrics.get("training_samples", 0),
            training_duration_seconds=metrics.get("training_duration", 0.0),
            config=config or {},
            notes=notes,
            parent_version=parent_version,
            file_path=str(model_file),
            file_hash=file_hash
        )

        # 保存元數據
        meta_file = version_dir / "metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)

        # 保存到數據庫
        await self._save_version_to_db(version)

        self._versions[version_id] = version
        logger.info(f"Model version created: {version_id}")

        return version

    async def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """獲取版本詳情"""
        # 先檢查緩存
        if version_id in self._versions:
            return self._versions[version_id]

        # 從數據庫獲取
        try:
            row = await async_db_manager.execute_one(
                "SELECT * FROM model_versions WHERE version_id = $1",
                version_id
            )
            if row:
                version = ModelVersion(
                    version_id=row["version_id"],
                    model_id=row["model_id"],
                    version_number=row["version_number"],
                    status=ModelStatus(row["status"]),
                    created_at=row["created_at"],
                    accuracy=float(row.get("accuracy", 0)),
                    precision=float(row.get("precision_score", 0)),
                    recall=float(row.get("recall", 0)),
                    f1_score=float(row.get("f1_score", 0)),
                    config=json.loads(row.get("config", "{}")) if row.get("config") else {},
                    notes=row.get("notes", ""),
                    parent_version=row.get("parent_version"),
                    file_path=row.get("file_path", ""),
                    file_hash=row.get("file_hash", "")
                )
                self._versions[version_id] = version
                return version
        except Exception as e:
            logger.warning(f"Failed to get version from DB: {e}")

        # 從文件系統獲取
        for model_dir in self.versions_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for ver_dir in model_dir.iterdir():
                meta_file = ver_dir / "metadata.json"
                if meta_file.exists():
                    with open(meta_file) as f:
                        data = json.load(f)
                        if data.get("version_id") == version_id:
                            version = ModelVersion(**data)
                            self._versions[version_id] = version
                            return version

        return None

    async def list_versions(
        self,
        model_id: str,
        status: Optional[ModelStatus] = None,
        limit: int = 20
    ) -> List[ModelVersion]:
        """列出模型的所有版本"""
        versions = []

        # 從數據庫獲取
        try:
            if status:
                rows = await async_db_manager.execute(
                    """
                    SELECT * FROM model_versions
                    WHERE model_id = $1 AND status = $2
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    model_id, status.value, limit
                )
            else:
                rows = await async_db_manager.execute(
                    """
                    SELECT * FROM model_versions
                    WHERE model_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    model_id, limit
                )

            for row in rows:
                versions.append(ModelVersion(
                    version_id=row["version_id"],
                    model_id=row["model_id"],
                    version_number=row["version_number"],
                    status=ModelStatus(row["status"]),
                    created_at=row["created_at"],
                    accuracy=float(row.get("accuracy", 0)),
                    f1_score=float(row.get("f1_score", 0)),
                    file_path=row.get("file_path", "")
                ))

            return versions
        except Exception as e:
            logger.warning(f"Failed to list versions from DB: {e}")

        # 從文件系統獲取
        model_dir = self.versions_dir / model_id
        if model_dir.exists():
            for ver_dir in sorted(model_dir.iterdir(), reverse=True):
                meta_file = ver_dir / "metadata.json"
                if meta_file.exists():
                    with open(meta_file) as f:
                        data = json.load(f)
                        version = ModelVersion(**data)
                        if status is None or version.status == status:
                            versions.append(version)
                            if len(versions) >= limit:
                                break

        return versions

    async def update_status(
        self,
        version_id: str,
        new_status: ModelStatus,
        notes: str = ""
    ) -> bool:
        """更新版本狀態"""
        version = await self.get_version(version_id)
        if not version:
            return False

        old_status = version.status
        version.status = new_status
        if notes:
            version.notes = notes

        # 更新數據庫
        try:
            await async_db_manager.execute(
                """
                UPDATE model_versions
                SET status = $1, notes = $2, updated_at = CURRENT_TIMESTAMP
                WHERE version_id = $3
                """,
                new_status.value, notes, version_id
            )
        except Exception as e:
            logger.warning(f"Failed to update version status in DB: {e}")

        # 更新文件
        meta_file = Path(version.file_path).parent / "metadata.json"
        if meta_file.exists():
            with open(meta_file, 'w') as f:
                json.dump(version.to_dict(), f, indent=2)

        logger.info(f"Version {version_id} status: {old_status} -> {new_status}")
        return True

    async def deploy_version(
        self,
        version_id: str,
        environment: str = "production",
        deployed_by: str = "system"
    ) -> Optional[DeploymentRecord]:
        """
        部署版本

        Args:
            version_id: 版本 ID
            environment: 部署環境
            deployed_by: 部署者

        Returns:
            DeploymentRecord: 部署記錄
        """
        version = await self.get_version(version_id)
        if not version:
            logger.error(f"Version not found: {version_id}")
            return None

        # 創建部署記錄
        deployment_id = f"deploy-{version_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        try:
            # 將舊的 production 版本標記為 archived
            if environment == "production":
                old_versions = await self.list_versions(version.model_id, ModelStatus.PRODUCTION)
                for old in old_versions:
                    if old.version_id != version_id:
                        await self.update_status(old.version_id, ModelStatus.ARCHIVED)

            # 複製模型到生產目錄
            prod_dir = self.model_dir / "production" / version.model_id
            prod_dir.mkdir(parents=True, exist_ok=True)

            source_file = Path(version.file_path)
            if source_file.exists():
                shutil.copy(source_file, prod_dir / "model.json")

                # 保存版本信息
                with open(prod_dir / "version.json", 'w') as f:
                    json.dump({
                        "version_id": version_id,
                        "version_number": version.version_number,
                        "deployed_at": datetime.utcnow().isoformat()
                    }, f, indent=2)

            # 更新版本狀態
            await self.update_status(version_id, ModelStatus.PRODUCTION)

            deployment = DeploymentRecord(
                deployment_id=deployment_id,
                version_id=version_id,
                deployed_at=datetime.utcnow().isoformat(),
                deployed_by=deployed_by,
                environment=environment,
                status="success"
            )

            # 保存部署記錄
            await self._save_deployment_to_db(deployment)
            self._deployments.append(deployment)

            logger.info(f"Deployed {version_id} to {environment}")
            return deployment

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            deployment = DeploymentRecord(
                deployment_id=deployment_id,
                version_id=version_id,
                deployed_at=datetime.utcnow().isoformat(),
                deployed_by=deployed_by,
                environment=environment,
                status="failed",
                notes=str(e)
            )
            self._deployments.append(deployment)
            return deployment

    async def rollback(
        self,
        model_id: str,
        target_version_id: Optional[str] = None
    ) -> bool:
        """
        回滾到指定版本

        Args:
            model_id: 模型 ID
            target_version_id: 目標版本 ID（如果為空，回滾到上一個版本）

        Returns:
            是否成功
        """
        if target_version_id:
            target = await self.get_version(target_version_id)
        else:
            # 獲取上一個 archived 版本
            archived = await self.list_versions(model_id, ModelStatus.ARCHIVED, limit=1)
            target = archived[0] if archived else None

        if not target:
            logger.error("No version to rollback to")
            return False

        # 部署目標版本
        deployment = await self.deploy_version(
            target.version_id,
            "production",
            "rollback"
        )

        return deployment is not None and deployment.status == "success"

    async def compare_versions(
        self,
        version_id_1: str,
        version_id_2: str
    ) -> Dict[str, Any]:
        """比較兩個版本"""
        v1 = await self.get_version(version_id_1)
        v2 = await self.get_version(version_id_2)

        if not v1 or not v2:
            return {"error": "Version not found"}

        return {
            "version_1": {
                "version_id": v1.version_id,
                "version_number": v1.version_number,
                "accuracy": v1.accuracy,
                "f1_score": v1.f1_score,
                "created_at": v1.created_at
            },
            "version_2": {
                "version_id": v2.version_id,
                "version_number": v2.version_number,
                "accuracy": v2.accuracy,
                "f1_score": v2.f1_score,
                "created_at": v2.created_at
            },
            "comparison": {
                "accuracy_diff": v2.accuracy - v1.accuracy,
                "f1_diff": v2.f1_score - v1.f1_score,
                "better_version": v2.version_id if v2.f1_score > v1.f1_score else v1.version_id
            }
        }

    async def get_deployment_history(
        self,
        model_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """獲取部署歷史"""
        try:
            if model_id:
                rows = await async_db_manager.execute(
                    """
                    SELECT d.* FROM model_deployments d
                    JOIN model_versions v ON d.version_id = v.version_id
                    WHERE v.model_id = $1
                    ORDER BY d.deployed_at DESC
                    LIMIT $2
                    """,
                    model_id, limit
                )
            else:
                rows = await async_db_manager.execute(
                    """
                    SELECT * FROM model_deployments
                    ORDER BY deployed_at DESC
                    LIMIT $1
                    """,
                    limit
                )

            return [dict(row) for row in rows]
        except Exception as e:
            logger.warning(f"Failed to get deployment history: {e}")
            return [d.to_dict() for d in self._deployments[-limit:]]

    async def _save_version_to_db(self, version: ModelVersion):
        """保存版本到數據庫"""
        try:
            await async_db_manager.execute(
                """
                INSERT INTO model_versions
                (version_id, model_id, version_number, status, created_at,
                 accuracy, precision_score, recall, f1_score, config, notes,
                 parent_version, file_path, file_hash)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (version_id) DO UPDATE SET
                    status = excluded.status,
                    notes = excluded.notes,
                    updated_at = CURRENT_TIMESTAMP
                """,
                version.version_id, version.model_id, version.version_number,
                version.status.value, version.created_at,
                version.accuracy, version.precision, version.recall, version.f1_score,
                json.dumps(version.config), version.notes,
                version.parent_version, version.file_path, version.file_hash
            )
        except Exception as e:
            logger.warning(f"Failed to save version to DB: {e}")

    async def _save_deployment_to_db(self, deployment: DeploymentRecord):
        """保存部署記錄到數據庫"""
        try:
            await async_db_manager.execute(
                """
                INSERT INTO model_deployments
                (deployment_id, version_id, deployed_at, deployed_by, environment, status, notes)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                deployment.deployment_id, deployment.version_id,
                deployment.deployed_at, deployment.deployed_by,
                deployment.environment, deployment.status, deployment.notes
            )
        except Exception as e:
            logger.warning(f"Failed to save deployment to DB: {e}")


# 單例實例
model_versioning_service = ModelVersioningService()
