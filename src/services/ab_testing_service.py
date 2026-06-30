"""
A/B Testing Service - A/B 測試框架

提供：
- 模型 A/B 測試
- 流量分配
- 統計顯著性分析
- 實驗管理
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import math

from src.logger import logger


class ExperimentStatus(str, Enum):
    """實驗狀態"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VariantType(str, Enum):
    """變體類型"""
    CONTROL = "control"      # 對照組
    TREATMENT = "treatment"  # 實驗組


@dataclass
class Variant:
    """實驗變體"""
    variant_id: str
    name: str
    variant_type: VariantType
    model_version: str
    traffic_percentage: float  # 流量百分比 (0-1)

    # 統計數據
    total_signals: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl_pct: float = 0.0
    trades_pnl: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "name": self.name,
            "variant_type": self.variant_type.value if isinstance(self.variant_type, VariantType) else self.variant_type,
            "model_version": self.model_version,
            "traffic_percentage": self.traffic_percentage,
            "total_signals": self.total_signals,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "total_pnl_pct": self.total_pnl_pct,
            "win_rate": self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            "avg_pnl_pct": self.total_pnl_pct / self.total_trades if self.total_trades > 0 else 0
        }


@dataclass
class Experiment:
    """A/B 測試實驗"""
    experiment_id: str
    name: str
    description: str
    status: ExperimentStatus
    variants: List[Variant]
    symbol: str
    created_at: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    min_sample_size: int = 100          # 最小樣本量
    confidence_level: float = 0.95       # 信心水平
    auto_stop_on_significance: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value if isinstance(self.status, ExperimentStatus) else self.status,
            "variants": [v.to_dict() for v in self.variants],
            "symbol": self.symbol,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "min_sample_size": self.min_sample_size,
            "confidence_level": self.confidence_level
        }


@dataclass
class StatisticalResult:
    """統計分析結果"""
    experiment_id: str
    timestamp: str
    control_win_rate: float
    treatment_win_rate: float
    control_avg_pnl: float
    treatment_avg_pnl: float
    lift_percentage: float           # 提升百分比
    p_value: float                   # p 值
    is_significant: bool             # 是否統計顯著
    confidence_interval: Tuple[float, float]
    recommended_action: str
    analysis_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "control_win_rate": self.control_win_rate,
            "treatment_win_rate": self.treatment_win_rate,
            "control_avg_pnl": self.control_avg_pnl,
            "treatment_avg_pnl": self.treatment_avg_pnl,
            "lift_percentage": self.lift_percentage,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "confidence_interval": list(self.confidence_interval),
            "recommended_action": self.recommended_action,
            "analysis_summary": self.analysis_summary
        }


class ABTestingService:
    """A/B 測試服務"""

    def __init__(self):
        self._experiments: Dict[str, Experiment] = {}
        self._experiment_counter = 0
        logger.info("ABTestingService initialized")

    def create_experiment(
        self,
        name: str,
        description: str,
        symbol: str,
        control_model: str,
        treatment_model: str,
        traffic_split: float = 0.5,
        min_sample_size: int = 100,
        confidence_level: float = 0.95
    ) -> Experiment:
        """
        創建 A/B 測試實驗

        Args:
            name: 實驗名稱
            description: 實驗描述
            symbol: 交易對
            control_model: 對照組模型
            treatment_model: 實驗組模型
            traffic_split: 實驗組流量比例 (0-1)
            min_sample_size: 最小樣本量
            confidence_level: 信心水平

        Returns:
            Experiment: 創建的實驗
        """
        self._experiment_counter += 1
        experiment_id = f"exp_{self._experiment_counter}_{datetime.utcnow().strftime('%Y%m%d')}"

        # 創建變體
        control = Variant(
            variant_id=f"{experiment_id}_control",
            name="Control (Current)",
            variant_type=VariantType.CONTROL,
            model_version=control_model,
            traffic_percentage=1 - traffic_split
        )

        treatment = Variant(
            variant_id=f"{experiment_id}_treatment",
            name="Treatment (New)",
            variant_type=VariantType.TREATMENT,
            model_version=treatment_model,
            traffic_percentage=traffic_split
        )

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            status=ExperimentStatus.DRAFT,
            variants=[control, treatment],
            symbol=symbol,
            created_at=datetime.utcnow().isoformat(),
            min_sample_size=min_sample_size,
            confidence_level=confidence_level
        )

        self._experiments[experiment_id] = experiment

        logger.info(f"Experiment created: {experiment_id}")
        return experiment

    def start_experiment(self, experiment_id: str) -> bool:
        """啟動實驗"""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return False

        if experiment.status != ExperimentStatus.DRAFT:
            logger.warning(f"Cannot start experiment {experiment_id}: status is {experiment.status}")
            return False

        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.utcnow().isoformat()

        logger.info(f"Experiment started: {experiment_id}")
        return True

    def pause_experiment(self, experiment_id: str) -> bool:
        """暫停實驗"""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return False

        if experiment.status != ExperimentStatus.RUNNING:
            return False

        experiment.status = ExperimentStatus.PAUSED
        logger.info(f"Experiment paused: {experiment_id}")
        return True

    def resume_experiment(self, experiment_id: str) -> bool:
        """恢復實驗"""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return False

        if experiment.status != ExperimentStatus.PAUSED:
            return False

        experiment.status = ExperimentStatus.RUNNING
        logger.info(f"Experiment resumed: {experiment_id}")
        return True

    def stop_experiment(self, experiment_id: str) -> bool:
        """停止實驗"""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return False

        if experiment.status not in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            return False

        experiment.status = ExperimentStatus.COMPLETED
        experiment.ended_at = datetime.utcnow().isoformat()

        logger.info(f"Experiment stopped: {experiment_id}")
        return True

    def get_variant_for_request(
        self,
        experiment_id: str,
        request_id: Optional[str] = None
    ) -> Optional[Variant]:
        """
        為請求分配變體

        Args:
            experiment_id: 實驗 ID
            request_id: 請求 ID（用於確保一致性）

        Returns:
            Variant: 分配的變體
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.RUNNING:
            return None

        # 使用隨機分配或基於 request_id 的一致性分配
        if request_id:
            # 基於 hash 的一致性分配
            hash_value = hash(request_id) % 100 / 100
        else:
            hash_value = random.random()

        # 根據流量比例分配
        cumulative = 0
        for variant in experiment.variants:
            cumulative += variant.traffic_percentage
            if hash_value < cumulative:
                return variant

        # 默認返回最後一個變體
        return experiment.variants[-1]

    async def record_signal(
        self,
        experiment_id: str,
        variant_id: str
    ):
        """記錄信號"""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return

        for variant in experiment.variants:
            if variant.variant_id == variant_id:
                variant.total_signals += 1
                break

    async def record_trade_result(
        self,
        experiment_id: str,
        variant_id: str,
        pnl_pct: float
    ):
        """
        記錄交易結果

        Args:
            experiment_id: 實驗 ID
            variant_id: 變體 ID
            pnl_pct: 盈虧百分比
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return

        for variant in experiment.variants:
            if variant.variant_id == variant_id:
                variant.total_trades += 1
                variant.total_pnl_pct += pnl_pct
                variant.trades_pnl.append(pnl_pct)

                if pnl_pct > 0:
                    variant.winning_trades += 1

                break

        # 檢查是否達到統計顯著性
        if experiment.auto_stop_on_significance:
            result = await self.analyze_experiment(experiment_id)
            if result and result.is_significant:
                min_trades = min(v.total_trades for v in experiment.variants)
                if min_trades >= experiment.min_sample_size:
                    logger.info(f"Experiment {experiment_id} reached significance, auto-stopping")
                    self.stop_experiment(experiment_id)

    async def analyze_experiment(
        self,
        experiment_id: str
    ) -> Optional[StatisticalResult]:
        """
        分析實驗結果

        Args:
            experiment_id: 實驗 ID

        Returns:
            StatisticalResult: 統計分析結果
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return None

        # 找到對照組和實驗組
        control = None
        treatment = None

        for variant in experiment.variants:
            if variant.variant_type == VariantType.CONTROL:
                control = variant
            elif variant.variant_type == VariantType.TREATMENT:
                treatment = variant

        if not control or not treatment:
            return None

        # 計算基本指標
        control_win_rate = control.winning_trades / control.total_trades if control.total_trades > 0 else 0
        treatment_win_rate = treatment.winning_trades / treatment.total_trades if treatment.total_trades > 0 else 0

        control_avg_pnl = control.total_pnl_pct / control.total_trades if control.total_trades > 0 else 0
        treatment_avg_pnl = treatment.total_pnl_pct / treatment.total_trades if treatment.total_trades > 0 else 0

        # 計算提升
        if control_win_rate > 0:
            lift = (treatment_win_rate - control_win_rate) / control_win_rate * 100
        else:
            lift = 0

        # 計算 p 值（使用簡化的 z-test）
        p_value = self._calculate_p_value(control, treatment)

        # 計算信賴區間
        confidence_interval = self._calculate_confidence_interval(
            control, treatment, experiment.confidence_level
        )

        # 判斷顯著性
        alpha = 1 - experiment.confidence_level
        is_significant = p_value < alpha and min(control.total_trades, treatment.total_trades) >= experiment.min_sample_size

        # 生成建議
        recommended_action, analysis_summary = self._generate_recommendation(
            control, treatment, is_significant, lift
        )

        return StatisticalResult(
            experiment_id=experiment_id,
            timestamp=datetime.utcnow().isoformat(),
            control_win_rate=control_win_rate,
            treatment_win_rate=treatment_win_rate,
            control_avg_pnl=control_avg_pnl,
            treatment_avg_pnl=treatment_avg_pnl,
            lift_percentage=lift,
            p_value=p_value,
            is_significant=is_significant,
            confidence_interval=confidence_interval,
            recommended_action=recommended_action,
            analysis_summary=analysis_summary
        )

    def _calculate_p_value(self, control: Variant, treatment: Variant) -> float:
        """計算 p 值（兩比例 z-test）"""
        n1 = control.total_trades
        n2 = treatment.total_trades

        if n1 == 0 or n2 == 0:
            return 1.0

        p1 = control.winning_trades / n1
        p2 = treatment.winning_trades / n2

        # 合併比例
        p_pool = (control.winning_trades + treatment.winning_trades) / (n1 + n2)

        # 標準誤差
        se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

        if se == 0:
            return 1.0

        # z 統計量
        z = (p2 - p1) / se

        # 計算 p 值（雙尾檢驗）
        # 使用簡化的正態分佈近似
        p_value = 2 * (1 - self._normal_cdf(abs(z)))

        return p_value

    def _normal_cdf(self, x: float) -> float:
        """標準正態分佈 CDF 近似"""
        # 使用 Abramowitz and Stegun 近似
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x) / math.sqrt(2)

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

        return 0.5 * (1.0 + sign * y)

    def _calculate_confidence_interval(
        self,
        control: Variant,
        treatment: Variant,
        confidence_level: float
    ) -> Tuple[float, float]:
        """計算信賴區間"""
        n1 = control.total_trades
        n2 = treatment.total_trades

        if n1 == 0 or n2 == 0:
            return (0.0, 0.0)

        p1 = control.winning_trades / n1
        p2 = treatment.winning_trades / n2

        diff = p2 - p1

        # 標準誤差
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

        # z 值（根據信心水平）
        z_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_values.get(confidence_level, 1.96)

        margin = z * se

        return (diff - margin, diff + margin)

    def _generate_recommendation(
        self,
        control: Variant,
        treatment: Variant,
        is_significant: bool,
        lift: float
    ) -> Tuple[str, str]:
        """生成建議"""
        min_trades = min(control.total_trades, treatment.total_trades)

        if min_trades < 30:
            return (
                "CONTINUE",
                f"樣本量不足（對照組：{control.total_trades}，實驗組：{treatment.total_trades}），需要更多數據。"
            )

        if not is_significant:
            return (
                "CONTINUE",
                f"結果尚未達到統計顯著性。目前實驗組相對提升 {lift:.1f}%。"
            )

        if lift > 10:
            return (
                "DEPLOY_TREATMENT",
                f"實驗組表現顯著優於對照組（提升 {lift:.1f}%），建議部署實驗組模型。"
            )
        elif lift < -10:
            return (
                "KEEP_CONTROL",
                f"對照組表現顯著優於實驗組（實驗組下降 {-lift:.1f}%），建議保持現有模型。"
            )
        else:
            return (
                "NO_DIFFERENCE",
                f"兩組表現相近（差異 {lift:.1f}%），可根據其他因素決定。"
            )

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """獲取實驗詳情"""
        experiment = self._experiments.get(experiment_id)
        if experiment:
            return experiment.to_dict()
        return None

    def get_all_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """獲取所有實驗"""
        experiments = list(self._experiments.values())

        if status:
            experiments = [e for e in experiments if e.status == status]

        # 按創建時間倒序
        experiments.sort(key=lambda e: e.created_at, reverse=True)

        return [e.to_dict() for e in experiments[:limit]]

    def get_running_experiments_for_symbol(self, symbol: str) -> List[Experiment]:
        """獲取指定交易對的運行中實驗"""
        return [
            e for e in self._experiments.values()
            if e.status == ExperimentStatus.RUNNING and e.symbol == symbol
        ]

    def delete_experiment(self, experiment_id: str) -> bool:
        """刪除實驗"""
        if experiment_id in self._experiments:
            experiment = self._experiments[experiment_id]
            if experiment.status in [ExperimentStatus.DRAFT, ExperimentStatus.COMPLETED, ExperimentStatus.CANCELLED]:
                del self._experiments[experiment_id]
                logger.info(f"Experiment deleted: {experiment_id}")
                return True
        return False


# 單例實例
ab_testing_service = ABTestingService()
