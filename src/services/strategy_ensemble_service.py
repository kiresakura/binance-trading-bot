"""
Strategy Ensemble Service - 策略組合服務

支持多種策略/模型組合方法：
- 投票法（Voting）：多數決定
- 加權平均（Weighted Average）：根據模型準確率加權
- 信心加權（Confidence Weighted）：根據信號信心度加權
- 堆疊法（Stacking）：使用元模型組合基礎模型
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Any, Optional
from decimal import Decimal
import math

from src.logger import logger


class EnsembleMethod(str, Enum):
    """組合方法"""
    VOTING = "voting"  # 投票法
    WEIGHTED_AVG = "weighted_average"  # 加權平均
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # 信心加權
    BEST_MODEL = "best_model"  # 選擇最佳模型


@dataclass
class ModelPrediction:
    """單個模型的預測結果"""
    model_id: str
    model_type: str
    signal: int  # -1=賣, 0=持有, 1=買
    confidence: float  # 0-1
    strength: float  # 信號強度
    accuracy: float  # 模型歷史準確率
    features_used: int  # 使用的特徵數量
    raw_probability: Optional[float] = None  # 原始概率輸出

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnsemblePrediction:
    """組合預測結果"""
    final_signal: int  # -1=賣, 0=持有, 1=買
    final_confidence: float  # 組合信心度
    final_strength: float  # 組合強度
    method_used: str  # 使用的組合方法
    participating_models: int  # 參與的模型數量
    agreement_ratio: float  # 意見一致度
    buy_votes: int
    sell_votes: int
    hold_votes: int
    individual_predictions: List[Dict[str, Any]]
    recommendation: str  # 交易建議文字
    reasoning: str  # 決策原因

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StrategyEnsembleService:
    """策略組合服務"""

    def __init__(self):
        # 默認模型權重（可以根據歷史表現動態調整）
        self.model_weights: Dict[str, float] = {}
        # 最小信心度閾值
        self.min_confidence_threshold = 0.5
        # 最小一致性比例
        self.min_agreement_ratio = 0.6

        logger.info("StrategyEnsembleService initialized")

    def set_model_weight(self, model_id: str, weight: float):
        """設置模型權重"""
        self.model_weights[model_id] = max(0.0, min(1.0, weight))
        logger.info(f"Set weight for {model_id}: {self.model_weights[model_id]}")

    def get_model_weight(self, model_id: str, accuracy: float = 0.5) -> float:
        """獲取模型權重（如果沒有設定，根據準確率計算）"""
        if model_id in self.model_weights:
            return self.model_weights[model_id]
        # 根據準確率計算默認權重：準確率越高權重越大
        # 使用 sigmoid 函數將 0.4-0.8 的準確率映射到 0.2-0.8 的權重
        normalized = (accuracy - 0.4) * 5  # 將 0.4-0.6 映射到 0-1
        weight = 0.2 + 0.6 * (1 / (1 + math.exp(-normalized)))
        return weight

    def combine_predictions(
        self,
        predictions: List[ModelPrediction],
        method: EnsembleMethod = EnsembleMethod.CONFIDENCE_WEIGHTED,
        min_models: int = 2
    ) -> EnsemblePrediction:
        """
        組合多個模型的預測結果

        Args:
            predictions: 模型預測列表
            method: 組合方法
            min_models: 最少需要的模型數量

        Returns:
            EnsemblePrediction: 組合後的預測結果
        """
        if not predictions:
            return self._create_empty_prediction("no_predictions")

        if len(predictions) < min_models:
            # 模型數量不足，使用最佳模型
            return self._use_best_model(predictions)

        # 根據方法選擇組合策略
        if method == EnsembleMethod.VOTING:
            return self._voting_ensemble(predictions)
        elif method == EnsembleMethod.WEIGHTED_AVG:
            return self._weighted_average_ensemble(predictions)
        elif method == EnsembleMethod.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted_ensemble(predictions)
        elif method == EnsembleMethod.BEST_MODEL:
            return self._use_best_model(predictions)
        else:
            return self._confidence_weighted_ensemble(predictions)

    def _voting_ensemble(self, predictions: List[ModelPrediction]) -> EnsemblePrediction:
        """投票法組合"""
        buy_votes = sum(1 for p in predictions if p.signal > 0)
        sell_votes = sum(1 for p in predictions if p.signal < 0)
        hold_votes = sum(1 for p in predictions if p.signal == 0)

        total = len(predictions)

        # 多數決定
        if buy_votes > sell_votes and buy_votes > hold_votes:
            final_signal = 1
            agreement = buy_votes / total
            recommendation = "買入"
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            final_signal = -1
            agreement = sell_votes / total
            recommendation = "賣出"
        else:
            final_signal = 0
            agreement = hold_votes / total if hold_votes > 0 else max(buy_votes, sell_votes) / total
            recommendation = "持有觀望"

        # 平均信心度和強度
        avg_confidence = sum(p.confidence for p in predictions) / total
        avg_strength = sum(p.strength for p in predictions) / total

        # 如果一致性太低，降低信心度
        if agreement < self.min_agreement_ratio:
            avg_confidence *= agreement
            final_signal = 0
            recommendation = "持有觀望（意見分歧）"

        reasoning = f"投票結果：買{buy_votes}票，賣{sell_votes}票，持有{hold_votes}票"

        return EnsemblePrediction(
            final_signal=final_signal,
            final_confidence=avg_confidence,
            final_strength=avg_strength,
            method_used="voting",
            participating_models=total,
            agreement_ratio=agreement,
            buy_votes=buy_votes,
            sell_votes=sell_votes,
            hold_votes=hold_votes,
            individual_predictions=[p.to_dict() for p in predictions],
            recommendation=recommendation,
            reasoning=reasoning
        )

    def _weighted_average_ensemble(self, predictions: List[ModelPrediction]) -> EnsemblePrediction:
        """加權平均組合（根據模型準確率加權）"""
        total_weight = 0
        weighted_signal = 0
        weighted_confidence = 0
        weighted_strength = 0

        buy_votes = 0
        sell_votes = 0
        hold_votes = 0

        for p in predictions:
            weight = self.get_model_weight(p.model_id, p.accuracy)
            total_weight += weight
            weighted_signal += p.signal * weight
            weighted_confidence += p.confidence * weight
            weighted_strength += p.strength * weight

            if p.signal > 0:
                buy_votes += 1
            elif p.signal < 0:
                sell_votes += 1
            else:
                hold_votes += 1

        if total_weight == 0:
            return self._create_empty_prediction("zero_weight")

        # 計算加權結果
        avg_signal = weighted_signal / total_weight
        avg_confidence = weighted_confidence / total_weight
        avg_strength = weighted_strength / total_weight

        # 根據加權信號確定最終方向
        if avg_signal > 0.3:
            final_signal = 1
            recommendation = "買入"
        elif avg_signal < -0.3:
            final_signal = -1
            recommendation = "賣出"
        else:
            final_signal = 0
            recommendation = "持有觀望"

        total = len(predictions)
        agreement = max(buy_votes, sell_votes, hold_votes) / total

        reasoning = f"加權平均信號: {avg_signal:.3f}（>0.3買入, <-0.3賣出）"

        return EnsemblePrediction(
            final_signal=final_signal,
            final_confidence=avg_confidence,
            final_strength=avg_strength,
            method_used="weighted_average",
            participating_models=total,
            agreement_ratio=agreement,
            buy_votes=buy_votes,
            sell_votes=sell_votes,
            hold_votes=hold_votes,
            individual_predictions=[p.to_dict() for p in predictions],
            recommendation=recommendation,
            reasoning=reasoning
        )

    def _confidence_weighted_ensemble(self, predictions: List[ModelPrediction]) -> EnsemblePrediction:
        """信心加權組合（根據信號信心度加權）"""
        # 計算買/賣/持有的加權分數
        buy_score = 0
        sell_score = 0
        hold_score = 0

        buy_votes = 0
        sell_votes = 0
        hold_votes = 0

        for p in predictions:
            # 組合權重 = 準確率權重 * 信心度
            accuracy_weight = self.get_model_weight(p.model_id, p.accuracy)
            combined_weight = accuracy_weight * p.confidence

            if p.signal > 0:
                buy_score += combined_weight * p.strength
                buy_votes += 1
            elif p.signal < 0:
                sell_score += combined_weight * p.strength
                sell_votes += 1
            else:
                hold_score += combined_weight * 0.5  # 持有給一半分數
                hold_votes += 1

        total = len(predictions)

        # 選擇最高分數的方向
        max_score = max(buy_score, sell_score, hold_score)

        if max_score < 0.1:  # 所有分數都很低
            final_signal = 0
            final_confidence = 0.3
            recommendation = "持有觀望（信號不明確）"
        elif buy_score >= sell_score and buy_score > hold_score:
            final_signal = 1
            final_confidence = min(0.95, buy_score / (buy_score + sell_score + 0.01))
            recommendation = "買入"
        elif sell_score > buy_score and sell_score > hold_score:
            final_signal = -1
            final_confidence = min(0.95, sell_score / (buy_score + sell_score + 0.01))
            recommendation = "賣出"
        else:
            final_signal = 0
            final_confidence = 0.5
            recommendation = "持有觀望"

        # 計算一致性
        agreement = max(buy_votes, sell_votes, hold_votes) / total

        # 平均強度
        avg_strength = sum(p.strength for p in predictions) / total

        reasoning = f"信心加權分數：買入{buy_score:.3f}，賣出{sell_score:.3f}，持有{hold_score:.3f}"

        return EnsemblePrediction(
            final_signal=final_signal,
            final_confidence=final_confidence,
            final_strength=avg_strength,
            method_used="confidence_weighted",
            participating_models=total,
            agreement_ratio=agreement,
            buy_votes=buy_votes,
            sell_votes=sell_votes,
            hold_votes=hold_votes,
            individual_predictions=[p.to_dict() for p in predictions],
            recommendation=recommendation,
            reasoning=reasoning
        )

    def _use_best_model(self, predictions: List[ModelPrediction]) -> EnsemblePrediction:
        """使用最佳模型的預測"""
        if not predictions:
            return self._create_empty_prediction("no_predictions")

        # 選擇準確率和信心度綜合最高的模型
        best = max(predictions, key=lambda p: p.accuracy * p.confidence)

        if best.signal > 0:
            recommendation = "買入"
        elif best.signal < 0:
            recommendation = "賣出"
        else:
            recommendation = "持有觀望"

        buy_votes = sum(1 for p in predictions if p.signal > 0)
        sell_votes = sum(1 for p in predictions if p.signal < 0)
        hold_votes = sum(1 for p in predictions if p.signal == 0)

        reasoning = f"使用最佳模型 {best.model_id}（準確率 {best.accuracy:.1%}，信心度 {best.confidence:.1%}）"

        return EnsemblePrediction(
            final_signal=best.signal,
            final_confidence=best.confidence,
            final_strength=best.strength,
            method_used="best_model",
            participating_models=len(predictions),
            agreement_ratio=1.0 if len(predictions) == 1 else max(buy_votes, sell_votes, hold_votes) / len(predictions),
            buy_votes=buy_votes,
            sell_votes=sell_votes,
            hold_votes=hold_votes,
            individual_predictions=[p.to_dict() for p in predictions],
            recommendation=recommendation,
            reasoning=reasoning
        )

    def _create_empty_prediction(self, reason: str) -> EnsemblePrediction:
        """創建空預測結果"""
        return EnsemblePrediction(
            final_signal=0,
            final_confidence=0,
            final_strength=0,
            method_used="none",
            participating_models=0,
            agreement_ratio=0,
            buy_votes=0,
            sell_votes=0,
            hold_votes=0,
            individual_predictions=[],
            recommendation="持有觀望",
            reasoning=f"無法產生預測：{reason}"
        )

    def calculate_ensemble_metrics(
        self,
        predictions: List[ModelPrediction]
    ) -> Dict[str, Any]:
        """計算組合指標"""
        if not predictions:
            return {
                "avg_accuracy": 0,
                "avg_confidence": 0,
                "diversity_score": 0,
                "correlation": 0
            }

        n = len(predictions)

        # 平均準確率
        avg_accuracy = sum(p.accuracy for p in predictions) / n

        # 平均信心度
        avg_confidence = sum(p.confidence for p in predictions) / n

        # 多樣性分數（信號不一致程度）
        signals = [p.signal for p in predictions]
        signal_variance = sum((s - sum(signals)/n)**2 for s in signals) / n if n > 1 else 0
        diversity_score = signal_variance / 2  # 正規化到 0-1

        # 信號相關性（簡化版本）
        buy_ratio = sum(1 for s in signals if s > 0) / n
        sell_ratio = sum(1 for s in signals if s < 0) / n
        correlation = 1 - 2 * min(buy_ratio, sell_ratio)  # 1=完全一致，0=完全分歧

        return {
            "avg_accuracy": avg_accuracy,
            "avg_confidence": avg_confidence,
            "diversity_score": diversity_score,
            "correlation": correlation,
            "model_count": n
        }


# 單例實例
ensemble_service = StrategyEnsembleService()
