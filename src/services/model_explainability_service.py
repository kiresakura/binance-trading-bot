"""
Model Explainability Service - 模型解釋性服務

提供：
- SHAP 值計算
- 特徵重要性分析
- 預測解釋
- 模型決策可視化數據
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import json

from src.logger import logger


@dataclass
class FeatureImportance:
    """特徵重要性"""
    feature_name: str
    importance: float
    direction: str  # positive, negative, neutral
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionExplanation:
    """預測解釋"""
    model_version: str
    symbol: str
    timestamp: str
    prediction: str  # BUY, SELL, HOLD
    confidence: float
    base_value: float
    feature_contributions: List[FeatureImportance]
    top_positive_features: List[str]
    top_negative_features: List[str]
    explanation_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "base_value": self.base_value,
            "feature_contributions": [f.to_dict() for f in self.feature_contributions],
            "top_positive_features": self.top_positive_features,
            "top_negative_features": self.top_negative_features,
            "explanation_text": self.explanation_text
        }


@dataclass
class ModelExplainability:
    """模型解釋性結果"""
    model_version: str
    timestamp: str
    global_feature_importance: List[FeatureImportance]
    sample_explanations: List[PredictionExplanation]
    model_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "global_feature_importance": [f.to_dict() for f in self.global_feature_importance],
            "sample_explanations": [e.to_dict() for e in self.sample_explanations],
            "model_summary": self.model_summary
        }


class ModelExplainabilityService:
    """模型解釋性服務"""

    # 常見技術指標特徵名稱
    FEATURE_NAMES = [
        "rsi_14", "rsi_21",
        "macd", "macd_signal", "macd_hist",
        "ema_9", "ema_21", "ema_50",
        "bb_upper", "bb_middle", "bb_lower", "bb_width",
        "atr_14", "atr_21",
        "volume_sma", "volume_ratio",
        "price_change_1h", "price_change_4h", "price_change_24h",
        "high_low_range", "close_open_ratio",
        "stoch_k", "stoch_d",
        "adx", "plus_di", "minus_di",
        "obv", "obv_sma",
        "momentum_10", "momentum_20",
        "williams_r",
        "cci_20",
        "mfi_14"
    ]

    # 特徵解釋模板
    FEATURE_EXPLANATIONS = {
        "rsi_14": "14 週期相對強弱指標",
        "rsi_21": "21 週期相對強弱指標",
        "macd": "MACD 主線",
        "macd_signal": "MACD 信號線",
        "macd_hist": "MACD 柱狀圖",
        "ema_9": "9 週期指數移動平均",
        "ema_21": "21 週期指數移動平均",
        "ema_50": "50 週期指數移動平均",
        "bb_upper": "布林帶上軌",
        "bb_middle": "布林帶中軌",
        "bb_lower": "布林帶下軌",
        "bb_width": "布林帶寬度",
        "atr_14": "14 週期平均真實波幅",
        "volume_sma": "成交量移動平均",
        "volume_ratio": "成交量比率",
        "price_change_1h": "1 小時價格變化",
        "price_change_4h": "4 小時價格變化",
        "price_change_24h": "24 小時價格變化",
        "adx": "平均方向指數",
        "momentum_10": "10 週期動量",
        "momentum_20": "20 週期動量"
    }

    def __init__(self):
        self._cache: Dict[str, ModelExplainability] = {}
        self._cache_ttl = 600  # 10 分鐘緩存
        logger.info("ModelExplainabilityService initialized")

    async def explain_prediction(
        self,
        model_version: str,
        symbol: str,
        features: Dict[str, float],
        prediction: str,
        confidence: float,
        model_weights: Optional[Dict[str, float]] = None
    ) -> PredictionExplanation:
        """
        解釋單次預測

        Args:
            model_version: 模型版本
            symbol: 交易對
            features: 輸入特徵字典
            prediction: 預測結果
            confidence: 信心度
            model_weights: 模型權重（可選）

        Returns:
            PredictionExplanation: 預測解釋
        """
        # 計算特徵貢獻（簡化的 SHAP 近似）
        contributions = await self._calculate_feature_contributions(
            features, model_weights
        )

        # 排序特徵貢獻
        sorted_contributions = sorted(
            contributions,
            key=lambda x: abs(x.importance),
            reverse=True
        )

        # 找出正負貢獻最大的特徵
        top_positive = [
            f.feature_name for f in sorted_contributions
            if f.direction == "positive"
        ][:5]

        top_negative = [
            f.feature_name for f in sorted_contributions
            if f.direction == "negative"
        ][:5]

        # 生成解釋文字
        explanation_text = self._generate_explanation_text(
            prediction, confidence, sorted_contributions[:5]
        )

        return PredictionExplanation(
            model_version=model_version,
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat(),
            prediction=prediction,
            confidence=confidence,
            base_value=0.5,  # 基礎預測值
            feature_contributions=sorted_contributions,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            explanation_text=explanation_text
        )

    async def _calculate_feature_contributions(
        self,
        features: Dict[str, float],
        model_weights: Optional[Dict[str, float]] = None
    ) -> List[FeatureImportance]:
        """計算特徵貢獻（SHAP 近似）"""
        contributions = []

        # 如果沒有提供模型權重，使用默認權重
        if model_weights is None:
            model_weights = self._get_default_weights()

        for rank, (feature_name, value) in enumerate(features.items(), 1):
            # 獲取特徵權重
            weight = model_weights.get(feature_name, 0.01)

            # 計算貢獻（簡化計算：值 * 權重）
            # 實際 SHAP 會更複雜，這是簡化版本
            normalized_value = self._normalize_feature(feature_name, value)
            contribution = normalized_value * weight

            direction = "positive" if contribution > 0 else ("negative" if contribution < 0 else "neutral")

            contributions.append(FeatureImportance(
                feature_name=feature_name,
                importance=abs(contribution),
                direction=direction,
                rank=rank
            ))

        # 按重要性重新排序
        contributions.sort(key=lambda x: x.importance, reverse=True)
        for i, c in enumerate(contributions, 1):
            c.rank = i

        return contributions

    def _normalize_feature(self, feature_name: str, value: float) -> float:
        """標準化特徵值"""
        # RSI 類指標：0-100 範圍，50 為中性
        if "rsi" in feature_name.lower():
            return (value - 50) / 50

        # MACD 類指標
        if "macd" in feature_name.lower():
            return np.tanh(value / 100)

        # 價格變化：直接使用百分比
        if "change" in feature_name.lower():
            return np.tanh(value / 10)

        # 成交量比率
        if "volume_ratio" in feature_name.lower():
            return np.tanh(value - 1)

        # 其他指標：簡單標準化
        return np.tanh(value / 100)

    def _get_default_weights(self) -> Dict[str, float]:
        """獲取默認特徵權重"""
        return {
            # RSI 權重較高
            "rsi_14": 0.15,
            "rsi_21": 0.10,
            # MACD 權重
            "macd": 0.12,
            "macd_signal": 0.08,
            "macd_hist": 0.10,
            # EMA 權重
            "ema_9": 0.08,
            "ema_21": 0.08,
            "ema_50": 0.05,
            # 布林帶
            "bb_width": 0.06,
            # ATR
            "atr_14": 0.05,
            # 成交量
            "volume_ratio": 0.08,
            # 價格變化
            "price_change_1h": 0.10,
            "price_change_4h": 0.08,
            "price_change_24h": 0.05,
            # 動量
            "momentum_10": 0.06,
            "momentum_20": 0.04,
            # ADX
            "adx": 0.05,
        }

    def _generate_explanation_text(
        self,
        prediction: str,
        confidence: float,
        top_features: List[FeatureImportance]
    ) -> str:
        """生成解釋文字"""
        confidence_text = "高信心" if confidence > 0.7 else ("中等信心" if confidence > 0.5 else "低信心")

        if prediction == "BUY":
            action_text = "買入"
        elif prediction == "SELL":
            action_text = "賣出"
        else:
            action_text = "持有"

        explanation = f"模型以 {confidence_text}（{confidence:.1%}）建議{action_text}。\n\n主要影響因素：\n"

        for feat in top_features[:3]:
            feat_desc = self.FEATURE_EXPLANATIONS.get(feat.feature_name, feat.feature_name)
            direction_text = "支持" if feat.direction == "positive" else "反對"
            explanation += f"• {feat_desc}：{direction_text}此決策（重要性：{feat.importance:.2f}）\n"

        return explanation

    async def get_global_feature_importance(
        self,
        model_version: str,
        sample_predictions: Optional[List[Dict[str, Any]]] = None
    ) -> List[FeatureImportance]:
        """
        獲取全局特徵重要性

        Args:
            model_version: 模型版本
            sample_predictions: 樣本預測列表

        Returns:
            List[FeatureImportance]: 特徵重要性列表
        """
        if sample_predictions is None:
            # 使用默認權重
            weights = self._get_default_weights()
            importance_list = []

            for rank, (feature, weight) in enumerate(
                sorted(weights.items(), key=lambda x: x[1], reverse=True), 1
            ):
                importance_list.append(FeatureImportance(
                    feature_name=feature,
                    importance=weight,
                    direction="neutral",
                    rank=rank
                ))

            return importance_list

        # 聚合多個預測的特徵貢獻
        feature_totals: Dict[str, float] = {}
        feature_counts: Dict[str, int] = {}

        for pred in sample_predictions:
            features = pred.get("features", {})
            for feature_name, value in features.items():
                if feature_name not in feature_totals:
                    feature_totals[feature_name] = 0
                    feature_counts[feature_name] = 0
                feature_totals[feature_name] += abs(value)
                feature_counts[feature_name] += 1

        # 計算平均重要性
        avg_importance = {
            f: total / feature_counts[f]
            for f, total in feature_totals.items()
        }

        # 標準化
        max_imp = max(avg_importance.values()) if avg_importance else 1
        normalized = {f: v / max_imp for f, v in avg_importance.items()}

        importance_list = []
        for rank, (feature, importance) in enumerate(
            sorted(normalized.items(), key=lambda x: x[1], reverse=True), 1
        ):
            importance_list.append(FeatureImportance(
                feature_name=feature,
                importance=importance,
                direction="neutral",
                rank=rank
            ))

        return importance_list

    async def explain_model(
        self,
        model_version: str,
        sample_data: Optional[List[Dict[str, Any]]] = None
    ) -> ModelExplainability:
        """
        獲取完整模型解釋

        Args:
            model_version: 模型版本
            sample_data: 樣本數據

        Returns:
            ModelExplainability: 模型解釋結果
        """
        # 全局特徵重要性
        global_importance = await self.get_global_feature_importance(
            model_version, sample_data
        )

        # 生成樣本解釋
        sample_explanations = []
        if sample_data:
            for sample in sample_data[:5]:  # 最多 5 個樣本
                explanation = await self.explain_prediction(
                    model_version=model_version,
                    symbol=sample.get("symbol", "BTCUSDT"),
                    features=sample.get("features", {}),
                    prediction=sample.get("prediction", "HOLD"),
                    confidence=sample.get("confidence", 0.5)
                )
                sample_explanations.append(explanation)

        # 模型摘要
        model_summary = {
            "total_features": len(global_importance),
            "top_5_features": [f.feature_name for f in global_importance[:5]],
            "model_type": "gradient_boosting",  # 假設模型類型
            "training_samples": sample_data.__len__() if sample_data else 0,
            "feature_coverage": self._calculate_feature_coverage(global_importance)
        }

        return ModelExplainability(
            model_version=model_version,
            timestamp=datetime.utcnow().isoformat(),
            global_feature_importance=global_importance,
            sample_explanations=sample_explanations,
            model_summary=model_summary
        )

    def _calculate_feature_coverage(
        self,
        importance_list: List[FeatureImportance]
    ) -> Dict[str, float]:
        """計算特徵覆蓋率"""
        if not importance_list:
            return {"top_5": 0, "top_10": 0}

        total = sum(f.importance for f in importance_list)
        if total == 0:
            return {"top_5": 0, "top_10": 0}

        top_5 = sum(f.importance for f in importance_list[:5]) / total
        top_10 = sum(f.importance for f in importance_list[:10]) / total

        return {
            "top_5": round(top_5, 3),
            "top_10": round(top_10, 3)
        }

    async def compare_explanations(
        self,
        explanation1: PredictionExplanation,
        explanation2: PredictionExplanation
    ) -> Dict[str, Any]:
        """比較兩個預測的解釋"""
        # 找出共同和不同的主要特徵
        top_features_1 = set(explanation1.top_positive_features + explanation1.top_negative_features)
        top_features_2 = set(explanation2.top_positive_features + explanation2.top_negative_features)

        common_features = top_features_1.intersection(top_features_2)
        unique_to_1 = top_features_1 - top_features_2
        unique_to_2 = top_features_2 - top_features_1

        return {
            "prediction_1": explanation1.prediction,
            "prediction_2": explanation2.prediction,
            "confidence_1": explanation1.confidence,
            "confidence_2": explanation2.confidence,
            "common_important_features": list(common_features),
            "unique_to_prediction_1": list(unique_to_1),
            "unique_to_prediction_2": list(unique_to_2),
            "similarity_score": len(common_features) / max(
                len(top_features_1.union(top_features_2)), 1
            )
        }


# 單例實例
explainability_service = ModelExplainabilityService()
