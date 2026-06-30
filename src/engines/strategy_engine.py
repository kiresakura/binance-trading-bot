"""
策略引擎

負責：
- 管理多個策略
- 信號聚合
- 策略選擇
- 多時間框架確認 (v0.3.3)
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import pandas as pd

from src.logger import logger
from src.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from src.features.feature_store import feature_store
from src.strategies.market_regime import (
    MarketRegimeAnalyzer,
    MarketRegime,
    RegimeAnalysis,
    market_regime_analyzer
)

# 用於 CPU 密集型策略計算的線程池
_strategy_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="strategy_")


@dataclass
class StrategyWeight:
    """策略權重配置"""
    name: str
    weight: float = 1.0
    enabled: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)  # 啟用條件


@dataclass
class AggregatedSignal:
    """聚合後的信號"""
    timestamp: datetime
    symbol: str
    direction: int  # 1=做多, -1=做空, 0=觀望
    strength: float  # 0-1
    confidence: float  # 0-1
    contributing_strategies: List[str]
    individual_signals: List[Signal]
    recommendation: str
    regime_analysis: Optional[RegimeAnalysis] = None  # 市場狀態分析
    metadata: Dict[str, Any] = field(default_factory=dict)


class SignalConsistencyFilter:
    """
    信號一致性過濾器 (v0.3.2)

    功能：
    1. 信號平滑：避免頻繁翻轉
    2. 確認要求：新信號需要持續確認
    3. 反彈過濾：防止假突破後立即反向交易
    """

    def __init__(
        self,
        confirmation_bars: int = 2,        # 信號確認需要的K線數
        min_bars_between_trades: int = 3,  # 兩次反向交易之間的最小間隔
        signal_decay: float = 0.8,         # 信號強度衰減係數
    ):
        self.confirmation_bars = confirmation_bars
        self.min_bars_between_trades = min_bars_between_trades
        self.signal_decay = signal_decay

        # 歷史信號追蹤
        self._signal_history: List[Dict[str, Any]] = []
        self._last_trade_direction: int = 0
        self._bars_since_last_trade: int = 0
        self._pending_signal: Optional[Dict[str, Any]] = None
        self._pending_count: int = 0

    def filter_signal(
        self,
        direction: int,
        strength: float,
        confidence: float,
        symbol: str
    ) -> Tuple[int, float, float, str]:
        """
        過濾和驗證信號

        Args:
            direction: 信號方向 (1/-1/0)
            strength: 信號強度
            confidence: 置信度
            symbol: 交易對

        Returns:
            (過濾後方向, 調整後強度, 調整後置信度, 過濾原因)
        """
        self._bars_since_last_trade += 1

        # 記錄當前信號
        current_signal = {
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'symbol': symbol,
            'timestamp': datetime.utcnow()
        }

        # 無信號情況
        if direction == 0:
            self._pending_signal = None
            self._pending_count = 0
            return 0, 0, 0, "無信號"

        # 反翻轉過濾：如果與上次交易方向相反且間隔太短
        if (self._last_trade_direction != 0 and
            direction != self._last_trade_direction and
            self._bars_since_last_trade < self.min_bars_between_trades):

            # 降低反向信號的強度和置信度
            penalty = self._bars_since_last_trade / self.min_bars_between_trades
            adjusted_strength = strength * penalty * 0.7
            adjusted_confidence = confidence * penalty * 0.7

            if adjusted_strength < 0.3 or adjusted_confidence < 0.4:
                return 0, 0, 0, f"反翻轉過濾（距上次{self._bars_since_last_trade}根K線）"

            return direction, adjusted_strength, adjusted_confidence, "反翻轉懲罰"

        # 信號確認：新信號需要持續確認
        if self._pending_signal is None or self._pending_signal['direction'] != direction:
            # 新方向信號，開始確認計數
            self._pending_signal = current_signal
            self._pending_count = 1

            if self.confirmation_bars > 1:
                return 0, 0, 0, f"等待確認 ({self._pending_count}/{self.confirmation_bars})"
        else:
            # 同方向信號，增加確認計數
            self._pending_count += 1

            if self._pending_count < self.confirmation_bars:
                # 還在確認中，但可以使用衰減的信號強度
                decayed_strength = strength * (self.signal_decay ** (self.confirmation_bars - self._pending_count))
                return 0, decayed_strength, confidence * 0.5, f"等待確認 ({self._pending_count}/{self.confirmation_bars})"

        # 通過確認，信號有效
        self._signal_history.append(current_signal)
        if len(self._signal_history) > 100:
            self._signal_history = self._signal_history[-100:]

        # 信號一致性加成：如果與歷史信號方向一致
        consistency_bonus = self._calculate_consistency_bonus(direction)
        final_strength = min(1.0, strength * (1 + consistency_bonus * 0.1))
        final_confidence = min(1.0, confidence * (1 + consistency_bonus * 0.1))

        return direction, final_strength, final_confidence, "信號通過"

    def record_trade(self, direction: int):
        """記錄實際執行的交易"""
        if direction != 0:
            self._last_trade_direction = direction
            self._bars_since_last_trade = 0

    def _calculate_consistency_bonus(self, direction: int) -> float:
        """計算信號一致性加成"""
        if len(self._signal_history) < 3:
            return 0

        # 檢查最近5個信號的方向一致性
        recent = self._signal_history[-5:]
        same_direction = sum(1 for s in recent if s['direction'] == direction)

        return same_direction / len(recent)

    def get_status(self) -> Dict[str, Any]:
        """獲取過濾器狀態"""
        return {
            'pending_signal': self._pending_signal['direction'] if self._pending_signal else None,
            'pending_count': self._pending_count,
            'confirmation_required': self.confirmation_bars,
            'last_trade_direction': self._last_trade_direction,
            'bars_since_last_trade': self._bars_since_last_trade,
            'history_length': len(self._signal_history)
        }


# ==================== 多時間框架分析器 (v0.3.3) ====================

@dataclass
class MTFAnalysis:
    """多時間框架分析結果"""
    higher_tf_trend: int           # 高時間框架趨勢方向 (1=上漲, -1=下跌, 0=中性)
    trend_strength: float          # 趨勢強度 (0-1)
    trend_alignment: bool          # 是否與交易方向一致
    confidence_multiplier: float   # 置信度調整係數
    timeframes_analyzed: List[str] # 分析的時間框架
    details: Dict[str, Any] = field(default_factory=dict)


class MultiTimeframeAnalyzer:
    """
    多時間框架分析器 (v0.3.3)

    功能：
    1. 分析高時間框架的趨勢方向
    2. 確認交易信號是否與高時間框架趨勢一致
    3. 調整信號置信度

    核心原則：
    - 順勢交易：與高時間框架趨勢方向一致的交易更可靠
    - 逆勢警告：與高時間框架趨勢方向相反的交易需謹慎
    """

    # 時間框架權重（越高的時間框架權重越大）
    TIMEFRAME_WEIGHTS = {
        '5m': 0.1,
        '15m': 0.2,
        '1h': 0.3,
        '4h': 0.5,
        '1d': 0.7,
    }

    def __init__(
        self,
        higher_timeframes: List[str] = None,
        min_alignment_confidence: float = 0.6,
        counter_trend_penalty: float = 0.4,
        aligned_trend_bonus: float = 0.2,
    ):
        """
        初始化多時間框架分析器

        Args:
            higher_timeframes: 要分析的高時間框架列表
            min_alignment_confidence: 趨勢一致性確認的最低置信度
            counter_trend_penalty: 逆勢交易的懲罰係數
            aligned_trend_bonus: 順勢交易的加成係數
        """
        self.higher_timeframes = higher_timeframes or ['4h', '1d']
        self.min_alignment_confidence = min_alignment_confidence
        self.counter_trend_penalty = counter_trend_penalty
        self.aligned_trend_bonus = aligned_trend_bonus

        # 緩存高時間框架數據
        self._htf_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_expiry: Dict[str, datetime] = {}

        # 趨勢歷史追蹤
        self._trend_history: Dict[str, deque] = {
            tf: deque(maxlen=10) for tf in self.higher_timeframes
        }

        logger.info(f"MTF Analyzer initialized with timeframes: {self.higher_timeframes}")

    def analyze(
        self,
        features: pd.DataFrame,
        signal_direction: int,
        symbol: str = ""
    ) -> MTFAnalysis:
        """
        分析多時間框架趨勢

        Args:
            features: 當前時間框架的特徵數據
            signal_direction: 信號方向 (1=買入, -1=賣出)
            symbol: 交易對

        Returns:
            MTFAnalysis 分析結果
        """
        if len(features) < 20:
            return MTFAnalysis(
                higher_tf_trend=0,
                trend_strength=0,
                trend_alignment=True,
                confidence_multiplier=1.0,
                timeframes_analyzed=[],
                details={'error': 'insufficient_data'}
            )

        # 從當前數據推斷高時間框架趨勢
        htf_trends = self._analyze_implied_htf_trend(features)

        # 計算加權趨勢方向
        weighted_trend = 0
        total_weight = 0
        details = {}

        for tf, trend_info in htf_trends.items():
            weight = self.TIMEFRAME_WEIGHTS.get(tf, 0.3)
            weighted_trend += trend_info['direction'] * trend_info['strength'] * weight
            total_weight += weight

            details[tf] = {
                'direction': trend_info['direction'],
                'strength': trend_info['strength'],
                'weight': weight
            }

            # 更新趨勢歷史
            if tf in self._trend_history:
                self._trend_history[tf].append(trend_info['direction'])

        # 確定最終趨勢方向
        if total_weight > 0:
            avg_trend = weighted_trend / total_weight
            if avg_trend > 0.2:
                higher_tf_trend = 1
            elif avg_trend < -0.2:
                higher_tf_trend = -1
            else:
                higher_tf_trend = 0
            trend_strength = min(1.0, abs(avg_trend))
        else:
            higher_tf_trend = 0
            trend_strength = 0

        # 判斷趨勢一致性
        trend_alignment = (
            signal_direction == 0 or
            higher_tf_trend == 0 or
            signal_direction == higher_tf_trend
        )

        # 計算置信度調整係數
        if signal_direction == 0:
            confidence_multiplier = 1.0
        elif trend_alignment:
            # 順勢交易：加成
            confidence_multiplier = 1.0 + self.aligned_trend_bonus * trend_strength
        else:
            # 逆勢交易：懲罰
            confidence_multiplier = 1.0 - self.counter_trend_penalty * trend_strength

        # 檢查趨勢持續性（連續同向可增加置信度）
        trend_consistency = self._calculate_trend_consistency(higher_tf_trend)
        if trend_consistency > 0.7 and trend_alignment:
            confidence_multiplier *= 1.1  # 高一致性加成

        details['trend_consistency'] = trend_consistency
        details['weighted_trend'] = weighted_trend / total_weight if total_weight > 0 else 0

        analysis = MTFAnalysis(
            higher_tf_trend=higher_tf_trend,
            trend_strength=trend_strength,
            trend_alignment=trend_alignment,
            confidence_multiplier=max(0.5, min(1.5, confidence_multiplier)),
            timeframes_analyzed=list(htf_trends.keys()),
            details=details
        )

        logger.debug(f"MTF Analysis: trend={higher_tf_trend}, strength={trend_strength:.2f}, "
                    f"aligned={trend_alignment}, multiplier={confidence_multiplier:.2f}")

        return analysis

    def _analyze_implied_htf_trend(self, features: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        從當前數據推斷高時間框架趨勢

        使用長週期指標作為高時間框架的代理
        """
        latest = features.iloc[-1]
        results = {}

        # 4H 時間框架 - 使用中長期指標
        ema_21 = latest.get('ema_21', latest.get('sma_25', 0))
        sma_50 = latest.get('sma_50', 0)
        close = latest.get('close', 0)

        if close > 0 and sma_50 > 0:
            # 價格相對於長期均線的位置
            price_vs_sma50 = (close - sma_50) / sma_50

            if price_vs_sma50 > 0.02:  # 價格高於 SMA50 2%+
                direction_4h = 1
                strength_4h = min(1.0, abs(price_vs_sma50) * 10)
            elif price_vs_sma50 < -0.02:
                direction_4h = -1
                strength_4h = min(1.0, abs(price_vs_sma50) * 10)
            else:
                direction_4h = 0
                strength_4h = 0

            results['4h'] = {'direction': direction_4h, 'strength': strength_4h}

        # 1D 時間框架 - 使用更長期的趨勢指標
        adx = latest.get('adx', 20)
        trend_ema = latest.get('trend_ema_9_21', 0)
        sma_slope = latest.get('sma_25_slope', 0)

        # 綜合判斷日線趨勢
        if adx > 25:  # 有趨勢
            if trend_ema > 0 and sma_slope > 0:
                direction_1d = 1
                strength_1d = min(1.0, adx / 50)
            elif trend_ema < 0 and sma_slope < 0:
                direction_1d = -1
                strength_1d = min(1.0, adx / 50)
            else:
                direction_1d = 0
                strength_1d = 0.3
        else:
            direction_1d = 0
            strength_1d = 0

        results['1d'] = {'direction': direction_1d, 'strength': strength_1d}

        # 1H 時間框架（當前時間框架的確認）
        rsi = latest.get('rsi_14', 50)
        macd_hist = latest.get('macd_hist', 0)

        if rsi > 55 and macd_hist > 0:
            direction_1h = 1
            strength_1h = min(1.0, (rsi - 50) / 30 + abs(macd_hist) * 10)
        elif rsi < 45 and macd_hist < 0:
            direction_1h = -1
            strength_1h = min(1.0, (50 - rsi) / 30 + abs(macd_hist) * 10)
        else:
            direction_1h = 0
            strength_1h = 0.2

        results['1h'] = {'direction': direction_1h, 'strength': strength_1h}

        return results

    def _calculate_trend_consistency(self, current_trend: int) -> float:
        """計算趨勢一致性（歷史趨勢方向的一致程度）"""
        if not any(self._trend_history.values()):
            return 0.5

        total_same = 0
        total_count = 0

        for tf, history in self._trend_history.items():
            if history:
                same_direction = sum(1 for t in history if t == current_trend)
                total_same += same_direction
                total_count += len(history)

        return total_same / total_count if total_count > 0 else 0.5

    def update_htf_data(self, timeframe: str, data: Dict[str, Any]):
        """更新高時間框架數據緩存"""
        self._htf_cache[timeframe] = data
        self._cache_expiry[timeframe] = datetime.utcnow()

    def get_status(self) -> Dict[str, Any]:
        """獲取 MTF 分析器狀態"""
        return {
            'higher_timeframes': self.higher_timeframes,
            'counter_trend_penalty': self.counter_trend_penalty,
            'aligned_trend_bonus': self.aligned_trend_bonus,
            'trend_history': {
                tf: list(history) for tf, history in self._trend_history.items()
            },
            'cache_status': {
                tf: str(expiry) for tf, expiry in self._cache_expiry.items()
            }
        }


class MarketStateClassifier:
    """
    市場狀態分類器

    根據特徵判斷當前市場狀態：
    - TRENDING_UP: 上升趨勢
    - TRENDING_DOWN: 下降趨勢
    - RANGING: 震盪
    - VOLATILE: 高波動
    - BREAKOUT: 突破
    """

    def __init__(self):
        self.states = ['trending_up', 'trending_down', 'ranging', 'volatile', 'breakout']
        self._current_state = 'ranging'
        self._state_probabilities = {s: 0.2 for s in self.states}

    def classify(self, features: pd.DataFrame) -> str:
        """
        分類市場狀態

        Args:
            features: 特徵 DataFrame

        Returns:
            市場狀態字符串
        """
        if len(features) < 20:
            return 'ranging'

        latest = features.iloc[-1]

        # 計算各狀態概率
        probs = {}

        # 趨勢判斷
        adx = latest.get('adx', 20)
        trend_ema = latest.get('trend_ema_9_21', 0)
        sma_slope = latest.get('sma_25_slope', 0)

        if adx > 25:
            if trend_ema > 0 and sma_slope > 0:
                probs['trending_up'] = 0.6
                probs['trending_down'] = 0.1
            elif trend_ema < 0 and sma_slope < 0:
                probs['trending_up'] = 0.1
                probs['trending_down'] = 0.6
            else:
                probs['trending_up'] = 0.3
                probs['trending_down'] = 0.3
        else:
            probs['trending_up'] = 0.15
            probs['trending_down'] = 0.15

        # 波動判斷
        volatility = latest.get('volatility_10', 0)
        bb_width = latest.get('bb_width', 0.02)

        if volatility > 0.03 or bb_width > 0.06:
            probs['volatile'] = 0.5
        else:
            probs['volatile'] = 0.1

        # 震盪判斷
        bb_position = latest.get('bb_position', 0.5)
        rsi = latest.get('rsi_14', 50)

        if adx < 20 and 0.2 < bb_position < 0.8 and 40 < rsi < 60:
            probs['ranging'] = 0.5
        else:
            probs['ranging'] = 0.15

        # 突破判斷
        bb_squeeze = latest.get('bb_squeeze', 0)
        volume_ratio = latest.get('volume_ratio', 1)

        if bb_squeeze and volume_ratio > 1.5:
            probs['breakout'] = 0.4
        else:
            probs['breakout'] = 0.1

        # 正規化
        total = sum(probs.values())
        for k in probs:
            probs[k] /= total

        # 更新狀態
        self._state_probabilities = probs
        self._current_state = max(probs, key=probs.get)

        return self._current_state

    def get_probabilities(self) -> Dict[str, float]:
        """獲取狀態概率"""
        return self._state_probabilities.copy()


class StrategyEngine:
    """
    策略引擎

    功能：
    - 註冊和管理多個策略
    - 根據市場狀態選擇策略
    - 聚合多策略信號
    - 提供最終交易建議
    """

    def __init__(self):
        self._strategies: Dict[str, BaseStrategy] = {}
        self._weights: Dict[str, StrategyWeight] = {}
        self._base_weights: Dict[str, float] = {}  # 保存基礎權重
        self._market_classifier = MarketStateClassifier()
        self._regime_analyzer = market_regime_analyzer  # 使用新的市場狀態分析器
        self._signal_filter = SignalConsistencyFilter(
            confirmation_bars=2,         # 需要2根K線確認
            min_bars_between_trades=3,   # 反向交易最少間隔3根K線
            signal_decay=0.8
        )
        # v0.3.3: 多時間框架分析器
        self._mtf_analyzer = MultiTimeframeAnalyzer(
            higher_timeframes=['4h', '1d'],
            counter_trend_penalty=0.4,   # 逆勢懲罰 40%
            aligned_trend_bonus=0.2      # 順勢加成 20%
        )
        self._mtf_enabled = True  # 可以動態開關

        # 策略-市場狀態映射（使用新的 MarketRegime 枚舉）
        self._strategy_state_mapping: Dict[str, List[str]] = {
            'trending_up': ['trend_following', 'momentum', 'ppo_agent'],
            'trending_down': ['trend_following', 'momentum', 'ppo_agent'],
            'ranging': ['mean_reversion', 'grid'],
            'volatile': ['volatility_breakout', 'ppo_agent'],
            'breakout': ['momentum', 'breakout']
        }

        # 新的 MarketRegime 到策略的映射
        self._regime_strategy_mapping: Dict[MarketRegime, List[str]] = {
            MarketRegime.STRONG_TREND_UP: ['trend_following', 'momentum'],
            MarketRegime.TREND_UP: ['trend_following', 'momentum'],
            MarketRegime.WEAK_TREND_UP: ['trend_following', 'momentum', 'mean_reversion'],
            MarketRegime.STRONG_TREND_DOWN: ['trend_following', 'momentum'],
            MarketRegime.TREND_DOWN: ['trend_following', 'momentum'],
            MarketRegime.WEAK_TREND_DOWN: ['trend_following', 'momentum', 'mean_reversion'],
            MarketRegime.RANGING: ['mean_reversion'],
            MarketRegime.HIGH_VOLATILITY: ['momentum'],
            MarketRegime.LOW_VOLATILITY: ['mean_reversion'],
            MarketRegime.BREAKOUT_UP: ['momentum', 'trend_following'],
            MarketRegime.BREAKOUT_DOWN: ['momentum', 'trend_following'],
        }

        # 信號歷史
        self._signal_history: List[AggregatedSignal] = []

        # 衝突檢測配置
        self._conflict_strategies = {
            ('trend_following', 'mean_reversion'),  # 趨勢和均值回歸衝突
        }

        logger.info("StrategyEngine initialized")

    def register_strategy(
        self,
        strategy: BaseStrategy,
        weight: float = 1.0,
        market_states: Optional[List[str]] = None
    ):
        """
        註冊策略

        Args:
            strategy: 策略實例
            weight: 權重
            market_states: 適用的市場狀態列表
        """
        name = strategy.name
        self._strategies[name] = strategy
        self._weights[name] = StrategyWeight(
            name=name,
            weight=weight,
            enabled=strategy.config.enabled
        )
        # 保存基礎權重（用於市場狀態調整）
        self._base_weights[name] = weight

        # 更新策略-狀態映射
        if market_states:
            for state in market_states:
                if state not in self._strategy_state_mapping:
                    self._strategy_state_mapping[state] = []
                if name not in self._strategy_state_mapping[state]:
                    self._strategy_state_mapping[state].append(name)

        logger.info(f"Registered strategy: {name} with weight {weight}")

    def unregister_strategy(self, name: str):
        """取消註冊策略"""
        if name in self._strategies:
            del self._strategies[name]
            del self._weights[name]
            logger.info(f"Unregistered strategy: {name}")

    def enable_strategy(self, name: str):
        """啟用策略"""
        if name in self._weights:
            self._weights[name].enabled = True
            logger.info(f"Enabled strategy: {name}")

    def disable_strategy(self, name: str):
        """停用策略"""
        if name in self._weights:
            self._weights[name].enabled = False
            logger.info(f"Disabled strategy: {name}")

    def set_weight(self, name: str, weight: float):
        """設置策略權重"""
        if name in self._weights:
            self._weights[name].weight = weight
            logger.info(f"Set weight for {name}: {weight}")

    async def generate_signals(
        self,
        symbol: str,
        features: pd.DataFrame,
        current_price: float,
        position: float = 0.0
    ) -> AggregatedSignal:
        """
        生成聚合信號

        Args:
            symbol: 交易對
            features: 特徵數據
            current_price: 當前價格
            position: 當前持倉

        Returns:
            聚合後的信號
        """
        # 使用新的市場狀態分析器
        regime_analysis = self._regime_analyzer.analyze(features, current_price)
        logger.debug(f"Market regime: {regime_analysis.regime.value}, "
                    f"confidence: {regime_analysis.confidence:.2f}")

        # v0.9.0: 廣播市場狀態更新
        try:
            from src.api.websocket_manager import ws_manager
            import asyncio
            asyncio.create_task(ws_manager.broadcast_market_regime_update(
                symbol=symbol,
                regime=regime_analysis.regime.value,
                confidence=regime_analysis.confidence,
                volatility=regime_analysis.volatility_level,
                trend_strength=regime_analysis.trend_strength,
                strategy_weights=regime_analysis.strategy_weights
            ))
        except Exception as e:
            logger.debug(f"Failed to broadcast market regime: {e}")

        # 根據市場狀態動態調整策略權重
        self._adjust_strategy_weights(regime_analysis)

        # 舊的市場分類（向後兼容）
        market_state = self._market_classifier.classify(features)

        # 獲取適用的策略（結合新舊邏輯）
        applicable_strategies = self._get_applicable_strategies_v2(regime_analysis, market_state)

        # 收集信號 - 使用線程池避免阻塞事件循環
        signals: List[Signal] = []
        loop = asyncio.get_event_loop()

        # 準備並行執行的策略任務
        strategy_tasks = []
        strategy_names = []

        for name in applicable_strategies:
            strategy = self._strategies.get(name)
            weight_config = self._weights.get(name)

            if not strategy or not weight_config or not weight_config.enabled:
                continue

            # 將同步的 generate_signal 移到線程池執行
            strategy_names.append(name)
            task = loop.run_in_executor(
                _strategy_executor,
                partial(strategy.generate_signal, symbol, features, current_price, position)
            )
            strategy_tasks.append(task)

        # 並行等待所有策略完成
        if strategy_tasks:
            results = await asyncio.gather(*strategy_tasks, return_exceptions=True)

            for name, result in zip(strategy_names, results):
                if isinstance(result, Exception):
                    logger.error(f"Error generating signal from {name}: {result}")
                elif result is not None:
                    signals.append(result)

        # 檢測並處理策略衝突
        signals = self._resolve_signal_conflicts(signals, regime_analysis)

        # v0.3.3: 預計算信號方向用於 MTF 分析
        preliminary_direction = 0
        if signals:
            weighted_dir = sum(s.direction * s.strength for s in signals if s.direction != 0)
            if weighted_dir > 0.3:
                preliminary_direction = 1
            elif weighted_dir < -0.3:
                preliminary_direction = -1

        # v0.3.3: 多時間框架確認
        mtf_analysis = None
        if self._mtf_enabled and preliminary_direction != 0:
            mtf_analysis = self._mtf_analyzer.analyze(
                features=features,
                signal_direction=preliminary_direction,
                symbol=symbol
            )
            logger.debug(f"MTF Analysis for {symbol}: trend={mtf_analysis.higher_tf_trend}, "
                        f"aligned={mtf_analysis.trend_alignment}, mult={mtf_analysis.confidence_multiplier:.2f}")

            # v0.9.0: 廣播 MTF 分析
            try:
                from src.api.websocket_manager import ws_manager
                import asyncio
                # 準備時間框架數據
                timeframes_data = []
                if hasattr(mtf_analysis, 'timeframe_trends'):
                    timeframes_data = [
                        {
                            "timeframe": tf,
                            "trend": trend.get("trend", 0),
                            "strength": trend.get("strength", 0.5)
                        }
                        for tf, trend in mtf_analysis.timeframe_trends.items()
                    ]
                overall_trend_str = "bullish" if mtf_analysis.higher_tf_trend == 1 else ("bearish" if mtf_analysis.higher_tf_trend == -1 else "neutral")
                asyncio.create_task(ws_manager.broadcast_mtf_analysis(
                    symbol=symbol,
                    overall_trend=overall_trend_str,
                    trend_alignment=mtf_analysis.confidence_multiplier,
                    primary_trend=overall_trend_str,
                    timeframes=timeframes_data
                ))
            except Exception as e:
                logger.debug(f"Failed to broadcast MTF analysis: {e}")

        # 聚合信號（包含 MTF 分析）
        aggregated = self._aggregate_signals(symbol, signals, market_state, regime_analysis, mtf_analysis)

        # v0.3.2: 應用信號一致性過濾
        filtered_direction, filtered_strength, filtered_confidence, filter_reason = \
            self._signal_filter.filter_signal(
                aggregated.direction,
                aggregated.strength,
                aggregated.confidence,
                symbol
            )

        # 更新聚合信號
        if filtered_direction != aggregated.direction:
            logger.debug(f"Signal filtered: {aggregated.direction} -> {filtered_direction} ({filter_reason})")
            aggregated = AggregatedSignal(
                timestamp=aggregated.timestamp,
                symbol=aggregated.symbol,
                direction=filtered_direction,
                strength=filtered_strength,
                confidence=filtered_confidence,
                contributing_strategies=aggregated.contributing_strategies,
                individual_signals=aggregated.individual_signals,
                recommendation=f"{aggregated.recommendation} | {filter_reason}",
                regime_analysis=aggregated.regime_analysis,
                metadata={**aggregated.metadata, 'filter_reason': filter_reason}
            )

        # 記錄
        self._signal_history.append(aggregated)
        if len(self._signal_history) > 1000:
            self._signal_history = self._signal_history[-1000:]

        # WebSocket 廣播信號更新
        if aggregated.direction != 0 or signals:
            try:
                from src.api.websocket_manager import ws_manager
                import asyncio
                # 廣播聚合信號
                asyncio.create_task(ws_manager.broadcast_signal(
                    symbol=symbol,
                    direction=aggregated.direction,
                    strength=aggregated.strength,
                    confidence=aggregated.confidence,
                    strategy_name=', '.join(aggregated.contributing_strategies) if aggregated.contributing_strategies else 'none',
                    reason=aggregated.recommendation
                ))
            except ImportError:
                pass  # WebSocket 管理器尚未初始化
            except Exception as e:
                logger.error(f"Failed to broadcast signal: {e}")

        # v0.9.0: 廣播成交量分析
        try:
            from src.api.websocket_manager import ws_manager
            import asyncio
            volume_ratio = features.get('volume_ratio', 1.0)
            # 判斷成交量狀態
            if volume_ratio >= 2.0:
                volume_status = "surge"
            elif volume_ratio >= 1.2:
                volume_status = "confirmed"
            elif volume_ratio < 0.8:
                volume_status = "dry"
            else:
                volume_status = "normal"
            asyncio.create_task(ws_manager.broadcast_volume_analysis(
                symbol=symbol,
                volume_ratio=volume_ratio,
                volume_status=volume_status,
                is_surge=volume_ratio >= 2.0,
                is_dry=volume_ratio < 0.8
            ))
        except Exception as e:
            logger.debug(f"Failed to broadcast volume analysis: {e}")

        # v0.9.0: 廣播 Kelly 倉位建議
        try:
            from src.api.websocket_manager import ws_manager
            import asyncio
            # 從貢獻策略中計算 Kelly
            kelly_fraction = 0.1  # 默認值
            win_rate = 0.5
            avg_win = 0.02
            avg_loss = 0.01
            for strategy_name in aggregated.contributing_strategies or []:
                strategy = self._strategies.get(strategy_name)
                if strategy and hasattr(strategy, 'get_kelly_stats'):
                    kelly_stats = strategy.get_kelly_stats()
                    if kelly_stats.get('sufficient_data', False):
                        kelly_fraction = kelly_stats.get('kelly_fraction', 0.1)
                        win_rate = kelly_stats.get('win_rate', 0.5)
                        avg_win = kelly_stats.get('avg_win', 0.02)
                        avg_loss = kelly_stats.get('avg_loss', 0.01)
                        break
            # 根據信號強度和置信度調整 Kelly
            adjusted_fraction = kelly_fraction * aggregated.strength * aggregated.confidence
            asyncio.create_task(ws_manager.broadcast_kelly_position(
                symbol=symbol,
                kelly_fraction=kelly_fraction,
                adjusted_fraction=min(adjusted_fraction, 0.25),  # 最大 25%
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss
            ))
        except Exception as e:
            logger.debug(f"Failed to broadcast Kelly position: {e}")

        return aggregated

    def _get_applicable_strategies(self, market_state: str) -> List[str]:
        """獲取適用於當前市場狀態的策略"""
        # 獲取映射的策略
        mapped = self._strategy_state_mapping.get(market_state, [])

        # 過濾出已註冊且啟用的策略
        applicable = []
        for name in mapped:
            if name in self._strategies and self._weights.get(name, StrategyWeight(name)).enabled:
                applicable.append(name)

        # 如果沒有映射的策略，使用所有啟用的策略
        if not applicable:
            applicable = [
                name for name, w in self._weights.items()
                if w.enabled and name in self._strategies
            ]

        return applicable

    def _get_applicable_strategies_v2(
        self,
        regime_analysis: RegimeAnalysis,
        market_state: str
    ) -> List[str]:
        """
        獲取適用於當前市場狀態的策略（v2 版本，結合 MarketRegime）

        Args:
            regime_analysis: 市場狀態分析結果
            market_state: 舊的市場狀態字符串（向後兼容）

        Returns:
            適用的策略名稱列表
        """
        regime = regime_analysis.regime

        # 首先嘗試使用新的 regime 映射
        mapped = self._regime_strategy_mapping.get(regime, [])

        # 過濾出已註冊且啟用的策略
        applicable = []
        for name in mapped:
            if name in self._strategies and self._weights.get(name, StrategyWeight(name)).enabled:
                applicable.append(name)

        # 如果新映射沒有結果，回退到舊映射
        if not applicable:
            applicable = self._get_applicable_strategies(market_state)

        # 如果仍然沒有結果，使用所有啟用的策略
        if not applicable:
            applicable = [
                name for name, w in self._weights.items()
                if w.enabled and name in self._strategies
            ]

        return applicable

    def _adjust_strategy_weights(self, regime_analysis: RegimeAnalysis):
        """
        根據市場狀態動態調整策略權重 (v0.3.2 增強版)

        Args:
            regime_analysis: 市場狀態分析結果
        """
        regime = regime_analysis.regime
        confidence = regime_analysis.confidence
        trend_strength = regime_analysis.trend_strength
        volatility_level = regime_analysis.volatility_level

        # 獲取建議的權重調整
        suggested_weights = regime_analysis.strategy_weights

        # v0.3.2: 增強的權重調整邏輯
        regime_multipliers = {}

        # 強趨勢市場：大幅提升趨勢策略，降低均值回歸
        if regime in [MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN]:
            regime_multipliers = {
                'trend_following': 1.5 * confidence,
                'momentum': 1.3 * confidence,
                'mean_reversion': 0.3,  # 大幅降低
            }
        # 普通趨勢市場
        elif regime in [MarketRegime.TREND_UP, MarketRegime.TREND_DOWN]:
            regime_multipliers = {
                'trend_following': 1.3 * confidence,
                'momentum': 1.2 * confidence,
                'mean_reversion': 0.5,
            }
        # 弱趨勢/過渡市場
        elif regime in [MarketRegime.WEAK_TREND_UP, MarketRegime.WEAK_TREND_DOWN]:
            regime_multipliers = {
                'trend_following': 1.0,
                'momentum': 0.9,
                'mean_reversion': 0.8,
            }
        # 震盪市場：提升均值回歸，降低趨勢策略
        elif regime == MarketRegime.RANGING:
            regime_multipliers = {
                'trend_following': 0.4,
                'momentum': 0.5,
                'mean_reversion': 1.5 * confidence,
            }
        # 高波動市場：降低所有策略倉位，謹慎交易
        elif regime == MarketRegime.HIGH_VOLATILITY:
            high_vol_penalty = max(0.5, 1.0 - volatility_level * 0.5)
            regime_multipliers = {
                'trend_following': 0.7 * high_vol_penalty,
                'momentum': 0.8 * high_vol_penalty,  # 動量策略在高波動時有機會
                'mean_reversion': 0.5 * high_vol_penalty,
            }
        # 低波動市場：均值回歸最佳
        elif regime == MarketRegime.LOW_VOLATILITY:
            regime_multipliers = {
                'trend_following': 0.6,
                'momentum': 0.5,
                'mean_reversion': 1.4 * confidence,
            }
        # 突破市場：動量策略優先
        elif regime in [MarketRegime.BREAKOUT_UP, MarketRegime.BREAKOUT_DOWN]:
            regime_multipliers = {
                'trend_following': 1.2,
                'momentum': 1.6 * confidence,
                'mean_reversion': 0.3,
            }

        # 合併 suggested_weights 和 regime_multipliers
        for name in self._strategies.keys():
            if name in self._weights and name in self._base_weights:
                # 優先使用 regime_multipliers，否則用 suggested_weights
                multiplier = regime_multipliers.get(name, suggested_weights.get(name, 1.0))

                # 應用權重調整（基於基礎權重）
                new_weight = self._base_weights[name] * multiplier
                # 限制權重範圍
                new_weight = max(0.1, min(2.0, new_weight))
                old_weight = self._weights[name].weight

                if abs(new_weight - old_weight) > 0.01:  # 只在有明顯變化時更新
                    self._weights[name].weight = new_weight
                    logger.debug(f"Adjusted weight for {name}: {old_weight:.2f} -> {new_weight:.2f} "
                               f"(regime: {regime.value}, conf: {confidence:.2f})")

    def _resolve_signal_conflicts(
        self,
        signals: List[Signal],
        regime_analysis: RegimeAnalysis
    ) -> List[Signal]:
        """
        解決策略間的信號衝突

        當衝突的策略發出相反方向的信號時，根據市場狀態選擇更適合的策略

        Args:
            signals: 原始信號列表
            regime_analysis: 市場狀態分析結果

        Returns:
            處理後的信號列表
        """
        if len(signals) <= 1:
            return signals

        # 建立策略名稱到信號的映射
        signal_map = {s.strategy_name: s for s in signals}

        # 檢查衝突
        for strategy_pair in self._conflict_strategies:
            s1, s2 = strategy_pair
            if s1 in signal_map and s2 in signal_map:
                sig1, sig2 = signal_map[s1], signal_map[s2]

                # 檢查是否方向相反
                if sig1.direction != 0 and sig2.direction != 0 and sig1.direction != sig2.direction:
                    # 根據市場狀態決定保留哪個
                    regime = regime_analysis.regime

                    # 趨勢市場優先趨勢策略
                    if regime in [MarketRegime.STRONG_TREND_UP, MarketRegime.TREND_UP,
                                  MarketRegime.STRONG_TREND_DOWN, MarketRegime.TREND_DOWN]:
                        preferred = s1 if s1 == 'trend_following' else s2
                        discarded = s2 if s1 == 'trend_following' else s1
                    # 震盪市場優先均值回歸
                    elif regime in [MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY]:
                        preferred = s1 if s1 == 'mean_reversion' else s2
                        discarded = s2 if s1 == 'mean_reversion' else s1
                    else:
                        # 其他情況，選擇置信度更高的
                        if sig1.confidence >= sig2.confidence:
                            preferred, discarded = s1, s2
                        else:
                            preferred, discarded = s2, s1

                    # 移除被丟棄的信號
                    if discarded in signal_map:
                        logger.info(f"Signal conflict resolved: kept {preferred}, discarded {discarded} "
                                  f"(regime: {regime_analysis.regime.value})")
                        del signal_map[discarded]

        return list(signal_map.values())

    def _aggregate_signals(
        self,
        symbol: str,
        signals: List[Signal],
        market_state: str,
        regime_analysis: Optional[RegimeAnalysis] = None,
        mtf_analysis: Optional[MTFAnalysis] = None
    ) -> AggregatedSignal:
        """聚合多個策略信號"""
        if not signals:
            return AggregatedSignal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                direction=0,
                strength=0.0,
                confidence=0.0,
                contributing_strategies=[],
                individual_signals=[],
                recommendation="HOLD - No strategy signals",
                regime_analysis=regime_analysis,
                metadata={'market_state': market_state}
            )

        # 加權計算
        total_weight = 0
        weighted_direction = 0
        weighted_strength = 0
        weighted_confidence = 0
        contributing = []

        for signal in signals:
            weight = self._weights.get(signal.strategy_name, StrategyWeight(signal.strategy_name)).weight
            total_weight += weight

            weighted_direction += signal.direction * signal.strength * weight
            weighted_strength += signal.strength * weight
            weighted_confidence += signal.confidence * weight
            contributing.append(signal.strategy_name)

        if total_weight > 0:
            avg_direction = weighted_direction / total_weight
            avg_strength = weighted_strength / total_weight
            avg_confidence = weighted_confidence / total_weight
        else:
            avg_direction = 0
            avg_strength = 0
            avg_confidence = 0

        # 確定最終方向
        if avg_direction > 0.3:
            direction = 1
        elif avg_direction < -0.3:
            direction = -1
        else:
            direction = 0

        # 一致性檢查（如果信號一致，增加置信度）
        directions = [s.direction for s in signals if s.direction != 0]
        agreement_ratio = 0
        if directions:
            agreement_ratio = sum(1 for d in directions if d == direction) / len(directions)
            avg_confidence = min(1.0, avg_confidence * (0.5 + 0.5 * agreement_ratio))

        # v0.3.2: 基於市場狀態的置信度調整
        if regime_analysis:
            regime = regime_analysis.regime
            regime_conf = regime_analysis.confidence

            # 高置信度市場狀態增加整體置信度
            if regime_conf > 0.7:
                avg_confidence = min(1.0, avg_confidence * 1.1)

            # 信號方向與趨勢方向一致時增加置信度
            if direction > 0 and regime in [MarketRegime.STRONG_TREND_UP, MarketRegime.TREND_UP, MarketRegime.BREAKOUT_UP]:
                avg_confidence = min(1.0, avg_confidence * 1.15)
            elif direction < 0 and regime in [MarketRegime.STRONG_TREND_DOWN, MarketRegime.TREND_DOWN, MarketRegime.BREAKOUT_DOWN]:
                avg_confidence = min(1.0, avg_confidence * 1.15)
            # 信號方向與趨勢方向相反時降低置信度
            elif direction > 0 and regime in [MarketRegime.STRONG_TREND_DOWN, MarketRegime.TREND_DOWN]:
                avg_confidence *= 0.7
                avg_strength *= 0.8
            elif direction < 0 and regime in [MarketRegime.STRONG_TREND_UP, MarketRegime.TREND_UP]:
                avg_confidence *= 0.7
                avg_strength *= 0.8

            # 高波動市場降低置信度
            if regime == MarketRegime.HIGH_VOLATILITY:
                volatility_penalty = max(0.7, 1.0 - regime_analysis.volatility_level * 0.3)
                avg_confidence *= volatility_penalty

        # v0.3.3: MTF 調整
        mtf_reason = ""
        if mtf_analysis:
            # 應用 MTF 置信度調整係數
            mtf_multiplier = mtf_analysis.confidence_multiplier
            avg_confidence *= mtf_multiplier
            avg_strength *= (0.5 + 0.5 * mtf_multiplier)  # 強度也受影響但幅度較小

            # 確保範圍合理
            avg_confidence = max(0.1, min(1.0, avg_confidence))
            avg_strength = max(0.1, min(1.0, avg_strength))

            if not mtf_analysis.trend_alignment:
                mtf_reason = f" | MTF逆勢警告(trend={mtf_analysis.higher_tf_trend})"
            elif mtf_analysis.trend_strength > 0.5:
                mtf_reason = f" | MTF順勢確認(strength={mtf_analysis.trend_strength:.2f})"

        # 生成建議
        recommendation = self._generate_recommendation(
            direction, avg_strength, avg_confidence, market_state, len(signals)
        )
        # 添加 MTF 信息到建議
        if mtf_reason:
            recommendation += mtf_reason

        return AggregatedSignal(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            direction=direction,
            strength=avg_strength,
            confidence=avg_confidence,
            contributing_strategies=contributing,
            individual_signals=signals,
            recommendation=recommendation,
            regime_analysis=regime_analysis,
            metadata={
                'market_state': market_state,
                'state_probabilities': self._market_classifier.get_probabilities(),
                'signal_agreement': agreement_ratio if directions else 0,
                'regime': regime_analysis.regime.value if regime_analysis else None,
                'regime_confidence': regime_analysis.confidence if regime_analysis else 0,
                'trend_strength': regime_analysis.trend_strength if regime_analysis else 0,
                # v0.3.3: MTF 信息
                'mtf_trend': mtf_analysis.higher_tf_trend if mtf_analysis else 0,
                'mtf_trend_strength': mtf_analysis.trend_strength if mtf_analysis else 0,
                'mtf_aligned': mtf_analysis.trend_alignment if mtf_analysis else True,
                'mtf_multiplier': mtf_analysis.confidence_multiplier if mtf_analysis else 1.0,
            }
        )

    def _generate_recommendation(
        self,
        direction: int,
        strength: float,
        confidence: float,
        market_state: str,
        n_signals: int
    ) -> str:
        """生成交易建議文字"""
        if direction == 0:
            return f"HOLD - Market state: {market_state}, insufficient signal strength"

        action = "BUY" if direction > 0 else "SELL"

        if strength > 0.5 and confidence > 0.6:
            prefix = "STRONG "
            desc = f"Multiple indicators aligned ({n_signals} strategies agree)"
        elif strength > 0.3 and confidence > 0.4:
            prefix = ""
            desc = f"Moderate signal strength"
        else:
            prefix = "WEAK "
            desc = f"Consider waiting for confirmation"

        return f"{prefix}{action} - {desc}. Market: {market_state}, Confidence: {confidence:.0%}"

    def get_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """獲取所有策略績效"""
        return {
            name: strategy.get_performance()
            for name, strategy in self._strategies.items()
        }

    def get_status(self) -> Dict[str, Any]:
        """獲取引擎狀態"""
        # 獲取當前市場狀態
        current_regime = self._regime_analyzer.get_current_regime()

        return {
            'registered_strategies': list(self._strategies.keys()),
            'enabled_strategies': [
                name for name, w in self._weights.items() if w.enabled
            ],
            'weights': {name: w.weight for name, w in self._weights.items()},
            'base_weights': self._base_weights.copy(),
            'current_market_state': self._market_classifier._current_state,
            'state_probabilities': self._market_classifier.get_probabilities(),
            'signal_history_length': len(self._signal_history),
            # 新增市場狀態分析信息
            'market_regime': {
                'current': current_regime.value,
                'history': self._regime_analyzer.get_regime_history(5),
            },
            # 各策略的資金管理狀態
            'strategy_money_management': {
                name: strategy.get_money_management_status()
                for name, strategy in self._strategies.items()
            },
            # v0.3.2: 信號一致性過濾器狀態
            'signal_filter': self._signal_filter.get_status(),
            # v0.3.3: MTF 分析器狀態
            'mtf_analyzer': {
                'enabled': self._mtf_enabled,
                **self._mtf_analyzer.get_status()
            }
        }

    def record_executed_trade(self, direction: int, symbol: str):
        """
        記錄已執行的交易（用於信號一致性過濾）

        Args:
            direction: 交易方向 (1=買入, -1=賣出)
            symbol: 交易對
        """
        self._signal_filter.record_trade(direction)
        logger.debug(f"Recorded executed trade: {symbol} direction={direction}")

    def set_mtf_enabled(self, enabled: bool):
        """
        動態開關 MTF 分析 (v0.3.3)

        Args:
            enabled: 是否啟用 MTF 分析
        """
        self._mtf_enabled = enabled
        logger.info(f"MTF analysis {'enabled' if enabled else 'disabled'}")

    def update_all(self, market_data: Dict[str, Any]):
        """更新所有策略"""
        for strategy in self._strategies.values():
            try:
                strategy.update(market_data)
            except Exception as e:
                logger.error(f"Error updating strategy {strategy.name}: {e}")


# 全局實例
strategy_engine = StrategyEngine()
