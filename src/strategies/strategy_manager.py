"""
策略管理器

負責策略的註冊、加載、管理和信號聚合

v0.4.0 - Enterprise Edition

功能：
1. 策略註冊與發現
2. 策略配置管理
3. 信號聚合（多策略投票）
4. 策略績效追蹤
5. 策略超市功能

作者：Binance Trading Bot
"""

import os
import json
import importlib
import inspect
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Type, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np

from src.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from src.logger import logger


class StrategyCategory(Enum):
    """策略分類"""
    TREND_FOLLOWING = "trend_following"      # 趨勢跟蹤
    MEAN_REVERSION = "mean_reversion"        # 均值回歸
    MOMENTUM = "momentum"                     # 動量策略
    VOLATILITY = "volatility"                # 波動率策略
    BREAKOUT = "breakout"                    # 突破策略
    SCALPING = "scalping"                    # 剝頭皮
    SWING = "swing"                          # 波段交易
    MACHINE_LEARNING = "machine_learning"    # 機器學習
    COMMUNITY = "community"                  # 社區策略


class AggregationMethod(Enum):
    """信號聚合方法"""
    MAJORITY_VOTE = "majority_vote"          # 多數投票
    WEIGHTED_AVERAGE = "weighted_average"    # 加權平均
    UNANIMOUS = "unanimous"                  # 一致同意
    STRONGEST = "strongest"                  # 最強信號
    ENSEMBLE = "ensemble"                    # 集成方法


@dataclass
class StrategyMetadata:
    """策略元數據"""
    name: str
    display_name: str
    category: StrategyCategory
    description: str
    author: str = "Community"
    version: str = "1.0.0"
    timeframe: str = "1h"
    min_candles: int = 100
    tags: List[str] = field(default_factory=list)
    risk_level: int = 3  # 1-5, 5 最高風險
    recommended_pairs: List[str] = field(default_factory=list)
    hyperopt_enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['category'] = self.category.value
        return result


@dataclass
class StrategyPerformance:
    """策略績效"""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_holding_time: float = 0.0  # 小時
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregatedSignal:
    """聚合後的信號"""
    timestamp: datetime
    symbol: str
    direction: int
    strength: float
    confidence: float
    contributing_strategies: List[str]
    individual_signals: List[Dict[str, Any]]
    aggregation_method: AggregationMethod
    final_decision: str  # "buy", "sell", "hold"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'direction': self.direction,
            'strength': self.strength,
            'confidence': self.confidence,
            'contributing_strategies': self.contributing_strategies,
            'individual_signals': self.individual_signals,
            'aggregation_method': self.aggregation_method.value,
            'final_decision': self.final_decision
        }


class StrategyManager:
    """策略管理器"""

    def __init__(self, config_path: str = None):
        self._strategies: Dict[str, BaseStrategy] = {}
        self._metadata: Dict[str, StrategyMetadata] = {}
        self._performance: Dict[str, StrategyPerformance] = {}
        self._strategy_classes: Dict[str, Type[BaseStrategy]] = {}
        self._enabled_strategies: List[str] = []

        # 配置路徑
        self._config_path = config_path or "config/strategies.json"
        self._config: Dict[str, Any] = {}

        # 聚合設置
        self._aggregation_method = AggregationMethod.WEIGHTED_AVERAGE
        self._min_agreement = 0.5  # 最小同意比例
        self._signal_decay_minutes = 5  # 信號衰減時間

        # 加載配置
        self._load_config()

        logger.info("StrategyManager initialized")

    def _load_config(self):
        """加載策略配置"""
        try:
            if os.path.exists(self._config_path):
                with open(self._config_path, 'r') as f:
                    self._config = json.load(f)
                    self._enabled_strategies = self._config.get('enabled_strategies', [])
                    self._aggregation_method = AggregationMethod(
                        self._config.get('aggregation_method', 'weighted_average')
                    )
                    self._min_agreement = self._config.get('min_agreement', 0.5)
                logger.info(f"Loaded strategy config: {len(self._enabled_strategies)} enabled strategies")
        except Exception as e:
            logger.warning(f"Failed to load strategy config: {e}")
            self._config = {}

    def _save_config(self):
        """保存策略配置"""
        try:
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            config_data = {
                'enabled_strategies': self._enabled_strategies,
                'aggregation_method': self._aggregation_method.value,
                'min_agreement': self._min_agreement,
                'strategy_configs': {
                    name: s.config.__dict__ if hasattr(s.config, '__dict__') else {}
                    for name, s in self._strategies.items()
                },
                'updated_at': datetime.utcnow().isoformat()
            }
            with open(self._config_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            logger.info("Strategy config saved")
        except Exception as e:
            logger.error(f"Failed to save strategy config: {e}")

    # ==================== 策略註冊 ====================

    def register_strategy_class(
        self,
        strategy_class: Type[BaseStrategy],
        metadata: StrategyMetadata
    ):
        """
        註冊策略類

        Args:
            strategy_class: 策略類
            metadata: 策略元數據
        """
        name = metadata.name
        self._strategy_classes[name] = strategy_class
        self._metadata[name] = metadata
        logger.info(f"Registered strategy class: {name}")

    def register_strategy(
        self,
        strategy: BaseStrategy,
        metadata: Optional[StrategyMetadata] = None
    ):
        """
        註冊策略實例

        Args:
            strategy: 策略實例
            metadata: 策略元數據（可選）
        """
        name = strategy.name
        self._strategies[name] = strategy

        if metadata:
            self._metadata[name] = metadata
        elif name not in self._metadata:
            # 創建默認元數據
            self._metadata[name] = StrategyMetadata(
                name=name,
                display_name=name.replace('_', ' ').title(),
                category=StrategyCategory.COMMUNITY,
                description=f"Strategy: {name}"
            )

        # 初始化績效追蹤
        if name not in self._performance:
            self._performance[name] = StrategyPerformance(strategy_name=name)

        logger.info(f"Registered strategy instance: {name}")

    def unregister_strategy(self, name: str):
        """取消註冊策略"""
        if name in self._strategies:
            del self._strategies[name]
        if name in self._enabled_strategies:
            self._enabled_strategies.remove(name)
        logger.info(f"Unregistered strategy: {name}")

    def create_strategy(
        self,
        name: str,
        config: Optional[StrategyConfig] = None
    ) -> Optional[BaseStrategy]:
        """
        從已註冊的類創建策略實例

        Args:
            name: 策略名稱
            config: 策略配置

        Returns:
            策略實例
        """
        if name not in self._strategy_classes:
            logger.error(f"Strategy class not found: {name}")
            return None

        try:
            strategy_class = self._strategy_classes[name]
            strategy = strategy_class(config) if config else strategy_class()
            self.register_strategy(strategy, self._metadata.get(name))
            return strategy
        except Exception as e:
            logger.error(f"Failed to create strategy {name}: {e}")
            return None

    # ==================== 策略發現 ====================

    def discover_strategies(self, search_paths: List[str] = None):
        """
        自動發現策略

        Args:
            search_paths: 搜索路徑列表
        """
        if search_paths is None:
            search_paths = [
                'src/strategies/rule_based',
                'src/strategies/ml_based',
                'src/strategies/community'
            ]

        for path in search_paths:
            if not os.path.exists(path):
                continue

            for filename in os.listdir(path):
                if filename.endswith('.py') and not filename.startswith('_'):
                    module_name = filename[:-3]
                    try:
                        # 動態導入模塊
                        module_path = path.replace('/', '.') + '.' + module_name
                        module = importlib.import_module(module_path)

                        # 查找策略類
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and
                                issubclass(obj, BaseStrategy) and
                                obj is not BaseStrategy):

                                # 獲取或創建元數據
                                metadata = getattr(obj, 'METADATA', None)
                                if metadata is None:
                                    metadata = StrategyMetadata(
                                        name=module_name,
                                        display_name=name,
                                        category=StrategyCategory.COMMUNITY,
                                        description=obj.__doc__ or f"Strategy: {name}"
                                    )

                                self.register_strategy_class(obj, metadata)
                                logger.info(f"Discovered strategy: {name} from {path}")

                    except Exception as e:
                        logger.warning(f"Failed to load strategy from {path}/{filename}: {e}")

    # ==================== 策略啟用/禁用 ====================

    def enable_strategy(self, name: str) -> bool:
        """啟用策略"""
        if name in self._strategies:
            self._strategies[name].config.enabled = True
            if name not in self._enabled_strategies:
                self._enabled_strategies.append(name)
            self._save_config()
            logger.info(f"Enabled strategy: {name}")
            return True
        return False

    def disable_strategy(self, name: str) -> bool:
        """禁用策略"""
        if name in self._strategies:
            self._strategies[name].config.enabled = False
            if name in self._enabled_strategies:
                self._enabled_strategies.remove(name)
            self._save_config()
            logger.info(f"Disabled strategy: {name}")
            return True
        return False

    def set_strategy_weight(self, name: str, weight: float) -> bool:
        """設置策略權重"""
        if name in self._strategies and 0 <= weight <= 10:
            self._strategies[name].config.weight = weight
            self._save_config()
            logger.info(f"Set {name} weight to {weight}")
            return True
        return False

    # ==================== 信號生成與聚合 ====================

    def generate_signals(
        self,
        symbol: str,
        features: pd.DataFrame,
        current_price: float,
        position: float = 0.0
    ) -> List[Signal]:
        """
        從所有啟用的策略生成信號

        Args:
            symbol: 交易對
            features: 特徵 DataFrame
            current_price: 當前價格
            position: 當前持倉

        Returns:
            信號列表
        """
        signals = []

        for name, strategy in self._strategies.items():
            if not strategy.config.enabled:
                continue

            try:
                signal = strategy.generate_signal(symbol, features, current_price, position)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Strategy {name} signal generation failed: {e}")

        return signals

    def aggregate_signals(
        self,
        signals: List[Signal],
        symbol: str,
        method: Optional[AggregationMethod] = None
    ) -> Optional[AggregatedSignal]:
        """
        聚合多個策略的信號

        Args:
            signals: 信號列表
            symbol: 交易對
            method: 聚合方法

        Returns:
            聚合後的信號
        """
        if not signals:
            return None

        method = method or self._aggregation_method

        # 過濾有效信號
        valid_signals = [s for s in signals if s.direction != 0]
        if not valid_signals:
            return None

        if method == AggregationMethod.MAJORITY_VOTE:
            return self._aggregate_majority_vote(valid_signals, symbol)
        elif method == AggregationMethod.WEIGHTED_AVERAGE:
            return self._aggregate_weighted_average(valid_signals, symbol)
        elif method == AggregationMethod.UNANIMOUS:
            return self._aggregate_unanimous(valid_signals, symbol)
        elif method == AggregationMethod.STRONGEST:
            return self._aggregate_strongest(valid_signals, symbol)
        elif method == AggregationMethod.ENSEMBLE:
            return self._aggregate_ensemble(valid_signals, symbol)

        return None

    def _aggregate_majority_vote(
        self,
        signals: List[Signal],
        symbol: str
    ) -> Optional[AggregatedSignal]:
        """多數投票聚合"""
        buy_count = sum(1 for s in signals if s.direction > 0)
        sell_count = sum(1 for s in signals if s.direction < 0)
        total = len(signals)

        if buy_count > sell_count and buy_count / total >= self._min_agreement:
            direction = 1
            final_decision = "buy"
        elif sell_count > buy_count and sell_count / total >= self._min_agreement:
            direction = -1
            final_decision = "sell"
        else:
            return None

        # 計算平均強度和置信度
        relevant_signals = [s for s in signals if s.direction == direction]
        avg_strength = np.mean([s.strength for s in relevant_signals])
        avg_confidence = np.mean([s.confidence for s in relevant_signals])

        return AggregatedSignal(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            direction=direction,
            strength=float(avg_strength),
            confidence=float(avg_confidence) * (max(buy_count, sell_count) / total),
            contributing_strategies=[s.strategy_name for s in relevant_signals],
            individual_signals=[s.to_dict() for s in signals],
            aggregation_method=AggregationMethod.MAJORITY_VOTE,
            final_decision=final_decision
        )

    def _aggregate_weighted_average(
        self,
        signals: List[Signal],
        symbol: str
    ) -> Optional[AggregatedSignal]:
        """加權平均聚合"""
        total_weight = 0
        weighted_direction = 0
        weighted_strength = 0
        weighted_confidence = 0

        for signal in signals:
            strategy = self._strategies.get(signal.strategy_name)
            weight = strategy.config.weight if strategy else 1.0

            # 考慮信號強度和置信度
            effective_weight = weight * signal.strength * signal.confidence
            total_weight += effective_weight

            weighted_direction += signal.direction * effective_weight
            weighted_strength += signal.strength * effective_weight
            weighted_confidence += signal.confidence * effective_weight

        if total_weight == 0:
            return None

        avg_direction = weighted_direction / total_weight
        avg_strength = weighted_strength / total_weight
        avg_confidence = weighted_confidence / total_weight

        # 確定最終方向
        if avg_direction > 0.3:
            direction = 1
            final_decision = "buy"
        elif avg_direction < -0.3:
            direction = -1
            final_decision = "sell"
        else:
            return None

        return AggregatedSignal(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            direction=direction,
            strength=float(abs(avg_strength)),
            confidence=float(avg_confidence),
            contributing_strategies=[s.strategy_name for s in signals],
            individual_signals=[s.to_dict() for s in signals],
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            final_decision=final_decision
        )

    def _aggregate_unanimous(
        self,
        signals: List[Signal],
        symbol: str
    ) -> Optional[AggregatedSignal]:
        """一致同意聚合"""
        directions = set(s.direction for s in signals)

        if len(directions) != 1:
            return None  # 不一致

        direction = signals[0].direction
        final_decision = "buy" if direction > 0 else "sell"

        avg_strength = np.mean([s.strength for s in signals])
        avg_confidence = np.mean([s.confidence for s in signals])

        return AggregatedSignal(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            direction=direction,
            strength=float(avg_strength),
            confidence=float(avg_confidence) * 1.2,  # 一致同意加成
            contributing_strategies=[s.strategy_name for s in signals],
            individual_signals=[s.to_dict() for s in signals],
            aggregation_method=AggregationMethod.UNANIMOUS,
            final_decision=final_decision
        )

    def _aggregate_strongest(
        self,
        signals: List[Signal],
        symbol: str
    ) -> Optional[AggregatedSignal]:
        """最強信號"""
        strongest = max(signals, key=lambda s: s.strength * s.confidence)

        return AggregatedSignal(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            direction=strongest.direction,
            strength=strongest.strength,
            confidence=strongest.confidence,
            contributing_strategies=[strongest.strategy_name],
            individual_signals=[s.to_dict() for s in signals],
            aggregation_method=AggregationMethod.STRONGEST,
            final_decision="buy" if strongest.direction > 0 else "sell"
        )

    def _aggregate_ensemble(
        self,
        signals: List[Signal],
        symbol: str
    ) -> Optional[AggregatedSignal]:
        """集成方法 - 結合多種聚合策略"""
        # 獲取各種方法的結果
        results = []

        majority = self._aggregate_majority_vote(signals, symbol)
        if majority:
            results.append(('majority', majority))

        weighted = self._aggregate_weighted_average(signals, symbol)
        if weighted:
            results.append(('weighted', weighted))

        strongest = self._aggregate_strongest(signals, symbol)
        if strongest:
            results.append(('strongest', strongest))

        if not results:
            return None

        # 計算最終方向
        directions = [r[1].direction for r in results]
        final_direction = 1 if sum(directions) > 0 else -1

        # 只有多數方法同意時才產生信號
        agreeing = sum(1 for d in directions if d == final_direction)
        if agreeing < len(results) * 0.5:
            return None

        # 計算加權平均強度和置信度
        avg_strength = np.mean([r[1].strength for r in results if r[1].direction == final_direction])
        avg_confidence = np.mean([r[1].confidence for r in results if r[1].direction == final_direction])

        return AggregatedSignal(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            direction=final_direction,
            strength=float(avg_strength),
            confidence=float(avg_confidence),
            contributing_strategies=list(set(
                s for r in results for s in r[1].contributing_strategies
            )),
            individual_signals=[s.to_dict() for s in signals],
            aggregation_method=AggregationMethod.ENSEMBLE,
            final_decision="buy" if final_direction > 0 else "sell"
        )

    # ==================== 績效追蹤 ====================

    def record_trade_result(
        self,
        strategy_name: str,
        pnl: float,
        holding_time_hours: float
    ):
        """記錄交易結果"""
        if strategy_name not in self._performance:
            self._performance[strategy_name] = StrategyPerformance(strategy_name=strategy_name)

        perf = self._performance[strategy_name]
        perf.total_trades += 1
        perf.total_pnl += pnl

        if pnl > 0:
            perf.winning_trades += 1
        else:
            perf.losing_trades += 1

        perf.win_rate = perf.winning_trades / perf.total_trades if perf.total_trades > 0 else 0
        perf.avg_pnl = perf.total_pnl / perf.total_trades if perf.total_trades > 0 else 0
        perf.avg_holding_time = (
            (perf.avg_holding_time * (perf.total_trades - 1) + holding_time_hours)
            / perf.total_trades
        )
        perf.last_updated = datetime.utcnow().isoformat()

        # 更新策略本身的績效
        if strategy_name in self._strategies:
            self._strategies[strategy_name].record_trade_result(pnl)

    def get_strategy_performance(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """獲取策略績效"""
        if strategy_name in self._performance:
            return self._performance[strategy_name].to_dict()
        return None

    def get_all_performance(self) -> Dict[str, Dict[str, Any]]:
        """獲取所有策略績效"""
        return {name: perf.to_dict() for name, perf in self._performance.items()}

    # ==================== 策略信息查詢 ====================

    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """獲取策略實例"""
        return self._strategies.get(name)

    def get_strategy_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """獲取策略元數據"""
        if name in self._metadata:
            return self._metadata[name].to_dict()
        return None

    def list_strategies(
        self,
        category: Optional[StrategyCategory] = None,
        enabled_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        列出策略

        Args:
            category: 按分類過濾
            enabled_only: 只顯示啟用的策略

        Returns:
            策略列表
        """
        result = []

        for name, metadata in self._metadata.items():
            if category and metadata.category != category:
                continue

            strategy = self._strategies.get(name)
            if enabled_only and (not strategy or not strategy.config.enabled):
                continue

            strategy_info = metadata.to_dict()
            strategy_info['enabled'] = strategy.config.enabled if strategy else False
            strategy_info['weight'] = strategy.config.weight if strategy else 1.0
            strategy_info['performance'] = self.get_strategy_performance(name)

            result.append(strategy_info)

        return result

    def get_available_categories(self) -> List[Dict[str, Any]]:
        """獲取可用的策略分類"""
        categories = {}
        for metadata in self._metadata.values():
            cat = metadata.category
            if cat not in categories:
                categories[cat] = {
                    'name': cat.value,
                    'display_name': cat.value.replace('_', ' ').title(),
                    'count': 0
                }
            categories[cat]['count'] += 1

        return list(categories.values())

    def get_strategy_count(self) -> Dict[str, int]:
        """獲取策略數量統計"""
        return {
            'total': len(self._metadata),
            'registered': len(self._strategies),
            'enabled': len([s for s in self._strategies.values() if s.config.enabled])
        }

    # ==================== 策略配置 ====================

    def update_strategy_config(
        self,
        name: str,
        config_updates: Dict[str, Any]
    ) -> bool:
        """
        更新策略配置

        Args:
            name: 策略名稱
            config_updates: 配置更新

        Returns:
            是否成功
        """
        if name not in self._strategies:
            return False

        strategy = self._strategies[name]

        for key, value in config_updates.items():
            if hasattr(strategy.config, key):
                setattr(strategy.config, key, value)
            elif hasattr(strategy.config, 'params') and isinstance(strategy.config.params, dict):
                strategy.config.params[key] = value

        self._save_config()
        logger.info(f"Updated strategy config for {name}: {config_updates}")
        return True

    def get_strategy_config(self, name: str) -> Optional[Dict[str, Any]]:
        """獲取策略配置"""
        if name not in self._strategies:
            return None

        strategy = self._strategies[name]
        config = strategy.config

        return {
            'name': config.name,
            'enabled': config.enabled,
            'weight': config.weight,
            'max_position_size': config.max_position_size,
            'stop_loss_pct': config.stop_loss_pct,
            'take_profit_pct': config.take_profit_pct,
            'min_signal_strength': config.min_signal_strength,
            'min_confidence': config.min_confidence,
            'cooldown_minutes': config.cooldown_minutes,
            'params': config.params
        }

    def set_aggregation_method(self, method: str) -> bool:
        """設置聚合方法"""
        try:
            self._aggregation_method = AggregationMethod(method)
            self._save_config()
            return True
        except ValueError:
            return False

    def get_aggregation_settings(self) -> Dict[str, Any]:
        """獲取聚合設置"""
        return {
            'method': self._aggregation_method.value,
            'min_agreement': self._min_agreement,
            'signal_decay_minutes': self._signal_decay_minutes,
            'available_methods': [m.value for m in AggregationMethod]
        }

    # ==================== 策略超市功能 ====================

    def get_strategy_store(self) -> Dict[str, Any]:
        """獲取策略超市信息"""
        return {
            'total_strategies': len(self._metadata),
            'categories': self.get_available_categories(),
            'strategies': self.list_strategies(),
            'featured': self._get_featured_strategies(),
            'top_performers': self._get_top_performers(),
            'recently_added': self._get_recently_added()
        }

    def _get_featured_strategies(self, limit: int = 3) -> List[Dict[str, Any]]:
        """獲取推薦策略"""
        # 根據績效和使用情況推薦
        strategies = self.list_strategies()

        # 按績效排序
        def score(s):
            perf = s.get('performance') or {}
            return (perf.get('win_rate', 0) * 0.4 +
                    perf.get('profit_factor', 0) * 0.3 +
                    (1 if s['enabled'] else 0) * 0.3)

        strategies.sort(key=score, reverse=True)
        return strategies[:limit]

    def _get_top_performers(self, limit: int = 5) -> List[Dict[str, Any]]:
        """獲取績效最佳策略"""
        performances = []

        for name, perf in self._performance.items():
            if perf.total_trades >= 10:  # 至少 10 筆交易
                performances.append({
                    'name': name,
                    'win_rate': perf.win_rate,
                    'profit_factor': perf.profit_factor,
                    'total_pnl': perf.total_pnl,
                    'total_trades': perf.total_trades
                })

        performances.sort(key=lambda x: x['win_rate'] * x.get('profit_factor', 1), reverse=True)
        return performances[:limit]

    def _get_recently_added(self, limit: int = 5) -> List[Dict[str, Any]]:
        """獲取最近添加的策略"""
        strategies = list(self._metadata.values())
        strategies.sort(key=lambda x: x.created_at, reverse=True)
        return [s.to_dict() for s in strategies[:limit]]


# 全局策略管理器實例
strategy_manager = StrategyManager()
