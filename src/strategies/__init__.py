"""
交易策略模組

包含：
- BaseStrategy: 策略基類
- 規則策略: TrendFollowing, MeanReversion, Momentum
- ML 策略: XGBoostStrategy
- RL 策略: PPOAgent, TradingEnv
- 市場狀態分析: MarketRegimeAnalyzer
"""

from .base import BaseStrategy, Signal, SignalType, StrategyConfig
from .rule_based import TrendFollowingStrategy, MeanReversionStrategy, MomentumStrategy
from .market_regime import (
    MarketRegime,
    MarketRegimeAnalyzer,
    RegimeAnalysis,
    market_regime_analyzer
)

__all__ = [
    'BaseStrategy',
    'Signal',
    'SignalType',
    'StrategyConfig',
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'MomentumStrategy',
    'MarketRegime',
    'MarketRegimeAnalyzer',
    'RegimeAnalysis',
    'market_regime_analyzer',
]
