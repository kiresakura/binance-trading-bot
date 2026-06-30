"""
社區策略模組

包含從 Freqtrade 社區移植的經典策略

v0.4.0 - Enterprise Edition

策略列表：
1. BBRSIStrategy - 布林帶 + RSI 策略
2. MACDCrossStrategy - MACD 交叉策略
3. GoldenCrossStrategy - 黃金交叉策略
4. ScalpingStrategy - 剝頭皮策略
5. BreakoutStrategy - 突破策略
6. ADXMomentumStrategy - ADX 動量策略
7. IchimokuStrategy - 一目均衡表策略
8. SuperTrendStrategy - 超級趨勢策略
9. RangeFilterStrategy - 範圍過濾策略
10. MultiTimeframeStrategy - 多時間框架策略
"""

from .bb_rsi_strategy import BBRSIStrategy
from .macd_cross_strategy import MACDCrossStrategy
from .golden_cross_strategy import GoldenCrossStrategy
from .scalping_strategy import ScalpingStrategy
from .breakout_strategy import BreakoutStrategy
from .adx_momentum_strategy import ADXMomentumStrategy
from .ichimoku_strategy import IchimokuStrategy
from .supertrend_strategy import SuperTrendStrategy
from .range_filter_strategy import RangeFilterStrategy
from .multi_timeframe_strategy import MultiTimeframeStrategy

__all__ = [
    'BBRSIStrategy',
    'MACDCrossStrategy',
    'GoldenCrossStrategy',
    'ScalpingStrategy',
    'BreakoutStrategy',
    'ADXMomentumStrategy',
    'IchimokuStrategy',
    'SuperTrendStrategy',
    'RangeFilterStrategy',
    'MultiTimeframeStrategy',
]

# 策略元數據列表
COMMUNITY_STRATEGIES = [
    {
        'name': 'bb_rsi',
        'class': BBRSIStrategy,
        'display_name': 'BB + RSI 策略',
        'description': '結合布林帶和 RSI 指標的經典策略',
        'category': 'mean_reversion',
        'risk_level': 2,
        'timeframe': '1h'
    },
    {
        'name': 'macd_cross',
        'class': MACDCrossStrategy,
        'display_name': 'MACD 交叉策略',
        'description': '基於 MACD 金叉死叉的趨勢策略',
        'category': 'trend_following',
        'risk_level': 2,
        'timeframe': '4h'
    },
    {
        'name': 'golden_cross',
        'class': GoldenCrossStrategy,
        'display_name': '黃金交叉策略',
        'description': '經典的 MA50/MA200 黃金交叉策略',
        'category': 'trend_following',
        'risk_level': 1,
        'timeframe': '1d'
    },
    {
        'name': 'scalping',
        'class': ScalpingStrategy,
        'display_name': '剝頭皮策略',
        'description': '高頻短線交易策略',
        'category': 'scalping',
        'risk_level': 4,
        'timeframe': '5m'
    },
    {
        'name': 'breakout',
        'class': BreakoutStrategy,
        'display_name': '突破策略',
        'description': '唐奇安通道突破策略',
        'category': 'breakout',
        'risk_level': 3,
        'timeframe': '1h'
    },
    {
        'name': 'adx_momentum',
        'class': ADXMomentumStrategy,
        'display_name': 'ADX 動量策略',
        'description': '結合 ADX 和動量指標的策略',
        'category': 'momentum',
        'risk_level': 3,
        'timeframe': '1h'
    },
    {
        'name': 'ichimoku',
        'class': IchimokuStrategy,
        'display_name': '一目均衡表策略',
        'description': '日本經典技術分析策略',
        'category': 'trend_following',
        'risk_level': 2,
        'timeframe': '4h'
    },
    {
        'name': 'supertrend',
        'class': SuperTrendStrategy,
        'display_name': '超級趨勢策略',
        'description': '基於 SuperTrend 指標的趨勢策略',
        'category': 'trend_following',
        'risk_level': 2,
        'timeframe': '1h'
    },
    {
        'name': 'range_filter',
        'class': RangeFilterStrategy,
        'display_name': '範圍過濾策略',
        'description': '基於價格範圍過濾的策略',
        'category': 'volatility',
        'risk_level': 3,
        'timeframe': '1h'
    },
    {
        'name': 'multi_timeframe',
        'class': MultiTimeframeStrategy,
        'display_name': '多時間框架策略',
        'description': '結合多個時間框架分析的策略',
        'category': 'swing',
        'risk_level': 2,
        'timeframe': '1h'
    },
]
