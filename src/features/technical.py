"""
技術指標特徵計算模組

計算 50+ 技術指標用於機器學習和策略信號
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import ta
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

from src.logger import logger


@dataclass
class FeatureConfig:
    """特徵配置"""
    # 移動平均週期
    sma_periods: List[int] = None
    ema_periods: List[int] = None

    # RSI 週期
    rsi_periods: List[int] = None

    # 布林帶
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR
    atr_period: int = 14

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [7, 25, 50, 99, 200]
        if self.ema_periods is None:
            self.ema_periods = [9, 12, 21, 26, 50]
        if self.rsi_periods is None:
            self.rsi_periods = [7, 14, 21]


class TechnicalFeatures:
    """
    技術指標特徵計算器

    計算完整的技術指標套件用於：
    - 機器學習模型輸入
    - 規則策略信號
    - 市場狀態分類
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        logger.info("TechnicalFeatures initialized")

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算所有技術指標特徵

        Args:
            df: OHLCV DataFrame，需要包含 columns: ['open', 'high', 'low', 'close', 'volume']

        Returns:
            包含所有特徵的 DataFrame
        """
        df = df.copy()

        # 確保列名小寫
        df.columns = df.columns.str.lower()

        # 計算各類特徵
        df = self._add_price_features(df)
        df = self._add_trend_features(df)
        df = self._add_momentum_features(df)
        df = self._add_volatility_features(df)
        df = self._add_volume_features(df)
        df = self._add_pattern_features(df)

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """價格相關特徵"""
        # 收益率
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        df['returns_20'] = df['close'].pct_change(20)

        # 對數收益
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # 價格位置（相對於高低點）
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        # 價格動量
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['momentum_20'] = df['close'] - df['close'].shift(20)

        # 價格加速度
        df['acceleration'] = df['momentum_5'] - df['momentum_5'].shift(5)

        # 波動率（滾動標準差）
        df['volatility_10'] = df['returns_1'].rolling(10).std()
        df['volatility_20'] = df['returns_1'].rolling(20).std()
        df['volatility_50'] = df['returns_1'].rolling(50).std()

        # 波動率比率（短期/長期）
        df['volatility_ratio'] = df['volatility_10'] / (df['volatility_50'] + 1e-10)

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """趨勢指標"""
        # SMA
        for period in self.config.sma_periods:
            df[f'sma_{period}'] = SMAIndicator(df['close'], window=period).sma_indicator()
            # 價格相對於 SMA 的位置
            df[f'close_sma_{period}_ratio'] = df['close'] / (df[f'sma_{period}'] + 1e-10)

        # EMA
        for period in self.config.ema_periods:
            df[f'ema_{period}'] = EMAIndicator(df['close'], window=period).ema_indicator()
            df[f'close_ema_{period}_ratio'] = df['close'] / (df[f'ema_{period}'] + 1e-10)

        # MACD
        macd = MACD(df['close'],
                    window_fast=self.config.macd_fast,
                    window_slow=self.config.macd_slow,
                    window_sign=self.config.macd_signal)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()

        # ADX（趨勢強度）
        adx = ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()

        # 趨勢方向（基於 EMA 交叉）
        df['trend_ema_9_21'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)
        df['trend_ema_12_26'] = np.where(df['ema_12'] > df['ema_26'], 1, -1)

        # SMA 斜率（趨勢方向和強度）
        df['sma_25_slope'] = df['sma_25'].diff(5) / 5
        df['sma_50_slope'] = df['sma_50'].diff(5) / 5

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """動量指標"""
        # RSI
        for period in self.config.rsi_periods:
            df[f'rsi_{period}'] = RSIIndicator(df['close'], window=period).rsi()

        # Stochastic Oscillator
        stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # Williams %R
        df['williams_r'] = WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()

        # ROC (Rate of Change)
        df['roc_10'] = ROCIndicator(df['close'], window=10).roc()
        df['roc_20'] = ROCIndicator(df['close'], window=20).roc()

        # RSI 超買超賣區域
        df['rsi_14_overbought'] = (df['rsi_14'] > 65).astype(int)
        df['rsi_14_oversold'] = (df['rsi_14'] < 35).astype(int)

        # RSI 背離（簡化版：價格新高但 RSI 沒有新高）
        df['price_high_20'] = df['close'].rolling(20).max()
        df['rsi_high_20'] = df['rsi_14'].rolling(20).max()
        df['bearish_divergence'] = ((df['close'] >= df['price_high_20'] * 0.99) &
                                    (df['rsi_14'] < df['rsi_high_20'] * 0.95)).astype(int)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """波動性指標"""
        # Bollinger Bands
        bb = BollingerBands(df['close'], window=self.config.bb_period, window_dev=self.config.bb_std)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # ATR
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=self.config.atr_period)
        df['atr'] = atr.average_true_range()
        df['atr_percent'] = df['atr'] / df['close'] * 100

        # Keltner Channel
        kc = KeltnerChannel(df['high'], df['low'], df['close'], window=20, window_atr=10)
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_lower'] = kc.keltner_channel_lband()
        df['kc_position'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'] + 1e-10)

        # True Range
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )

        # 波動率突破（BB Squeeze）
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).mean() * 0.75).astype(int)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量指標"""
        # 成交量 MA
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_sma_20'] = df['volume'].rolling(20).mean()

        # 成交量比率
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)

        # OBV
        df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['obv_sma'] = df['obv'].rolling(20).mean()
        df['obv_trend'] = np.where(df['obv'] > df['obv_sma'], 1, -1)

        # VWAP（需要 intraday 數據，這裡用簡化版）
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['close_vwap_ratio'] = df['close'] / (df['vwap'] + 1e-10)

        # 成交量價格趨勢確認
        df['volume_price_trend'] = np.where(
            (df['close'] > df['close'].shift(1)) & (df['volume'] > df['volume_sma_10']), 1,
            np.where(
                (df['close'] < df['close'].shift(1)) & (df['volume'] > df['volume_sma_10']), -1, 0
            )
        )

        # 成交量峰值（可能的反轉信號）
        df['volume_spike'] = (df['volume'] > df['volume_sma_20'] * 2).astype(int)

        return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """K線形態特徵"""
        # 實體大小
        df['body'] = abs(df['close'] - df['open'])
        df['body_percent'] = df['body'] / (df['high'] - df['low'] + 1e-10)

        # 上下影線
        df['upper_shadow'] = df['high'] - np.maximum(df['close'], df['open'])
        df['lower_shadow'] = np.minimum(df['close'], df['open']) - df['low']

        # Doji（十字星）
        df['doji'] = (df['body_percent'] < 0.1).astype(int)

        # 錘子線（下影線長，實體小，在下跌趨勢末端）
        df['hammer'] = (
            (df['lower_shadow'] > df['body'] * 2) &
            (df['upper_shadow'] < df['body'] * 0.3) &
            (df['body_percent'] > 0.1)
        ).astype(int)

        # 吞沒形態
        df['bullish_engulfing'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &  # 前一根陰線
            (df['close'] > df['open']) &  # 當前陽線
            (df['open'] < df['close'].shift(1)) &  # 開盤低於前收
            (df['close'] > df['open'].shift(1))  # 收盤高於前開
        ).astype(int)

        df['bearish_engulfing'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &  # 前一根陽線
            (df['close'] < df['open']) &  # 當前陰線
            (df['open'] > df['close'].shift(1)) &  # 開盤高於前收
            (df['close'] < df['open'].shift(1))  # 收盤低於前開
        ).astype(int)

        # 連續陽線/陰線
        df['consecutive_up'] = (df['close'] > df['open']).astype(int)
        df['consecutive_up_count'] = df['consecutive_up'].groupby(
            (df['consecutive_up'] != df['consecutive_up'].shift()).cumsum()
        ).cumsum()
        df.loc[df['consecutive_up'] == 0, 'consecutive_up_count'] = 0

        df['consecutive_down'] = (df['close'] < df['open']).astype(int)
        df['consecutive_down_count'] = df['consecutive_down'].groupby(
            (df['consecutive_down'] != df['consecutive_down'].shift()).cumsum()
        ).cumsum()
        df.loc[df['consecutive_down'] == 0, 'consecutive_down_count'] = 0

        return df

    def get_feature_names(self) -> List[str]:
        """獲取所有特徵名稱列表"""
        # 創建示例數據來獲取特徵名
        sample_data = pd.DataFrame({
            'open': [100] * 250,
            'high': [101] * 250,
            'low': [99] * 250,
            'close': [100.5] * 250,
            'volume': [1000] * 250
        })

        result = self.calculate_all(sample_data)

        # 排除 OHLCV 原始列
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_names = [col for col in result.columns if col not in exclude_cols]

        return feature_names

    def calculate_signal_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        計算用於即時信號生成的精簡特徵集

        Returns:
            最新一行的關鍵特徵字典
        """
        df = self.calculate_all(df)
        latest = df.iloc[-1]

        return {
            # 趨勢
            'trend_score': self._calculate_trend_score(df),
            'macd_signal': 1 if latest['macd'] > latest['macd_signal'] else -1,
            'adx': latest['adx'],

            # 動量
            'rsi_14': latest['rsi_14'],
            'stoch_k': latest['stoch_k'],
            'momentum': latest['momentum_10'],

            # 波動
            'volatility': latest['volatility_20'],
            'bb_position': latest['bb_position'],
            'atr_percent': latest['atr_percent'],

            # 成交量
            'volume_ratio': latest['volume_ratio'],
            'obv_trend': latest['obv_trend'],

            # 綜合評分
            'overall_signal': self._calculate_overall_signal(df)
        }

    def _calculate_trend_score(self, df: pd.DataFrame) -> float:
        """計算趨勢評分 (-1 到 1)"""
        latest = df.iloc[-1]

        score = 0
        # EMA 排列
        if latest['ema_9'] > latest['ema_21'] > latest['ema_50']:
            score += 0.3
        elif latest['ema_9'] < latest['ema_21'] < latest['ema_50']:
            score -= 0.3

        # 價格在 SMA 上方/下方
        if latest['close'] > latest['sma_50']:
            score += 0.2
        else:
            score -= 0.2

        # MACD
        if latest['macd'] > 0 and latest['macd'] > latest['macd_signal']:
            score += 0.25
        elif latest['macd'] < 0 and latest['macd'] < latest['macd_signal']:
            score -= 0.25

        # ADX 趨勢強度
        if latest['adx'] > 25:
            score *= 1.2  # 強趨勢放大

        return np.clip(score, -1, 1)

    def _calculate_overall_signal(self, df: pd.DataFrame) -> float:
        """計算綜合交易信號 (-1 到 1)"""
        latest = df.iloc[-1]

        signals = []
        weights = []

        # 趨勢信號 (權重 0.3)
        trend = self._calculate_trend_score(df)
        signals.append(trend)
        weights.append(0.3)

        # RSI 信號 (權重 0.2)
        rsi = latest['rsi_14']
        if rsi > 65:
            rsi_signal = -0.5 - (rsi - 70) / 60  # 超買
        elif rsi < 35:
            rsi_signal = 0.5 + (30 - rsi) / 60  # 超賣
        else:
            rsi_signal = (rsi - 50) / 40  # 中性區域
        signals.append(rsi_signal)
        weights.append(0.2)

        # BB 位置信號 (權重 0.15)
        bb_pos = latest['bb_position']
        if bb_pos > 0.95:
            bb_signal = -0.5
        elif bb_pos < 0.05:
            bb_signal = 0.5
        else:
            bb_signal = (bb_pos - 0.5) * 0.5
        signals.append(bb_signal)
        weights.append(0.15)

        # 成交量確認 (權重 0.15)
        vol_signal = latest['volume_price_trend'] * 0.5
        signals.append(vol_signal)
        weights.append(0.15)

        # 動量 (權重 0.2)
        momentum = latest['momentum_10']
        mom_signal = np.clip(momentum / (latest['close'] * 0.05), -1, 1)
        signals.append(mom_signal)
        weights.append(0.2)

        # 加權平均
        overall = sum(s * w for s, w in zip(signals, weights))

        return np.clip(overall, -1, 1)


# 便捷函數
def calculate_features(df: pd.DataFrame, config: Optional[FeatureConfig] = None) -> pd.DataFrame:
    """計算所有技術指標特徵的便捷函數"""
    calculator = TechnicalFeatures(config)
    return calculator.calculate_all(df)
