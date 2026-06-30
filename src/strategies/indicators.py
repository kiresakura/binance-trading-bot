"""
技術指標庫

提供所有交易策略共用的技術指標計算函數
借鑒 Freqtrade 的指標設計，支援向量化操作

v0.4.0 - Enterprise Edition

作者：Binance Trading Bot
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


# ==================== 趨勢指標 ====================

def sma(series: pd.Series, period: int) -> pd.Series:
    """簡單移動平均線 (Simple Moving Average)"""
    return series.rolling(window=period, min_periods=1).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """指數移動平均線 (Exponential Moving Average)"""
    return series.ewm(span=period, adjust=False).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """加權移動平均線 (Weighted Moving Average)"""
    weights = np.arange(1, period + 1)
    return series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def hma(series: pd.Series, period: int) -> pd.Series:
    """Hull 移動平均線 (Hull Moving Average)"""
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    wma_half = wma(series, half_period)
    wma_full = wma(series, period)
    return wma(2 * wma_half - wma_full, sqrt_period)


def tema(series: pd.Series, period: int) -> pd.Series:
    """三重指數移動平均線 (Triple EMA)"""
    ema1 = ema(series, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3


def dema(series: pd.Series, period: int) -> pd.Series:
    """雙重指數移動平均線 (Double EMA)"""
    ema1 = ema(series, period)
    ema2 = ema(ema1, period)
    return 2 * ema1 - ema2


def vwma(df: pd.DataFrame, period: int, price_col: str = 'close') -> pd.Series:
    """成交量加權移動平均線 (Volume Weighted MA)"""
    pv = df[price_col] * df['volume']
    return pv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()


# ==================== 動量指標 ====================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """相對強弱指標 (Relative Strength Index)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def stoch_rsi(series: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
    """隨機 RSI (Stochastic RSI)"""
    rsi_values = rsi(series, period)
    rsi_min = rsi_values.rolling(window=period).min()
    rsi_max = rsi_values.rolling(window=period).max()

    stoch = (rsi_values - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    k = stoch.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()

    return k, d


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD 指標 (Moving Average Convergence Divergence)"""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """動量指標 (Momentum)"""
    return series.diff(period)


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    """變動率指標 (Rate of Change)"""
    return (series / series.shift(period) - 1) * 100


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """威廉指標 (Williams %R)"""
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    return -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-10)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """商品通道指標 (Commodity Channel Index)"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = sma(typical_price, period)
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (typical_price - sma_tp) / (0.015 * mean_deviation + 1e-10)


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """資金流量指標 (Money Flow Index)"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']

    delta = typical_price.diff()
    positive_flow = money_flow.where(delta > 0, 0.0)
    negative_flow = money_flow.where(delta < 0, 0.0)

    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()

    mfi_ratio = positive_sum / (negative_sum + 1e-10)
    return 100 - (100 / (1 + mfi_ratio))


def ultimate_oscillator(df: pd.DataFrame, period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
    """終極震盪指標 (Ultimate Oscillator)"""
    bp = df['close'] - df[['low', 'close']].shift(1).min(axis=1)
    tr = df[['high', 'close']].shift(1).max(axis=1) - df[['low', 'close']].shift(1).min(axis=1)

    avg1 = bp.rolling(period1).sum() / (tr.rolling(period1).sum() + 1e-10)
    avg2 = bp.rolling(period2).sum() / (tr.rolling(period2).sum() + 1e-10)
    avg3 = bp.rolling(period3).sum() / (tr.rolling(period3).sum() + 1e-10)

    return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7


# ==================== 波動率指標 ====================

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """平均真實範圍 (Average True Range)"""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)

    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def natr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """標準化 ATR (Normalized ATR)"""
    return (atr(df, period) / df['close']) * 100


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """布林通道 (Bollinger Bands)"""
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def bollinger_width(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """布林帶寬度"""
    upper, middle, lower = bollinger_bands(series, period, std_dev)
    return (upper - lower) / middle


def bollinger_percent(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """布林帶百分比位置 (%B)"""
    upper, middle, lower = bollinger_bands(series, period, std_dev)
    return (series - lower) / (upper - lower + 1e-10)


def keltner_channel(df: pd.DataFrame, period: int = 20, atr_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """肯特納通道 (Keltner Channel)"""
    middle = ema(df['close'], period)
    atr_value = atr(df, period)
    upper = middle + atr_mult * atr_value
    lower = middle - atr_mult * atr_value
    return upper, middle, lower


def donchian_channel(df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """唐奇安通道 (Donchian Channel)"""
    upper = df['high'].rolling(window=period).max()
    lower = df['low'].rolling(window=period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower


def stddev(series: pd.Series, period: int = 20) -> pd.Series:
    """標準差"""
    return series.rolling(window=period).std()


# ==================== 趨勢強度指標 ====================

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """平均趨向指標 (Average Directional Index)"""
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr_value = atr(df, period)

    plus_di = 100 * ema(plus_dm, period) / (atr_value + 1e-10)
    minus_di = 100 * ema(minus_dm, period) / (atr_value + 1e-10)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    return ema(dx, period)


def di_plus(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """正向趨向指標 (+DI)"""
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    atr_value = atr(df, period)
    return 100 * ema(plus_dm, period) / (atr_value + 1e-10)


def di_minus(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """負向趨向指標 (-DI)"""
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr_value = atr(df, period)
    return 100 * ema(minus_dm, period) / (atr_value + 1e-10)


def aroon(df: pd.DataFrame, period: int = 25) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """阿隆指標 (Aroon)"""
    aroon_up = 100 * df['high'].rolling(window=period + 1).apply(
        lambda x: x.argmax(), raw=True
    ) / period
    aroon_down = 100 * df['low'].rolling(window=period + 1).apply(
        lambda x: x.argmin(), raw=True
    ) / period
    aroon_osc = aroon_up - aroon_down
    return aroon_up, aroon_down, aroon_osc


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """超級趨勢指標 (SuperTrend)"""
    hl2 = (df['high'] + df['low']) / 2
    atr_value = atr(df, period)

    upper_band = hl2 + (multiplier * atr_value)
    lower_band = hl2 - (multiplier * atr_value)

    supertrend_val = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    supertrend_val.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(df)):
        if df['close'].iloc[i] > supertrend_val.iloc[i-1]:
            supertrend_val.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            supertrend_val.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1

    return supertrend_val, direction


def parabolic_sar(df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """拋物線 SAR (Parabolic SAR)"""
    high = df['high']
    low = df['low']
    close = df['close']

    sar = pd.Series(index=df.index, dtype=float)
    af = af_start
    trend = 1  # 1 = 上升, -1 = 下降

    sar.iloc[0] = low.iloc[0]
    ep = high.iloc[0]  # 極端點

    for i in range(1, len(df)):
        if trend == 1:
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])

            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + af_step, af_max)

            if low.iloc[i] < sar.iloc[i]:
                trend = -1
                sar.iloc[i] = ep
                ep = low.iloc[i]
                af = af_start
        else:
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])

            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + af_step, af_max)

            if high.iloc[i] > sar.iloc[i]:
                trend = 1
                sar.iloc[i] = ep
                ep = high.iloc[i]
                af = af_start

    return sar


# ==================== 成交量指標 ====================

def obv(df: pd.DataFrame) -> pd.Series:
    """能量潮指標 (On-Balance Volume)"""
    direction = np.sign(df['close'].diff())
    direction.iloc[0] = 0
    return (direction * df['volume']).cumsum()


def ad_line(df: pd.DataFrame) -> pd.Series:
    """累積派發線 (Accumulation/Distribution Line)"""
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
    return (clv * df['volume']).cumsum()


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """蔡金資金流量 (Chaikin Money Flow)"""
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
    return (clv * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()


def force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
    """強力指數 (Force Index)"""
    fi = df['close'].diff() * df['volume']
    return ema(fi, period)


def eom(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """簡易波動指標 (Ease of Movement)"""
    dm = ((df['high'] + df['low']) / 2) - ((df['high'].shift(1) + df['low'].shift(1)) / 2)
    br = df['volume'] / (df['high'] - df['low'] + 1e-10)
    eom_val = dm / (br + 1e-10)
    return sma(eom_val, period)


def vwap(df: pd.DataFrame) -> pd.Series:
    """成交量加權平均價 (Volume Weighted Average Price)"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cumulative_tp_vol = (typical_price * df['volume']).cumsum()
    cumulative_vol = df['volume'].cumsum()
    return cumulative_tp_vol / (cumulative_vol + 1e-10)


def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """成交量比率"""
    volume_sma = sma(df['volume'], period)
    return df['volume'] / (volume_sma + 1e-10)


# ==================== 價格形態指標 ====================

def pivot_points(df: pd.DataFrame) -> dict:
    """樞軸點 (Pivot Points)"""
    pp = (df['high'] + df['low'] + df['close']) / 3
    r1 = 2 * pp - df['low']
    s1 = 2 * pp - df['high']
    r2 = pp + (df['high'] - df['low'])
    s2 = pp - (df['high'] - df['low'])
    r3 = df['high'] + 2 * (pp - df['low'])
    s3 = df['low'] - 2 * (df['high'] - pp)

    return {
        'pp': pp,
        'r1': r1, 's1': s1,
        'r2': r2, 's2': s2,
        'r3': r3, 's3': s3
    }


def fibonacci_retracement(high: float, low: float) -> dict:
    """費波那契回撤位"""
    diff = high - low
    return {
        '0.0': low,
        '0.236': low + 0.236 * diff,
        '0.382': low + 0.382 * diff,
        '0.5': low + 0.5 * diff,
        '0.618': low + 0.618 * diff,
        '0.786': low + 0.786 * diff,
        '1.0': high
    }


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> dict:
    """一目均衡表 (Ichimoku Cloud)"""
    tenkan_sen = (df['high'].rolling(window=tenkan).max() + df['low'].rolling(window=tenkan).min()) / 2
    kijun_sen = (df['high'].rolling(window=kijun).max() + df['low'].rolling(window=kijun).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_span_b = ((df['high'].rolling(window=senkou).max() + df['low'].rolling(window=senkou).min()) / 2).shift(kijun)
    chikou_span = df['close'].shift(-kijun)

    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }


# ==================== 統計指標 ====================

def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Z-Score 標準化"""
    mean = sma(series, period)
    std = stddev(series, period)
    return (series - mean) / (std + 1e-10)


def correlation(series1: pd.Series, series2: pd.Series, period: int = 20) -> pd.Series:
    """滾動相關係數"""
    return series1.rolling(window=period).corr(series2)


def beta(returns: pd.Series, benchmark_returns: pd.Series, period: int = 60) -> pd.Series:
    """滾動 Beta 係數"""
    covariance = returns.rolling(window=period).cov(benchmark_returns)
    variance = benchmark_returns.rolling(window=period).var()
    return covariance / (variance + 1e-10)


def sharpe_rolling(returns: pd.Series, period: int = 252, risk_free: float = 0.02) -> pd.Series:
    """滾動夏普比率"""
    excess_returns = returns - risk_free / 252
    return np.sqrt(252) * excess_returns.rolling(window=period).mean() / (excess_returns.rolling(window=period).std() + 1e-10)


def max_drawdown_rolling(returns: pd.Series, period: int = 252) -> pd.Series:
    """滾動最大回撤"""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.rolling(window=period, min_periods=1).max()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.rolling(window=period, min_periods=1).min()


# ==================== 輔助函數 ====================

def cross_above(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """向上穿越"""
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))


def cross_below(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """向下穿越"""
    return (series1 < series2) & (series1.shift(1) >= series2.shift(1))


def crossed(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """任意方向穿越"""
    return cross_above(series1, series2) | cross_below(series1, series2)


def highest(series: pd.Series, period: int) -> pd.Series:
    """滾動最高值"""
    return series.rolling(window=period).max()


def lowest(series: pd.Series, period: int) -> pd.Series:
    """滾動最低值"""
    return series.rolling(window=period).min()


def range_high_low(df: pd.DataFrame, period: int) -> pd.Series:
    """價格範圍（最高-最低）"""
    return highest(df['high'], period) - lowest(df['low'], period)


def true_range(df: pd.DataFrame) -> pd.Series:
    """真實範圍"""
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)

    tr1 = high - low
    tr2 = abs(high - close_prev)
    tr3 = abs(low - close_prev)

    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def returns(series: pd.Series, period: int = 1) -> pd.Series:
    """收益率"""
    return series.pct_change(periods=period)


def log_returns(series: pd.Series, period: int = 1) -> pd.Series:
    """對數收益率"""
    return np.log(series / series.shift(period))


# ==================== 指標組合函數 ====================

def populate_all_indicators(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    填充所有常用指標

    Args:
        df: OHLCV DataFrame
        config: 指標配置字典

    Returns:
        添加了指標的 DataFrame
    """
    config = config or {}

    # 複製 DataFrame
    df = df.copy()

    # 確保有基本列
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # === 移動平均線 ===
    for period in config.get('sma_periods', [5, 10, 20, 50, 200]):
        df[f'sma_{period}'] = sma(df['close'], period)

    for period in config.get('ema_periods', [9, 12, 26, 50]):
        df[f'ema_{period}'] = ema(df['close'], period)

    # === RSI ===
    for period in config.get('rsi_periods', [14]):
        df[f'rsi_{period}'] = rsi(df['close'], period)

    # === MACD ===
    macd_config = config.get('macd', {'fast': 12, 'slow': 26, 'signal': 9})
    df['macd'], df['macd_signal'], df['macd_hist'] = macd(
        df['close'],
        macd_config['fast'],
        macd_config['slow'],
        macd_config['signal']
    )

    # === 布林通道 ===
    bb_config = config.get('bollinger', {'period': 20, 'std': 2.0})
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = bollinger_bands(
        df['close'],
        bb_config['period'],
        bb_config['std']
    )
    df['bb_width'] = bollinger_width(df['close'], bb_config['period'], bb_config['std'])
    df['bb_percent'] = bollinger_percent(df['close'], bb_config['period'], bb_config['std'])

    # === ATR ===
    for period in config.get('atr_periods', [14]):
        df[f'atr_{period}'] = atr(df, period)
        df[f'natr_{period}'] = natr(df, period)

    # === ADX ===
    adx_period = config.get('adx_period', 14)
    df['adx'] = adx(df, adx_period)
    df['di_plus'] = di_plus(df, adx_period)
    df['di_minus'] = di_minus(df, adx_period)

    # === 成交量指標 ===
    df['obv'] = obv(df)
    df['cmf'] = cmf(df, config.get('cmf_period', 20))
    df['volume_sma_20'] = sma(df['volume'], 20)
    df['volume_ratio'] = volume_ratio(df, 20)

    # === MFI ===
    df['mfi'] = mfi(df, config.get('mfi_period', 14))

    # === 收益率 ===
    df['returns_1'] = returns(df['close'], 1)
    df['returns_3'] = returns(df['close'], 3)
    df['returns_5'] = returns(df['close'], 5)

    # === Stochastic RSI ===
    df['stoch_k'], df['stoch_d'] = stoch_rsi(df['close'])

    # === Williams %R ===
    df['willr'] = williams_r(df)

    # === CCI ===
    df['cci'] = cci(df)

    return df
