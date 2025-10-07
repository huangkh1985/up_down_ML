import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy import stats
from tqdm import tqdm
import math
from utils.wavelet_denoise import wavelet_denoising, ensure_length_match
from utils.pattern_recognition import add_pattern_features

# 忽略警告
warnings.filterwarnings("ignore")

# 尝试导入 talib，如果失败则使用替代实现
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib 不可用，使用纯 Python 实现")

# TA-Lib 替代函数（纯 pandas/numpy 实现）
def _ema(data, timeperiod):
    """EMA 指数移动平均线"""
    return data.ewm(span=timeperiod, adjust=False).mean()

def _rsi(data, timeperiod=14):
    """RSI 相对强弱指数"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def _atr(high, low, close, timeperiod=14):
    """ATR 真实波幅"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(timeperiod).mean()

def _cci(high, low, close, timeperiod=14):
    """CCI 商品通道指数"""
    tp = (high + low + close) / 3
    sma = tp.rolling(timeperiod).mean()
    mad = tp.rolling(timeperiod).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad)

def _adx(high, low, close, timeperiod=14):
    """ADX 平均趋向指数"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = _atr(high, low, close, 1)
    plus_di = 100 * (plus_dm.rolling(timeperiod).mean() / tr.rolling(timeperiod).mean())
    minus_di = 100 * (minus_dm.rolling(timeperiod).mean() / tr.rolling(timeperiod).mean())
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(timeperiod).mean()
    return adx

def _plus_di(high, low, close, timeperiod=14):
    """DI+ 正向指标"""
    plus_dm = high.diff()
    plus_dm[plus_dm < 0] = 0
    tr = _atr(high, low, close, 1)
    return 100 * (plus_dm.rolling(timeperiod).mean() / tr.rolling(timeperiod).mean())

def _minus_di(high, low, close, timeperiod=14):
    """DI- 负向指标"""
    minus_dm = -low.diff()
    minus_dm[minus_dm < 0] = 0
    tr = _atr(high, low, close, 1)
    return 100 * (minus_dm.rolling(timeperiod).mean() / tr.rolling(timeperiod).mean())

def _mfi(high, low, close, volume, timeperiod=14):
    """MFI 资金流量指数"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_mf = positive_flow.rolling(timeperiod).sum()
    negative_mf = negative_flow.rolling(timeperiod).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    return mfi

def _ad(high, low, close, volume):
    """AD 累积/派发线"""
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0)
    ad = (clv * volume).cumsum()
    return ad

def _obv(close, volume):
    """OBV 能量潮"""
    obv = volume.copy()
    obv[close < close.shift(1)] = -volume[close < close.shift(1)]
    return obv.cumsum()

def _trix(close, timeperiod=30):
    """TRIX 三重指数平滑移动平均"""
    ema1 = close.ewm(span=timeperiod, adjust=False).mean()
    ema2 = ema1.ewm(span=timeperiod, adjust=False).mean()
    ema3 = ema2.ewm(span=timeperiod, adjust=False).mean()
    return ema3.pct_change() * 100

def _tema(close, timeperiod=30):
    """TEMA 三重指数移动平均"""
    ema1 = close.ewm(span=timeperiod, adjust=False).mean()
    ema2 = ema1.ewm(span=timeperiod, adjust=False).mean()
    ema3 = ema2.ewm(span=timeperiod, adjust=False).mean()
    return 3 * ema1 - 3 * ema2 + ema3

# 统一的 TA-Lib 函数接口
if TALIB_AVAILABLE:
    EMA = talib.EMA
    RSI = talib.RSI
    ATR = talib.ATR
    CCI = talib.CCI
    ADX = talib.ADX
    PLUS_DI = talib.PLUS_DI
    MINUS_DI = talib.MINUS_DI
    MFI = talib.MFI
    AD = talib.AD
    OBV = talib.OBV
    TRIX = talib.TRIX
    TEMA = talib.TEMA
else:
    EMA = _ema
    RSI = _rsi
    ATR = _atr
    CCI = _cci
    ADX = _adx
    PLUS_DI = _plus_di
    MINUS_DI = _minus_di
    MFI = _mfi
    AD = _ad
    OBV = _obv
    TRIX = _trix
    TEMA = _tema

def add_moving_averages(df):
    """
    添加各种移动平均线指标
    
    参数:
    df: 包含股票数据的DataFrame
    
    返回:
    添加了移动平均线指标的DataFrame
    """
    # 添加移动平均线，价+时
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['AVG_VOL'] = df['Volume'].rolling(window=10).mean()
    df['AVG_TR'] = df['TurnoverRate'].rolling(window=10).mean()
    
    # 添加EMA (指数移动平均线)
    df['EMA_12'] = EMA(df['Close'], timeperiod=12)
    df['EMA_26'] = EMA(df['Close'], timeperiod=26)
    
    return df

def add_price_indicators(df):
    """
    添加价格相关指标
    
    参数:
    df: 包含股票数据的DataFrame
    
    返回:
    添加了价格相关指标的DataFrame
    """
    # 添加降噪后的价格变化率
    df['Denoised_PriceChange'] = df['Close'].pct_change()
    # 应用小波降噪处理价格变化率
    denoised_data = wavelet_denoising(
        df['Denoised_PriceChange'].fillna(0).values, 
        wavelet='sym8', 
        level=2
    )
    # 使用辅助函数确保长度匹配
    df['Denoised_PriceChange'] = ensure_length_match(denoised_data, len(df), 'Denoised_PriceChange')
    
    # 添加价格动量指标
    df['Momentum'] = df['Close'].diff(periods=5)
    momentum_denoised = wavelet_denoising(df['Momentum'].fillna(0).values, wavelet='db4', level=1)
    df['Momentum'] = ensure_length_match(momentum_denoised, len(df), 'Momentum')
    
    # 添加价格波动率
    df['Volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
    
    # 添加ROC (变动率)
    df['ROC'] = df['Close'].pct_change(periods=10) * 100
    
    return df

def add_bollinger_bands(df):
    """
    添加布林带相关指标
    
    参数:
    df: 包含股票数据的DataFrame
    
    返回:
    添加了布林带相关指标的DataFrame
    """
    # 计算布林带，价+空
    df['STD'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (df['STD'] * 2.5)
    df['Lower_Band'] = df['MA20'] - (df['STD'] * 2.5)

    # 计算%B (Bollinger Band %)
    df['%B'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])
    
    return df

def add_macd(df):
    """
    添加MACD指标
    
    参数:
    df: 包含股票数据的DataFrame
    
    返回:
    添加了MACD指标的DataFrame
    """
    # 添加MACD，价+时
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 对MACD应用小波降噪
    macd_denoised = wavelet_denoising(df['MACD'].fillna(0).values, wavelet='db4', level=1)
    signal_denoised = wavelet_denoising(df['Signal'].fillna(0).values, wavelet='db4', level=1)
    
    # 使用辅助函数确保长度匹配
    df['MACD'] = ensure_length_match(macd_denoised, len(df), 'MACD')
    df['Signal'] = ensure_length_match(signal_denoised, len(df), 'Signal')
    
    return df

def add_kdj(df):
    """
    添加KDJ指标
    
    参数:
    df: 包含股票数据的DataFrame
    
    返回:
    添加了KDJ指标的DataFrame
    """
    # 添加KDJ指标，价+空
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['D'] = df['K'].rolling(window=3).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # 添加Stochastic Oscillator (随机振荡器)
    df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    return df

def add_dmi(df):
    """
    添加DMI指标
    
    参数:
    df: 包含股票数据的DataFrame
    
    返回:
    添加了DMI指标的DataFrame
    """
    # 添加DMI指标
    df['DI_plus'] = PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['DI_minus'] = MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ADX'] = ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    return df

def add_volume_indicators(df):
    """
    添加成交量相关指标
    
    参数:
    df: 包含股票数据的DataFrame
    
    返回:
    添加了成交量相关指标的DataFrame
    """
    # 添加VWAP (成交量加权平均价格)
    df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    
    # 添加MFI (资金流量指标)
    df['MFI'] = MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)

    # 添加AD (累积/派发)
    df['AD'] = AD(df['High'], df['Low'], df['Close'], df['Volume'])

    # 添加OBV (On-Balance Volume)
    df['OBV'] = OBV(df['Close'], df['Volume'])
    
    return df

def add_other_indicators(df):
    """
    添加其他技术指标
    
    参数:
    df: 包含股票数据的DataFrame
    
    返回:
    添加了其他技术指标的DataFrame
    """
    # 添加TRIX指标
    df['TRIX'] = TRIX(df['Close'], timeperiod=30)
    
    # 添加TEMA指标
    df['TEMA'] = TEMA(df['Close'], timeperiod=30)
    
    # 添加Williams %R
    highest_high = df['High'].rolling(window=14).max()
    lowest_low = df['Low'].rolling(window=14).min()
    df['Williams_R'] = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
    
    # 添加ATR (真实波幅)
    df['ATR'] = ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    # 添加CCI (commodity channel index)
    df['CCI'] = CCI(df['High'], df['Low'], df['Close'], timeperiod=14)

    # 添加RSI (相对强弱指数)
    df['RSI'] = RSI(df['Close'], timeperiod=14)
    
    return df

def add_correlation_indicators(df):
    """
    添加相关性指标
    
    参数:
    df: 包含股票数据的DataFrame
    
    返回:
    添加了相关性指标的DataFrame
    """
    # 添加行业相关指标
    if 'high_industry_index' in df.columns:
        # 个股与行业指数的相对强度
        df['Stock_Industry_RS'] = df['Close'] / df['close_industry_index']
        # 个股与行业指数的相关性(20日滚动)
        df['Industry_Correlation'] = df['Close'].rolling(window=20).corr(df['close_industry_index'])
    
    # 添加与上证指数相关的指标
    if 'close_SZ_index' in df.columns:
        # 个股与上证指数的相对强度
        df['Stock_SZ_RS'] = df['Close'] / df['close_SZ_index']
        # 个股与上证指数的相关性(20日滚动)
        df['SZ_Correlation'] = df['Close'].rolling(window=20).corr(df['close_SZ_index'])
    
    return df

def add_volatility_indicators(df):
    """
    添加更多高级波动率指标
    
    参数:
    df: 包含股票数据的DataFrame
    
    返回:
    添加了波动率指标的DataFrame
    """
    # 计算历史波动率（Historical Volatility）- 20日
    df['HV_20'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    
    # 计算Garman-Klass波动率 - 考虑日内高低点的波动率估计
    df['GK_Volatility'] = 0.5 * np.log(df['High'] / df['Low'])**2 - (2*np.log(2)-1) * np.log(df['Close'] / df['Open'])**2
    df['GK_Volatility'] = np.sqrt(df['GK_Volatility'].rolling(window=20).mean() * 252)
    
    # 计算波动率比率 - 短期/长期波动率比较
    df['Vol_Ratio'] = df['Close'].pct_change().rolling(window=10).std() / \
                     df['Close'].pct_change().rolling(window=30).std()
    
    # 添加True Range变化率
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['TR'] = true_range
    df['TR_ROC'] = df['TR'].pct_change(periods=5) * 100  # 5日TR变化率
    
    # 添加UVOL/DVOL (上涨/下跌成交量比率)
    df['UD_Ratio'] = np.where(df['Close'] > df['Close'].shift(), df['Volume'], 0) / \
                    np.where(df['Close'] < df['Close'].shift(), df['Volume'], 1)
    # 平滑处理
    df['UD_Ratio'] = df['UD_Ratio'].replace([np.inf, -np.inf], np.nan).fillna(1)
    df['UD_Ratio'] = df['UD_Ratio'].rolling(window=10).mean()
    
    # 添加归一化ATR (ATR/价格)
    df['Normalized_ATR'] = df['ATR'] / df['Close']
    
    return df

def add_sentiment_indicators(df):
    """
    添加市场情绪指标
    
    参数:
    df: 包含股票数据的DataFrame
    
    返回:
    添加了情绪指标的DataFrame
    """
    # 计算布林带宽度作为市场不确定性指标
    if 'Upper_Band' in df.columns and 'Lower_Band' in df.columns:
        df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['MA20']
    
    # 计算价格振幅
    df['Amplitude'] = (df['High'] - df['Low']) / df['Close'].shift(1) * 100
    
    # 计算涨跌天数比率
    df['Price_Up'] = np.where(df['Close'] > df['Close'].shift(1), 1, 0)
    df['Up_Days_Ratio'] = df['Price_Up'].rolling(window=20).sum() / 20
    
    # 添加极端波动日计数
    avg_amplitude = df['Amplitude'].rolling(window=20).mean()
    df['Extreme_Day'] = np.where(df['Amplitude'] > 2 * avg_amplitude, 1, 0)
    df['Extreme_Days_Count'] = df['Extreme_Day'].rolling(window=20).sum()
    
    # 添加"超买超卖"指标
    if 'RSI' in df.columns:
        df['Overbought'] = np.where(df['RSI'] > 80, 1, 0)
        df['Oversold'] = np.where(df['RSI'] < 20, 1, 0)
        df['OB_OS_Signal'] = df['Overbought'] - df['Oversold']
    
    # 价格趋势强度
    df['Trend_Strength'] = np.abs(df['Close'].pct_change(periods=20)) / \
                         (df['Close'].pct_change().rolling(window=20).std() * np.sqrt(20))
    
    return df

def add_advanced_momentum_indicators(df):
    """
    添加高级动量指标
    
    参数:
    df: 包含股票数据的DataFrame
    
    返回:
    添加了高级动量指标的DataFrame
    """
    # 添加TSI (True Strength Index)
    momentum = df['Close'].diff()
    abs_momentum = np.abs(momentum)
    
    # 双重平滑动量和绝对动量
    momentum_smooth1 = momentum.ewm(span=25, adjust=False).mean()
    momentum_smooth2 = momentum_smooth1.ewm(span=13, adjust=False).mean()
    abs_momentum_smooth1 = abs_momentum.ewm(span=25, adjust=False).mean()
    abs_momentum_smooth2 = abs_momentum_smooth1.ewm(span=13, adjust=False).mean()
    
    df['TSI'] = 100 * momentum_smooth2 / abs_momentum_smooth2
    
    # 添加CMO (Chande Momentum Oscillator)
    # 收盘价上涨和下跌的绝对值之和
    up_sum = np.zeros(len(df))
    down_sum = np.zeros(len(df))
    
    # 计算上涨和下跌
    for i in range(1, len(df)):
        change = df['Close'].iloc[i] - df['Close'].iloc[i-1]
        if change > 0:
            up_sum[i] = change
        else:
            down_sum[i] = abs(change)
    
    # 转换为Series以便使用rolling
    up_sum_series = pd.Series(up_sum)
    down_sum_series = pd.Series(down_sum)
    
    # 计算14天上涨和下跌之和
    up_sum_14 = up_sum_series.rolling(window=14).sum()
    down_sum_14 = down_sum_series.rolling(window=14).sum()
    
    # 计算CMO
    df['CMO'] = 100 * ((up_sum_14 - down_sum_14) / (up_sum_14 + down_sum_14))
    
    # 添加Fisher Transform应用于RSI
    if 'RSI' in df.columns:
        # 将RSI范围从[0, 100]转换为[-1, 1]
        scaled_rsi = 2 * (df['RSI'] / 100 - 0.5)
        # 应用Fisher变换
        df['Fisher_Transform_RSI'] = 0.5 * np.log((1 + scaled_rsi) / (1 - scaled_rsi))
        # 处理无穷大值
        df['Fisher_Transform_RSI'] = df['Fisher_Transform_RSI'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 添加KST (Know Sure Thing)
    # 价格变动率的加权合计，以捕捉多个时间周期的动量
    roc10 = df['Close'].pct_change(periods=10)
    roc15 = df['Close'].pct_change(periods=15)
    roc20 = df['Close'].pct_change(periods=20)
    roc30 = df['Close'].pct_change(periods=30)
    
    # 移动平均
    ma10 = roc10.rolling(window=10).mean()
    ma15 = roc15.rolling(window=10).mean()
    ma20 = roc20.rolling(window=10).mean()
    ma30 = roc30.rolling(window=15).mean()
    
    # 计算KST
    df['KST'] = 1*ma10 + 2*ma15 + 3*ma20 + 4*ma30
    df['KST_Signal'] = df['KST'].rolling(window=9).mean()
    
    return df

def add_technical_indicators(df, include_patterns=True):
    """
    添加所有技术指标
    
    参数:
    df: 包含股票数据的DataFrame
    include_patterns: 是否包含形态识别特征（默认True）
    
    返回:
    添加了所有技术指标的DataFrame
    """
    # 应用每个指标函数
    df = add_moving_averages(df)
    df = add_price_indicators(df)
    df = add_bollinger_bands(df)
    df = add_macd(df)
    df = add_kdj(df)
    df = add_dmi(df)
    df = add_volume_indicators(df)
    df = add_other_indicators(df)
    df = add_correlation_indicators(df)
    
    # 调用新增的指标函数
    df = add_volatility_indicators(df)
    df = add_sentiment_indicators(df)
    df = add_advanced_momentum_indicators(df)
    
    # 添加形态识别特征（反转、回调、反弹等）
    if include_patterns:
        try:
            df = add_pattern_features(df)
        except Exception as e:
            print(f"  ⚠ 形态识别失败: {e}")
            print(f"  继续使用其他技术指标...")
    
    # 填充NaN值
    df = df.fillna(method='bfill').fillna(0)
    
    return df

def add_chouma_indicators(data):
    # 计算PPSCM和SHCM
    data['WINNER_CLOSE'] = data['Close'].rolling(window=3).apply(lambda x: np.sum(x > x[-1]), raw=True)
    data['PPSCM'] = data['WINNER_CLOSE'] * 70
    data['WINNER_CLOSE_1.1'] = data['Close'].rolling(window=3).apply(lambda x: np.sum(x > x[-1] * 1.1), raw=True)
    data['WINNER_CLOSE_0.9'] = data['Close'].rolling(window=3).apply(lambda x: np.sum(x > x[-1] * 0.9), raw=True)
    data['SHCM'] = (data['WINNER_CLOSE_1.1'] - data['WINNER_CLOSE_0.9']) * 80
    # 计算ZSHTL和TTSLKP
    data['ZSHTL'] = data['SHCM'] / (data['PPSCM'] + data['SHCM']) * 100
    data['TTSLKP'] = data['PPSCM'] / (data['PPSCM'] + data['SHCM']) * 100

    # 计算ZCMPPS
    data['ZCMPPS'] = data['PPSCM'] + data['SHCM'].rolling(window=13).mean()

    # 计算TTSNTS
    data['TTSNTS'] = (data['ZSHTL'] < 90) & (data['ZSHTL'].shift(1) > 90)
    data['TTSNTS'] = data['TTSNTS'].cumsum()

    # 计算ZSHJJ和TTSLJJ
    data['ZSHJJ'] = data['ZSHTL'].ewm(span=89, adjust=False).mean()
    data['TTSLJJ'] = data['TTSLKP'].ewm(span=89, adjust=False).mean()

    # 计算ZJLRQD
    data['ZJLRQD'] = np.floor(data['TTSLKP'] - data['TTSLJJ'])

    # 计算SH8
    data['SH8'] = data['ZSHTL'].ewm(span=8, adjust=False).mean()

    # 计算DKB
    data['DKB'] = np.where(data['TTSLKP'] - data['TTSLKP'].shift(1) > data['ZSHTL'] - data['ZSHTL'].shift(1), 1, 0)

    # 计算lijinz1和lijinz4
    data['lijinz1'] = (data['Close'] - data['Low'].rolling(window=9).min()) / (data['High'].rolling(window=9).max() - data['Low'].rolling(window=9).min()) * 100
    data['lijinz4'] = (data['Close'] - data['Low'].rolling(window=89).min()) / (data['High'].rolling(window=89).max() - data['Low'].rolling(window=89).min()) * 100
    # 填充NaN值
    data.fillna(method='bfill', inplace=True)
    return data