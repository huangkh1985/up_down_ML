"""
股票技术形态识别模块
识别反转、回调、反弹等重要技术形态，并提取为特征
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import warnings

warnings.filterwarnings('ignore')


def identify_trend(df, window=20):
    """
    识别当前趋势状态
    
    参数:
    df: DataFrame with price data
    window: 趋势判断窗口期
    
    返回:
    trend: 1=上升趋势, -1=下降趋势, 0=横盘
    """
    # 计算短期和长期均线
    ma_short = df['Close'].rolling(window=window//2).mean()
    ma_long = df['Close'].rolling(window=window).mean()
    
    # 判断趋势
    trend = np.where(ma_short > ma_long, 1, -1)
    
    # 横盘判断：价格变化小于2%
    price_change = abs(df['Close'].pct_change(periods=window))
    trend = np.where(price_change < 0.02, 0, trend)
    
    return trend


def identify_reversal_points(df, order=5):
    """
    识别价格反转点（趋势改变）
    
    参数:
    df: DataFrame with price data
    order: 局部极值识别的窗口大小
    
    返回:
    DataFrame with reversal indicators
    """
    # 使用scipy找到局部极大值和极小值
    close_prices = df['Close'].values
    
    # 找到局部最高点（顶部）
    local_max_idx = argrelextrema(close_prices, np.greater, order=order)[0]
    df['Peak'] = 0
    df.loc[df.index[local_max_idx], 'Peak'] = 1
    
    # 找到局部最低点（底部）
    local_min_idx = argrelextrema(close_prices, np.less, order=order)[0]
    df['Trough'] = 0
    df.loc[df.index[local_min_idx], 'Trough'] = 1
    
    # 识别上升反转（从底部反转向上）
    df['Bullish_Reversal'] = 0
    for i in range(1, len(df)):
        if df['Trough'].iloc[i] == 1:
            # 检查是否从下跌趋势反转
            before_trend = df['Close'].iloc[max(0, i-10):i].diff().mean()
            after_trend = df['Close'].iloc[i:min(len(df), i+10)].diff().mean()
            
            if before_trend < 0 and after_trend > 0:
                df.loc[df.index[i], 'Bullish_Reversal'] = 1
    
    # 识别下降反转（从顶部反转向下）
    df['Bearish_Reversal'] = 0
    for i in range(1, len(df)):
        if df['Peak'].iloc[i] == 1:
            # 检查是否从上涨趋势反转
            before_trend = df['Close'].iloc[max(0, i-10):i].diff().mean()
            after_trend = df['Close'].iloc[i:min(len(df), i+10)].diff().mean()
            
            if before_trend > 0 and after_trend < 0:
                df.loc[df.index[i], 'Bearish_Reversal'] = 1
    
    # 计算距离上一个反转点的天数
    df['Days_Since_Reversal'] = 0
    last_reversal = 0
    for i in range(len(df)):
        if df['Bullish_Reversal'].iloc[i] == 1 or df['Bearish_Reversal'].iloc[i] == 1:
            last_reversal = i
        df.loc[df.index[i], 'Days_Since_Reversal'] = i - last_reversal
    
    return df


def identify_pullback(df, trend_window=20, pullback_threshold=0.05):
    """
    识别回调（在上升趋势中的短期下跌）
    
    参数:
    df: DataFrame with price data
    trend_window: 趋势判断窗口
    pullback_threshold: 回调幅度阈值（默认5%）
    
    返回:
    DataFrame with pullback indicators
    """
    # 识别趋势
    trend = identify_trend(df, window=trend_window)
    df['Trend'] = trend
    
    # 计算最近N日最高价
    df['Recent_High'] = df['Close'].rolling(window=trend_window).max()
    
    # 计算回撤幅度
    df['Drawdown_From_High'] = (df['Recent_High'] - df['Close']) / df['Recent_High']
    
    # 识别回调：在上升趋势中，价格回撤超过阈值
    df['Is_Pullback'] = 0
    df['Pullback_Depth'] = 0.0
    
    for i in range(trend_window, len(df)):
        # 当前在上升趋势中
        if trend[i] == 1:
            drawdown = df['Drawdown_From_High'].iloc[i]
            # 回撤超过阈值但不超过20%（超过20%可能是反转）
            if pullback_threshold < drawdown < 0.20:
                df.loc[df.index[i], 'Is_Pullback'] = 1
                df.loc[df.index[i], 'Pullback_Depth'] = drawdown
    
    # 识别回调结束（价格开始回升）
    df['Pullback_Recovery'] = 0
    for i in range(1, len(df)):
        if df['Is_Pullback'].iloc[i-1] == 1 and df['Is_Pullback'].iloc[i] == 0:
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                df.loc[df.index[i], 'Pullback_Recovery'] = 1
    
    # 计算回调持续天数
    df['Pullback_Days'] = 0
    pullback_count = 0
    for i in range(len(df)):
        if df['Is_Pullback'].iloc[i] == 1:
            pullback_count += 1
            df.loc[df.index[i], 'Pullback_Days'] = pullback_count
        else:
            pullback_count = 0
    
    return df


def identify_bounce(df, trend_window=20, bounce_threshold=0.05):
    """
    识别反弹（在下降趋势中的短期上涨）
    
    参数:
    df: DataFrame with price data
    trend_window: 趋势判断窗口
    bounce_threshold: 反弹幅度阈值（默认5%）
    
    返回:
    DataFrame with bounce indicators
    """
    # 识别趋势
    if 'Trend' not in df.columns:
        trend = identify_trend(df, window=trend_window)
        df['Trend'] = trend
    
    # 计算最近N日最低价
    df['Recent_Low'] = df['Close'].rolling(window=trend_window).min()
    
    # 计算反弹幅度
    df['Rally_From_Low'] = (df['Close'] - df['Recent_Low']) / df['Recent_Low']
    
    # 识别反弹：在下降趋势中，价格反弹超过阈值
    df['Is_Bounce'] = 0
    df['Bounce_Height'] = 0.0
    
    for i in range(trend_window, len(df)):
        # 当前在下降趋势中
        if df['Trend'].iloc[i] == -1:
            rally = df['Rally_From_Low'].iloc[i]
            # 反弹超过阈值但不超过20%（超过20%可能是反转）
            if bounce_threshold < rally < 0.20:
                df.loc[df.index[i], 'Is_Bounce'] = 1
                df.loc[df.index[i], 'Bounce_Height'] = rally
    
    # 识别反弹结束（价格继续下跌）
    df['Bounce_End'] = 0
    for i in range(1, len(df)):
        if df['Is_Bounce'].iloc[i-1] == 1 and df['Is_Bounce'].iloc[i] == 0:
            if df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                df.loc[df.index[i], 'Bounce_End'] = 1
    
    # 计算反弹持续天数
    df['Bounce_Days'] = 0
    bounce_count = 0
    for i in range(len(df)):
        if df['Is_Bounce'].iloc[i] == 1:
            bounce_count += 1
            df.loc[df.index[i], 'Bounce_Days'] = bounce_count
        else:
            bounce_count = 0
    
    return df


def calculate_pattern_statistics(df):
    """
    计算形态统计特征
    
    参数:
    df: DataFrame with pattern indicators
    
    返回:
    DataFrame with pattern statistics
    """
    # 反转强度（反转后的价格变化幅度）
    df['Reversal_Strength'] = 0.0
    for i in range(10, len(df)):
        if df['Bullish_Reversal'].iloc[i] == 1:
            # 反转后10天的涨幅
            future_return = (df['Close'].iloc[min(len(df)-1, i+10)] - df['Close'].iloc[i]) / df['Close'].iloc[i]
            df.loc[df.index[i], 'Reversal_Strength'] = future_return
        elif df['Bearish_Reversal'].iloc[i] == 1:
            # 反转后10天的跌幅（负值）
            future_return = (df['Close'].iloc[min(len(df)-1, i+10)] - df['Close'].iloc[i]) / df['Close'].iloc[i]
            df.loc[df.index[i], 'Reversal_Strength'] = future_return
    
    # 回调/反弹频率（过去N天出现的次数）
    window = 60
    df['Pullback_Frequency'] = df['Is_Pullback'].rolling(window=window).sum()
    df['Bounce_Frequency'] = df['Is_Bounce'].rolling(window=window).sum()
    
    # 平均回调/反弹深度
    df['Avg_Pullback_Depth'] = df['Pullback_Depth'].rolling(window=window).mean()
    df['Avg_Bounce_Height'] = df['Bounce_Height'].rolling(window=window).mean()
    
    # 形态成功率（回调后是否继续上涨，反弹后是否继续下跌）
    df['Pullback_Success'] = 0
    for i in range(5, len(df)-5):
        if df['Pullback_Recovery'].iloc[i] == 1:
            # 检查恢复后5天是否价格更高
            if df['Close'].iloc[min(len(df)-1, i+5)] > df['Close'].iloc[i]:
                df.loc[df.index[i], 'Pullback_Success'] = 1
    
    df['Bounce_Success'] = 0
    for i in range(5, len(df)-5):
        if df['Bounce_End'].iloc[i] == 1:
            # 检查结束后5天是否价格更低
            if df['Close'].iloc[min(len(df)-1, i+5)] < df['Close'].iloc[i]:
                df.loc[df.index[i], 'Bounce_Success'] = 1
    
    # 累计成功率
    df['Pullback_Success_Rate'] = df['Pullback_Success'].rolling(window=window).mean()
    df['Bounce_Success_Rate'] = df['Bounce_Success'].rolling(window=window).mean()
    
    return df


def identify_v_reversal(df, window=10, threshold=0.10):
    """
    识别V型反转（快速反转）
    
    参数:
    df: DataFrame with price data
    window: 检测窗口
    threshold: 价格变化阈值（默认10%）
    
    返回:
    DataFrame with V-reversal indicators
    """
    df['V_Reversal_Bullish'] = 0  # V型底部反转
    df['V_Reversal_Bearish'] = 0  # 倒V型顶部反转
    
    for i in range(window, len(df)-window):
        # V型底部：快速下跌后快速上涨
        before_drop = (df['Close'].iloc[i] - df['Close'].iloc[i-window]) / df['Close'].iloc[i-window]
        after_rise = (df['Close'].iloc[i+window] - df['Close'].iloc[i]) / df['Close'].iloc[i]
        
        if before_drop < -threshold and after_rise > threshold:
            df.loc[df.index[i], 'V_Reversal_Bullish'] = 1
        
        # 倒V型顶部：快速上涨后快速下跌
        before_rise = (df['Close'].iloc[i] - df['Close'].iloc[i-window]) / df['Close'].iloc[i-window]
        after_drop = (df['Close'].iloc[i+window] - df['Close'].iloc[i]) / df['Close'].iloc[i]
        
        if before_rise > threshold and after_drop < -threshold:
            df.loc[df.index[i], 'V_Reversal_Bearish'] = 1
    
    return df


def identify_double_top_bottom(df, window=20, tolerance=0.02):
    """
    识别双顶和双底形态
    
    参数:
    df: DataFrame with price data
    window: 检测窗口
    tolerance: 价格相似度容忍度（默认2%）
    
    返回:
    DataFrame with double pattern indicators
    """
    df['Double_Top'] = 0
    df['Double_Bottom'] = 0
    
    # 获取局部极值
    if 'Peak' not in df.columns:
        df = identify_reversal_points(df, order=5)
    
    peaks = df[df['Peak'] == 1].index
    troughs = df[df['Trough'] == 1].index
    
    # 识别双顶
    for i in range(len(peaks)-1):
        idx1 = df.index.get_loc(peaks[i])
        idx2 = df.index.get_loc(peaks[i+1])
        
        # 两个顶部之间的距离在合理范围内
        if window < (idx2 - idx1) < window * 3:
            price1 = df['Close'].iloc[idx1]
            price2 = df['Close'].iloc[idx2]
            
            # 两个顶部价格相近
            if abs(price2 - price1) / price1 < tolerance:
                # 中间有明显回调
                middle_low = df['Close'].iloc[idx1:idx2].min()
                if (price1 - middle_low) / price1 > 0.05:
                    df.loc[peaks[i+1], 'Double_Top'] = 1
    
    # 识别双底
    for i in range(len(troughs)-1):
        idx1 = df.index.get_loc(troughs[i])
        idx2 = df.index.get_loc(troughs[i+1])
        
        # 两个底部之间的距离在合理范围内
        if window < (idx2 - idx1) < window * 3:
            price1 = df['Close'].iloc[idx1]
            price2 = df['Close'].iloc[idx2]
            
            # 两个底部价格相近
            if abs(price2 - price1) / price1 < tolerance:
                # 中间有明显反弹
                middle_high = df['Close'].iloc[idx1:idx2].max()
                if (middle_high - price1) / price1 > 0.05:
                    df.loc[troughs[i+1], 'Double_Bottom'] = 1
    
    return df


def add_pattern_features(df):
    """
    添加所有形态识别特征
    
    参数:
    df: DataFrame with stock data (必须包含 Open, High, Low, Close, Volume)
    
    返回:
    DataFrame with all pattern features
    """
    print("  识别技术形态...")
    
    # 创建副本以避免警告
    df = df.copy()
    
    # 1. 识别反转点
    df = identify_reversal_points(df, order=5)
    print(f"    - 反转点识别完成")
    
    # 2. 识别回调
    df = identify_pullback(df, trend_window=20, pullback_threshold=0.05)
    print(f"    - 回调形态识别完成")
    
    # 3. 识别反弹
    df = identify_bounce(df, trend_window=20, bounce_threshold=0.05)
    print(f"    - 反弹形态识别完成")
    
    # 4. 计算形态统计特征
    df = calculate_pattern_statistics(df)
    print(f"    - 形态统计计算完成")
    
    # 5. 识别V型反转
    df = identify_v_reversal(df, window=10, threshold=0.10)
    print(f"    - V型反转识别完成")
    
    # 6. 识别双顶双底
    df = identify_double_top_bottom(df, window=20, tolerance=0.02)
    print(f"    - 双顶双底识别完成")
    
    # 填充NaN值
    pattern_columns = [
        'Peak', 'Trough', 'Bullish_Reversal', 'Bearish_Reversal', 'Days_Since_Reversal',
        'Trend', 'Recent_High', 'Drawdown_From_High', 'Is_Pullback', 'Pullback_Depth',
        'Pullback_Recovery', 'Pullback_Days', 'Recent_Low', 'Rally_From_Low',
        'Is_Bounce', 'Bounce_Height', 'Bounce_End', 'Bounce_Days',
        'Reversal_Strength', 'Pullback_Frequency', 'Bounce_Frequency',
        'Avg_Pullback_Depth', 'Avg_Bounce_Height', 'Pullback_Success', 'Bounce_Success',
        'Pullback_Success_Rate', 'Bounce_Success_Rate',
        'V_Reversal_Bullish', 'V_Reversal_Bearish', 'Double_Top', 'Double_Bottom'
    ]
    
    for col in pattern_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    print(f"  [OK] 形态特征提取完成，新增 {len(pattern_columns)} 个特征")
    
    return df


def summarize_patterns(df):
    """
    汇总形态统计信息
    
    参数:
    df: DataFrame with pattern features
    
    返回:
    统计摘要字典
    """
    summary = {
        '反转次数': {
            '牛市反转': df['Bullish_Reversal'].sum(),
            '熊市反转': df['Bearish_Reversal'].sum(),
            'V型底部': df['V_Reversal_Bullish'].sum(),
            '倒V顶部': df['V_Reversal_Bearish'].sum(),
        },
        '回调统计': {
            '回调次数': df['Is_Pullback'].sum(),
            '平均回调深度': f"{df[df['Pullback_Depth'] > 0]['Pullback_Depth'].mean():.2%}",
            '最大回调深度': f"{df['Pullback_Depth'].max():.2%}",
            '回调成功率': f"{df['Pullback_Success_Rate'].mean():.2%}",
        },
        '反弹统计': {
            '反弹次数': df['Is_Bounce'].sum(),
            '平均反弹高度': f"{df[df['Bounce_Height'] > 0]['Bounce_Height'].mean():.2%}",
            '最大反弹高度': f"{df['Bounce_Height'].max():.2%}",
            '反弹成功率': f"{df['Bounce_Success_Rate'].mean():.2%}",
        },
        '特殊形态': {
            '双顶形态': df['Double_Top'].sum(),
            '双底形态': df['Double_Bottom'].sum(),
        }
    }
    
    return summary
