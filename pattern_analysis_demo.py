"""
形态识别分析演示脚本
展示如何提取和使用反转、回调、反弹等技术形态作为统计分析的自变量
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from utils.pattern_recognition import add_pattern_features, summarize_patterns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_analyze_patterns(stock_code='all'):
    """
    加载股票数据并分析技术形态
    
    参数:
    stock_code: 股票代码，'all'表示分析所有股票
    """
    print("="*80)
    print("股票技术形态识别与分析")
    print("="*80)
    
    # 1. 加载数据
    print("\n步骤1: 加载股票数据")
    print("-" * 50)
    
    try:
        with open('data/all_stock_data.pkl', 'rb') as f:
            all_data = pickle.load(f)
        print(f"✅ 成功加载 {len(all_data)} 只股票的数据")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("请先运行 python stock_data_downloader.py 下载数据")
        return None, None
    
    # 2. 选择要分析的股票
    if stock_code == 'all':
        stock_codes = list(all_data.keys())[:5]  # 演示用，只取前5只
        print(f"分析前 {len(stock_codes)} 只股票: {stock_codes}")
    else:
        stock_codes = [stock_code] if stock_code in all_data else []
        print(f"分析股票: {stock_codes}")
    
    if not stock_codes:
        print("❌ 未找到有效股票")
        return None, None
    
    # 3. 提取形态特征
    print("\n步骤2: 提取技术形态特征")
    print("-" * 50)
    
    pattern_data = {}
    pattern_summaries = {}
    
    for code in stock_codes:
        print(f"\n处理股票: {code}")
        df = all_data[code].copy()
        
        # 添加形态识别特征
        df_with_patterns = add_pattern_features(df)
        pattern_data[code] = df_with_patterns
        
        # 生成统计摘要
        summary = summarize_patterns(df_with_patterns)
        pattern_summaries[code] = summary
        
        # 打印摘要
        print(f"\n{code} 形态统计:")
        print(f"  反转: 牛市{summary['反转次数']['牛市反转']}次, "
              f"熊市{summary['反转次数']['熊市反转']}次")
        print(f"  回调: {summary['回调统计']['回调次数']}次, "
              f"平均深度{summary['回调统计']['平均回调深度']}")
        print(f"  反弹: {summary['反弹统计']['反弹次数']}次, "
              f"平均高度{summary['反弹统计']['平均反弹高度']}")
        print(f"  双顶: {summary['特殊形态']['双顶形态']}次, "
              f"双底: {summary['特殊形态']['双底形态']}次")
    
    return pattern_data, pattern_summaries


def extract_pattern_samples(pattern_data, pattern_type='all'):
    """
    提取特定形态的样本数据用于统计分析
    
    参数:
    pattern_data: 包含形态特征的数据字典
    pattern_type: 形态类型 ('reversal', 'pullback', 'bounce', 'all')
    
    返回:
    提取的形态样本DataFrame
    """
    print("\n步骤3: 提取形态样本作为自变量")
    print("-" * 50)
    
    all_samples = []
    
    for stock_code, df in pattern_data.items():
        # 提取反转样本
        if pattern_type in ['reversal', 'all']:
            # 牛市反转样本
            bullish_reversals = df[df['Bullish_Reversal'] == 1].copy()
            bullish_reversals['Pattern_Type'] = 'Bullish_Reversal'
            bullish_reversals['Stock_Code'] = stock_code
            all_samples.append(bullish_reversals)
            
            # 熊市反转样本
            bearish_reversals = df[df['Bearish_Reversal'] == 1].copy()
            bearish_reversals['Pattern_Type'] = 'Bearish_Reversal'
            bearish_reversals['Stock_Code'] = stock_code
            all_samples.append(bearish_reversals)
        
        # 提取回调样本
        if pattern_type in ['pullback', 'all']:
            pullbacks = df[df['Is_Pullback'] == 1].copy()
            pullbacks['Pattern_Type'] = 'Pullback'
            pullbacks['Stock_Code'] = stock_code
            all_samples.append(pullbacks)
        
        # 提取反弹样本
        if pattern_type in ['bounce', 'all']:
            bounces = df[df['Is_Bounce'] == 1].copy()
            bounces['Pattern_Type'] = 'Bounce'
            bounces['Stock_Code'] = stock_code
            all_samples.append(bounces)
    
    if not all_samples:
        print("⚠ 未找到任何形态样本")
        return pd.DataFrame()
    
    # 合并所有样本
    pattern_samples = pd.concat(all_samples, ignore_index=True)
    
    print(f"✅ 提取完成:")
    print(f"  总样本数: {len(pattern_samples)}")
    print(f"  形态分布:")
    for pattern, count in pattern_samples['Pattern_Type'].value_counts().items():
        print(f"    - {pattern}: {count} 个样本")
    
    return pattern_samples


def build_pattern_features_for_analysis(pattern_samples):
    """
    构建用于统计分析的特征矩阵
    
    参数:
    pattern_samples: 形态样本DataFrame
    
    返回:
    特征矩阵和目标变量
    """
    print("\n步骤4: 构建统计分析特征矩阵")
    print("-" * 50)
    
    # 选择关键特征作为自变量
    feature_columns = [
        # 基础价格特征
        'Close', 'Volume', 'TurnoverRate',
        
        # 趋势特征
        'Trend', 'Days_Since_Reversal',
        
        # 回调/反弹特征
        'Pullback_Depth', 'Pullback_Days', 'Pullback_Frequency',
        'Bounce_Height', 'Bounce_Days', 'Bounce_Frequency',
        
        # 反转强度
        'Reversal_Strength',
        
        # 成功率特征
        'Pullback_Success_Rate', 'Bounce_Success_Rate',
        
        # 技术指标
        'RSI', 'MACD', 'ATR', 'ADX',
        
        # 均线特征
        'MA5', 'MA10', 'MA20', 'MA50',
        
        # 波动率特征
        'Volatility', 'HV_20',
    ]
    
    # 提取存在的特征
    available_features = [col for col in feature_columns if col in pattern_samples.columns]
    
    print(f"可用特征数: {len(available_features)}")
    print(f"特征列表: {available_features[:10]}...")  # 只显示前10个
    
    # 创建特征矩阵
    X = pattern_samples[available_features].copy()
    
    # 创建目标变量（示例：未来5天收益率）
    if 'Close' in pattern_samples.columns:
        # 计算未来收益（如果数据允许）
        pattern_samples['Future_Return_5D'] = 0.0  # 需要根据实际数据计算
    
    print(f"✅ 特征矩阵构建完成: {X.shape}")
    
    return X, pattern_samples


def visualize_patterns(pattern_data, stock_code=None):
    """
    可视化技术形态
    
    参数:
    pattern_data: 包含形态特征的数据字典
    stock_code: 要可视化的股票代码
    """
    print("\n步骤5: 可视化技术形态")
    print("-" * 50)
    
    if stock_code is None:
        stock_code = list(pattern_data.keys())[0]
    
    df = pattern_data[stock_code]
    
    # 创建图表
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(f'股票 {stock_code} 技术形态分析', fontsize=16, fontweight='bold')
    
    # 1. 价格图 + 反转点
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='收盘价', color='black', linewidth=1)
    
    # 标记牛市反转点
    bullish_idx = df[df['Bullish_Reversal'] == 1].index
    if len(bullish_idx) > 0:
        ax1.scatter(bullish_idx, df.loc[bullish_idx, 'Close'], 
                   color='green', marker='^', s=100, label='牛市反转', zorder=5)
    
    # 标记熊市反转点
    bearish_idx = df[df['Bearish_Reversal'] == 1].index
    if len(bearish_idx) > 0:
        ax1.scatter(bearish_idx, df.loc[bearish_idx, 'Close'], 
                   color='red', marker='v', s=100, label='熊市反转', zorder=5)
    
    ax1.set_title('价格走势与反转点')
    ax1.set_ylabel('价格')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. 回调区域
    ax2 = axes[1]
    ax2.plot(df.index, df['Close'], label='收盘价', color='black', linewidth=1)
    ax2.fill_between(df.index, df['Close'].min(), df['Close'].max(), 
                     where=df['Is_Pullback']==1, alpha=0.3, color='orange', label='回调区域')
    ax2.set_title('回调形态识别')
    ax2.set_ylabel('价格')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. 反弹区域
    ax3 = axes[2]
    ax3.plot(df.index, df['Close'], label='收盘价', color='black', linewidth=1)
    ax3.fill_between(df.index, df['Close'].min(), df['Close'].max(), 
                     where=df['Is_Bounce']==1, alpha=0.3, color='cyan', label='反弹区域')
    ax3.set_title('反弹形态识别')
    ax3.set_ylabel('价格')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 4. 形态频率统计
    ax4 = axes[3]
    if 'Pullback_Frequency' in df.columns and 'Bounce_Frequency' in df.columns:
        ax4.plot(df.index, df['Pullback_Frequency'], label='回调频率', color='orange')
        ax4.plot(df.index, df['Bounce_Frequency'], label='反弹频率', color='cyan')
        ax4.set_title('形态频率变化（60日窗口）')
        ax4.set_ylabel('频率')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/pattern_analysis_{stock_code}.png', dpi=300, bbox_inches='tight')
    print(f"✅ 可视化图表已保存: results/pattern_analysis_{stock_code}.png")
    plt.close()


def analyze_pattern_effectiveness(pattern_samples):
    """
    分析各种形态的有效性
    
    参数:
    pattern_samples: 形态样本DataFrame
    """
    print("\n步骤6: 分析形态有效性")
    print("-" * 50)
    
    if 'Pattern_Type' not in pattern_samples.columns:
        print("⚠ 数据中缺少形态类型信息")
        return
    
    # 按形态类型分组统计
    pattern_stats = []
    
    for pattern_type in pattern_samples['Pattern_Type'].unique():
        subset = pattern_samples[pattern_samples['Pattern_Type'] == pattern_type]
        
        stats = {
            '形态类型': pattern_type,
            '样本数量': len(subset),
            '平均RSI': subset['RSI'].mean() if 'RSI' in subset.columns else np.nan,
            '平均成交量': subset['Volume'].mean() if 'Volume' in subset.columns else np.nan,
            '平均波动率': subset['Volatility'].mean() if 'Volatility' in subset.columns else np.nan,
        }
        
        # 特定形态的特定指标
        if pattern_type == 'Pullback':
            stats['平均回调深度'] = subset['Pullback_Depth'].mean()
            stats['平均持续天数'] = subset['Pullback_Days'].mean()
        elif pattern_type == 'Bounce':
            stats['平均反弹高度'] = subset['Bounce_Height'].mean()
            stats['平均持续天数'] = subset['Bounce_Days'].mean()
        elif 'Reversal' in pattern_type:
            stats['平均反转强度'] = subset['Reversal_Strength'].mean() if 'Reversal_Strength' in subset.columns else np.nan
        
        pattern_stats.append(stats)
    
    # 创建统计表
    stats_df = pd.DataFrame(pattern_stats)
    
    print("\n形态有效性统计:")
    print(stats_df.to_string(index=False))
    
    # 保存统计结果
    stats_df.to_csv('results/pattern_effectiveness.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 统计结果已保存: results/pattern_effectiveness.csv")
    
    return stats_df


def main():
    """
    主函数：完整的形态识别与分析流程
    """
    import os
    
    # 创建结果目录
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 1. 加载数据并提取形态
    pattern_data, pattern_summaries = load_and_analyze_patterns(stock_code='all')
    
    if pattern_data is None:
        return
    
    # 2. 提取形态样本
    pattern_samples = extract_pattern_samples(pattern_data, pattern_type='all')
    
    if pattern_samples.empty:
        print("⚠ 未提取到形态样本，程序结束")
        return
    
    # 3. 构建特征矩阵
    X, samples_with_target = build_pattern_features_for_analysis(pattern_samples)
    
    # 4. 可视化（选第一只股票）
    if pattern_data:
        first_stock = list(pattern_data.keys())[0]
        visualize_patterns(pattern_data, stock_code=first_stock)
    
    # 5. 分析形态有效性
    stats_df = analyze_pattern_effectiveness(pattern_samples)
    
    # 6. 保存特征矩阵用于后续分析
    print("\n步骤7: 保存形态特征数据")
    print("-" * 50)
    X.to_csv('results/pattern_features.csv', index=False)
    pattern_samples.to_csv('results/pattern_samples_full.csv', index=False)
    print(f"✅ 特征矩阵已保存: results/pattern_features.csv")
    print(f"✅ 完整样本已保存: results/pattern_samples_full.csv")
    
    print("\n" + "="*80)
    print("形态识别与分析完成！")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  1. results/pattern_features.csv - 用于统计分析的特征矩阵")
    print(f"  2. results/pattern_samples_full.csv - 完整的形态样本数据")
    print(f"  3. results/pattern_effectiveness.csv - 形态有效性统计")
    print(f"  4. results/pattern_analysis_*.png - 可视化图表")
    print(f"\n这些数据可以直接用于:")
    print(f"  - 机器学习模型训练（以形态为特征）")
    print(f"  - 统计分析（相关性、回归分析等）")
    print(f"  - 交易策略回测")


if __name__ == '__main__':
    main()
