"""
形态识别可视化演示
展示各种技术形态的实际识别效果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from utils.pattern_recognition import (
    add_pattern_features, 
    summarize_patterns
)

# 配置中文显示
try:
    from utils.matplotlib_config import configure_chinese_font
    configure_chinese_font()
except:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False


def visualize_pattern_on_chart(df, stock_code):
    """
    在价格图表上标注识别出的形态
    """
    print(f"\n正在生成 {stock_code} 的形态识别图表...")
    
    # 只显示最近120天
    df_recent = df.tail(120).copy()
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f'技术形态识别 - {stock_code}', fontsize=16, fontweight='bold')
    
    # ==================== 子图1: 价格走势 + 反转点 ====================
    ax1 = axes[0]
    
    # 绘制价格
    ax1.plot(df_recent.index, df_recent['Close'], 
            label='收盘价', linewidth=2, color='blue', alpha=0.7)
    
    # 标注顶部 (Peak)
    peaks = df_recent[df_recent['Peak'] == 1]
    ax1.scatter(peaks.index, peaks['Close'], 
               color='red', s=100, marker='v', label='顶部', zorder=5)
    
    # 标注底部 (Trough)
    troughs = df_recent[df_recent['Trough'] == 1]
    ax1.scatter(troughs.index, troughs['Close'], 
               color='green', s=100, marker='^', label='底部', zorder=5)
    
    # 标注牛市反转 (Bullish Reversal)
    bull_rev = df_recent[df_recent['Bullish_Reversal'] == 1]
    ax1.scatter(bull_rev.index, bull_rev['Close'], 
               color='darkgreen', s=200, marker='*', 
               label='牛市反转', zorder=6, edgecolors='yellow', linewidths=2)
    
    # 标注熊市反转 (Bearish Reversal)
    bear_rev = df_recent[df_recent['Bearish_Reversal'] == 1]
    ax1.scatter(bear_rev.index, bear_rev['Close'], 
               color='darkred', s=200, marker='*', 
               label='熊市反转', zorder=6, edgecolors='black', linewidths=2)
    
    # 标注V型反转
    v_bull = df_recent[df_recent['V_Reversal_Bullish'] == 1]
    for idx in v_bull.index:
        ax1.annotate('V底', xy=(idx, v_bull.loc[idx, 'Close']),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='green'))
    
    v_bear = df_recent[df_recent['V_Reversal_Bearish'] == 1]
    for idx in v_bear.index:
        ax1.annotate('倒V顶', xy=(idx, v_bear.loc[idx, 'Close']),
                    xytext=(10, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    ax1.set_title('价格走势 + 反转点识别', fontsize=13, fontweight='bold')
    ax1.set_ylabel('价格', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ==================== 子图2: 回调和反弹 ====================
    ax2 = axes[1]
    
    # 绘制价格
    ax2.plot(df_recent.index, df_recent['Close'], 
            label='收盘价', linewidth=2, color='blue', alpha=0.5)
    
    # 绘制趋势线
    trend_colors = {1: 'green', -1: 'red', 0: 'gray'}
    for trend_val, color in trend_colors.items():
        trend_data = df_recent[df_recent['Trend'] == trend_val]
        if not trend_data.empty:
            ax2.scatter(trend_data.index, trend_data['Close'], 
                       c=color, s=20, alpha=0.5)
    
    # 标注回调期间
    pullback_data = df_recent[df_recent['Is_Pullback'] == 1]
    if not pullback_data.empty:
        ax2.scatter(pullback_data.index, pullback_data['Close'],
                   color='orange', s=80, marker='o', 
                   label=f'回调 ({len(pullback_data)}次)', alpha=0.7)
    
    # 标注回调恢复
    recovery_data = df_recent[df_recent['Pullback_Recovery'] == 1]
    if not recovery_data.empty:
        ax2.scatter(recovery_data.index, recovery_data['Close'],
                   color='green', s=150, marker='D', 
                   label=f'回调恢复 ({len(recovery_data)}次)', 
                   edgecolors='darkgreen', linewidths=2)
    
    # 标注反弹期间
    bounce_data = df_recent[df_recent['Is_Bounce'] == 1]
    if not bounce_data.empty:
        ax2.scatter(bounce_data.index, bounce_data['Close'],
                   color='purple', s=80, marker='s', 
                   label=f'反弹 ({len(bounce_data)}次)', alpha=0.7)
    
    ax2.set_title('回调和反弹识别', fontsize=13, fontweight='bold')
    ax2.set_ylabel('价格', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ==================== 子图3: 双顶双底 ====================
    ax3 = axes[2]
    
    # 绘制价格
    ax3.plot(df_recent.index, df_recent['Close'], 
            label='收盘价', linewidth=2, color='blue', alpha=0.7)
    
    # 标注双顶
    double_top = df_recent[df_recent['Double_Top'] == 1]
    if not double_top.empty:
        for idx in double_top.index:
            ax3.annotate('双顶(M)', xy=(idx, double_top.loc[idx, 'Close']),
                        xytext=(0, 20), textcoords='offset points',
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                        fontsize=10, color='white', fontweight='bold')
    
    # 标注双底
    double_bottom = df_recent[df_recent['Double_Bottom'] == 1]
    if not double_bottom.empty:
        for idx in double_bottom.index:
            ax3.annotate('双底(W)', xy=(idx, double_bottom.loc[idx, 'Close']),
                        xytext=(0, -30), textcoords='offset points',
                        bbox=dict(boxstyle='round', facecolor='green', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                        fontsize=10, color='white', fontweight='bold')
    
    ax3.set_title('双顶双底识别', fontsize=13, fontweight='bold')
    ax3.set_ylabel('价格', fontsize=11)
    ax3.set_xlabel('日期', fontsize=11)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    os.makedirs('results', exist_ok=True)
    save_path = f'results/pattern_visualization_{stock_code}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 图表已保存: {save_path}")
    
    plt.show()


def print_pattern_summary(df, stock_code):
    """打印形态统计摘要"""
    
    print(f"\n{'='*80}")
    print(f"形态统计摘要 - {stock_code}")
    print(f"{'='*80}")
    
    summary = summarize_patterns(df)
    
    print(f"\n【反转形态统计】")
    for key, value in summary['反转次数'].items():
        print(f"  {key}: {value} 次")
    
    print(f"\n【回调形态统计】")
    for key, value in summary['回调统计'].items():
        print(f"  {key}: {value}")
    
    print(f"\n【反弹形态统计】")
    for key, value in summary['反弹统计'].items():
        print(f"  {key}: {value}")
    
    print(f"\n【特殊形态统计】")
    for key, value in summary['特殊形态'].items():
        print(f"  {key}: {value} 次")
    
    # 最近的交易信号
    print(f"\n{'='*80}")
    print(f"最近10天的交易信号")
    print(f"{'='*80}")
    
    recent = df.tail(10)
    
    for idx, row in recent.iterrows():
        date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
        signals = []
        
        if row.get('Bullish_Reversal', 0) == 1:
            signals.append('[OK] 牛市反转 (强烈买入)')
        if row.get('Bearish_Reversal', 0) == 1:
            signals.append('[!] 熊市反转 (强烈卖出)')
        if row.get('Pullback_Recovery', 0) == 1:
            signals.append('[+] 回调恢复 (买入)')
        if row.get('V_Reversal_Bullish', 0) == 1:
            signals.append('[++] V型底部 (强烈买入)')
        if row.get('V_Reversal_Bearish', 0) == 1:
            signals.append('[!!] 倒V顶部 (强烈卖出)')
        if row.get('Double_Bottom', 0) == 1:
            signals.append('[++] 双底形态 (买入)')
        if row.get('Double_Top', 0) == 1:
            signals.append('[--] 双顶形态 (卖出)')
        
        if signals:
            print(f"\n{date_str}:")
            for signal in signals:
                print(f"  {signal}")


def main():
    """主函数"""
    
    print("="*80)
    print("形态识别可视化演示")
    print("="*80)
    
    # 加载股票数据
    print("\n加载股票数据...")
    
    try:
        with open('data/all_stock_data.pkl', 'rb') as f:
            all_data = pickle.load(f)
        print(f"[OK] 成功加载 {len(all_data)} 只股票")
    except FileNotFoundError:
        print("[X] 数据文件不存在")
        print("请先运行: python stock_data_downloader.py")
        return
    
    # 选择要分析的股票
    stock_codes = list(all_data.keys())[:3]  # 前3只股票
    
    print(f"\n将演示以下股票的形态识别:")
    for i, code in enumerate(stock_codes, 1):
        print(f"  {i}. {code}")
    
    # 对每只股票进行形态识别和可视化
    for stock_code in stock_codes:
        print(f"\n{'='*80}")
        print(f"分析股票: {stock_code}")
        print(f"{'='*80}")
        
        df = all_data[stock_code].copy()
        
        # 检查是否已有形态特征
        if 'Bullish_Reversal' not in df.columns:
            print("添加形态识别特征...")
            df = add_pattern_features(df)
            # 更新数据
            all_data[stock_code] = df
        else:
            print("已有形态特征，直接使用")
        
        # 打印统计摘要
        print_pattern_summary(df, stock_code)
        
        # 生成可视化图表
        visualize_pattern_on_chart(df, stock_code)
        
        print(f"\n{'='*80}\n")
    
    print("="*80)
    print("演示完成！")
    print("="*80)
    print(f"\n生成的图表保存在: results/pattern_visualization_*.png")
    print(f"\n详细说明文档: PATTERN_RECOGNITION_EXPLAINED.md")


if __name__ == '__main__':
    main()
