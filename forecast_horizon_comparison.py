"""
预测时间窗口对比实验
测试不同预测天数（1天、3天、5天、10天）的准确率
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import warnings

from utils.pattern_recognition import add_pattern_features

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def prepare_data_with_horizon(all_data, lookback_days=60, forecast_horizon=5):
    """
    准备指定预测窗口的数据
    """
    print(f"\n准备{forecast_horizon}天预测数据...")
    print("-"*60)
    
    all_X = []
    all_y = []
    
    for stock_code, df in all_data.items():
        if 'Bullish_Reversal' not in df.columns:
            df = add_pattern_features(df)
        
        df['Any_Reversal'] = ((df['Bullish_Reversal'] == 1) | 
                              (df['Bearish_Reversal'] == 1)).astype(int)
        
        # 提取特征
        feature_columns = ['Close', 'Volume', 'RSI', 'MACD', 'ATR', 'MA20']
        available_features = [col for col in feature_columns if col in df.columns]
        
        for i in range(lookback_days, len(df) - forecast_horizon):
            # 预测未来forecast_horizon天的信号
            future_signal = df['Any_Reversal'].iloc[i + forecast_horizon]
            
            # 提取前lookback_days天的统计特征
            lookback_data = df.iloc[i-lookback_days:i]
            
            feature_dict = {}
            for col in available_features:
                if col in lookback_data.columns:
                    values = lookback_data[col].values
                    feature_dict[f'{col}_mean'] = np.mean(values)
                    feature_dict[f'{col}_std'] = np.std(values)
                    feature_dict[f'{col}_last'] = values[-1]
            
            all_X.append(feature_dict)
            all_y.append(int(future_signal))
    
    X = pd.DataFrame(all_X)
    y = pd.Series(all_y)
    
    print(f"  样本数: {len(X)}")
    print(f"  信号样本: {y.sum()} ({y.sum()/len(y):.2%})")
    print(f"  特征数: {X.shape[1]}")
    
    return X, y


def test_single_horizon(all_data, forecast_horizon, lookback_days=60):
    """
    测试单个预测窗口的性能
    """
    print(f"\n{'='*80}")
    print(f"测试预测窗口: {forecast_horizon}天后")
    print(f"{'='*80}")
    
    try:
        # 准备数据
        X, y = prepare_data_with_horizon(all_data, lookback_days, forecast_horizon)
        
        if len(X) < 100:
            print(f"⚠️ 样本数过少 ({len(X)})，跳过")
            return None
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 处理类别不平衡
        minority_ratio = y_train.value_counts().min() / y_train.value_counts().max()
        
        if minority_ratio < 0.3:
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"  使用SMOTE平衡数据")
            except:
                pass
        
        # 训练模型
        print(f"  训练模型...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n  结果:")
        print(f"    准确率: {accuracy:.4f} ({accuracy:.2%})")
        print(f"    精确率: {precision:.4f} ({precision:.2%})")
        print(f"    召回率: {recall:.4f} ({recall:.2%})")
        print(f"    F1分数: {f1:.4f}")
        
        return {
            'forecast_horizon': forecast_horizon,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'samples': len(X),
            'signal_ratio': y.sum() / len(y),
            'test_samples': len(X_test)
        }
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_comparison(results_df):
    """
    可视化对比结果
    """
    print(f"\n生成对比图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('不同预测窗口性能对比', fontsize=16, fontweight='bold')
    
    # 1. 准确率趋势
    ax1 = axes[0, 0]
    ax1.plot(results_df['forecast_horizon'], results_df['accuracy'], 
             marker='o', linewidth=2, markersize=10, color='#2E86DE', label='准确率')
    ax1.set_xlabel('预测天数', fontsize=12)
    ax1.set_ylabel('准确率', fontsize=12)
    ax1.set_title('准确率 vs 预测天数', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.0])
    
    # 添加数值标签
    for x, y in zip(results_df['forecast_horizon'], results_df['accuracy']):
        ax1.text(x, y + 0.02, f'{y:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. 精确率趋势
    ax2 = axes[0, 1]
    ax2.plot(results_df['forecast_horizon'], results_df['precision'], 
             marker='s', linewidth=2, markersize=10, color='#FF6348', label='精确率')
    ax2.set_xlabel('预测天数', fontsize=12)
    ax2.set_ylabel('精确率', fontsize=12)
    ax2.set_title('精确率 vs 预测天数', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.5, 1.0])
    
    for x, y in zip(results_df['forecast_horizon'], results_df['precision']):
        ax2.text(x, y + 0.02, f'{y:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. 召回率趋势
    ax3 = axes[1, 0]
    ax3.plot(results_df['forecast_horizon'], results_df['recall'], 
             marker='^', linewidth=2, markersize=10, color='#26C281', label='召回率')
    ax3.set_xlabel('预测天数', fontsize=12)
    ax3.set_ylabel('召回率', fontsize=12)
    ax3.set_title('召回率 vs 预测天数', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.5, 1.0])
    
    for x, y in zip(results_df['forecast_horizon'], results_df['recall']):
        ax3.text(x, y + 0.02, f'{y:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. 综合对比
    ax4 = axes[1, 1]
    x = np.arange(len(results_df))
    width = 0.2
    
    ax4.bar(x - width, results_df['accuracy'], width, label='准确率', alpha=0.8, color='#2E86DE')
    ax4.bar(x, results_df['precision'], width, label='精确率', alpha=0.8, color='#FF6348')
    ax4.bar(x + width, results_df['recall'], width, label='召回率', alpha=0.8, color='#26C281')
    
    ax4.set_xlabel('预测天数', fontsize=12)
    ax4.set_ylabel('指标值', fontsize=12)
    ax4.set_title('综合指标对比', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{h}天' for h in results_df['forecast_horizon']])
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/forecast_horizon_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ 对比图已保存: results/forecast_horizon_comparison.png")
    
    plt.show()


def main():
    """
    主函数：对比不同预测窗口
    """
    from datetime import datetime
    
    print("="*80)
    print("预测时间窗口对比实验")
    print("测试不同预测天数的准确率")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载数据
    print("\n步骤1: 加载股票数据")
    print("-"*60)
    
    data_path = 'data/all_stock_data.pkl'
    if not os.path.exists(data_path):
        print(f"错误: {data_path} 不存在")
        print("请先运行: python stock_data_downloader.py")
        return
    
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    
    print(f"✅ 成功加载 {len(all_data)} 只股票")
    
    # 2. 测试不同的预测窗口
    print("\n步骤2: 测试不同预测窗口")
    print("-"*60)
    
    horizons = [1, 3, 5, 10]  # 测试1天、3天、5天、10天
    results = []
    
    for horizon in horizons:
        result = test_single_horizon(all_data, horizon, lookback_days=60)
        if result:
            results.append(result)
    
    if not results:
        print("\n❌ 所有测试都失败了")
        return
    
    # 3. 汇总结果
    print("\n步骤3: 汇总对比结果")
    print("-"*60)
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("对比结果汇总")
    print("="*80)
    print(results_df[['forecast_horizon', 'accuracy', 'precision', 'recall', 'f1_score']].to_string(index=False))
    
    # 4. 可视化
    print("\n步骤4: 生成可视化图表")
    print("-"*60)
    
    visualize_comparison(results_df)
    
    # 5. 保存结果
    print("\n步骤5: 保存结果")
    print("-"*60)
    
    os.makedirs('results', exist_ok=True)
    results_path = 'results/forecast_horizon_comparison.csv'
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"✅ 结果已保存: {results_path}")
    
    # 6. 分析结论
    print("\n步骤6: 分析结论")
    print("-"*60)
    
    best_accuracy_idx = results_df['accuracy'].idxmax()
    best_precision_idx = results_df['precision'].idxmax()
    best_f1_idx = results_df['f1_score'].idxmax()
    
    print(f"\n📊 性能排名:")
    print(f"  最高准确率: {results_df.loc[best_accuracy_idx, 'forecast_horizon']}天后 "
          f"({results_df.loc[best_accuracy_idx, 'accuracy']:.2%})")
    print(f"  最高精确率: {results_df.loc[best_precision_idx, 'forecast_horizon']}天后 "
          f"({results_df.loc[best_precision_idx, 'precision']:.2%})")
    print(f"  最高F1分数: {results_df.loc[best_f1_idx, 'forecast_horizon']}天后 "
          f"({results_df.loc[best_f1_idx, 'f1_score']:.2%})")
    
    # 性能衰减分析
    if len(results_df) > 1:
        first_acc = results_df.iloc[0]['accuracy']
        last_acc = results_df.iloc[-1]['accuracy']
        acc_decay = (first_acc - last_acc) / first_acc
        
        print(f"\n📉 准确率衰减:")
        print(f"  {results_df.iloc[0]['forecast_horizon']}天: {first_acc:.2%}")
        print(f"  {results_df.iloc[-1]['forecast_horizon']}天: {last_acc:.2%}")
        print(f"  衰减幅度: {acc_decay:.2%}")
    
    # 推荐配置
    print(f"\n💡 推荐配置:")
    best_horizon = results_df.loc[best_accuracy_idx, 'forecast_horizon']
    
    if best_horizon <= 1:
        print(f"  ✅ {best_horizon}天预测准确率最高")
        print(f"  适合: 日内交易、短线操作")
        print(f"  特点: 准确率高但需频繁交易")
    elif best_horizon <= 3:
        print(f"  ✅ {best_horizon}天预测效果很好")
        print(f"  适合: 短线交易")
        print(f"  特点: 准确率高且交易频率适中")
    elif best_horizon <= 5:
        print(f"  ✅ {best_horizon}天预测平衡性好")
        print(f"  适合: 波段交易（推荐）")
        print(f"  特点: 过滤噪音，符合T+1规则")
    else:
        print(f"  ✅ {best_horizon}天预测稳定")
        print(f"  适合: 中线投资")
        print(f"  特点: 捕捉中期趋势")
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == '__main__':
    main()
