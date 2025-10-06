"""
股票形态信号预测模块
目标：预测反转信号的发生
使用反转信号前60天的数据作为自变量
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
import warnings

from utils.pattern_recognition import add_pattern_features

warnings.filterwarnings('ignore')


def extract_pre_signal_features(df, signal_column, lookback_days=60, target_type='reversal'):
    """
    提取信号发生前N天的特征数据
    
    参数:
    df: 包含形态特征的DataFrame
    signal_column: 信号列名（如'Bullish_Reversal', 'Bearish_Reversal'）
    lookback_days: 回看天数（信号前多少天）
    target_type: 目标类型 ('reversal', 'pullback', 'bounce')
    
    返回:
    X: 特征矩阵（信号前60天的数据）
    y: 目标变量（1=信号发生, 0=信号未发生）
    sample_info: 样本信息（用于分析）
    """
    print(f"提取 {signal_column} 信号前 {lookback_days} 天的特征...")
    
    X_samples = []
    y_samples = []
    sample_info = []
    
    # 选择要使用的特征列（技术指标 + 形态特征）
    feature_columns = [
        # 基础价格特征
        'Close', 'Volume', 'TurnoverRate', 'PriceChangeRate',
        
        # 均线特征
        'MA5', 'MA10', 'MA20', 'MA50',
        
        # 技术指标
        'RSI', 'MACD', 'Signal', 'ATR', 'ADX',
        'K', 'D', 'J',
        'DI_plus', 'DI_minus',
        'Williams_R', 'CCI',
        
        # 成交量指标
        'OBV', 'MFI', 'VWAP',
        
        # 波动率指标
        'Volatility', 'HV_20', 'BB_Width',
        
        # 动量指标
        'Momentum', 'ROC', 'TSI', 'CMO',
        
        # 形态特征
        'Trend', 'Drawdown_From_High', 'Rally_From_Low',
        'Days_Since_Reversal',
        'Pullback_Frequency', 'Bounce_Frequency',
        'Avg_Pullback_Depth', 'Avg_Bounce_Height',
        'Pullback_Success_Rate', 'Bounce_Success_Rate',
    ]
    
    # 筛选存在的特征
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"  可用特征数: {len(available_features)}")
    
    # 遍历每一天，检查是否为信号点
    for i in range(lookback_days, len(df)):
        # 检查当前时点是否有信号
        is_signal = df[signal_column].iloc[i]
        
        # 提取前lookback_days天的数据
        lookback_data = df.iloc[i-lookback_days:i]
        
        # 计算这lookback_days天的统计特征
        feature_dict = {}
        
        for col in available_features:
            if col in lookback_data.columns:
                values = lookback_data[col].values
                
                # 多种统计特征
                feature_dict[f'{col}_mean'] = np.mean(values)
                feature_dict[f'{col}_std'] = np.std(values)
                feature_dict[f'{col}_min'] = np.min(values)
                feature_dict[f'{col}_max'] = np.max(values)
                feature_dict[f'{col}_last'] = values[-1]  # 最后一天的值
                
                # 趋势特征
                if len(values) > 1:
                    feature_dict[f'{col}_trend'] = (values[-1] - values[0]) / (values[0] + 1e-8)
                    feature_dict[f'{col}_momentum'] = values[-1] - values[-5] if len(values) >= 5 else 0
                else:
                    feature_dict[f'{col}_trend'] = 0
                    feature_dict[f'{col}_momentum'] = 0
        
        X_samples.append(feature_dict)
        y_samples.append(int(is_signal))
        
        # 记录样本信息
        sample_info.append({
            'date': df.index[i],
            'signal': is_signal,
            'close_price': df['Close'].iloc[i] if 'Close' in df.columns else np.nan
        })
    
    # 转换为DataFrame
    X = pd.DataFrame(X_samples)
    y = pd.Series(y_samples)
    sample_info_df = pd.DataFrame(sample_info)
    
    print(f"  提取完成: {len(X)} 个样本")
    print(f"  信号样本数: {y.sum()} ({y.sum()/len(y):.2%})")
    print(f"  非信号样本数: {len(y)-y.sum()} ({(len(y)-y.sum())/len(y):.2%})")
    print(f"  特征维度: {X.shape[1]}")
    
    return X, y, sample_info_df


def prepare_reversal_prediction_data(all_data, lookback_days=60):
    """
    准备反转信号预测数据
    
    参数:
    all_data: 股票数据字典
    lookback_days: 回看天数
    
    返回:
    X: 特征矩阵
    y: 目标变量（1=反转信号, 0=非反转信号）
    sample_info: 样本信息
    """
    print("\n" + "="*80)
    print("准备反转信号预测数据")
    print("="*80)
    print(f"回看天数: {lookback_days}")
    print(f"目标: 预测反转信号（牛市反转 + 熊市反转）")
    
    all_X = []
    all_y = []
    all_info = []
    
    for stock_code, df in all_data.items():
        print(f"\n处理股票: {stock_code}")
        
        # 添加形态特征（如果还没有）
        if 'Bullish_Reversal' not in df.columns:
            print(f"  添加形态特征...")
            df = add_pattern_features(df)
        
        # 创建综合反转信号（牛市反转 OR 熊市反转）
        df['Any_Reversal'] = ((df['Bullish_Reversal'] == 1) | 
                              (df['Bearish_Reversal'] == 1)).astype(int)
        
        print(f"  反转信号数: {df['Any_Reversal'].sum()}")
        
        # 提取特征
        X_stock, y_stock, info_stock = extract_pre_signal_features(
            df, 
            signal_column='Any_Reversal',
            lookback_days=lookback_days
        )
        
        # 添加股票代码信息
        info_stock['stock_code'] = stock_code
        
        all_X.append(X_stock)
        all_y.append(y_stock)
        all_info.append(info_stock)
    
    # 合并所有股票的数据
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)
    info_combined = pd.concat(all_info, ignore_index=True)
    
    print("\n" + "="*80)
    print("数据准备完成")
    print("="*80)
    print(f"总样本数: {len(X_combined)}")
    print(f"反转信号样本: {y_combined.sum()} ({y_combined.sum()/len(y_combined):.2%})")
    print(f"非反转信号样本: {len(y_combined)-y_combined.sum()} ({(len(y_combined)-y_combined.sum())/len(y_combined):.2%})")
    print(f"特征数: {X_combined.shape[1]}")
    print(f"股票数: {len(all_data)}")
    
    return X_combined, y_combined, info_combined


def prepare_pullback_prediction_data(all_data, lookback_days=60):
    """
    准备回调信号预测数据
    """
    print("\n" + "="*80)
    print("准备回调信号预测数据")
    print("="*80)
    
    all_X = []
    all_y = []
    all_info = []
    
    for stock_code, df in all_data.items():
        print(f"\n处理股票: {stock_code}")
        
        if 'Is_Pullback' not in df.columns:
            df = add_pattern_features(df)
        
        print(f"  回调信号数: {df['Is_Pullback'].sum()}")
        
        X_stock, y_stock, info_stock = extract_pre_signal_features(
            df, 
            signal_column='Is_Pullback',
            lookback_days=lookback_days
        )
        
        info_stock['stock_code'] = stock_code
        
        all_X.append(X_stock)
        all_y.append(y_stock)
        all_info.append(info_stock)
    
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)
    info_combined = pd.concat(all_info, ignore_index=True)
    
    print("\n" + "="*80)
    print(f"总样本数: {len(X_combined)}")
    print(f"回调信号样本: {y_combined.sum()} ({y_combined.sum()/len(y_combined):.2%})")
    print(f"特征数: {X_combined.shape[1]}")
    
    return X_combined, y_combined, info_combined


def prepare_bounce_prediction_data(all_data, lookback_days=60):
    """
    准备反弹信号预测数据
    """
    print("\n" + "="*80)
    print("准备反弹信号预测数据")
    print("="*80)
    
    all_X = []
    all_y = []
    all_info = []
    
    for stock_code, df in all_data.items():
        print(f"\n处理股票: {stock_code}")
        
        if 'Is_Bounce' not in df.columns:
            df = add_pattern_features(df)
        
        print(f"  反弹信号数: {df['Is_Bounce'].sum()}")
        
        X_stock, y_stock, info_stock = extract_pre_signal_features(
            df, 
            signal_column='Is_Bounce',
            lookback_days=lookback_days
        )
        
        info_stock['stock_code'] = stock_code
        
        all_X.append(X_stock)
        all_y.append(y_stock)
        all_info.append(info_stock)
    
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)
    info_combined = pd.concat(all_info, ignore_index=True)
    
    print("\n" + "="*80)
    print(f"总样本数: {len(X_combined)}")
    print(f"反弹信号样本: {y_combined.sum()} ({y_combined.sum()/len(y_combined):.2%})")
    print(f"特征数: {X_combined.shape[1]}")
    
    return X_combined, y_combined, info_combined


def train_signal_prediction_model(X_train, X_test, y_train, y_test, signal_type='reversal'):
    """
    训练信号预测模型
    
    参数:
    X_train, X_test: 训练和测试特征
    y_train, y_test: 训练和测试标签
    signal_type: 信号类型
    
    返回:
    model: 训练好的模型
    y_pred: 预测结果
    metrics: 性能指标
    """
    print(f"\n训练{signal_type}信号预测模型...")
    print("="*80)
    
    # 打印类别分布
    print("\n训练集类别分布:")
    train_counts = pd.Series(y_train).value_counts().sort_index()
    print(f"  无信号: {train_counts.get(0, 0)} ({train_counts.get(0, 0)/len(y_train):.2%})")
    print(f"  有信号: {train_counts.get(1, 0)} ({train_counts.get(1, 0)/len(y_train):.2%})")
    
    print("\n测试集类别分布:")
    test_counts = pd.Series(y_test).value_counts().sort_index()
    print(f"  无信号: {test_counts.get(0, 0)} ({test_counts.get(0, 0)/len(y_test):.2%})")
    print(f"  有信号: {test_counts.get(1, 0)} ({test_counts.get(1, 0)/len(y_test):.2%})")
    
    # 处理类别不平衡
    minority_ratio = train_counts.min() / train_counts.max()
    print(f"\n类别不平衡比率: {minority_ratio:.2%}")
    
    # 使用SMOTE处理不平衡（如果比例过于悬殊）
    if minority_ratio < 0.3:
        try:
            from imblearn.over_sampling import SMOTE
            print("\n使用SMOTE平衡数据集...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            balanced_counts = pd.Series(y_train_balanced).value_counts().sort_index()
            print("平衡后类别分布:")
            print(f"  无信号: {balanced_counts.get(0, 0)} ({balanced_counts.get(0, 0)/len(y_train_balanced):.2%})")
            print(f"  有信号: {balanced_counts.get(1, 0)} ({balanced_counts.get(1, 0)/len(y_train_balanced):.2%})")
        except ImportError:
            print("⚠️ imbalanced-learn未安装，使用类别权重")
            X_train_balanced = X_train
            y_train_balanced = y_train
    else:
        X_train_balanced = X_train
        y_train_balanced = y_train
    
    # 训练随机森林模型
    print("\n训练Random Forest模型...")
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',  # 自动平衡类别权重
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_balanced, y_train_balanced)
    print("✅ 模型训练完成")
    
    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算性能指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    print("\n" + "="*80)
    print("模型性能评估")
    print("="*80)
    print(f"准确率: {accuracy:.4f} ({accuracy:.2%})")
    print(f"精确率: {precision:.4f} ({precision:.2%})")
    print(f"召回率: {recall:.4f} ({recall:.2%})")
    
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, 
                              target_names=['无信号', f'{signal_type}信号'],
                              zero_division=0))
    
    # 特征重要性分析
    print("\n特征重要性排名（Top 20）:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(20).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'feature_importance': feature_importance
    }
    
    return model, y_pred, y_proba, metrics


def save_prediction_results(X, y, y_pred, y_proba, sample_info, signal_type, output_dir='results'):
    """
    保存预测结果
    """
    print(f"\n保存{signal_type}预测结果...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 组合结果
    results_df = sample_info.copy()
    results_df['true_signal'] = y.values
    results_df['pred_signal'] = y_pred
    results_df['signal_probability'] = y_proba
    results_df['prediction_correct'] = (y.values == y_pred).astype(int)
    
    # 保存详细结果
    result_path = f'{output_dir}/{signal_type}_prediction_results.csv'
    results_df.to_csv(result_path, index=False, encoding='utf-8-sig')
    print(f"✅ 详细结果已保存: {result_path}")
    
    # 保存特征矩阵
    feature_path = f'{output_dir}/{signal_type}_features.csv'
    X.to_csv(feature_path, index=False)
    print(f"✅ 特征矩阵已保存: {feature_path}")
    
    return results_df


def main():
    """
    主函数：反转信号预测完整流程
    """
    print("="*80)
    print("股票形态信号预测系统")
    print("使用信号前60天数据预测反转/回调/反弹信号")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 加载股票数据
        print("\n步骤1: 加载股票数据")
        print("-"*50)
        
        data_path = 'data/all_stock_data.pkl'
        if not os.path.exists(data_path):
            print(f"错误: 数据文件不存在 - {data_path}")
            print("请先运行: python stock_data_downloader.py")
            return
        
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        
        print(f"✅ 成功加载 {len(all_data)} 只股票的数据")
        
        # 2. 选择要预测的信号类型
        signal_type = 'reversal'  # 可选: 'reversal', 'pullback', 'bounce'
        lookback_days = 60
        
        print(f"\n步骤2: 准备{signal_type}信号预测数据")
        print("-"*50)
        
        if signal_type == 'reversal':
            X, y, sample_info = prepare_reversal_prediction_data(all_data, lookback_days)
        elif signal_type == 'pullback':
            X, y, sample_info = prepare_pullback_prediction_data(all_data, lookback_days)
        elif signal_type == 'bounce':
            X, y, sample_info = prepare_bounce_prediction_data(all_data, lookback_days)
        else:
            print(f"错误: 不支持的信号类型 {signal_type}")
            return
        
        # 3. 数据分割
        print(f"\n步骤3: 数据分割")
        print("-"*50)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y
        )
        
        print(f"训练集: {len(X_train)} 样本")
        print(f"测试集: {len(X_test)} 样本")
        
        # 4. 训练模型
        print(f"\n步骤4: 训练预测模型")
        print("-"*50)
        
        model, y_pred, y_proba, metrics = train_signal_prediction_model(
            X_train, X_test, y_train, y_test, signal_type
        )
        
        # 5. 保存结果
        print(f"\n步骤5: 保存预测结果")
        print("-"*50)
        
        # 准备测试集的sample_info
        test_indices = X_test.index
        sample_info_test = sample_info.iloc[test_indices].reset_index(drop=True)
        
        results_df = save_prediction_results(
            X_test, y_test, y_pred, y_proba, 
            sample_info_test, signal_type
        )
        
        # 6. 保存模型
        print(f"\n步骤6: 保存训练模型")
        print("-"*50)
        
        os.makedirs('models', exist_ok=True)
        
        model_path = f'models/{signal_type}_prediction_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_columns': X.columns.tolist(),
                'metrics': metrics,
                'lookback_days': lookback_days,
                'signal_type': signal_type,
                'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f)
        
        print(f"✅ 模型已保存: {model_path}")
        
        # 7. 总结
        print("\n" + "="*80)
        print("预测分析完成！")
        print("="*80)
        print(f"\n信号类型: {signal_type}")
        print(f"回看天数: {lookback_days}")
        print(f"总样本数: {len(X)}")
        print(f"特征数: {X.shape[1]}")
        print(f"准确率: {metrics['accuracy']:.2%}")
        print(f"精确率: {metrics['precision']:.2%}")
        print(f"召回率: {metrics['recall']:.2%}")
        
        print(f"\n生成文件:")
        print(f"  - results/{signal_type}_prediction_results.csv")
        print(f"  - results/{signal_type}_features.csv")
        print(f"  - models/{signal_type}_prediction_model.pkl")
        
        print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 8. 分析预测效果
        print(f"\n步骤7: 分析预测效果")
        print("-"*50)
        
        # 找出高置信度的预测
        high_confidence = results_df[results_df['signal_probability'] > 0.7]
        print(f"\n高置信度预测（概率>70%）:")
        print(f"  样本数: {len(high_confidence)}")
        print(f"  准确率: {high_confidence['prediction_correct'].mean():.2%}")
        
        # 按股票统计
        stock_stats = results_df.groupby('stock_code').agg({
            'true_signal': 'sum',
            'pred_signal': 'sum',
            'prediction_correct': 'mean'
        }).round(3)
        
        print(f"\n各股票预测统计:")
        print(stock_stats.to_string())
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
