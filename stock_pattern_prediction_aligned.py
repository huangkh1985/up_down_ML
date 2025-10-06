"""
股票形态信号预测模块（时间对齐版本）
目标：预测未来5天的反转信号（与MA20分类方法时间对齐）
使用信号前60天的数据作为自变量
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


def extract_pre_signal_features_aligned(df, signal_column, lookback_days=60, 
                                        forecast_horizon=5, target_type='reversal'):
    """
    提取信号发生前N天的特征数据（时间对齐版本）
    
    ⭐ 关键改进：预测未来forecast_horizon天的信号，而不是当天
    
    参数:
    df: 包含形态特征的DataFrame
    signal_column: 信号列名（如'Bullish_Reversal', 'Bearish_Reversal'）
    lookback_days: 回看天数（信号前多少天）
    forecast_horizon: 预测未来天数（默认5天，与MA20分类一致）
    target_type: 目标类型 ('reversal', 'pullback', 'bounce')
    
    返回:
    X: 特征矩阵（第i-lookback_days天到第i天的数据）
    y: 目标变量（第i+forecast_horizon天是否有信号）
    sample_info: 样本信息（用于分析）
    """
    print(f"提取特征预测{forecast_horizon}天后的{signal_column}信号...")
    print(f"时间线: [第i-{lookback_days}天 → 第i天] → 预测 → [第i+{forecast_horizon}天]")
    
    X_samples = []
    y_samples = []
    sample_info = []
    
    # 选择要使用的特征列
    feature_columns = [
        'Close', 'Volume', 'TurnoverRate', 'PriceChangeRate',
        'MA5', 'MA10', 'MA20', 'MA50',
        'RSI', 'MACD', 'Signal', 'ATR', 'ADX',
        'K', 'D', 'J', 'DI_plus', 'DI_minus',
        'Williams_R', 'CCI', 'OBV', 'MFI', 'VWAP',
        'Volatility', 'HV_20', 'BB_Width',
        'Momentum', 'ROC', 'TSI', 'CMO',
        'Trend', 'Drawdown_From_High', 'Rally_From_Low',
        'Days_Since_Reversal',
        'Pullback_Frequency', 'Bounce_Frequency',
        'Avg_Pullback_Depth', 'Avg_Bounce_Height',
        'Pullback_Success_Rate', 'Bounce_Success_Rate',
    ]
    
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"  可用特征数: {len(available_features)}")
    
    # ⭐ 关键修改：遍历时要留出forecast_horizon的空间
    for i in range(lookback_days, len(df) - forecast_horizon):
        # ✅ 预测未来第forecast_horizon天的信号（而不是当天）
        future_signal = df[signal_column].iloc[i + forecast_horizon]
        
        # 提取第i-lookback_days天到第i天的数据
        lookback_data = df.iloc[i-lookback_days:i]
        
        # 计算统计特征
        feature_dict = {}
        
        for col in available_features:
            if col in lookback_data.columns:
                values = lookback_data[col].values
                
                # 多种统计特征
                feature_dict[f'{col}_mean'] = np.mean(values)
                feature_dict[f'{col}_std'] = np.std(values)
                feature_dict[f'{col}_min'] = np.min(values)
                feature_dict[f'{col}_max'] = np.max(values)
                feature_dict[f'{col}_last'] = values[-1]
                
                # 趋势特征
                if len(values) > 1:
                    feature_dict[f'{col}_trend'] = (values[-1] - values[0]) / (values[0] + 1e-8)
                    feature_dict[f'{col}_momentum'] = values[-1] - values[-5] if len(values) >= 5 else 0
                else:
                    feature_dict[f'{col}_trend'] = 0
                    feature_dict[f'{col}_momentum'] = 0
        
        X_samples.append(feature_dict)
        y_samples.append(int(future_signal))
        
        # 记录样本信息（包含当前日期和预测日期）
        sample_info.append({
            'current_date': df.index[i],
            'prediction_date': df.index[i + forecast_horizon],
            'signal': future_signal,
            'close_price': df['Close'].iloc[i] if 'Close' in df.columns else np.nan
        })
    
    X = pd.DataFrame(X_samples)
    y = pd.Series(y_samples)
    sample_info_df = pd.DataFrame(sample_info)
    
    print(f"  提取完成: {len(X)} 个样本")
    print(f"  信号样本数: {y.sum()} ({y.sum()/len(y):.2%})")
    print(f"  非信号样本数: {len(y)-y.sum()} ({(len(y)-y.sum())/len(y):.2%})")
    print(f"  特征维度: {X.shape[1]}")
    
    return X, y, sample_info_df


def prepare_reversal_prediction_data_aligned(all_data, lookback_days=60, forecast_horizon=5):
    """
    准备反转信号预测数据（时间对齐版本）
    
    参数:
    all_data: 股票数据字典
    lookback_days: 回看天数（默认60）
    forecast_horizon: 预测未来天数（默认5，与MA20分类一致）
    
    返回:
    X: 特征矩阵
    y: 目标变量（未来forecast_horizon天是否有反转信号）
    sample_info: 样本信息
    """
    print("\n" + "="*80)
    print("准备反转信号预测数据（时间对齐版本）")
    print("="*80)
    print(f"回看天数: {lookback_days}")
    print(f"预测天数: {forecast_horizon}天后")
    print(f"目标: 预测未来{forecast_horizon}天是否出现反转信号")
    print(f"时间对齐: 与MA20分类方法保持一致")
    
    all_X = []
    all_y = []
    all_info = []
    
    for stock_code, df in all_data.items():
        print(f"\n处理股票: {stock_code}")
        
        if 'Bullish_Reversal' not in df.columns:
            print(f"  添加形态特征...")
            df = add_pattern_features(df)
        
        # 创建综合反转信号
        df['Any_Reversal'] = ((df['Bullish_Reversal'] == 1) | 
                              (df['Bearish_Reversal'] == 1)).astype(int)
        
        print(f"  反转信号总数: {df['Any_Reversal'].sum()}")
        
        # 提取特征（时间对齐版本）
        X_stock, y_stock, info_stock = extract_pre_signal_features_aligned(
            df, 
            signal_column='Any_Reversal',
            lookback_days=lookback_days,
            forecast_horizon=forecast_horizon
        )
        
        info_stock['stock_code'] = stock_code
        
        all_X.append(X_stock)
        all_y.append(y_stock)
        all_info.append(info_stock)
    
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)
    info_combined = pd.concat(all_info, ignore_index=True)
    
    print("\n" + "="*80)
    print("数据准备完成")
    print("="*80)
    print(f"总样本数: {len(X_combined)}")
    print(f"未来{forecast_horizon}天有信号: {y_combined.sum()} ({y_combined.sum()/len(y_combined):.2%})")
    print(f"未来{forecast_horizon}天无信号: {len(y_combined)-y_combined.sum()} ({(len(y_combined)-y_combined.sum())/len(y_combined):.2%})")
    print(f"特征数: {X_combined.shape[1]}")
    print(f"股票数: {len(all_data)}")
    
    return X_combined, y_combined, info_combined


def train_signal_prediction_model(X_train, X_test, y_train, y_test, 
                                  signal_type='reversal', forecast_horizon=5):
    """
    训练信号预测模型
    """
    print(f"\n训练{signal_type}信号预测模型（预测{forecast_horizon}天后）...")
    print("="*80)
    
    # 打印类别分布
    print("\n训练集类别分布:")
    train_counts = pd.Series(y_train).value_counts().sort_index()
    print(f"  无信号: {train_counts.get(0, 0)} ({train_counts.get(0, 0)/len(y_train):.2%})")
    print(f"  有信号: {train_counts.get(1, 0)} ({train_counts.get(1, 0)/len(y_train):.2%})")
    
    minority_ratio = train_counts.min() / train_counts.max()
    print(f"\n类别不平衡比率: {minority_ratio:.2%}")
    
    # SMOTE处理
    if minority_ratio < 0.3:
        try:
            from imblearn.over_sampling import SMOTE
            print("\n使用SMOTE平衡数据集...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"平衡后样本数: {len(y_train_balanced)}")
        except ImportError:
            print("⚠️ imbalanced-learn未安装")
            X_train_balanced = X_train
            y_train_balanced = y_train
    else:
        X_train_balanced = X_train
        y_train_balanced = y_train
    
    # 训练模型
    print("\n训练Random Forest模型...")
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_balanced, y_train_balanced)
    print("✅ 模型训练完成")
    
    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    print("\n" + "="*80)
    print(f"模型性能评估（预测{forecast_horizon}天后）")
    print("="*80)
    print(f"准确率: {accuracy:.4f} ({accuracy:.2%})")
    print(f"精确率: {precision:.4f} ({precision:.2%})")
    print(f"召回率: {recall:.4f} ({recall:.2%})")
    
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, 
                              target_names=[f'无信号({forecast_horizon}天后)', 
                                          f'有信号({forecast_horizon}天后)'],
                              zero_division=0))
    
    # 特征重要性
    print("\n特征重要性排名（Top 15）:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'feature_importance': feature_importance
    }
    
    return model, y_pred, y_proba, metrics


def main():
    """
    主函数：时间对齐的反转信号预测
    """
    print("="*80)
    print("股票形态信号预测系统（时间对齐版本）")
    print("预测未来5天的反转信号（与MA20分类方法时间对齐）")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 加载数据
        print("\n步骤1: 加载股票数据")
        print("-"*50)
        
        data_path = 'data/all_stock_data.pkl'
        if not os.path.exists(data_path):
            print(f"错误: {data_path} 不存在")
            print("请先运行: python stock_data_downloader.py")
            return
        
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        
        print(f"✅ 成功加载 {len(all_data)} 只股票")
        
        # 2. 准备数据（时间对齐）
        print(f"\n步骤2: 准备预测数据（时间对齐）")
        print("-"*50)
        
        LOOKBACK_DAYS = 60
        FORECAST_HORIZON = 5  # ⭐ 与MA20分类保持一致
        
        X, y, sample_info = prepare_reversal_prediction_data_aligned(
            all_data, 
            lookback_days=LOOKBACK_DAYS,
            forecast_horizon=FORECAST_HORIZON
        )
        
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
            X_train, X_test, y_train, y_test,
            signal_type='reversal',
            forecast_horizon=FORECAST_HORIZON
        )
        
        # 5. 保存结果
        print(f"\n步骤5: 保存结果")
        print("-"*50)
        
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # 保存预测结果
        test_indices = X_test.index
        sample_info_test = sample_info.iloc[test_indices].reset_index(drop=True)
        
        results_df = sample_info_test.copy()
        results_df['true_signal'] = y_test.values
        results_df['pred_signal'] = y_pred
        results_df['signal_probability'] = y_proba
        results_df['prediction_correct'] = (y_test.values == y_pred).astype(int)
        
        results_path = 'results/reversal_prediction_aligned_results.csv'
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f"✅ 结果已保存: {results_path}")
        
        # 保存特征矩阵
        features_path = 'results/reversal_features_aligned.csv'
        X_test.to_csv(features_path, index=False)
        print(f"✅ 特征已保存: {features_path}")
        
        # 保存模型
        model_path = 'models/reversal_prediction_model_aligned.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_columns': X.columns.tolist(),
                'metrics': metrics,
                'lookback_days': LOOKBACK_DAYS,
                'forecast_horizon': FORECAST_HORIZON,
                'signal_type': 'reversal',
                'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'note': '时间对齐版本：预测未来5天的信号，与MA20分类方法一致'
            }, f)
        
        print(f"✅ 模型已保存: {model_path}")
        
        # 6. 总结
        print("\n" + "="*80)
        print("预测分析完成（时间对齐版本）")
        print("="*80)
        print(f"\n配置信息:")
        print(f"  回看天数: {LOOKBACK_DAYS}")
        print(f"  预测天数: {FORECAST_HORIZON}天后")
        print(f"  总样本数: {len(X)}")
        print(f"  特征数: {X.shape[1]}")
        
        print(f"\n性能指标:")
        print(f"  准确率: {metrics['accuracy']:.2%}")
        print(f"  精确率: {metrics['precision']:.2%}")
        print(f"  召回率: {metrics['recall']:.2%}")
        
        print(f"\n生成文件:")
        print(f"  - {results_path}")
        print(f"  - {features_path}")
        print(f"  - {model_path}")
        
        print(f"\n⭐ 时间对齐说明:")
        print(f"  方法1 (MA20分类): 使用20天数据 → 预测5天后状态")
        print(f"  方法2 (信号预测): 使用60天数据 → 预测5天后信号")
        print(f"  两者现在预测的是同一时间点，可以直接组合使用！")
        
        print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
