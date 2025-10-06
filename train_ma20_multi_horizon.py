"""
MA20多时间窗口预测模型训练脚本
为1天、3天、5天、10天分别训练独立的MA20状态预测模型
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings

warnings.filterwarnings('ignore')

def load_stock_data():
    """加载股票数据"""
    print("=" * 80)
    print("[数据加载] 加载股票数据...")
    
    data_path = 'data/all_stock_data.pkl'
    if not os.path.exists(data_path):
        print(f"[错误] 找不到数据文件 {data_path}")
        return None
    
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    
    print(f"[成功] 成功加载 {len(all_data)} 只股票的数据")
    return all_data


def extract_basic_features(stock_data):
    """
    提取基本技术指标特征
    直接使用数据中已有的技术指标
    """
    # 定义基本特征列表
    feature_columns = [
        'Close', 'Open', 'High', 'Low', 'Volume', 
        'TurnoverRate', 'PriceChangeRate', 'Amplitude',
        'MA5', 'MA10', 'MA20', 'MA50', 'MA100',
        'RSI', 'MACD', 'ATR', 'ADX', 'CCI',
        'Volatility', 'Momentum', 'ROC',
        'MainNetInflow', 'MainNetInflowRatio',
        'K', 'D', 'J',
        '%B', 'BB_Width',
        'MFI', 'Williams_R',
        'HV_20', 'Vol_Ratio'
    ]
    
    # 选择存在的特征
    available_features = [col for col in feature_columns if col in stock_data.columns]
    
    if len(available_features) < 10:
        return pd.DataFrame()
    
    # 提取特征数据
    features_df = stock_data[available_features].copy()
    
    # 计算额外的衍生特征
    if 'Close' in features_df.columns and 'MA20' in features_df.columns:
        features_df['Close_MA20_Ratio'] = features_df['Close'] / (features_df['MA20'] + 1e-8)
        features_df['Close_MA20_Diff'] = (features_df['Close'] - features_df['MA20']) / (features_df['MA20'] + 1e-8)
    
    if 'MA5' in features_df.columns and 'MA20' in features_df.columns:
        features_df['MA5_MA20_Ratio'] = features_df['MA5'] / (features_df['MA20'] + 1e-8)
    
    if 'Volume' in features_df.columns:
        features_df['Volume_MA5'] = features_df['Volume'].rolling(window=5, min_periods=1).mean()
        features_df['Volume_Ratio'] = features_df['Volume'] / (features_df['Volume_MA5'] + 1e-8)
    
    # 填充缺失值
    features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return features_df


def create_ma20_target(stock_data, forecast_horizon):
    """
    创建MA20状态标签
    
    Parameters:
    -----------
    stock_data: DataFrame
        股票数据
    forecast_horizon: int
        预测时间窗口（1, 3, 5, 10天）
    
    Returns:
    --------
    target: Series
        1表示未来N天收盘价>=MA20（强势），0表示<MA20（弱势）
    """
    target = []
    
    for i in range(len(stock_data) - forecast_horizon):
        # 获取未来第N天的数据
        future_idx = i + forecast_horizon
        future_close = stock_data['Close'].iloc[future_idx]
        future_ma20 = stock_data['MA20'].iloc[future_idx]
        
        # 判断未来收盘价是否>=MA20
        if pd.notna(future_close) and pd.notna(future_ma20) and future_ma20 > 0:
            label = 1 if future_close >= future_ma20 else 0
            target.append(label)
        else:
            target.append(np.nan)
    
    # 补充最后forecast_horizon个数据点为NaN
    target.extend([np.nan] * forecast_horizon)
    
    return pd.Series(target, index=stock_data.index)


def prepare_training_data(all_stock_data, forecast_horizon):
    """
    准备训练数据
    
    Parameters:
    -----------
    all_stock_data: dict
        所有股票数据字典
    forecast_horizon: int
        预测时间窗口
    
    Returns:
    --------
    X: DataFrame
        特征矩阵
    y: Series
        标签
    """
    print(f"\n{'='*80}")
    print(f"[数据准备] 准备 {forecast_horizon} 天预测的训练数据...")
    
    all_X = []
    all_y = []
    
    for stock_code, stock_data in all_stock_data.items():
        try:
            print(f"  处理股票 {stock_code}...", end=' ')
            
            # 确保有MA20列
            if 'MA20' not in stock_data.columns or 'Close' not in stock_data.columns:
                print("[跳过] 缺少MA20或Close列")
                continue
            
            # 提取基本技术特征
            features_df = extract_basic_features(stock_data)
            
            if features_df.empty:
                print("[跳过] 特征计算失败")
                continue
            
            # 创建标签
            target = create_ma20_target(stock_data, forecast_horizon)
            
            # 对齐特征和标签
            combined = pd.concat([features_df, target.rename('target')], axis=1)
            combined = combined.dropna()
            
            if len(combined) < 50:  # 至少需要50个样本
                print(f"[跳过] 有效样本不足({len(combined)})")
                continue
            
            X_stock = combined.drop('target', axis=1)
            y_stock = combined['target']
            
            all_X.append(X_stock)
            all_y.append(y_stock)
            
            print(f"[OK] 获得 {len(y_stock)} 个样本 (强势: {int(y_stock.sum())}, 弱势: {len(y_stock)-int(y_stock.sum())})")
            
        except Exception as e:
            print(f"[错误] {e}")
            continue
    
    if not all_X:
        print("[错误] 没有有效的训练数据！")
        return None, None
    
    # 合并所有股票数据
    X = pd.concat(all_X, axis=0, ignore_index=True)
    y = pd.concat(all_y, axis=0, ignore_index=True)
    
    print(f"\n{'='*80}")
    print(f"[完成] {forecast_horizon}天数据准备完成:")
    print(f"   总样本数: {len(X)}")
    print(f"   特征数量: {X.shape[1]}")
    print(f"   强势样本: {int(y.sum())} ({y.sum()/len(y)*100:.1f}%)")
    print(f"   弱势样本: {len(y)-int(y.sum())} ({(len(y)-y.sum())/len(y)*100:.1f}%)")
    
    return X, y


def train_model(X, y, forecast_horizon):
    """
    训练XGBoost模型
    
    Parameters:
    -----------
    X: DataFrame
        特征矩阵
    y: Series
        标签
    forecast_horizon: int
        预测时间窗口
    
    Returns:
    --------
    model: XGBClassifier
        训练好的模型
    metrics: dict
        评估指标
    feature_list: list
        特征列表
    """
    print(f"\n{'='*80}")
    print(f"[模型训练] 训练 {forecast_horizon} 天预测模型...")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   训练集: {len(X_train)} 样本")
    print(f"   测试集: {len(X_test)} 样本")
    
    # 创建XGBoost模型
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    # 训练模型
    print("   开始训练...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算评估指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0)
    }
    
    print(f"\n   [完成] 模型训练完成！")
    print(f"   [评估指标]:")
    print(f"      准确率 (Accuracy):  {metrics['accuracy']:.4f}")
    print(f"      精确率 (Precision): {metrics['precision']:.4f}")
    print(f"      召回率 (Recall):    {metrics['recall']:.4f}")
    print(f"      F1分数 (F1-Score):  {metrics['f1_score']:.4f}")
    
    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n   [Top 10 重要特征]:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"      {row['feature']:<30} {row['importance']:.6f}")
    
    return model, metrics, X.columns.tolist()


def main():
    """主函数"""
    print("\n" + "="*80)
    print("MA20多时间窗口模型训练系统")
    print("="*80)
    print(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"训练目标: 为1天、3天、5天、10天分别训练独立的MA20预测模型")
    print("="*80)
    
    # 加载数据
    all_stock_data = load_stock_data()
    if all_stock_data is None:
        return
    
    # 训练多个时间窗口的模型
    horizons = [1, 3, 5, 10]
    trained_models = {}
    
    for horizon in horizons:
        print(f"\n\n{'#'*80}")
        print(f"#  训练 {horizon} 天预测模型")
        print(f"{'#'*80}")
        
        # 准备数据
        X, y = prepare_training_data(all_stock_data, horizon)
        
        if X is None or y is None:
            print(f"[错误] {horizon}天数据准备失败，跳过")
            continue
        
        # 训练模型
        model, metrics, feature_list = train_model(X, y, horizon)
        
        # 保存模型信息
        trained_models[horizon] = {
            'model': model,
            'metrics': metrics,
            'feature_list': feature_list,
            'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_samples': len(X)
        }
    
    # 保存所有模型
    if trained_models:
        print(f"\n\n{'='*80}")
        print("[保存] 保存模型...")
        
        save_path = 'models/ma20_multi_horizon_models.pkl'
        os.makedirs('models', exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(trained_models, f)
        
        print(f"[成功] 模型已保存到: {save_path}")
        
        # 打印总结
        print(f"\n{'='*80}")
        print("[训练总结]:")
        print(f"{'='*80}")
        for horizon, info in trained_models.items():
            print(f"\n[{horizon}天预测模型]:")
            print(f"   训练样本数: {info['train_samples']}")
            print(f"   特征数量: {len(info['feature_list'])}")
            print(f"   准确率: {info['metrics']['accuracy']:.4f}")
            print(f"   F1分数: {info['metrics']['f1_score']:.4f}")
            print(f"   训练时间: {info['train_date']}")
        
        print(f"\n{'='*80}")
        print("[成功] 训练完成！")
        print(f"{'='*80}\n")
        
    else:
        print("\n[错误] 没有成功训练任何模型！")


if __name__ == "__main__":
    main()

