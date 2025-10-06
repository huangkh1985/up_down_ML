"""
多时间窗口预测系统
同时使用1天、3天、5天、10天的预测结果进行互相验证
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import warnings

from utils.pattern_recognition import add_pattern_features

warnings.filterwarnings('ignore')

# 设置中文显示 - 使用统一配置
try:
    from utils.matplotlib_config import configure_chinese_font
    configure_chinese_font()
except:
    # 备用配置
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False


class MultiHorizonPredictor:
    """多时间窗口预测器"""
    
    def __init__(self, horizons=[1, 3, 5, 10], lookback_days=60):
        """
        初始化多时间窗口预测器
        
        参数:
        horizons: 预测时间窗口列表
        lookback_days: 回看天数
        """
        self.horizons = horizons
        self.lookback_days = lookback_days
        self.models = {}  # 存储各个时间窗口的模型
        self.feature_columns = None
        
    def prepare_data_for_horizon(self, all_data, forecast_horizon):
        """
        为指定时间窗口准备数据
        """
        print(f"\n准备{forecast_horizon}天预测数据...")
        
        all_X = []
        all_y = []
        all_info = []
        
        for stock_code, df in all_data.items():
            if 'Bullish_Reversal' not in df.columns:
                df = add_pattern_features(df)
            
            df['Any_Reversal'] = ((df['Bullish_Reversal'] == 1) | 
                                  (df['Bearish_Reversal'] == 1)).astype(int)
            
            # 定义特征
            feature_columns = [
                'Close', 'Volume', 'TurnoverRate', 'PriceChangeRate',
                'MA5', 'MA10', 'MA20', 'MA50',
                'RSI', 'MACD', 'ATR', 'ADX',
                'Volatility', 'Momentum', 'ROC'
            ]
            
            available_features = [col for col in feature_columns if col in df.columns]
            
            for i in range(self.lookback_days, len(df) - forecast_horizon):
                # 预测未来forecast_horizon天的信号
                future_signal = df['Any_Reversal'].iloc[i + forecast_horizon]
                
                # 提取特征
                lookback_data = df.iloc[i-self.lookback_days:i]
                
                feature_dict = {}
                for col in available_features:
                    if col in lookback_data.columns:
                        values = lookback_data[col].values
                        feature_dict[f'{col}_mean'] = np.mean(values)
                        feature_dict[f'{col}_std'] = np.std(values)
                        feature_dict[f'{col}_min'] = np.min(values)
                        feature_dict[f'{col}_max'] = np.max(values)
                        feature_dict[f'{col}_last'] = values[-1]
                        
                        if len(values) > 1:
                            feature_dict[f'{col}_trend'] = (values[-1] - values[0]) / (values[0] + 1e-8)
                
                all_X.append(feature_dict)
                all_y.append(int(future_signal))
                all_info.append({
                    'stock_code': stock_code,
                    'current_date': df.index[i],
                    'prediction_date': df.index[i + forecast_horizon]
                })
        
        X = pd.DataFrame(all_X)
        y = pd.Series(all_y)
        info = pd.DataFrame(all_info)
        
        print(f"  样本数: {len(X)}, 信号比例: {y.sum()/len(y):.2%}")
        
        return X, y, info
    
    def train_all_models(self, all_data):
        """
        训练所有时间窗口的模型
        """
        print("\n" + "="*80)
        print("训练多时间窗口模型")
        print("="*80)
        
        for horizon in self.horizons:
            print(f"\n{'='*80}")
            print(f"训练{horizon}天预测模型")
            print(f"{'='*80}")
            
            # 准备数据
            X, y, info = self.prepare_data_for_horizon(all_data, horizon)
            
            if len(X) < 100:
                print(f"[!] 样本数不足，跳过{horizon}天模型")
                continue
            
            # 保存特征列（首次）
            if self.feature_columns is None:
                self.feature_columns = X.columns.tolist()
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # SMOTE处理类别不平衡
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"  使用SMOTE平衡数据")
            except:
                pass
            
            # 训练模型
            print(f"  训练Random Forest...")
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # 评估
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            
            print(f"\n  性能指标:")
            print(f"    准确率: {accuracy:.2%}")
            print(f"    精确率: {precision:.2%}")
            print(f"    召回率: {recall:.2%}")
            
            # 保存模型
            self.models[horizon] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'X_test': X_test,
                'y_test': y_test
            }
            
            print(f"  [OK] {horizon}天模型训练完成")
        
        print("\n" + "="*80)
        print(f"[OK] 所有模型训练完成！共{len(self.models)}个模型")
        print("="*80)
    
    def predict_multi_horizon(self, stock_data):
        """
        对单只股票进行多时间窗口预测
        
        参数:
        stock_data: 股票DataFrame（至少包含lookback_days天数据）
        
        返回:
        predictions: 各时间窗口的预测结果
        """
        if len(stock_data) < self.lookback_days:
            raise ValueError(f"数据不足{self.lookback_days}天")
        
        # 提取特征
        lookback_data = stock_data.iloc[-self.lookback_days:]
        
        feature_dict = {}
        for col in self.feature_columns:
            base_col = col.rsplit('_', 1)[0]  # 去掉后缀（如_mean）
            
            if base_col in lookback_data.columns:
                values = lookback_data[base_col].values
                
                if col.endswith('_mean'):
                    feature_dict[col] = np.mean(values)
                elif col.endswith('_std'):
                    feature_dict[col] = np.std(values)
                elif col.endswith('_min'):
                    feature_dict[col] = np.min(values)
                elif col.endswith('_max'):
                    feature_dict[col] = np.max(values)
                elif col.endswith('_last'):
                    feature_dict[col] = values[-1]
                elif col.endswith('_trend'):
                    if len(values) > 1:
                        feature_dict[col] = (values[-1] - values[0]) / (values[0] + 1e-8)
                    else:
                        feature_dict[col] = 0
        
        # 创建特征DataFrame
        X_pred = pd.DataFrame([feature_dict])[self.feature_columns]
        
        # 对每个时间窗口进行预测
        predictions = {}
        
        for horizon in self.horizons:
            if horizon not in self.models:
                continue
            
            model_data = self.models[horizon]
            model = model_data['model']
            
            # 预测
            pred_proba = model.predict_proba(X_pred)[0, 1]
            pred_label = 1 if pred_proba > 0.5 else 0
            
            predictions[horizon] = {
                'probability': pred_proba,
                'prediction': pred_label,
                'signal': '有信号' if pred_label == 1 else '无信号',
                'confidence': pred_proba if pred_label == 1 else (1 - pred_proba)
            }
        
        return predictions
    
    def make_综合_decision(self, predictions):
        """
        基于多时间窗口预测做出综合决策
        
        参数:
        predictions: 各时间窗口的预测结果
        
        返回:
        decision: 综合决策信息
        """
        # 统计信号一致性
        signal_counts = sum(1 for p in predictions.values() if p['prediction'] == 1)
        total_windows = len(predictions)
        
        # 计算平均置信度
        avg_confidence = np.mean([p['confidence'] for p in predictions.values()])
        
        # 决策逻辑
        if signal_counts == total_windows:
            # 所有窗口都预测有信号
            decision_level = "强烈"
            action = "强烈建议交易"
            confidence_level = "极高"
            color = "[+]"
        elif signal_counts >= total_windows * 0.75:
            # 75%以上窗口预测有信号
            decision_level = "推荐"
            action = "建议交易"
            confidence_level = "高"
            color = "[+]"
        elif signal_counts >= total_windows * 0.5:
            # 50%以上窗口预测有信号
            decision_level = "谨慎"
            action = "可以考虑"
            confidence_level = "中"
            color = "[*]"
        elif signal_counts > 0:
            # 少数窗口预测有信号
            decision_level = "观望"
            action = "暂不建议"
            confidence_level = "低"
            color = "[*]"
        else:
            # 所有窗口都预测无信号
            decision_level = "避免"
            action = "不建议交易"
            confidence_level = "高（反向）"
            color = "[-]"
        
        return {
            'decision_level': decision_level,
            'action': action,
            'confidence_level': confidence_level,
            'signal_consistency': f"{signal_counts}/{total_windows}",
            'avg_confidence': avg_confidence,
            'color': color
        }
    
    def save_models(self, output_dir='models'):
        """保存所有模型"""
        os.makedirs(output_dir, exist_ok=True)
        
        save_path = f'{output_dir}/multi_horizon_models.pkl'
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'horizons': self.horizons,
                'lookback_days': self.lookback_days,
                'feature_columns': self.feature_columns,
                'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f)
        
        print(f"\n[OK] 多时间窗口模型已保存: {save_path}")
    
    def load_models(self, model_path='models/multi_horizon_models.pkl'):
        """加载已保存的模型"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.models = data['models']
        self.horizons = data['horizons']
        self.lookback_days = data['lookback_days']
        self.feature_columns = data['feature_columns']
        
        print(f"[OK] 已加载{len(self.models)}个时间窗口模型")


def visualize_multi_horizon_prediction(predictions, decision, stock_code):
    """可视化多时间窗口预测结果"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'多时间窗口预测分析 - {stock_code}', fontsize=16, fontweight='bold')
    
    horizons = list(predictions.keys())
    
    # 1. 信号概率对比
    ax1 = axes[0, 0]
    probs = [predictions[h]['probability'] for h in horizons]
    colors = ['green' if p > 0.5 else 'red' for p in probs]
    bars1 = ax1.bar(range(len(horizons)), probs, color=colors, alpha=0.7)
    ax1.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='决策阈值')
    ax1.set_xlabel('预测窗口', fontsize=12)
    ax1.set_ylabel('信号概率', fontsize=12)
    ax1.set_title('各时间窗口信号概率', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(horizons)))
    ax1.set_xticklabels([f'{h}天' for h in horizons])
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, prob) in enumerate(zip(bars1, probs)):
        ax1.text(bar.get_x() + bar.get_width()/2, prob + 0.03,
                f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 置信度对比
    ax2 = axes[0, 1]
    confidences = [predictions[h]['confidence'] for h in horizons]
    bars2 = ax2.bar(range(len(horizons)), confidences, color='skyblue', alpha=0.7)
    ax2.set_xlabel('预测窗口', fontsize=12)
    ax2.set_ylabel('置信度', fontsize=12)
    ax2.set_title('各时间窗口置信度', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(horizons)))
    ax2.set_xticklabels([f'{h}天' for h in horizons])
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, conf) in enumerate(zip(bars2, confidences)):
        ax2.text(bar.get_x() + bar.get_width()/2, conf + 0.03,
                f'{conf:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 预测结果汇总
    ax3 = axes[1, 0]
    ax3.axis('off')
    
    summary_text = f"""
    {decision['color']} 综合决策: {decision['decision_level']}
    
    信号一致性: {decision['signal_consistency']}
    平均置信度: {decision['avg_confidence']:.1%}
    置信水平: {decision['confidence_level']}
    
    建议行动: {decision['action']}
    
    各窗口预测:
    """
    
    for h in horizons:
        p = predictions[h]
        symbol = "[V]" if p['prediction'] == 1 else "[X]"
        summary_text += f"\n    {h}天: {symbol} {p['signal']} ({p['probability']:.1%})"
    
    ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 4. 决策置信度雷达图
    ax4 = axes[1, 1]
    
    # 准备雷达图数据
    angles = np.linspace(0, 2 * np.pi, len(horizons), endpoint=False).tolist()
    values = [predictions[h]['confidence'] for h in horizons]
    angles += angles[:1]
    values += values[:1]
    
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    ax4.plot(angles, values, 'o-', linewidth=2, color='blue', label='置信度')
    ax4.fill(angles, values, alpha=0.25, color='blue')
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels([f'{h}天' for h in horizons])
    ax4.set_ylim(0, 1)
    ax4.set_title('多窗口置信度雷达图', fontsize=13, fontweight='bold', pad=20)
    ax4.grid(True)
    
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/multi_horizon_prediction_{stock_code}.png', 
               dpi=300, bbox_inches='tight')
    print(f"[OK] 可视化图表已保存: results/multi_horizon_prediction_{stock_code}.png")
    plt.show()


def demo_prediction():
    """演示：对股票进行多时间窗口预测"""
    
    print("\n" + "="*80)
    print("多时间窗口预测演示")
    print("="*80)
    
    # 加载模型
    predictor = MultiHorizonPredictor()
    
    try:
        predictor.load_models('models/multi_horizon_models.pkl')
    except:
        print("[!] 模型文件不存在，请先运行训练")
        return
    
    # 加载股票数据
    with open('data/all_stock_data.pkl', 'rb') as f:
        all_data = pickle.load(f)
    
    # 选择一只股票进行演示
    stock_code = list(all_data.keys())[0]
    stock_data = all_data[stock_code]
    
    print(f"\n对股票 {stock_code} 进行多时间窗口预测")
    print("-"*60)
    
    # 进行预测
    predictions = predictor.predict_multi_horizon(stock_data)
    
    # 输出预测结果
    print(f"\n各时间窗口预测结果:")
    print("-"*60)
    for horizon, pred in predictions.items():
        symbol = "[+]" if pred['prediction'] == 1 else "[-]"
        print(f"{symbol} {horizon}天后: {pred['signal']}")
        print(f"   概率: {pred['probability']:.2%}, 置信度: {pred['confidence']:.2%}")
    
    # 综合决策
    decision = predictor.make_综合_decision(predictions)
    
    print(f"\n{decision['color']} 综合决策:")
    print("-"*60)
    print(f"决策级别: {decision['decision_level']}")
    print(f"建议行动: {decision['action']}")
    print(f"信号一致性: {decision['signal_consistency']}")
    print(f"平均置信度: {decision['avg_confidence']:.1%}")
    print(f"置信水平: {decision['confidence_level']}")
    
    # 可视化
    visualize_multi_horizon_prediction(predictions, decision, stock_code)


def main():
    """主函数"""
    
    print("="*80)
    print("多时间窗口预测系统")
    print("同时使用1天、3天、5天、10天预测进行互相验证")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载数据
    print("\n步骤1: 加载股票数据")
    print("-"*60)
    
    with open('data/all_stock_data.pkl', 'rb') as f:
        all_data = pickle.load(f)
    
    print(f"[OK] 成功加载 {len(all_data)} 只股票")
    
    # 创建多时间窗口预测器
    print("\n步骤2: 创建多时间窗口预测器")
    print("-"*60)
    
    predictor = MultiHorizonPredictor(
        horizons=[1, 3, 5, 10],  # 四个时间窗口
        lookback_days=60
    )
    
    # 训练所有模型
    print("\n步骤3: 训练多时间窗口模型")
    print("-"*60)
    
    predictor.train_all_models(all_data)
    
    # 保存模型
    print("\n步骤4: 保存模型")
    print("-"*60)
    
    predictor.save_models('models')
    
    # 演示预测
    print("\n步骤5: 演示预测")
    print("-"*60)
    
    # 随机选择3只股票进行演示
    demo_stocks = list(all_data.keys())[:3]
    
    for stock_code in demo_stocks:
        print(f"\n{'='*80}")
        print(f"预测股票: {stock_code}")
        print(f"{'='*80}")
        
        stock_data = all_data[stock_code]
        
        # 多时间窗口预测
        predictions = predictor.predict_multi_horizon(stock_data)
        
        # 输出结果
        print(f"\n各时间窗口预测:")
        for horizon, pred in predictions.items():
            symbol = "[+]" if pred['prediction'] == 1 else "[-]"
            print(f"  {symbol} {horizon}天: {pred['signal']} "
                  f"(概率:{pred['probability']:.1%}, 置信:{pred['confidence']:.1%})")
        
        # 综合决策
        decision = predictor.make_综合_decision(predictions)
        
        print(f"\n{decision['color']} 综合决策:")
        print(f"  级别: {decision['decision_level']}")
        print(f"  行动: {decision['action']}")
        print(f"  一致性: {decision['signal_consistency']}")
        print(f"  置信度: {decision['avg_confidence']:.1%}")
        
        # 可视化第一只股票
        if stock_code == demo_stocks[0]:
            visualize_multi_horizon_prediction(predictions, decision, stock_code)
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == '__main__':
    main()
