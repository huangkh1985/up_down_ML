"""
实盘多时间窗口预测
结合MA20状态预测和形态信号预测，使用多时间窗口互相验证
"""

import pandas as pd
import numpy as np
import pickle
import os
import efinance as ef
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示 - 使用统一配置
try:
    from utils.matplotlib_config import configure_chinese_font
    configure_chinese_font()
except:
    # 备用配置
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False


class IntegratedMultiHorizonPredictor:
    """
    集成多时间窗口预测器
    结合MA20状态分类和形态信号预测
    """
    
    def __init__(self):
        self.ma20_models = {}  # MA20状态预测模型（1,3,5,10天）
        self.pattern_models = {}  # 形态信号预测模型（1,3,5,10天）
        self.feature_columns = None
        
    def load_existing_models(self):
        """加载已有的模型"""
        
        # 尝试加载多时间窗口模型
        try:
            with open('models/multi_horizon_models.pkl', 'rb') as f:
                data = pickle.load(f)
                self.pattern_models = data['models']
                self.feature_columns = data['feature_columns']
            print("[OK] 成功加载多时间窗口模型")
        except:
            print("[!] 未找到多时间窗口模型")
        
        # 加载标准MA20预测模型
        try:
            with open('models/trained_model.pkl', 'rb') as f:
                self.standard_ma20_model = pickle.load(f)
            with open('models/feature_list.pkl', 'rb') as f:
                self.ma20_feature_list = pickle.load(f)
            print("[OK] 成功加载MA20状态预测模型")
        except:
            print("[!] 未找到MA20预测模型")
            self.standard_ma20_model = None
    
    def predict_ma20_multi_horizon(self, stock_code, stock_data, horizons=[1, 3, 5, 10]):
        """
        MA20状态多时间窗口预测
        预测未来N天股价是否在MA20之上
        """
        from stock_feature_engineering import create_multi_feature_tsfresh_data, extract_enhanced_features
        
        results = {}
        
        for horizon in horizons:
            print(f"\n预测{horizon}天后MA20状态...")
            
            try:
                # 使用现有的特征提取流程
                if len(stock_data) < 30 + horizon:
                    print(f"  数据不足，跳过{horizon}天预测")
                    continue
                
                # 简化特征：使用最近的技术指标
                recent_data = stock_data.iloc[-30:]
                
                # 提取简单特征
                features = {
                    'close_mean': recent_data['Close'].mean(),
                    'close_std': recent_data['Close'].std(),
                    'volume_mean': recent_data['Volume'].mean(),
                    'ma5': recent_data['Close'].rolling(5).mean().iloc[-1] if len(recent_data) >= 5 else recent_data['Close'].mean(),
                    'ma10': recent_data['Close'].rolling(10).mean().iloc[-1] if len(recent_data) >= 10 else recent_data['Close'].mean(),
                    'ma20': recent_data['Close'].rolling(20).mean().iloc[-1] if len(recent_data) >= 20 else recent_data['Close'].mean(),
                }
                
                # 预测（这里简化使用当前价格相对MA20的位置）
                current_close = stock_data['Close'].iloc[-1]
                current_ma20 = features['ma20']
                
                # 简单预测逻辑（实际应该用训练好的模型）
                prediction_score = (current_close / current_ma20 - 1) * 10
                probability = 1 / (1 + np.exp(-prediction_score))  # Sigmoid
                
                results[horizon] = {
                    'probability': probability,
                    'prediction': 1 if probability > 0.5 else 0,
                    'signal': '强势(≥MA20)' if probability > 0.5 else '弱势(<MA20)',
                    'confidence': max(probability, 1-probability)
                }
                
            except Exception as e:
                print(f"  预测{horizon}天失败: {e}")
                continue
        
        return results
    
    def predict_pattern_multi_horizon(self, stock_data, horizons=[1, 3, 5, 10]):
        """
        形态信号多时间窗口预测
        """
        from multi_horizon_prediction_system import MultiHorizonPredictor
        
        results = {}
        
        # 使用已加载的形态模型
        if not self.pattern_models:
            print("[!] 形态预测模型未加载")
            return results
        
        # 提取特征（简化版）
        lookback_days = 60
        if len(stock_data) < lookback_days:
            print(f"[!] 数据不足{lookback_days}天")
            return results
        
        lookback_data = stock_data.iloc[-lookback_days:]
        
        for horizon in horizons:
            if horizon not in self.pattern_models:
                continue
            
            try:
                # 提取特征
                feature_dict = {}
                for col in ['Close', 'Volume', 'TurnoverRate']:
                    if col in lookback_data.columns:
                        values = lookback_data[col].values
                        feature_dict[f'{col}_mean'] = np.mean(values)
                        feature_dict[f'{col}_std'] = np.std(values)
                        feature_dict[f'{col}_last'] = values[-1]
                
                # 如果特征不匹配，跳过
                if not self.feature_columns:
                    continue
                
                # 填充缺失特征
                for col in self.feature_columns:
                    if col not in feature_dict:
                        feature_dict[col] = 0
                
                X_pred = pd.DataFrame([feature_dict])[self.feature_columns]
                
                # 预测
                model = self.pattern_models[horizon]['model']
                pred_proba = model.predict_proba(X_pred)[0, 1]
                pred_label = 1 if pred_proba > 0.5 else 0
                
                results[horizon] = {
                    'probability': pred_proba,
                    'prediction': pred_label,
                    'signal': '有信号' if pred_label == 1 else '无信号',
                    'confidence': pred_proba if pred_label == 1 else (1 - pred_proba)
                }
                
            except Exception as e:
                print(f"  预测{horizon}天形态失败: {e}")
                continue
        
        return results
    
    def 综合_prediction(self, stock_code):
        """
        对单只股票进行综合的多时间窗口预测
        """
        print("\n" + "="*80)
        print(f"多时间窗口综合预测 - {stock_code}")
        print("="*80)
        
        # 获取股票数据
        print(f"\n获取{stock_code}实时数据...")
        try:
            stock_data_dict = ef.stock.get_quote_history([stock_code], beg='20240101', end='20250930')
            
            if stock_code not in stock_data_dict:
                print(f"[X] 无法获取{stock_code}数据")
                return None
            
            df = stock_data_dict[stock_code]
            
            # 重命名列
            df = df.rename(columns={
                '日期': 'Date',
                '开盘': 'Open',
                '收盘': 'Close',
                '最高': 'High',
                '最低': 'Low',
                '成交量': 'Volume',
                '成交额': 'Amount',
                '涨跌幅': 'PriceChangeRate',
                '换手率': 'TurnoverRate'
            })
            
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            print(f"[OK] 获取到 {len(df)} 条记录")
            
        except Exception as e:
            print(f"[X] 获取数据失败: {e}")
            return None
        
        # 添加技术指标
        from utils.technical_indicators import add_technical_indicators
        df = add_technical_indicators(df, include_patterns=True)
        
        horizons = [1, 3, 5, 10]
        
        # 1. MA20状态预测
        print(f"\n{'='*80}")
        print("1️⃣ MA20状态多时间窗口预测（价格相对MA20）")
        print(f"{'='*80}")
        
        ma20_predictions = self.predict_ma20_multi_horizon(stock_code, df, horizons)
        
        if ma20_predictions:
            print(f"\nMA20状态预测结果:")
            for h, pred in ma20_predictions.items():
                symbol = "[+]" if pred['prediction'] == 1 else "[-]"
                print(f"  {symbol} {h}天后: {pred['signal']} (置信度:{pred['confidence']:.1%})")
        
        # 2. 形态信号预测
        print(f"\n{'='*80}")
        print("2️⃣ 形态信号多时间窗口预测（反转/回调/反弹）")
        print(f"{'='*80}")
        
        pattern_predictions = self.predict_pattern_multi_horizon(df, horizons)
        
        if pattern_predictions:
            print(f"\n形态信号预测结果:")
            for h, pred in pattern_predictions.items():
                symbol = "[+]" if pred['prediction'] == 1 else "[-]"
                print(f"  {symbol} {h}天后: {pred['signal']} (置信度:{pred['confidence']:.1%})")
        
        # 3. 综合决策
        print(f"\n{'='*80}")
        print("3️⃣ 综合决策")
        print(f"{'='*80}")
        
        comprehensive_decision = self.make_comprehensive_decision(
            ma20_predictions, pattern_predictions, horizons
        )
        
        # 打印综合决策
        self.print_comprehensive_decision(comprehensive_decision)
        
        # 4. 可视化
        self.visualize_comprehensive_prediction(
            stock_code, ma20_predictions, pattern_predictions, 
            comprehensive_decision, df
        )
        
        return {
            'ma20_predictions': ma20_predictions,
            'pattern_predictions': pattern_predictions,
            'comprehensive_decision': comprehensive_decision,
            'stock_data': df
        }
    
    def make_comprehensive_decision(self, ma20_preds, pattern_preds, horizons):
        """综合两种预测方法做出决策"""
        
        decisions = {}
        
        for horizon in horizons:
            ma20_pred = ma20_preds.get(horizon, {})
            pattern_pred = pattern_preds.get(horizon, {})
            
            if not ma20_pred and not pattern_pred:
                continue
            
            # 统计信号
            signals = []
            confidences = []
            
            if ma20_pred:
                signals.append(ma20_pred['prediction'])
                confidences.append(ma20_pred['confidence'])
            
            if pattern_pred:
                signals.append(pattern_pred['prediction'])
                confidences.append(pattern_pred['confidence'])
            
            if not signals:
                continue
            
            # 信号一致性
            signal_count = sum(signals)
            total_signals = len(signals)
            avg_confidence = np.mean(confidences)
            
            # 决策逻辑
            if signal_count == total_signals and total_signals >= 2:
                # 所有信号一致且至少有两个信号
                level = "强烈"
                action = "强烈建议"
                color = "[+]"
            elif signal_count >= total_signals * 0.5:
                level = "推荐"
                action = "建议关注"
                color = "[*]"
            else:
                level = "观望"
                action = "暂不建议"
                color = "[-]"
            
            decisions[horizon] = {
                'level': level,
                'action': action,
                'signal_consistency': f"{signal_count}/{total_signals}",
                'avg_confidence': avg_confidence,
                'color': color,
                'ma20_signal': ma20_pred.get('signal', 'N/A'),
                'pattern_signal': pattern_pred.get('signal', 'N/A')
            }
        
        return decisions
    
    def print_comprehensive_decision(self, decisions):
        """打印综合决策结果"""
        
        print(f"\n综合决策结果:")
        print("-"*80)
        
        for horizon, decision in decisions.items():
            print(f"\n{decision['color']} {horizon}天预测:")
            print(f"  决策级别: {decision['level']}")
            print(f"  建议行动: {decision['action']}")
            print(f"  信号一致性: {decision['signal_consistency']}")
            print(f"  平均置信度: {decision['avg_confidence']:.1%}")
            print(f"  MA20预测: {decision['ma20_signal']}")
            print(f"  形态预测: {decision['pattern_signal']}")
    
    def visualize_comprehensive_prediction(self, stock_code, ma20_preds, 
                                          pattern_preds, decisions, stock_data):
        """可视化综合预测结果"""
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'多时间窗口综合预测分析 - {stock_code}', 
                    fontsize=16, fontweight='bold')
        
        horizons = [1, 3, 5, 10]
        
        # 1. 价格走势图
        ax1 = fig.add_subplot(gs[0, :])
        recent_data = stock_data.tail(60)
        ax1.plot(recent_data.index, recent_data['Close'], 
                label='收盘价', linewidth=2, color='blue')
        
        if 'MA20' in recent_data.columns:
            ax1.plot(recent_data.index, recent_data['MA20'], 
                    label='MA20', linewidth=1.5, color='orange', linestyle='--')
        elif 'SMA_20' in recent_data.columns:
            ax1.plot(recent_data.index, recent_data['SMA_20'], 
                    label='MA20', linewidth=1.5, color='orange', linestyle='--')
        
        ax1.set_title('近60日价格走势', fontsize=12, fontweight='bold')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('价格')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. MA20预测对比
        ax2 = fig.add_subplot(gs[1, 0])
        if ma20_preds:
            probs = [ma20_preds[h]['probability'] for h in horizons if h in ma20_preds]
            h_labels = [f'{h}天' for h in horizons if h in ma20_preds]
            colors = ['green' if p > 0.5 else 'red' for p in probs]
            bars = ax2.bar(range(len(probs)), probs, color=colors, alpha=0.7)
            ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
            ax2.set_xticks(range(len(probs)))
            ax2.set_xticklabels(h_labels)
            ax2.set_ylim([0, 1])
            ax2.set_title('MA20状态预测', fontweight='bold')
            ax2.set_ylabel('强势概率')
            ax2.grid(True, alpha=0.3, axis='y')
            
            for bar, prob in zip(bars, probs):
                ax2.text(bar.get_x() + bar.get_width()/2, prob + 0.03,
                        f'{prob:.0%}', ha='center', fontsize=9)
        
        # 3. 形态信号预测对比
        ax3 = fig.add_subplot(gs[1, 1])
        if pattern_preds:
            probs = [pattern_preds[h]['probability'] for h in horizons if h in pattern_preds]
            h_labels = [f'{h}天' for h in horizons if h in pattern_preds]
            colors = ['green' if p > 0.5 else 'red' for p in probs]
            bars = ax3.bar(range(len(probs)), probs, color=colors, alpha=0.7)
            ax3.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
            ax3.set_xticks(range(len(probs)))
            ax3.set_xticklabels(h_labels)
            ax3.set_ylim([0, 1])
            ax3.set_title('形态信号预测', fontweight='bold')
            ax3.set_ylabel('信号概率')
            ax3.grid(True, alpha=0.3, axis='y')
            
            for bar, prob in zip(bars, probs):
                ax3.text(bar.get_x() + bar.get_width()/2, prob + 0.03,
                        f'{prob:.0%}', ha='center', fontsize=9)
        
        # 4. 综合决策置信度
        ax4 = fig.add_subplot(gs[1, 2])
        if decisions:
            confidences = [decisions[h]['avg_confidence'] for h in horizons if h in decisions]
            h_labels = [f'{h}天' for h in horizons if h in decisions]
            bars = ax4.bar(range(len(confidences)), confidences, 
                          color='skyblue', alpha=0.7)
            ax4.set_xticks(range(len(confidences)))
            ax4.set_xticklabels(h_labels)
            ax4.set_ylim([0, 1])
            ax4.set_title('综合置信度', fontweight='bold')
            ax4.set_ylabel('置信度')
            ax4.grid(True, alpha=0.3, axis='y')
            
            for bar, conf in zip(bars, confidences):
                ax4.text(bar.get_x() + bar.get_width()/2, conf + 0.03,
                        f'{conf:.0%}', ha='center', fontsize=9)
        
        # 5. 综合决策文本摘要
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        summary_text = f"综合决策摘要\n{'='*70}\n\n"
        
        for horizon in horizons:
            if horizon not in decisions:
                continue
            
            d = decisions[horizon]
            ma20_sig = ma20_preds.get(horizon, {}).get('signal', 'N/A')
            pattern_sig = pattern_preds.get(horizon, {}).get('signal', 'N/A')
            
            summary_text += f"{d['color']} {horizon}天预测: {d['level']} | {d['action']}\n"
            summary_text += f"   MA20: {ma20_sig} | 形态: {pattern_sig}\n"
            summary_text += f"   一致性: {d['signal_consistency']} | 置信: {d['avg_confidence']:.1%}\n\n"
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.2))
        
        plt.tight_layout()
        
        os.makedirs('results', exist_ok=True)
        save_path = f'results/comprehensive_multi_horizon_{stock_code}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[OK] 综合预测图表已保存: {save_path}")
        plt.show()


def main():
    """主函数"""
    
    print("="*80)
    print("股票实盘多时间窗口综合预测")
    print("结合MA20状态 + 形态信号 + 多时间窗口验证")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 创建预测器
    predictor = IntegratedMultiHorizonPredictor()
    
    # 加载模型
    print("加载预测模型...")
    predictor.load_existing_models()
    
    # 输入股票代码
    print("\n请输入要预测的股票代码（多个用逗号分隔）:")
    print("示例: 600519,000001,002415")
    
    # 默认使用一些示例股票
    default_stocks = ['600519', '000001', '600036']
    
    user_input = input("股票代码（回车使用默认）: ").strip()
    
    if user_input:
        stock_codes = [s.strip() for s in user_input.split(',')]
    else:
        stock_codes = default_stocks
        print(f"使用默认股票: {stock_codes}")
    
    # 对每只股票进行预测
    for stock_code in stock_codes:
        try:
            result = predictor.综合_prediction(stock_code)
            
            if result:
                print(f"\n[OK] {stock_code} 预测完成")
            else:
                print(f"\n[X] {stock_code} 预测失败")
        
        except Exception as e:
            print(f"\n[X] {stock_code} 预测异常: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*80 + "\n")
    
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
