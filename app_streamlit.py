"""
股票多时间窗口预测系统 - Streamlit Web应用
支持PC端和移动端访问
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import efinance as ef
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys

warnings.filterwarnings('ignore')

# 配置页面
st.set_page_config(
    page_title="股票多时间窗口预测",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式（优化移动端显示）
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-signal {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .warning-signal {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .danger-signal {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    /* 移动端优化 */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.5rem;
            padding: 0.8rem;
        }
        .stButton button {
            width: 100%;
            padding: 0.8rem;
            font-size: 1.1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# 配置中文显示
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


class StreamlitPredictor:
    """Streamlit预测器"""
    
    def __init__(self):
        self.pattern_models = {}
        self.feature_columns = None
        self.standard_ma20_model = None
        self.ma20_feature_list = None
        self.ma20_multi_horizon_models = {}  # 新增：多时间窗口MA20模型
        self.ma10_multi_horizon_models = {}  # 新增：多时间窗口MA10模型
        
    @st.cache_resource
    def load_models(_self):
        """加载模型（使用缓存）"""
        status = {}
        
        # 加载多时间窗口模型
        try:
            with open('models/multi_horizon_models.pkl', 'rb') as f:
                data = pickle.load(f)
                _self.pattern_models = data['models']
                _self.feature_columns = data['feature_columns']
            status['pattern'] = True
        except:
            status['pattern'] = False
        
        # 加载MA20模型（旧版单一模型）
        try:
            with open('models/trained_model.pkl', 'rb') as f:
                _self.standard_ma20_model = pickle.load(f)
            with open('models/feature_list.pkl', 'rb') as f:
                _self.ma20_feature_list = pickle.load(f)
            status['ma20'] = True
        except:
            status['ma20'] = False
        
        # 加载MA20多时间窗口模型（新版独立模型）
        try:
            with open('models/ma20_multi_horizon_models.pkl', 'rb') as f:
                _self.ma20_multi_horizon_models = pickle.load(f)
            status['ma20_multi'] = True
        except:
            status['ma20_multi'] = False
        
        # 加载MA10多时间窗口模型（新增）
        try:
            with open('models/ma10_multi_horizon_models.pkl', 'rb') as f:
                _self.ma10_multi_horizon_models = pickle.load(f)
            status['ma10_multi'] = True
        except:
            status['ma10_multi'] = False
        
        return status
    
    @st.cache_data(ttl=3600)
    def get_stock_data(_self, stock_code, start_date='20240101', end_date='20250930'):
        """获取股票数据（缓存1小时）"""
        try:
            stock_data_dict = ef.stock.get_quote_history(
                [stock_code], 
                beg=start_date, 
                end=end_date
            )
            
            if stock_code not in stock_data_dict:
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
            
            # 添加技术指标
            from utils.technical_indicators import add_technical_indicators
            df = add_technical_indicators(df, include_patterns=True)
            
            return df
        except Exception as e:
            st.error(f"获取数据失败: {e}")
            return None
    
    def predict_ma20_rule_based(self, stock_data, horizons=[1, 3, 5, 10], ma_strategy='dynamic'):
        """MA状态预测 - 基于规则的技术分析方法（支持多种MA策略）"""
        results = {}
        
        # MA周期选择策略
        if ma_strategy == 'dynamic':
            # 动态选择：根据预测窗口选择最优MA
            ma_selection = {
                1: 10,   # 1天预测使用MA10
                3: 10,   # 3天预测使用MA10
                5: 20,   # 5天预测使用MA20
                10: 20   # 10天预测使用MA20
            }
        elif ma_strategy == 'ma10':
            # 统一使用MA10
            ma_selection = {1: 10, 3: 10, 5: 10, 10: 10}
        else:  # ma20
            # 统一使用MA20
            ma_selection = {1: 20, 3: 20, 5: 20, 10: 20}
        
        for horizon in horizons:
            try:
                # 选择适合该预测窗口的MA周期
                ma_period = ma_selection.get(horizon, 20)
                
                if len(stock_data) < ma_period + 10 + horizon:
                    continue
                
                # 获取不同长度的历史数据
                lookback = max(60, ma_period + 30)
                recent_data = stock_data.iloc[-lookback:] if len(stock_data) >= lookback else stock_data.copy()
                
                # 计算基础特征
                current_close = stock_data['Close'].iloc[-1]
                current_ma = recent_data['Close'].rolling(ma_period).mean().iloc[-1] if len(recent_data) >= ma_period else recent_data['Close'].mean()
                
                # 计算价格相对MA的位置（基础分数）
                price_position = (current_close / current_ma - 1) * 100  # 转换为百分比
                
                # 计算动量因子（最近N天的价格变化）
                momentum_days = min(horizon, 5)  # 根据时间窗口调整动量周期
                if len(recent_data) >= momentum_days + 1:
                    momentum = (current_close / recent_data['Close'].iloc[-(momentum_days+1)] - 1) * 100
                else:
                    momentum = 0
                
                # 计算趋势因子（MA5相对当前MA的位置）
                if len(recent_data) >= max(5, ma_period):
                    ma5 = recent_data['Close'].rolling(5).mean().iloc[-1]
                    trend = (ma5 / current_ma - 1) * 100
                else:
                    trend = 0
                
                # 计算成交量因子
                if 'Volume' in stock_data.columns and len(recent_data) >= 5:
                    current_volume = recent_data['Volume'].iloc[-1]
                    avg_volume = recent_data['Volume'].rolling(5).mean().iloc[-1]
                    volume_factor = (current_volume / avg_volume - 1) * 50  # 成交量放大系数
                else:
                    volume_factor = 0
                
                # 根据不同时间窗口调整权重
                if horizon == 1:
                    # 1天预测：重点关注动量和成交量
                    prediction_score = price_position * 0.4 + momentum * 0.4 + volume_factor * 0.2
                elif horizon == 3:
                    # 3天预测：平衡考虑位置、动量和趋势
                    prediction_score = price_position * 0.4 + momentum * 0.3 + trend * 0.3
                elif horizon == 5:
                    # 5天预测：更重视趋势
                    prediction_score = price_position * 0.3 + momentum * 0.2 + trend * 0.5
                else:  # horizon >= 10
                    # 10天预测：主要看趋势和位置
                    prediction_score = price_position * 0.5 + trend * 0.5
                
                # 转换为概率（使用sigmoid函数）
                probability = 1 / (1 + np.exp(-prediction_score / 10))
                
                # 根据预测结果生成信号描述
                is_strong = probability > 0.5
                signal_text = f'强势(≥MA{ma_period})' if is_strong else f'弱势(<MA{ma_period})'
                
                results[horizon] = {
                    'probability': probability,
                    'prediction': 1 if is_strong else 0,
                    'signal': signal_text,
                    'confidence': max(probability, 1-probability),
                    'ma_period': ma_period,  # 记录使用的MA周期
                    'method': f'规则分析(MA{ma_period})'
                }
            except Exception as e:
                # 如果出错，使用简单逻辑（回退到MA20）
                try:
                    current_close = stock_data['Close'].iloc[-1]
                    fallback_ma = stock_data['Close'].rolling(20).mean().iloc[-1]
                    simple_score = (current_close / fallback_ma - 1) * 10
                    probability = 1 / (1 + np.exp(-simple_score))
                    is_strong = probability > 0.5
                    results[horizon] = {
                        'probability': probability,
                        'prediction': 1 if is_strong else 0,
                        'signal': f'强势(≥MA20)' if is_strong else '弱势(<MA20)',
                        'confidence': max(probability, 1-probability),
                        'ma_period': 20,
                        'method': '规则分析(MA20-回退)'
                    }
                except:
                    continue
        
        return results
    
    def predict_ma20_ml(self, stock_data, horizons=[1, 3, 5, 10], ma_strategy='dynamic'):
        """MA状态预测 - 基于机器学习模型（使用独立的多时间窗口模型，支持MA10/MA20）"""
        results = {}
        
        # 根据ma_strategy选择模型
        if ma_strategy == 'ma10' and self.ma10_multi_horizon_models:
            # 使用MA10模型
            return self._predict_ma20_ml_multi(stock_data, horizons, ma_type='MA10')
        elif self.ma20_multi_horizon_models:
            # 使用MA20模型（默认或ma_strategy='ma20'或'dynamic'）
            return self._predict_ma20_ml_multi(stock_data, horizons, ma_type='MA20')
        elif self.standard_ma20_model is not None:
            # 如果没有多时间窗口模型，回退到旧的单一模型
            return self._predict_ma20_ml_single(stock_data, horizons)
        
        return results
    
    def _predict_ma20_ml_multi(self, stock_data, horizons=[1, 3, 5, 10], ma_type='MA20'):
        """使用独立的多时间窗口模型进行预测（支持MA10/MA20）"""
        results = {}
        
        # 根据ma_type选择模型集合
        if ma_type == 'MA10':
            models_dict = self.ma10_multi_horizon_models
            ma_label = 'MA10'
        else:
            models_dict = self.ma20_multi_horizon_models
            ma_label = 'MA20'
        
        if not models_dict:
            return results
        
        # 提取基本特征
        features_df = self._extract_basic_features(stock_data, ma_type=ma_type)
        if features_df.empty:
            return results
        
        # 获取最新一行的特征
        latest_row = features_df.iloc[-1]
        
        for horizon in horizons:
            # 检查是否有该时间窗口的模型
            if horizon not in models_dict:
                continue
            
            try:
                model_info = models_dict[horizon]
                model = model_info['model']
                feature_list = model_info['feature_list']
                
                # 提取模型需要的特征
                available_features = [f for f in feature_list if f in features_df.columns]
                if len(available_features) < len(feature_list) * 0.8:  # 至少80%的特征可用
                    continue
                
                # 准备特征数据（填充缺失的特征为0）
                feature_values = []
                for feat in feature_list:
                    if feat in latest_row.index:
                        feature_values.append(latest_row[feat])
                    else:
                        feature_values.append(0.0)
                
                X_pred = np.array(feature_values).reshape(1, -1)
                
                # 使用独立模型预测
                prediction = model.predict(X_pred)[0]
                probability = model.predict_proba(X_pred)[0, 1]
                
                results[horizon] = {
                    'probability': float(probability),
                    'prediction': int(prediction),
                    'signal': f'强势(≥{ma_label})' if prediction == 1 else f'弱势(低于{ma_label})',
                    'confidence': float(probability if prediction == 1 else (1 - probability)),
                    'method': f'独立ML模型-{horizon}天({ma_label})'
                }
            except Exception as e:
                continue
        
        return results
    
    def _predict_ma20_ml_single(self, stock_data, horizons=[1, 3, 5, 10]):
        """使用单一模型进行预测（旧版本，带概率调整）"""
        results = {}
        
        try:
            import stock_feature_engineering
            if hasattr(stock_feature_engineering, 'calculate_technical_features'):
                calculate_technical_features = stock_feature_engineering.calculate_technical_features
            else:
                return results
            
            for horizon in horizons:
                try:
                    min_data_length = 60 + horizon
                    if len(stock_data) < min_data_length:
                        continue
                    
                    if horizon == 1:
                        lookback_data = stock_data.iloc[-30:].copy()
                    elif horizon == 3:
                        lookback_data = stock_data.iloc[-60:].copy()
                    elif horizon == 5:
                        lookback_data = stock_data.iloc[-90:].copy() if len(stock_data) >= 90 else stock_data.copy()
                    else:
                        lookback_data = stock_data.iloc[-120:].copy() if len(stock_data) >= 120 else stock_data.copy()
                    
                    features_df = calculate_technical_features(lookback_data)
                    if features_df.empty or len(features_df) == 0:
                        continue
                    
                    latest_features = features_df.iloc[-1:][self.ma20_feature_list]
                    prediction = self.standard_ma20_model.predict(latest_features)[0]
                    probability = self.standard_ma20_model.predict_proba(latest_features)[0, 1]
                    
                    if horizon == 1:
                        adjusted_prob = probability * 1.1 if probability > 0.5 else probability * 0.9
                    elif horizon == 3:
                        adjusted_prob = probability * 1.05 if probability > 0.5 else probability * 0.95
                    elif horizon == 5:
                        adjusted_prob = probability
                    else:
                        adjusted_prob = probability * 0.9 if probability > 0.5 else probability * 1.1
                    
                    adjusted_prob = np.clip(adjusted_prob, 0.0, 1.0)
                    adjusted_prediction = 1 if adjusted_prob > 0.5 else 0
                    
                    results[horizon] = {
                        'probability': adjusted_prob,
                        'prediction': int(adjusted_prediction),
                        'signal': '强势(≥MA20)' if adjusted_prediction == 1 else '弱势(低于MA20)',
                        'confidence': adjusted_prob if adjusted_prediction == 1 else (1 - adjusted_prob),
                        'method': f'单一ML模型({horizon}天调整)'
                    }
                except Exception as e:
                    continue
                    
        except ImportError:
            pass
        
        return results
    
    def _extract_basic_features(self, stock_data, ma_type='MA20'):
        """提取基本技术指标特征（与训练脚本一致，支持MA10/MA20）"""
        try:
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
            
            available_features = [col for col in feature_columns if col in stock_data.columns]
            if len(available_features) < 10:
                return pd.DataFrame()
            
            features_df = stock_data[available_features].copy()
            
            # 计算衍生特征（根据ma_type选择）
            if ma_type == 'MA10':
                # MA10相关衍生特征
                if 'Close' in features_df.columns and 'MA10' in features_df.columns:
                    features_df['Close_MA10_Ratio'] = features_df['Close'] / (features_df['MA10'] + 1e-8)
                    features_df['Close_MA10_Diff'] = (features_df['Close'] - features_df['MA10']) / (features_df['MA10'] + 1e-8)
                
                if 'MA5' in features_df.columns and 'MA10' in features_df.columns:
                    features_df['MA5_MA10_Ratio'] = features_df['MA5'] / (features_df['MA10'] + 1e-8)
                
                if 'MA10' in features_df.columns and 'MA20' in features_df.columns:
                    features_df['MA10_MA20_Ratio'] = features_df['MA10'] / (features_df['MA20'] + 1e-8)
            else:
                # MA20相关衍生特征（默认）
                if 'Close' in features_df.columns and 'MA20' in features_df.columns:
                    features_df['Close_MA20_Ratio'] = features_df['Close'] / (features_df['MA20'] + 1e-8)
                    features_df['Close_MA20_Diff'] = (features_df['Close'] - features_df['MA20']) / (features_df['MA20'] + 1e-8)
                
                if 'MA5' in features_df.columns and 'MA20' in features_df.columns:
                    features_df['MA5_MA20_Ratio'] = features_df['MA5'] / (features_df['MA20'] + 1e-8)
            
            # 通用特征
            if 'Volume' in features_df.columns:
                features_df['Volume_MA5'] = features_df['Volume'].rolling(window=5, min_periods=1).mean()
                features_df['Volume_Ratio'] = features_df['Volume'] / (features_df['Volume_MA5'] + 1e-8)
            
            # 填充缺失值
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return features_df
        except Exception as e:
            return pd.DataFrame()
    
    def predict_ma20(self, stock_data, horizons=[1, 3, 5, 10], method='both', ma_strategy='dynamic'):
        """
        MA状态预测 - 统一接口
        
        Parameters:
        -----------
        method: str
            'ml' - 仅使用机器学习模型
            'rule' - 仅使用规则方法
            'both' - 同时使用两种方法（默认）
        ma_strategy: str
            'dynamic' - 动态选择MA周期（1-3天用MA10，5-10天用MA20）
            'ma10' - 统一使用MA10
            'ma20' - 统一使用MA20
        
        Returns:
        --------
        dict: 根据method参数返回不同格式
            - 'ml' 或 'rule': 返回单一方法的结果
            - 'both': 返回 {'ml': {...}, 'rule': {...}} 格式
        """
        if method == 'ml':
            return self.predict_ma20_ml(stock_data, horizons, ma_strategy)
        elif method == 'rule':
            return self.predict_ma20_rule_based(stock_data, horizons, ma_strategy)
        else:  # both
            ml_results = self.predict_ma20_ml(stock_data, horizons, ma_strategy)
            rule_results = self.predict_ma20_rule_based(stock_data, horizons, ma_strategy)
            
            return {
                'ml': ml_results,
                'rule': rule_results
            }
    
    def predict_pattern(self, stock_data, horizons=[1, 3, 5, 10]):
        """形态信号预测"""
        results = {}
        
        if not self.pattern_models:
            return results
        
        lookback_days = 60
        if len(stock_data) < lookback_days:
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
                
                if not self.feature_columns:
                    continue
                
                # 填充特征
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
            except:
                continue
        
        return results
    
    def make_decision(self, ma20_preds, pattern_preds, horizons=[1, 3, 5, 10]):
        """综合决策"""
        decisions = {}
        
        # 处理ma20_preds的不同格式
        # 如果是新格式（包含'ml'和'rule'），则优先使用ML结果，如果ML为空则使用规则结果
        if 'ml' in ma20_preds and 'rule' in ma20_preds:
            ml_preds = ma20_preds['ml']
            rule_preds = ma20_preds['rule']
            # 优先使用ML预测，如果ML预测为空则使用规则预测
            ma20_preds_to_use = ml_preds if ml_preds else rule_preds
        else:
            ma20_preds_to_use = ma20_preds
        
        for horizon in horizons:
            ma20_pred = ma20_preds_to_use.get(horizon, {})
            pattern_pred = pattern_preds.get(horizon, {})
            
            if not ma20_pred and not pattern_pred:
                continue
            
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
            
            signal_count = sum(signals)
            total_signals = len(signals)
            avg_confidence = np.mean(confidences)
            
            # 决策
            if signal_count == total_signals and total_signals >= 2:
                level = "强烈"
                action = "强烈建议"
                color = "success"
            elif signal_count >= total_signals * 0.5:
                level = "推荐"
                action = "建议关注"
                color = "warning"
            else:
                level = "观望"
                action = "暂不建议"
                color = "danger"
            
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


def check_system_status():
    """检查系统状态"""
    st.sidebar.header("🔍 系统状态")
    
    status = {}
    
    # 检查数据文件
    status['data'] = os.path.exists('data/all_stock_data.pkl')
    
    # 检查模型文件
    status['pattern_model'] = os.path.exists('models/multi_horizon_models.pkl')
    status['ma20_model'] = os.path.exists('models/trained_model.pkl')
    status['ma20_multi_model'] = os.path.exists('models/ma20_multi_horizon_models.pkl')
    status['ma10_multi_model'] = os.path.exists('models/ma10_multi_horizon_models.pkl')
    
    # 显示状态
    st.sidebar.markdown("### 📁 文件状态")
    st.sidebar.write("✅ 数据文件" if status['data'] else "❌ 数据文件")
    st.sidebar.write("✅ 形态识别模型" if status['pattern_model'] else "❌ 形态识别模型")
    st.sidebar.write("✅ MA20多窗口模型" if status['ma20_multi_model'] else "❌ MA20多窗口模型")
    st.sidebar.write("✅ MA10多窗口模型" if status['ma10_multi_model'] else "❌ MA10多窗口模型")
    
    # 可选：显示旧模型状态
    if status['ma20_model']:
        st.sidebar.caption("✓ 旧版MA20模型（兼容）")
    
    # 检查必需文件（旧版MA20模型是可选的）
    required_status = {
        'data': status['data'],
        'pattern_model': status['pattern_model'],
        'ma20_multi_model': status['ma20_multi_model'],
        'ma10_multi_model': status['ma10_multi_model']
    }
    
    # 统计缺失的文件
    missing_files = []
    if not status['data']:
        missing_files.append("📊 数据文件")
    if not status['pattern_model']:
        missing_files.append("🔄 形态识别模型")
    if not status['ma20_multi_model']:
        missing_files.append("📈 MA20多窗口模型")
    if not status['ma10_multi_model']:
        missing_files.append("📊 MA10多窗口模型")
    
    if missing_files:
        st.sidebar.error(f"⚠️ 缺失 {len(missing_files)} 个文件")
        
        # 显示缺失的具体文件
        with st.sidebar.expander("查看缺失文件", expanded=True):
            for file in missing_files:
                st.write(f"❌ {file}")
        
        # 提供训练建议
        st.sidebar.markdown("### 🚀 快速开始")
        
        # 根据缺失情况给出不同建议
        if not status['data']:
            st.sidebar.info("💡 首次使用，请运行完整训练")
            st.sidebar.code("retrain_all_models_complete.bat", language="bash")
        elif not status['ma10_multi_model'] and status['ma20_multi_model']:
            st.sidebar.info("💡 需要训练MA10模型")
            st.sidebar.code("train_ma10_models.bat", language="bash")
        elif len(missing_files) >= 2:
            st.sidebar.info("💡 推荐运行完整训练")
            st.sidebar.code("retrain_all_models_complete.bat", language="bash")
        else:
            st.sidebar.info("💡 运行快速训练")
            st.sidebar.code("retrain_all_models.bat", language="bash")
        
        # 显示详细步骤
        with st.sidebar.expander("查看详细步骤"):
            st.markdown("""
            **步骤1: 下载数据**
            ```bash
            python stock_data_downloader.py
            ```
            
            **步骤2: 训练所有模型**
            ```bash
            retrain_all_models_complete.bat
            ```
            
            **或分别训练:**
            ```bash
            python train_ma20_multi_horizon.py
            python train_ma10_multi_horizon.py
            python multi_horizon_prediction_system.py
            ```
            """)
    else:
        st.sidebar.success("✅ 所有文件就绪")
    
    return status


def display_prediction_results(stock_code, ma20_preds, pattern_preds, decisions, stock_data, start_date=None, end_date=None):
    """显示预测结果"""
    
    st.markdown(f"<div class='main-header'>📈 {stock_code} - 多时间窗口综合预测</div>", unsafe_allow_html=True)
    
    # 股票基本信息
    col1, col2, col3, col4, col5 = st.columns(5)
    
    current_price = stock_data['Close'].iloc[-1]
    price_change = stock_data['PriceChangeRate'].iloc[-1] if 'PriceChangeRate' in stock_data.columns else 0
    
    col1.metric("当前价格", f"¥{current_price:.2f}", f"{price_change:.2f}%")
    col2.metric("数据天数", f"{len(stock_data)}天")
    col3.metric("数据时间段", f"{stock_data.index[0].strftime('%Y-%m-%d')}\n至\n{stock_data.index[-1].strftime('%Y-%m-%d')}")
    col4.metric("最新日期", stock_data.index[-1].strftime('%Y-%m-%d'))
    col5.metric("成交量", f"{stock_data['Volume'].iloc[-1]/10000:.0f}万手")
    
    st.markdown("---")
    
    # Tab布局
    tab1, tab2, tab3, tab4 = st.tabs(["📊 综合决策", "📈 MA预测", "🔄 形态信号", "📉 价格走势"])
    
    with tab1:
        st.subheader("综合决策建议")
        
        if decisions:
            for horizon in [1, 3, 5, 10]:
                if horizon not in decisions:
                    continue
                
                d = decisions[horizon]
                
                # 根据决策级别选择样式
                if d['color'] == 'success':
                    css_class = 'success-signal'
                    icon = "✅"
                elif d['color'] == 'warning':
                    css_class = 'warning-signal'
                    icon = "⚠️"
                else:
                    css_class = 'danger-signal'
                    icon = "🚫"
                
                st.markdown(f"""
                <div class='{css_class}'>
                    <h4>{icon} {horizon}天预测: {d['level']} | {d['action']}</h4>
                    <p><strong>信号一致性:</strong> {d['signal_consistency']}</p>
                    <p><strong>平均置信度:</strong> {d['avg_confidence']:.1%}</p>
                    <p><strong>MA预测:</strong> {d['ma20_signal']}</p>
                    <p><strong>形态预测:</strong> {d['pattern_signal']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("暂无综合决策数据")
    
    with tab2:
        st.subheader("MA状态预测 (MA10/MA20)")
        
        if ma20_preds:
            # 检查是否为对比模式（包含'ml'和'rule'键）
            if 'ml' in ma20_preds and 'rule' in ma20_preds:
                ml_preds = ma20_preds['ml']
                rule_preds = ma20_preds['rule']
                
                # 创建两列布局
                col_ml, col_rule = st.columns(2)
                
                with col_ml:
                    st.markdown("#### 🤖 机器学习模型")
                    if ml_preds:
                        df_ml = pd.DataFrame(ml_preds).T
                        df_ml.index.name = '时间窗口'
                        # 格式化索引显示
                        df_ml.index = [f'{h}天' for h in df_ml.index]
                        st.dataframe(df_ml, use_container_width=True)
                        
                        # ML模型可视化
                        fig, ax = plt.subplots(figsize=(5, 4))
                        horizons = list(ml_preds.keys())
                        probs = [ml_preds[h]['probability'] for h in horizons]
                        colors = ['green' if p > 0.5 else 'red' for p in probs]
                        
                        ax.bar(range(len(horizons)), probs, color=colors, alpha=0.7)
                        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
                        ax.set_xticks(range(len(horizons)))
                        ax.set_xticklabels([f'{h}天' for h in horizons])
                        ax.set_ylim([0, 1])
                        ax.set_ylabel('强势概率')
                        ax.set_title('ML模型预测')
                        ax.grid(True, alpha=0.3, axis='y')
                        
                        for i, prob in enumerate(probs):
                            ax.text(i, prob + 0.03, f'{prob:.0%}', ha='center', fontsize=9)
                        
                        st.pyplot(fig)
                    else:
                        st.warning("ML模型未加载或预测失败")
                
                with col_rule:
                    st.markdown("#### 📊 规则技术分析")
                    if rule_preds:
                        # 显示MA周期使用情况
                        ma_info = []
                        for h in sorted(rule_preds.keys()):
                            if 'ma_period' in rule_preds[h]:
                                ma_info.append(f"{h}天→MA{rule_preds[h]['ma_period']}")
                        if ma_info:
                            st.caption(f"📈 使用策略: {' | '.join(ma_info)}")
                        
                        df_rule = pd.DataFrame(rule_preds).T
                        df_rule.index.name = '时间窗口'
                        # 格式化索引显示
                        df_rule.index = [f'{h}天' for h in df_rule.index]
                        st.dataframe(df_rule, use_container_width=True)
                        
                        # 规则方法可视化
                        fig, ax = plt.subplots(figsize=(5, 4))
                        horizons = list(rule_preds.keys())
                        probs = [rule_preds[h]['probability'] for h in horizons]
                        colors = ['green' if p > 0.5 else 'red' for p in probs]
                        
                        ax.bar(range(len(horizons)), probs, color=colors, alpha=0.7)
                        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
                        ax.set_xticks(range(len(horizons)))
                        ax.set_xticklabels([f'{h}天' for h in horizons])
                        ax.set_ylim([0, 1])
                        ax.set_ylabel('强势概率')
                        ax.set_title('规则方法预测')
                        ax.grid(True, alpha=0.3, axis='y')
                        
                        for i, prob in enumerate(probs):
                            ax.text(i, prob + 0.03, f'{prob:.0%}', ha='center', fontsize=9)
                        
                        st.pyplot(fig)
                    else:
                        st.warning("规则方法预测失败")
                
                # 对比分析
                if ml_preds and rule_preds:
                    st.markdown("---")
                    st.markdown("#### 🔍 方法对比分析")
                    
                    comparison_data = []
                    for h in [1, 3, 5, 10]:
                        if h in ml_preds and h in rule_preds:
                            ml_prob = ml_preds[h]['probability']
                            rule_prob = rule_preds[h]['probability']
                            diff = abs(ml_prob - rule_prob)
                            agreement = "一致✅" if (ml_prob > 0.5) == (rule_prob > 0.5) else "分歧⚠️"
                            
                            comparison_data.append({
                                '时间窗口': f'{h}天',
                                'ML概率': f'{ml_prob:.1%}',
                                '规则概率': f'{rule_prob:.1%}',
                                '差异': f'{diff:.1%}',
                                '结论': agreement
                            })
                    
                    if comparison_data:
                        df_comp = pd.DataFrame(comparison_data)
                        st.dataframe(df_comp, use_container_width=True)
                        
            else:
                # 单一方法模式（兼容旧代码）
                df_ma20 = pd.DataFrame(ma20_preds).T
                df_ma20.index.name = '时间窗口'
                
                st.dataframe(df_ma20, use_container_width=True)
                
                # 可视化
                fig, ax = plt.subplots(figsize=(10, 5))
                horizons = list(ma20_preds.keys())
                probs = [ma20_preds[h]['probability'] for h in horizons]
                colors = ['green' if p > 0.5 else 'red' for p in probs]
                
                ax.bar(range(len(horizons)), probs, color=colors, alpha=0.7)
                ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
                ax.set_xticks(range(len(horizons)))
                ax.set_xticklabels([f'{h}天' for h in horizons])
                ax.set_ylim([0, 1])
                ax.set_ylabel('强势概率')
                ax.set_title('MA状态预测')
                ax.grid(True, alpha=0.3, axis='y')
                
                for i, prob in enumerate(probs):
                    ax.text(i, prob + 0.03, f'{prob:.0%}', ha='center')
                
                st.pyplot(fig)
        else:
            st.info("暂无MA预测数据")
    
    with tab3:
        st.subheader("形态信号预测")
        
        if pattern_preds:
            # 创建数据框
            df_pattern = pd.DataFrame(pattern_preds).T
            df_pattern.index.name = '时间窗口'
            
            st.dataframe(df_pattern, use_container_width=True)
            
            # 可视化
            fig, ax = plt.subplots(figsize=(10, 5))
            horizons = list(pattern_preds.keys())
            probs = [pattern_preds[h]['probability'] for h in horizons]
            colors = ['green' if p > 0.5 else 'red' for p in probs]
            
            ax.bar(range(len(horizons)), probs, color=colors, alpha=0.7)
            ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
            ax.set_xticks(range(len(horizons)))
            ax.set_xticklabels([f'{h}天' for h in horizons])
            ax.set_ylim([0, 1])
            ax.set_ylabel('信号概率')
            ax.set_title('形态信号预测')
            ax.grid(True, alpha=0.3, axis='y')
            
            for i, prob in enumerate(probs):
                ax.text(i, prob + 0.03, f'{prob:.0%}', ha='center')
            
            st.pyplot(fig)
        else:
            st.info("暂无形态预测数据")
    
    with tab4:
        st.subheader("价格走势图")
        
        # 绘制价格走势
        fig, ax = plt.subplots(figsize=(12, 6))
        
        recent_data = stock_data.tail(60)
        ax.plot(recent_data.index, recent_data['Close'], label='收盘价', linewidth=2)
        
        # 添加MA20
        if 'SMA_20' in recent_data.columns:
            ax.plot(recent_data.index, recent_data['SMA_20'], 
                   label='MA20', linewidth=1.5, linestyle='--', alpha=0.7)
        
        ax.set_xlabel('日期')
        ax.set_ylabel('价格')
        ax.set_title('近60日价格走势')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        st.pyplot(fig)


def main():
    """主函数"""
    
    # 侧边栏
    st.sidebar.title("📊 股票预测系统")
    st.sidebar.markdown("---")
    
    # 检查系统状态
    status = check_system_status()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📅 数据时间段")
    
    # 初始化 session state
    if 'date_range' not in st.session_state:
        st.session_state.date_range = {
            'start': datetime(2024, 1, 1),
            'end': datetime.now()
        }
    
    # 快速时间段选择（放在前面）
    st.sidebar.markdown("**快速选择：**")
    quick_period_cols1 = st.sidebar.columns(3)
    quick_period_cols2 = st.sidebar.columns(3)
    
    if quick_period_cols1[0].button("近1月", use_container_width=True):
        st.session_state.date_range['start'] = datetime.now() - pd.DateOffset(months=1)
        st.session_state.date_range['end'] = datetime.now()
        st.rerun()
    
    if quick_period_cols1[1].button("近3月", use_container_width=True):
        st.session_state.date_range['start'] = datetime.now() - pd.DateOffset(months=3)
        st.session_state.date_range['end'] = datetime.now()
        st.rerun()
    
    if quick_period_cols1[2].button("近6月", use_container_width=True):
        st.session_state.date_range['start'] = datetime.now() - pd.DateOffset(months=6)
        st.session_state.date_range['end'] = datetime.now()
        st.rerun()
    
    if quick_period_cols2[0].button("近1年", use_container_width=True):
        st.session_state.date_range['start'] = datetime.now() - pd.DateOffset(years=1)
        st.session_state.date_range['end'] = datetime.now()
        st.rerun()
    
    if quick_period_cols2[1].button("近2年", use_container_width=True):
        st.session_state.date_range['start'] = datetime.now() - pd.DateOffset(years=2)
        st.session_state.date_range['end'] = datetime.now()
        st.rerun()
    
    if quick_period_cols2[2].button("2024至今", use_container_width=True):
        st.session_state.date_range['start'] = datetime(2024, 1, 1)
        st.session_state.date_range['end'] = datetime.now()
        st.rerun()
    
    # 日期选择器
    col_date1, col_date2 = st.sidebar.columns(2)
    
    with col_date1:
        start_date = st.date_input(
            "开始日期",
            value=st.session_state.date_range['start'],
            min_value=datetime(2020, 1, 1),
            max_value=datetime.now(),
            help="选择数据开始日期",
            key='start_date_input'
        )
        st.session_state.date_range['start'] = datetime.combine(start_date, datetime.min.time())
    
    with col_date2:
        end_date = st.date_input(
            "结束日期",
            value=st.session_state.date_range['end'],
            min_value=datetime(2020, 1, 1),
            max_value=datetime.now(),
            help="选择数据结束日期",
            key='end_date_input'
        )
        st.session_state.date_range['end'] = datetime.combine(end_date, datetime.min.time())
    
    # 日期验证
    if st.session_state.date_range['start'] >= st.session_state.date_range['end']:
        st.sidebar.error("⚠️ 开始日期必须早于结束日期")
    
    # 显示时间跨度
    date_diff = (st.session_state.date_range['end'] - st.session_state.date_range['start']).days
    st.sidebar.info(f"📊 数据跨度: **{date_diff}天** ({date_diff//30}个月)")
    
    if date_diff < 60:
        st.sidebar.warning("⚠️ 建议选择至少2个月的数据以提高预测准确性")
    
    # 转换为字符串格式
    start_date_str = st.session_state.date_range['start'].strftime('%Y%m%d')
    end_date_str = st.session_state.date_range['end'].strftime('%Y%m%d')
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ 预测方法")
    
    prediction_method = st.sidebar.radio(
        "选择MA预测方法:",
        options=['both', 'ml', 'rule'],
        format_func=lambda x: {
            'both': '🔄 两种方法对比（推荐）',
            'ml': '🤖 仅机器学习模型',
            'rule': '📊 仅规则技术分析'
        }[x],
        index=0
    )
    
    # MA周期选择策略
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 MA周期策略")
    
    ma_strategy = st.sidebar.radio(
        "选择MA周期策略:",
        options=['dynamic', 'ma10', 'ma20'],
        format_func=lambda x: {
            'dynamic': '🎯 动态选择（推荐）',
            'ma10': '📊 统一MA10（短期敏感）',
            'ma20': '📈 统一MA20（中期稳定）'
        }[x],
        index=0,
        help="动态选择：1-3天用MA10，5-10天用MA20"
    )
    
    # 显示当前策略说明
    if ma_strategy == 'dynamic':
        st.sidebar.info("📌 1-3天预测用MA10\n📌 5-10天预测用MA20")
    elif ma_strategy == 'ma10':
        st.sidebar.info("📌 所有预测窗口统一使用MA10\n✅ 更敏感，适合短期交易")
    else:
        st.sidebar.info("📌 所有预测窗口统一使用MA20\n✅ 更稳定，假信号较少")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 使用说明")
    st.sidebar.info("""
    1. 选择数据时间段（默认2024年至今）
    2. 输入股票代码（支持单个或多个）
       - 单个: 600519
       - 多个: 600519,000001,002415
       - 或用空格分隔: 600519 000001
    3. 选择预测方法和MA策略
    4. 点击"开始预测"按钮
    5. 查看多时间窗口预测结果
    6. 支持1天、3天、5天、10天预测
    
    💡 提示：
    - 支持批量分析多只股票 ✨
    - 建议至少选择3个月以上的数据
    - 数据越充分，预测越准确
    """)
    
    # 主页面
    st.title("📈 股票多时间窗口预测系统")
    st.markdown("### 结合MA10/MA20状态 + 形态信号 + 多时间窗口验证")
    
    # 创建预测器
    if 'predictor' not in st.session_state:
        st.session_state.predictor = StreamlitPredictor()
        st.session_state.predictor.load_models()
    
    predictor = st.session_state.predictor
    
    # 输入区域
    col1, col2 = st.columns([3, 1])
    
    with col1:
        stock_codes_input = st.text_input(
            "请输入股票代码（支持多个）",
            value="600519",
            placeholder="单个: 600519  或  多个: 600519,000001,002415  或  600519 000001 002415",
            help="输入单个或多个6位股票代码，用逗号、空格或分号分隔"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🚀 开始预测", use_container_width=True)
    
    # 常用股票快速选择
    st.markdown("### 📌 快速选择")
    quick_stocks = {
        "贵州茅台": "600519",
        "平安银行": "000001",
        "招商银行": "600036",
        "五粮液": "000858",
        "海康威视": "002415",
        "宁德时代": "300750"
    }
    
    cols = st.columns(len(quick_stocks))
    for i, (name, code) in enumerate(quick_stocks.items()):
        if cols[i].button(f"{name}\n{code}", use_container_width=True):
            stock_codes_input = code
            predict_button = True
    
    st.markdown("---")
    
    # 执行预测
    if predict_button and stock_codes_input:
        # 解析股票代码（支持逗号、空格、分号分隔）
        import re
        stock_codes = re.split(r'[,，;\s]+', stock_codes_input.strip())
        stock_codes = [code.strip() for code in stock_codes if code.strip()]
        
        # 验证股票代码
        valid_codes = []
        invalid_codes = []
        for code in stock_codes:
            if code.isdigit() and len(code) == 6:
                valid_codes.append(code)
            else:
                invalid_codes.append(code)
        
        if invalid_codes:
            st.warning(f"⚠️ 以下股票代码格式不正确，已忽略: {', '.join(invalid_codes)}")
        
        if not valid_codes:
            st.error("❌ 没有有效的股票代码，请输入正确的6位股票代码")
            return
        
        # 显示处理信息
        if len(valid_codes) == 1:
            st.info(f"📊 正在分析 1 只股票...")
        else:
            st.info(f"📊 正在批量分析 {len(valid_codes)} 只股票: {', '.join(valid_codes)}")
        
        # 处理多个股票
        results_data = []
        
        for idx, stock_code in enumerate(valid_codes):
            with st.spinner(f"正在处理 {stock_code} ({idx+1}/{len(valid_codes)})..."):
                try:
                    # 获取股票数据
                    stock_data = predictor.get_stock_data(stock_code, start_date_str, end_date_str)
                    
                    if stock_data is None or stock_data.empty:
                        st.warning(f"⚠️ 无法获取 {stock_code} 的数据，跳过")
                        continue
                    
                    # 执行预测
                    ma20_preds = predictor.predict_ma20(stock_data, method=prediction_method, ma_strategy=ma_strategy)
                    pattern_preds = predictor.predict_pattern(stock_data)
                    decisions = predictor.make_decision(ma20_preds, pattern_preds)
                    
                    results_data.append({
                        'code': stock_code,
                        'data': stock_data,
                        'ma20_preds': ma20_preds,
                        'pattern_preds': pattern_preds,
                        'decisions': decisions
                    })
                    
                    # 保存到历史
                    if 'history' not in st.session_state:
                        st.session_state.history = []
                    
                    st.session_state.history.append({
                        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'stock_code': stock_code,
                        'decisions': decisions
                    })
                    
                except Exception as e:
                    st.error(f"❌ 处理 {stock_code} 时出错: {str(e)}")
                    continue
        
        # 显示结果
        if not results_data:
            st.error("❌ 没有成功获取任何股票数据")
            return
        
        st.success(f"✅ 成功分析 {len(results_data)} 只股票")
        
        # 如果只有一个股票，直接显示
        if len(results_data) == 1:
            result = results_data[0]
            display_prediction_results(
                result['code'], 
                result['ma20_preds'], 
                result['pattern_preds'], 
                result['decisions'], 
                result['data'],
                start_date_str,
                end_date_str
            )
        else:
            # 多个股票用标签页显示
            st.markdown("---")
            st.markdown("## 📊 批量分析结果")
            
            # 创建概览表格
            st.markdown("### 📋 快速概览")
            overview_data = []
            for result in results_data:
                code = result['code']
                decisions = result['decisions']
                data = result['data']
                
                # 统计各时间窗口的建议
                levels = [decisions.get(h, {}).get('level', 'N/A') for h in [1, 3, 5, 10]]
                
                overview_data.append({
                    '股票代码': code,
                    '当前价格': f"¥{data['Close'].iloc[-1]:.2f}",
                    '涨跌幅': f"{data['PriceChangeRate'].iloc[-1]:.2f}%" if 'PriceChangeRate' in data.columns else 'N/A',
                    '1天': levels[0],
                    '3天': levels[1],
                    '5天': levels[2],
                    '10天': levels[3]
                })
            
            df_overview = pd.DataFrame(overview_data)
            st.dataframe(df_overview, use_container_width=True)
            
            # 为每个股票创建标签页
            st.markdown("### 📈 详细分析")
            tabs = st.tabs([f"📊 {result['code']}" for result in results_data])
            
            for tab, result in zip(tabs, results_data):
                with tab:
                    display_prediction_results(
                        result['code'], 
                        result['ma20_preds'], 
                        result['pattern_preds'], 
                        result['decisions'], 
                        result['data'],
                        start_date_str,
                        end_date_str
                    )
    
    # 显示历史记录
    if 'history' in st.session_state and st.session_state.history:
        st.markdown("---")
        st.subheader("📜 预测历史")
        
        for record in reversed(st.session_state.history[-5:]):  # 只显示最近5条
            with st.expander(f"{record['time']} - {record['stock_code']}"):
                for horizon, d in record['decisions'].items():
                    st.write(f"**{horizon}天**: {d['level']} | {d['action']} (置信度: {d['avg_confidence']:.1%})")
    
    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p>📱 支持移动端访问 | 💡 实时数据更新 | 🔐 本地运行安全</p>
        <p>⚠️ 风险提示：预测结果仅供参考，投资需谨慎</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
