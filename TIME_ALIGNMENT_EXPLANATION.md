# 两种预测方法的时间对齐说明

## 🕐 当前时间设置分析

### 方法1: MA20状态分类（`stock_statistical_analysis.py`）

**时间线**:
```
[第i-20天 ——————> 第i天]  →  预测  →  [第i+5天的状态]
      ↑                                    ↑
   使用这20天的数据                   预测5天后
   提取TSFresh特征                    价格与MA20关系
```

**代码位置**: `stock_feature_engineering.py` 第111行
```python
future_close = float(data['Close'].iloc[i + forecast_horizon])
# forecast_horizon = 5，所以预测的是5天后
```

**预测目标**: 第i+5天，价格>=MA20(强势) 还是 <MA20(弱势)

---

### 方法2: 形态信号预测（`stock_pattern_prediction.py`）⚠️ 需要修正

**当前实现**:
```
[第i-60天 ——————> 第i天]  →  预测  →  [第i天的信号] ❌ 错误！
      ↑                                    ↑
   使用这60天的数据                    预测当天（已知）
```

**代码位置**: `stock_pattern_prediction.py` 第81行
```python
is_signal = df[signal_column].iloc[i]
# 这是当天的信号，不是未来的！
```

**问题**: 预测的是"当天"而不是"未来"，这样没有实用价值！

---

## ✅ 修正方案

### 统一预测目标：预测未来5天

两种方法都应该预测**未来第5天**的状态/信号，保持时间一致。

**修正后的方法2**:
```
[第i-60天 ——————> 第i天]  →  预测  →  [第i+5天的信号] ✓
      ↑                                    ↑
   使用这60天的数据                    预测5天后
                                      是否出现信号
```

---

## 🔧 修正代码

### 修正版：`extract_pre_signal_features()`

关键修改：
```python
# 原代码（错误）：
is_signal = df[signal_column].iloc[i]  # 预测当天

# 修正代码（正确）：
forecast_horizon = 5  # 预测5天后
is_signal = df[signal_column].iloc[i + forecast_horizon]  # 预测5天后
```

完整的修正逻辑：
```python
def extract_pre_signal_features(df, signal_column, lookback_days=60, 
                               forecast_horizon=5, target_type='reversal'):
    """
    提取信号发生前N天的特征数据（时间对齐版本）
    
    参数:
    df: 包含形态特征的DataFrame
    signal_column: 信号列名
    lookback_days: 回看天数（默认60）
    forecast_horizon: 预测未来天数（默认5，与MA20分类一致）
    
    返回:
    X: 特征矩阵（第i-60天到第i天的数据）
    y: 目标变量（第i+5天是否有信号）
    """
    print(f"提取特征预测{forecast_horizon}天后的{signal_column}信号...")
    
    X_samples = []
    y_samples = []
    sample_info = []
    
    # 遍历每一天，但要留出forecast_horizon的空间
    for i in range(lookback_days, len(df) - forecast_horizon):
        # ✅ 关键修正：预测未来第forecast_horizon天的信号
        future_signal = df[signal_column].iloc[i + forecast_horizon]
        
        # 提取当前时点之前lookback_days天的数据
        lookback_data = df.iloc[i-lookback_days:i]
        
        # 计算统计特征
        feature_dict = {}
        for col in available_features:
            values = lookback_data[col].values
            feature_dict[f'{col}_mean'] = np.mean(values)
            feature_dict[f'{col}_std'] = np.std(values)
            # ... 其他特征
        
        X_samples.append(feature_dict)
        y_samples.append(int(future_signal))  # 未来的信号
        
        sample_info.append({
            'current_date': df.index[i],
            'prediction_date': df.index[i + forecast_horizon],
            'signal': future_signal
        })
    
    return X, y, sample_info
```

---

## 🎯 统一后的时间线对比

### 方法1和方法2时间对齐

```
时间轴: ... 第i-60天 ... 第i-20天 ... 第i天 ... 第i+5天 ...

方法1:              [——20天——]              [预测]
                       TSFresh特征            状态

方法2:  [————————60天————————————]          [预测]
              统计特征                        信号
```

**现在两者都预测第i+5天！**

---

## 🔄 两种方法结合使用

### 场景1: 顺序预测（推荐）

```python
# 时间点：第i天（今天）

# 步骤1：用方法2预测5天后是否有反转信号
signal_prob = predict_reversal_signal(
    data_from_i_minus_60_to_i  # 过去60天数据
)

if signal_prob > 0.7:  # 高置信度预测会有信号
    print("预测5天后会出现反转信号")
    
    # 步骤2：用方法1预测5天后的状态
    strength = predict_ma20_position(
        data_from_i_minus_20_to_i  # 过去20天数据
    )
    
    if strength == 0:  # 预测强势
        decision = "买入准备"
        reason = "5天后：反转信号 + 强势状态"
    else:
        decision = "观望"
        reason = "5天后：反转信号但弱势状态"
else:
    decision = "持有"
    reason = "5天后无明显信号"

print(f"决策: {decision}")
print(f"原因: {reason}")
```

### 场景2: 互相验证

```python
# 两个模型独立预测，然后比对
signal_pred = predict_signal(data_60days)      # 预测：有信号
strength_pred = predict_strength(data_20days)  # 预测：强势

# 构建决策矩阵
decision_matrix = {
    (1, 0): "强买入",   # 有信号 + 强势
    (1, 1): "谨慎买入", # 有信号 + 弱势
    (0, 0): "持有",     # 无信号 + 强势
    (0, 1): "卖出"      # 无信号 + 弱势
}

action = decision_matrix[(signal_pred, strength_pred)]
```

---

## 📊 实际应用示例

### 完整的交易决策流程

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时间对齐的综合预测系统
"""

import pickle
import pandas as pd
from datetime import datetime, timedelta

# ========== 配置 ==========
FORECAST_HORIZON = 5  # 统一预测5天后
LOOKBACK_SIGNAL = 60  # 信号预测回看60天
LOOKBACK_STATE = 20   # 状态预测回看20天

# ========== 加载模型 ==========
# 方法2模型（修正后）
with open('models/reversal_prediction_model_aligned.pkl', 'rb') as f:
    signal_model_data = pickle.load(f)
signal_model = signal_model_data['model']

# 方法1模型
with open('models/trained_model.pkl', 'rb') as f:
    state_model = pickle.load(f)

# ========== 获取当前数据 ==========
def get_current_stock_data(stock_code):
    """获取股票当前数据"""
    # 这里应该是实时数据接口
    # 示例：返回最近90天的数据（满足两个模型的需求）
    return stock_data_last_90days

# ========== 预测函数 ==========
def predict_future_5days(stock_code):
    """
    预测5天后的信号和状态
    
    返回:
    - signal_prob: 反转信号概率
    - strength_pred: 状态预测（0=强势, 1=弱势）
    - decision: 交易决策
    """
    # 获取数据
    data = get_current_stock_data(stock_code)
    current_date = data.index[-1]
    prediction_date = current_date + timedelta(days=5)
    
    print(f"\n{'='*60}")
    print(f"股票: {stock_code}")
    print(f"当前日期: {current_date}")
    print(f"预测日期: {prediction_date} (5天后)")
    print(f"{'='*60}")
    
    # ========== 方法2: 预测5天后是否有反转信号 ==========
    print("\n方法2: 预测反转信号...")
    
    # 提取过去60天的统计特征
    data_60days = data.iloc[-60:]
    signal_features = extract_statistics_features(data_60days)
    
    # 预测
    signal_prob = signal_model.predict_proba([signal_features])[0, 1]
    signal_pred = 1 if signal_prob > 0.5 else 0
    
    print(f"  信号概率: {signal_prob:.2%}")
    print(f"  预测结果: {'会出现反转信号' if signal_pred == 1 else '无反转信号'}")
    
    # ========== 方法1: 预测5天后的强弱状态 ==========
    print("\n方法1: 预测状态...")
    
    # 提取过去20天的TSFresh特征
    data_20days = data.iloc[-20:]
    state_features = extract_tsfresh_features(data_20days)
    
    # 预测
    strength_pred = state_model.predict([state_features])[0]
    strength_name = "强势(≥MA20)" if strength_pred == 0 else "弱势(<MA20)"
    
    print(f"  状态预测: {strength_name}")
    
    # ========== 综合决策 ==========
    print("\n综合分析:")
    print(f"  5天后预测: {prediction_date}")
    
    # 决策逻辑
    if signal_prob > 0.7 and strength_pred == 0:
        decision = "强烈买入"
        reason = "高置信度反转信号 + 预期强势"
        confidence = "高"
    elif signal_prob > 0.7 and strength_pred == 1:
        decision = "谨慎观望"
        reason = "有反转信号但预期弱势"
        confidence = "中"
    elif signal_prob > 0.5 and strength_pred == 0:
        decision = "考虑买入"
        reason = "可能有信号 + 预期强势"
        confidence = "中"
    elif signal_prob < 0.3 and strength_pred == 1:
        decision = "考虑卖出"
        reason = "无信号且预期弱势"
        confidence = "中"
    else:
        decision = "持有观望"
        reason = "信号不明确"
        confidence = "低"
    
    print(f"\n{'='*60}")
    print(f"交易决策: {decision}")
    print(f"决策依据: {reason}")
    print(f"置信程度: {confidence}")
    print(f"{'='*60}\n")
    
    return {
        'stock_code': stock_code,
        'current_date': current_date,
        'prediction_date': prediction_date,
        'signal_prob': signal_prob,
        'strength_pred': strength_pred,
        'decision': decision,
        'reason': reason,
        'confidence': confidence
    }

# ========== 批量预测 ==========
def batch_predict(stock_list):
    """批量预测多只股票"""
    results = []
    
    for stock_code in stock_list:
        try:
            result = predict_future_5days(stock_code)
            results.append(result)
        except Exception as e:
            print(f"预测 {stock_code} 失败: {e}")
    
    # 汇总结果
    results_df = pd.DataFrame(results)
    
    # 按决策分组
    print("\n决策汇总:")
    decision_summary = results_df.groupby('decision').size()
    print(decision_summary)
    
    # 保存结果
    results_df.to_csv('results/unified_predictions.csv', index=False)
    print("\n✅ 预测结果已保存到 results/unified_predictions.csv")
    
    return results_df

# ========== 主程序 ==========
if __name__ == '__main__':
    # 示例：预测单只股票
    result = predict_future_5days('600519')
    
    # 示例：批量预测
    # stock_list = ['600519', '000001', '000002']
    # results_df = batch_predict(stock_list)
```

---

## 📋 修改清单

### 需要修改的文件

1. **`stock_pattern_prediction.py`**
   - 修改 `extract_pre_signal_features()` 函数
   - 添加 `forecast_horizon` 参数
   - 修改第81行的信号提取逻辑

2. **创建新文件**
   - `stock_pattern_prediction_aligned.py` - 时间对齐版本
   - `unified_prediction_system.py` - 综合预测系统

3. **文档更新**
   - 更新 `PATTERN_SIGNAL_PREDICTION_GUIDE.md`
   - 添加时间对齐说明

---

## ⚠️ 重要注意事项

### 1. 数据要求

两种方法对数据长度的要求：
```python
# 方法1: 需要至少 20 + 5 = 25 天数据
# 方法2: 需要至少 60 + 5 = 65 天数据

# 综合使用: 至少需要 65 天历史数据
min_data_required = max(LOOKBACK_SIGNAL, LOOKBACK_STATE) + FORECAST_HORIZON
# = max(60, 20) + 5 = 65 天
```

### 2. 预测时间窗口

```python
# 统一设置
FORECAST_HORIZON = 5  # 都预测5天后

# 可以调整为：
FORECAST_HORIZON = 1   # 预测明天
FORECAST_HORIZON = 3   # 预测3天后
FORECAST_HORIZON = 10  # 预测10天后
```

### 3. 实时预测

```python
# 今天（第i天）的预测
今天的数据 → 模型 → 5天后（第i+5天）的结果

# 5天后验证
5天后对比预测值和实际值，评估准确性
```

---

## 🎯 总结

### 关键要点

1. ✅ **统一预测时间**: 两种方法都预测未来5天
2. ✅ **时间对齐**: 确保预测的是同一天
3. ✅ **互相验证**: 信号预测 + 状态预测 = 更可靠的决策
4. ✅ **清晰的时间线**: 明确当前、回看、预测三个时间点

### 使用建议

- **短期交易**: 设置 `forecast_horizon=1` 预测明天
- **中期交易**: 设置 `forecast_horizon=5` 预测5天后（默认）
- **长期投资**: 设置 `forecast_horizon=10` 预测10天后

---

**下一步**: 我将创建修正后的代码文件！
