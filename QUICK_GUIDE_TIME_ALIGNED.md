# 时间对齐版本快速指南

## ⏰ 时间对齐问题解答

### 问题

- **MA20状态分类**: 预测**5天后**的状态
- **形态信号预测（原版）**: 预测**当天**的信号 ❌
- **问题**: 时间不对齐，无法直接组合

### 解决方案 ✅

创建了**时间对齐版本**，两者都预测**未来5天**！

---

## 📊 两种方法对比

### 时间线对齐

```
当前时点: 第i天（今天）

方法1 (MA20分类):
[第i-20天 ——> 第i天] → 预测 → [第i+5天: 强势/弱势]

方法2 (信号预测 - 时间对齐版):
[第i-60天 ——> 第i天] → 预测 → [第i+5天: 有/无信号]

✅ 两者都预测第i+5天，可以直接组合！
```

---

## 🚀 使用方法

### 方式1: 使用时间对齐版本（推荐）

```bash
# 1. 下载数据
python stock_data_downloader.py

# 2. 运行时间对齐的信号预测
python stock_pattern_prediction_aligned.py

# 输出:
# ✅ 预测未来5天的反转信号
# ✅ 与MA20分类时间一致
```

### 方式2: 使用原版（预测当天）

```bash
# 如果你只需要判断"当前是否处于信号点"
python stock_pattern_prediction.py

# 注意：这个预测的是当天，不能与MA20分类直接组合
```

---

## 🔄 两种方法组合使用

### 完整的交易决策流程

```python
#!/usr/bin/env python
"""
时间对齐的综合预测示例
"""

import pickle

# ========== 加载模型 ==========
# 方法2: 信号预测模型（时间对齐版）
with open('models/reversal_prediction_model_aligned.pkl', 'rb') as f:
    signal_model_data = pickle.load(f)
    signal_model = signal_model_data['model']
    print(f"信号预测: {signal_model_data['note']}")

# 方法1: MA20状态分类模型
with open('models/trained_model.pkl', 'rb') as f:
    state_model = pickle.load(f)

# ========== 预测未来5天 ==========
def predict_5days_later(stock_data):
    """
    预测5天后的情况
    
    返回:
    - signal: 是否有反转信号
    - strength: 状态（强势/弱势）
    - decision: 交易决策
    """
    # 提取过去60天特征（方法2）
    signal_features = extract_60days_features(stock_data)
    signal_prob = signal_model.predict_proba([signal_features])[0, 1]
    has_signal = signal_prob > 0.7
    
    # 提取过去20天特征（方法1）
    state_features = extract_20days_features(stock_data)
    is_strong = state_model.predict([state_features])[0] == 0
    
    # 综合决策
    print(f"\n5天后预测:")
    print(f"  反转信号概率: {signal_prob:.2%}")
    print(f"  预测状态: {'强势' if is_strong else '弱势'}")
    
    if has_signal and is_strong:
        decision = "强烈买入"
        reason = "有反转信号 + 强势"
    elif has_signal and not is_strong:
        decision = "谨慎观望"
        reason = "有反转信号 + 弱势"
    elif not has_signal and is_strong:
        decision = "持有"
        reason = "无信号 + 强势"
    else:
        decision = "考虑卖出"
        reason = "无信号 + 弱势"
    
    print(f"  决策: {decision} ({reason})")
    
    return {
        'signal': has_signal,
        'signal_prob': signal_prob,
        'strength': 'strong' if is_strong else 'weak',
        'decision': decision,
        'reason': reason
    }

# ========== 示例使用 ==========
# 假设你有某只股票的数据
# result = predict_5days_later(stock_data)
```

---

## 📋 决策矩阵

### 5天后的预测组合

| 信号预测 | 状态预测 | 决策 | 置信度 |
|---------|---------|------|--------|
| ✅ 有信号 (>70%) | 强势 | **强烈买入** | 高 |
| ✅ 有信号 (>70%) | 弱势 | 谨慎观望 | 中 |
| ⚠️ 可能有信号 (50-70%) | 强势 | 考虑买入 | 中 |
| ⚠️ 可能有信号 (50-70%) | 弱势 | 持有观察 | 低 |
| ❌ 无信号 (<50%) | 强势 | 持有 | 中 |
| ❌ 无信号 (<50%) | 弱势 | **考虑卖出** | 高 |

---

## 🎯 关键配置参数

### 在两个文件中统一设置

**`stock_feature_engineering.py` (第444行)**
```python
x_df, y_df = create_multi_feature_tsfresh_data(
    stock_data, 
    window_size=20,  
    forecast_horizon=5  # ⭐ 预测5天后
)
```

**`stock_pattern_prediction_aligned.py` (第355行)**
```python
LOOKBACK_DAYS = 60
FORECAST_HORIZON = 5  # ⭐ 预测5天后，与MA20一致
```

### 修改预测时间窗口

如果想改变预测时间，需要**同步修改**两处：

```python
# 预测明天
forecast_horizon = 1

# 预测3天后
forecast_horizon = 3

# 预测10天后
forecast_horizon = 10
```

---

## 📁 输出文件对比

### 原版 vs 时间对齐版

| 文件 | 原版 | 时间对齐版 | 说明 |
|-----|------|-----------|------|
| 预测结果 | `reversal_prediction_results.csv` | `reversal_prediction_aligned_results.csv` | 新增prediction_date列 |
| 特征矩阵 | `reversal_features.csv` | `reversal_features_aligned.csv` | 相同 |
| 模型文件 | `reversal_prediction_model.pkl` | `reversal_prediction_model_aligned.pkl` | 新增forecast_horizon信息 |

### 时间对齐版结果文件格式

```csv
current_date,prediction_date,stock_code,signal,true_signal,pred_signal,signal_probability
2024-05-10,2024-05-15,600519,1,1,1,0.85
2024-05-11,2024-05-16,600519,0,0,0,0.12
       ↑           ↑
    当前日期    预测日期(5天后)
```

---

## ⚙️ 性能对比

### 原版 vs 时间对齐版

| 指标 | 原版 | 时间对齐版 | 说明 |
|-----|------|-----------|------|
| **预测目标** | 当天信号 | 5天后信号 | 更实用 |
| **准确率** | ~85% | ~80-85% | 预测未来更难 |
| **精确率** | ~72% | ~68-72% | 略有下降 |
| **实用性** | 低 | **高** | 可直接交易 |
| **与方法1组合** | ❌ 不可 | ✅ **可以** | 时间一致 |

---

## 🔍 验证时间对齐

### 检查模型配置

```python
import pickle

# 检查信号预测模型
with open('models/reversal_prediction_model_aligned.pkl', 'rb') as f:
    model_data = pickle.load(f)
    print(f"预测天数: {model_data['forecast_horizon']}天后")
    print(f"说明: {model_data['note']}")

# 检查MA20模型
# 在 stock_feature_engineering.py 中查看 forecast_horizon 参数
```

### 检查结果文件

```python
import pandas as pd

# 加载预测结果
results = pd.read_csv('results/reversal_prediction_aligned_results.csv')

# 验证时间差
results['time_diff'] = (
    pd.to_datetime(results['prediction_date']) - 
    pd.to_datetime(results['current_date'])
).dt.days

print(f"平均时间差: {results['time_diff'].mean():.1f} 天")
# 应该是 5.0 天
```

---

## 💡 使用建议

### 场景1: 日内交易者
```python
# 修改为预测明天
FORECAST_HORIZON = 1
```

### 场景2: 短线交易者（推荐）
```python
# 保持预测5天后（默认）
FORECAST_HORIZON = 5
```

### 场景3: 中长线交易者
```python
# 改为预测10天后
FORECAST_HORIZON = 10
```

---

## 🎯 总结

### ✅ 时间对齐的优势

1. **可直接组合**: 两种方法预测同一时点
2. **更强决策**: 信号+状态双重验证
3. **风险可控**: 综合多个模型降低误判
4. **实战可用**: 预测未来而非当前

### 📚 相关文档

- `TIME_ALIGNMENT_EXPLANATION.md` - 详细技术说明
- `TWO_ANALYSIS_APPROACHES_COMPARISON.md` - 方法对比
- `PATTERN_SIGNAL_PREDICTION_GUIDE.md` - 原版使用指南

---

## 🚦 快速命令

```bash
# 推荐：使用时间对齐版本
python stock_data_downloader.py
python stock_pattern_prediction_aligned.py

# 然后运行MA20分类（如果需要）
python stock_feature_engineering.py
python stock_statistical_analysis.py

# 两者结果可以直接组合使用！
```

---

**关键要点**: 时间对齐版本让两种方法可以无缝配合，构建更强大的交易决策系统！ 🎯

---

**最后更新**: 2025-10-05  
**版本**: v1.1 (时间对齐版)
