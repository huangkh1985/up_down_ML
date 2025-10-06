# 股票形态信号预测指南

## 🎯 功能概述

本系统专门用于**预测反转、回调、反弹信号的发生**，使用信号前60天的数据作为自变量。

### 核心思想

```
时间线: [第1天 ——————————> 第60天] → [第61天: 信号发生？]
                  ↑
              提取这60天的特征
              作为自变量(X)
                                        ↑
                                   目标变量(y)
                                   1 = 信号发生
                                   0 = 信号未发生
```

---

## 📊 预测目标

### 1. 反转信号预测（默认）
- **目标**: 预测反转信号（牛市反转 + 熊市反转）
- **应用**: 把握趋势转折点
- **标签定义**: 
  - `1` = 反转信号发生（从下跌转上涨 或 从上涨转下跌）
  - `0` = 无反转信号

### 2. 回调信号预测
- **目标**: 预测回调信号
- **应用**: 在上升趋势中寻找买入机会
- **标签定义**:
  - `1` = 回调信号发生
  - `0` = 无回调信号

### 3. 反弹信号预测
- **目标**: 预测反弹信号  
- **应用**: 在下降趋势中寻找卖出时机
- **标签定义**:
  - `1` = 反弹信号发生
  - `0` = 无反弹信号

---

## 🚀 快速开始

### 方式1: 使用默认配置（推荐）

```bash
# 1. 确保已下载股票数据
python stock_data_downloader.py

# 2. 运行反转信号预测
python stock_pattern_prediction.py
```

### 方式2: 自定义配置

编辑 `stock_pattern_prediction.py` 的 `main()` 函数：

```python
# 第2步：选择要预测的信号类型
signal_type = 'reversal'  # 可选: 'reversal', 'pullback', 'bounce'
lookback_days = 60         # 回看天数（可调整为30、90等）
```

---

## 📈 数据结构说明

### 自变量 (X) - 信号前60天的统计特征

对每个技术指标，提取以下统计量：

| 特征类型 | 说明 | 示例 |
|---------|------|------|
| `*_mean` | 60天均值 | `Close_mean`, `RSI_mean` |
| `*_std` | 60天标准差 | `Volume_std`, `MACD_std` |
| `*_min` | 60天最小值 | `Low_min`, `RSI_min` |
| `*_max` | 60天最大值 | `High_max`, `Volume_max` |
| `*_last` | 最后一天的值 | `Close_last`, `MA20_last` |
| `*_trend` | 趋势（首尾变化率） | `Close_trend`, `RSI_trend` |
| `*_momentum` | 动量（最近5天变化） | `Close_momentum` |

**包含的原始指标:**
- 基础价格: `Close`, `Volume`, `TurnoverRate`, `PriceChangeRate`
- 均线: `MA5`, `MA10`, `MA20`, `MA50`
- 技术指标: `RSI`, `MACD`, `Signal`, `ATR`, `ADX`, `K`, `D`, `J`
- 成交量: `OBV`, `MFI`, `VWAP`
- 波动率: `Volatility`, `HV_20`, `BB_Width`
- 形态特征: `Trend`, `Drawdown_From_High`, `Rally_From_Low`

**总特征数**: 约 **200-300个** （取决于可用指标）

### 目标变量 (y) - 信号标签

```python
y = [0, 0, 1, 0, 0, 1, 0, ...]
#    ↑        ↑        ↑
#    无信号   信号发生  无信号
```

---

## 📊 完整工作流程

### 步骤1: 数据加载
```python
# 加载股票数据
with open('data/all_stock_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

# 示例输出:
# ✅ 成功加载 30 只股票的数据
```

### 步骤2: 数据准备
```python
# 为每只股票提取形态特征
X, y, sample_info = prepare_reversal_prediction_data(
    all_data, 
    lookback_days=60
)

# 示例输出:
# 总样本数: 5000
# 反转信号样本: 200 (4.00%)
# 非反转信号样本: 4800 (96.00%)
# 特征数: 250
```

### 步骤3: 数据分割
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # 保持类别比例
)
```

### 步骤4: 模型训练
```python
model, y_pred, y_proba, metrics = train_signal_prediction_model(
    X_train, X_test, y_train, y_test, 
    signal_type='reversal'
)

# 自动处理类别不平衡（SMOTE过采样）
# 训练Random Forest模型
```

### 步骤5: 结果评估
```python
# 模型性能指标
准确率: 85.32%
精确率: 72.50%  # 预测的信号中，有72.5%是对的
召回率: 68.20%  # 真实信号中，识别出了68.2%
```

### 步骤6: 保存结果
```
results/
  ├── reversal_prediction_results.csv  # 详细预测结果
  ├── reversal_features.csv            # 特征矩阵
models/
  └── reversal_prediction_model.pkl    # 训练好的模型
```

---

## 📁 输出文件说明

### 1. `reversal_prediction_results.csv`

详细的预测结果，每行一个样本：

| 列名 | 说明 |
|-----|------|
| `date` | 预测日期 |
| `stock_code` | 股票代码 |
| `signal` | 真实信号（1/0） |
| `close_price` | 当天收盘价 |
| `true_signal` | 真实标签 |
| `pred_signal` | 预测标签 |
| `signal_probability` | 信号概率（0-1） |
| `prediction_correct` | 预测是否正确 |

**示例数据:**
```csv
date,stock_code,signal,close_price,true_signal,pred_signal,signal_probability,prediction_correct
2024-05-15,600519,1,1850.5,1,1,0.85,1
2024-05-16,600519,0,1845.2,0,0,0.12,1
2024-05-17,000001,1,12.35,1,0,0.45,0
```

### 2. `reversal_features.csv`

特征矩阵，每行一个样本，列为所有特征：

```csv
Close_mean,Close_std,Close_min,Close_max,RSI_mean,RSI_std,...
1845.3,15.2,1820.5,1870.8,62.5,8.3,...
12.4,0.3,12.0,12.8,45.2,12.1,...
```

### 3. `reversal_prediction_model.pkl`

保存的模型文件，包含：
- 训练好的模型
- 特征列表
- 性能指标
- 训练日期
- 回看天数

---

## 💡 使用示例

### 示例1: 基本预测流程

```python
from stock_pattern_prediction import prepare_reversal_prediction_data, train_signal_prediction_model
import pickle

# 1. 加载数据
with open('data/all_stock_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

# 2. 准备数据
X, y, sample_info = prepare_reversal_prediction_data(all_data, lookback_days=60)

# 3. 分割数据
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 训练模型
model, y_pred, y_proba, metrics = train_signal_prediction_model(
    X_train, X_test, y_train, y_test, 'reversal'
)

# 5. 查看结果
print(f"准确率: {metrics['accuracy']:.2%}")
print(f"精确率: {metrics['precision']:.2%}")
print(f"召回率: {metrics['recall']:.2%}")
```

### 示例2: 加载已训练模型进行预测

```python
import pickle
import pandas as pd

# 加载模型
with open('models/reversal_prediction_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_columns = model_data['feature_columns']

# 准备新数据（需要包含相同的特征）
new_X = pd.read_csv('new_data.csv')[feature_columns]

# 预测
predictions = model.predict(new_X)
probabilities = model.predict_proba(new_X)[:, 1]

# 找出高置信度的信号
high_conf_signals = probabilities > 0.7
print(f"高置信度信号数: {high_conf_signals.sum()}")
```

### 示例3: 分析特征重要性

```python
import pickle

# 加载模型
with open('models/reversal_prediction_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# 查看特征重要性
importance_df = model_data['metrics']['feature_importance']
print("\nTop 10 最重要特征:")
print(importance_df.head(10))

# 可视化
import matplotlib.pyplot as plt
importance_df.head(20).plot(x='feature', y='importance', kind='barh', figsize=(10, 8))
plt.title('特征重要性排名')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

### 示例4: 预测不同类型的信号

```python
# 预测回调信号
X_pullback, y_pullback, info_pullback = prepare_pullback_prediction_data(
    all_data, lookback_days=60
)

# 预测反弹信号
X_bounce, y_bounce, info_bounce = prepare_bounce_prediction_data(
    all_data, lookback_days=60
)

# 分别训练模型...
```

---

## ⚙️ 参数调整

### 调整回看天数

```python
# 在 main() 函数中修改
lookback_days = 30   # 短期：30天
lookback_days = 60   # 中期：60天（默认）
lookback_days = 90   # 长期：90天
```

**影响:**
- 更短: 捕捉短期信号，但特征信息可能不足
- 更长: 特征更丰富，但可能过度平滑

### 调整模型参数

在 `train_signal_prediction_model()` 函数中：

```python
model = RandomForestClassifier(
    n_estimators=500,      # 树的数量：越多越好，但速度慢
    max_depth=15,          # 树的深度：防止过拟合
    min_samples_split=10,  # 分裂最小样本数
    min_samples_leaf=5,    # 叶节点最小样本数
    max_features='sqrt',   # 每次分裂考虑的特征数
    class_weight='balanced', # 类别权重策略
    random_state=42,
    n_jobs=-1
)
```

### 调整阈值

```python
# 在使用模型预测时
threshold = 0.7  # 默认0.5，提高到0.7更保守

# 高置信度预测
high_conf_predictions = (y_proba > threshold).astype(int)
```

---

## 📊 性能评估指标

### 理解指标

| 指标 | 含义 | 理想值 | 实际意义 |
|-----|------|--------|---------|
| **准确率** | 预测正确的比例 | >80% | 整体表现 |
| **精确率** | 预测为信号中真正是信号的比例 | >70% | 信号可靠性 |
| **召回率** | 真实信号中被识别出的比例 | >65% | 信号覆盖率 |

### 应用建议

**保守型策略（高精确率）:**
```python
# 只采纳高置信度的信号
reliable_signals = results_df[
    (results_df['pred_signal'] == 1) & 
    (results_df['signal_probability'] > 0.75)
]
```

**激进型策略（高召回率）:**
```python
# 采纳所有预测的信号
all_signals = results_df[results_df['pred_signal'] == 1]
```

---

## 🔧 常见问题

### Q1: 为什么类别不平衡？
**A:** 信号事件通常较少（5-10%），这是正常的。系统会自动使用SMOTE处理。

### Q2: 准确率很高但实际效果不好？
**A:** 可能是因为样本不平衡导致。关注**精确率**和**召回率**更重要。

### Q3: 如何提高预测性能？
**A:** 
1. 增加训练数据（更多股票）
2. 调整回看天数
3. 添加更多特征
4. 尝试不同的模型算法

### Q4: 可以预测多种信号吗？
**A:** 可以！分别训练三个模型：
- `reversal_prediction_model.pkl`
- `pullback_prediction_model.pkl`
- `bounce_prediction_model.pkl`

---

## 📚 与原有系统的区别

| 对比项 | 原系统 (stock_statistical_analysis.py) | 新系统 (stock_pattern_prediction.py) |
|-------|--------------------------------------|--------------------------------------|
| **目标** | 预测价格与MA20关系 | 预测形态信号发生 |
| **自变量** | 单时点的TSFresh特征 | 信号前60天的统计特征 |
| **目标变量** | 价格>=MA20(0) vs <MA20(1) | 信号发生(1) vs 未发生(0) |
| **应用** | 判断股票强弱 | 捕捉交易时机 |
| **特征数** | ~100个TSFresh特征 | ~200-300个统计特征 |

**两者可以配合使用！**

---

## 🎯 实战应用流程

### 完整交易策略

1. **信号预测** → 使用 `stock_pattern_prediction.py`
   - 预测未来是否出现反转信号
   
2. **信号验证** → 结合实时数据
   - 观察信号前60天的特征是否匹配

3. **风险评估** → 使用 `stock_statistical_analysis.py`
   - 预测信号后股价相对MA20的位置

4. **决策执行**
   - 反转信号 + 高置信度 + 预期强势 → 买入
   - 反转信号 + 高置信度 + 预期弱势 → 卖出

---

## 💾 完整代码示例

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的形态信号预测示例
"""

import pickle
import pandas as pd
from stock_pattern_prediction import (
    prepare_reversal_prediction_data,
    train_signal_prediction_model
)
from sklearn.model_selection import train_test_split

# 1. 加载数据
print("加载股票数据...")
with open('data/all_stock_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

# 2. 准备反转信号预测数据
print("准备反转信号数据...")
X, y, sample_info = prepare_reversal_prediction_data(
    all_data, 
    lookback_days=60
)

# 3. 分割数据
print("分割训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# 4. 训练模型
print("训练预测模型...")
model, y_pred, y_proba, metrics = train_signal_prediction_model(
    X_train, X_test, y_train, y_test,
    signal_type='reversal'
)

# 5. 分析结果
print("\n预测性能:")
print(f"准确率: {metrics['accuracy']:.2%}")
print(f"精确率: {metrics['precision']:.2%}")
print(f"召回率: {metrics['recall']:.2%}")

# 6. 找出高置信度信号
high_conf_mask = y_proba > 0.7
print(f"\n高置信度信号数: {high_conf_mask.sum()}")

if high_conf_mask.sum() > 0:
    test_info = sample_info.iloc[X_test.index]
    high_conf_info = test_info[high_conf_mask].copy()
    high_conf_info['probability'] = y_proba[high_conf_mask]
    
    print("\n高置信度信号详情:")
    print(high_conf_info[['date', 'stock_code', 'close_price', 'probability']])

# 7. 保存模型
print("\n保存模型...")
with open('models/my_reversal_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'metrics': metrics,
        'feature_columns': X.columns.tolist()
    }, f)

print("✅ 完成！")
```

---

## 🎓 总结

这个系统提供了一个完整的框架来预测股票形态信号，关键优势：

1. ✅ **自动化**: 自动提取信号前60天特征
2. ✅ **灵活**: 支持多种信号类型（反转/回调/反弹）
3. ✅ **可调**: 回看天数、模型参数都可调整
4. ✅ **实用**: 处理类别不平衡、提供置信度评分
5. ✅ **完整**: 从数据准备到模型保存全流程

**下一步**: 结合实时数据，将预测结果应用到实际交易中！

---

**最后更新**: 2025-10-05  
**版本**: v1.0  
**状态**: ✅ 已测试
