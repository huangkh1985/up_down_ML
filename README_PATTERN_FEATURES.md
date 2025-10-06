# 股票技术形态识别系统

## 🎯 新功能概述

本系统已成功添加**自动识别股票反转、回调、反弹**等技术形态的功能，并将其提取为可用于统计分析的自变量。

### ✅ 测试结果
```
测试数据: 500天历史数据
识别形态数量: 244个样本
- 回调形态: 124个
- 反弹形态: 79个
- 反转形态: 41个
- V型反转: 8个
- 双顶/双底: 2个

提取特征: 31个形态相关特征
数据完整性: ✅ 无缺失值
功能状态: ✅ 正常工作
```

---

## 📦 新增文件

### 核心模块
1. **`utils/pattern_recognition.py`** - 形态识别核心算法
   - 反转点识别
   - 回调/反弹识别
   - V型反转识别
   - 双顶/双底识别
   - 形态统计分析

### 演示脚本
2. **`pattern_analysis_demo.py`** - 完整的形态分析演示
   - 自动加载股票数据
   - 批量提取形态特征
   - 生成可视化图表
   - 导出分析结果

3. **`test_pattern_recognition.py`** - 功能测试脚本
   - 快速验证功能
   - 生成测试报告
   - 无需真实数据

### 文档
4. **`PATTERN_FEATURES_GUIDE.md`** - 详细使用指南
   - 形态定义说明
   - 特征列表
   - 使用示例
   - 统计分析方法

5. **`STOCK_FILTER_CONFIG.md`** - 股票筛选配置
   - ST股过滤
   - 次新股过滤
   - 数据质量控制

---

## 🚀 快速开始

### 方式1: 使用现有数据（推荐首次测试）

```bash
# 1. 测试形态识别功能
python test_pattern_recognition.py

# 输出: 
# - 测试报告（控制台）
# - test_pattern_result.csv（测试结果）
```

### 方式2: 使用真实股票数据

```bash
# 1. 下载股票数据（已自动集成形态识别）
python stock_data_downloader.py

# 2. 运行形态分析
python pattern_analysis_demo.py

# 输出:
# - results/pattern_features.csv（特征矩阵）
# - results/pattern_samples_full.csv（完整样本）
# - results/pattern_effectiveness.csv（形态统计）
# - results/pattern_analysis_*.png（可视化图表）
```

### 方式3: 在代码中使用

```python
from utils.pattern_recognition import add_pattern_features
import pandas as pd

# 加载你的股票数据
df = pd.read_csv('your_stock_data.csv')

# 自动识别所有形态
df_with_patterns = add_pattern_features(df)

# 提取回调样本
pullback_samples = df_with_patterns[df_with_patterns['Is_Pullback'] == 1]
print(f"找到 {len(pullback_samples)} 个回调形态")

# 提取反转样本
reversal_samples = df_with_patterns[
    (df_with_patterns['Bullish_Reversal'] == 1) | 
    (df_with_patterns['Bearish_Reversal'] == 1)
]
print(f"找到 {len(reversal_samples)} 个反转形态")
```

---

## 📊 识别的形态类型

| 形态类型 | 说明 | 关键特征 | 应用场景 |
|---------|------|---------|---------|
| **牛市反转** | 从下跌转为上涨 | `Bullish_Reversal` | 买入信号 |
| **熊市反转** | 从上涨转为下跌 | `Bearish_Reversal` | 卖出信号 |
| **回调** | 上升趋势中的回撤 | `Is_Pullback`, `Pullback_Depth` | 加仓机会 |
| **反弹** | 下降趋势中的反弹 | `Is_Bounce`, `Bounce_Height` | 减仓机会 |
| **V型反转** | 快速反转 | `V_Reversal_Bullish/Bearish` | 强势信号 |
| **双顶/双底** | 经典反转形态 | `Double_Top/Bottom` | 趋势终结 |

---

## 🔧 提取的特征（31个）

### 反转特征 (8个)
```python
- Bullish_Reversal         # 牛市反转标记
- Bearish_Reversal         # 熊市反转标记
- V_Reversal_Bullish       # V型底部
- V_Reversal_Bearish       # 倒V型顶部
- Double_Top               # 双顶形态
- Double_Bottom            # 双底形态
- Reversal_Strength        # 反转强度
- Days_Since_Reversal      # 距上次反转天数
```

### 回调特征 (7个)
```python
- Is_Pullback              # 回调标记
- Pullback_Depth           # 回调深度
- Pullback_Days            # 回调天数
- Pullback_Recovery        # 回调恢复标记
- Pullback_Frequency       # 回调频率
- Avg_Pullback_Depth       # 平均回调深度
- Pullback_Success_Rate    # 回调成功率
```

### 反弹特征 (7个)
```python
- Is_Bounce                # 反弹标记
- Bounce_Height            # 反弹高度
- Bounce_Days              # 反弹天数
- Bounce_End               # 反弹结束标记
- Bounce_Frequency         # 反弹频率
- Avg_Bounce_Height        # 平均反弹高度
- Bounce_Success_Rate      # 反弹成功率
```

### 辅助特征 (9个)
```python
- Trend                    # 当前趋势
- Peak                     # 局部顶部
- Trough                   # 局部底部
- Recent_High              # 近期最高价
- Recent_Low               # 近期最低价
- Drawdown_From_High       # 从高点回撤
- Rally_From_Low           # 从低点反弹
- Pullback_Success         # 回调成功标记
- Bounce_Success           # 反弹成功标记
```

---

## 📈 作为统计分析自变量

### 1. 构建特征矩阵

```python
import pandas as pd

# 加载带形态的数据
df = pd.read_csv('results/pattern_samples_full.csv')

# 选择自变量
X_features = [
    'Pullback_Depth', 'Bounce_Height', 'Reversal_Strength',
    'Pullback_Frequency', 'Bounce_Frequency',
    'Days_Since_Reversal', 'Trend',
    'RSI', 'MACD', 'Volume'  # 配合技术指标
]

X = df[X_features]

# 构建因变量（例如：未来收益率）
y = df['Close'].pct_change(5).shift(-5)  # 未来5日收益

# 统计分析
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

print("特征重要性:")
for feature, coef in zip(X_features, model.coef_):
    print(f"  {feature}: {coef:.6f}")
```

### 2. 相关性分析

```python
# 计算形态特征与收益的相关性
correlation = df[X_features + ['y']].corr()['y'].sort_values()

print("与未来收益的相关性:")
print(correlation)
```

### 3. 分组比较

```python
# 比较不同形态的后续表现
groups = df.groupby('Pattern_Type')['Future_Return'].agg(['mean', 'std', 'count'])
print(groups)
```

---

## 🎨 可视化示例

系统自动生成以下可视化：

1. **价格图 + 反转点标注**
   - 绿色三角：牛市反转
   - 红色倒三角：熊市反转

2. **回调区域标注**
   - 橙色区域：回调发生时段

3. **反弹区域标注**
   - 青色区域：反弹发生时段

4. **形态频率变化**
   - 回调/反弹频率的时间序列

---

## 💡 实际应用示例

### 示例1: 寻找高质量回调买入机会

```python
# 筛选条件：
# 1. 正在回调
# 2. RSI超卖
# 3. 上升趋势
# 4. 历史回调成功率高
buy_signals = df[
    (df['Is_Pullback'] == 1) &
    (df['RSI'] < 40) &
    (df['Trend'] == 1) &
    (df['Pullback_Success_Rate'] > 0.6)
]

print(f"找到 {len(buy_signals)} 个买入信号")
print(f"平均后续收益: {buy_signals['Future_Return'].mean():.2%}")
```

### 示例2: 评估反转信号可靠性

```python
# 分析不同RSI区间的反转准确率
rsi_groups = pd.cut(df['RSI'], bins=[0, 30, 70, 100], labels=['超卖', '中性', '超买'])
accuracy = df.groupby(rsi_groups)['Reversal_Strength'].apply(lambda x: (abs(x) > 0.05).mean())

print("不同RSI区间的反转准确率:")
print(accuracy)
```

### 示例3: 回调深度与成功率分析

```python
# 回调深度分组
depth_bins = [0, 0.05, 0.10, 0.15, 0.20, 1.0]
labels = ['0-5%', '5-10%', '10-15%', '15-20%', '>20%']
df['Depth_Group'] = pd.cut(df['Pullback_Depth'], bins=depth_bins, labels=labels)

# 统计各组的成功率和后续收益
stats = df.groupby('Depth_Group').agg({
    'Pullback_Success': 'mean',
    'Future_Return': 'mean',
    'Pullback_Days': 'mean'
})

print("不同回调深度的统计:")
print(stats)
```

---

## ⚙️ 配置和调整

### 调整形态识别阈值

```python
from utils.pattern_recognition import identify_pullback, identify_bounce

# 更敏感的回调识别（3%阈值）
df = identify_pullback(df, pullback_threshold=0.03, trend_window=20)

# 更保守的反弹识别（8%阈值）
df = identify_bounce(df, bounce_threshold=0.08, trend_window=30)
```

### 在数据下载时禁用形态识别

```python
# 在 stock_data_downloader.py 中
df = add_technical_indicators(df, include_patterns=False)
```

---

## 📁 输出文件说明

| 文件 | 内容 | 用途 |
|------|------|------|
| `pattern_features.csv` | 特征矩阵 | 机器学习训练 |
| `pattern_samples_full.csv` | 完整形态样本 | 统计分析 |
| `pattern_effectiveness.csv` | 形态有效性统计 | 策略评估 |
| `pattern_analysis_*.png` | 可视化图表 | 直观分析 |
| `test_pattern_result.csv` | 测试结果 | 功能验证 |

---

## 🔍 常见问题

### Q1: 识别不到形态？
**A:** 检查数据长度（至少100天）和价格波动（不能全是0）

### Q2: 如何调整敏感度？
**A:** 修改 `pullback_threshold` 和 `bounce_threshold` 参数

### Q3: 可以用于期货/外汇吗？
**A:** 可以，只要数据包含 OHLCV 字段

### Q4: 形态特征可以和其他特征混合使用吗？
**A:** 可以，推荐与传统技术指标（RSI、MACD等）结合

---

## 📚 相关文档

- **详细指南**: `PATTERN_FEATURES_GUIDE.md`
- **股票筛选**: `STOCK_FILTER_CONFIG.md`
- **核心代码**: `utils/pattern_recognition.py`
- **演示脚本**: `pattern_analysis_demo.py`

---

## 🎯 下一步

1. ✅ 已完成：形态识别和特征提取
2. ✅ 已完成：统计分析框架
3. 🔄 进行中：特征工程集成
4. 📋 待完成：基于形态的交易策略回测
5. 📋 待完成：多股票形态联动分析

---

## 💬 反馈与改进

如需调整形态识别算法或添加新的形态类型，请编辑：
- `utils/pattern_recognition.py`

如需调整可视化效果，请编辑：
- `pattern_analysis_demo.py`

---

**最后更新**: 2025-10-05  
**版本**: v1.0  
**状态**: ✅ 测试通过，功能正常
