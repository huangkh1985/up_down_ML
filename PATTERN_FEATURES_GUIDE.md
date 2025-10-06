# 股票技术形态特征提取与分析指南

## 📊 概述

本指南介绍如何从股票数据中识别和提取**反转、回调、反弹**等技术形态，并将其作为**统计分析的自变量**使用。

---

## 🎯 识别的形态类型

### 1. **反转形态 (Reversal)**
趋势方向的改变，从上涨转下跌或从下跌转上涨

**子类型:**
- **牛市反转 (Bullish Reversal)**: 从下跌趋势转为上涨
- **熊市反转 (Bearish Reversal)**: 从上涨趋势转为下跌
- **V型反转**: 快速的趋势反转（V型底、倒V型顶）
- **双顶/双底**: 价格两次测试同一价位后反转

**提取的特征:**
```python
- Bullish_Reversal: 牛市反转标记 (0/1)
- Bearish_Reversal: 熊市反转标记 (0/1)
- V_Reversal_Bullish: V型底部反转 (0/1)
- V_Reversal_Bearish: 倒V型顶部反转 (0/1)
- Double_Bottom: 双底形态 (0/1)
- Double_Top: 双顶形态 (0/1)
- Reversal_Strength: 反转后的价格变化强度
- Days_Since_Reversal: 距离上次反转的天数
```

### 2. **回调形态 (Pullback)**
上升趋势中的短期价格回撤

**识别条件:**
- 当前处于上升趋势（MA短期 > MA长期）
- 价格从近期高点回落 5%-20%
- 未跌破关键支撑位

**提取的特征:**
```python
- Is_Pullback: 回调标记 (0/1)
- Pullback_Depth: 回调深度（回撤百分比）
- Pullback_Days: 回调持续天数
- Pullback_Recovery: 回调结束标记 (0/1)
- Pullback_Frequency: 60日内回调次数
- Avg_Pullback_Depth: 平均回调深度
- Pullback_Success_Rate: 回调后继续上涨的成功率
```

### 3. **反弹形态 (Bounce)**
下降趋势中的短期价格反弹

**识别条件:**
- 当前处于下降趋势（MA短期 < MA长期）
- 价格从近期低点反弹 5%-20%
- 未突破关键阻力位

**提取的特征:**
```python
- Is_Bounce: 反弹标记 (0/1)
- Bounce_Height: 反弹高度（反弹百分比）
- Bounce_Days: 反弹持续天数
- Bounce_End: 反弹结束标记 (0/1)
- Bounce_Frequency: 60日内反弹次数
- Avg_Bounce_Height: 平均反弹高度
- Bounce_Success_Rate: 反弹后继续下跌的成功率
```

### 4. **辅助特征**
```python
- Trend: 当前趋势 (1=上升, -1=下降, 0=横盘)
- Peak: 局部最高点标记 (0/1)
- Trough: 局部最低点标记 (0/1)
- Recent_High: 近期最高价
- Recent_Low: 近期最低价
- Drawdown_From_High: 从高点的回撤幅度
- Rally_From_Low: 从低点的反弹幅度
```

---

## 🚀 快速开始

### 方法1: 自动提取（推荐）

系统已经自动集成，下载数据时会自动提取形态特征：

```bash
# 1. 下载数据（已包含形态特征）
python stock_data_downloader.py

# 2. 运行形态分析演示
python pattern_analysis_demo.py
```

### 方法2: 手动提取

```python
from utils.pattern_recognition import add_pattern_features, summarize_patterns
import pandas as pd

# 加载你的股票数据
df = pd.read_csv('your_stock_data.csv')

# 提取形态特征
df_with_patterns = add_pattern_features(df)

# 查看形态统计
summary = summarize_patterns(df_with_patterns)
print(summary)

# 保存带形态的数据
df_with_patterns.to_csv('stock_data_with_patterns.csv', index=False)
```

---

## 📈 作为统计分析自变量使用

### 1. 提取形态样本

```python
import pandas as pd

# 加载带形态特征的数据
df = pd.read_csv('stock_data_with_patterns.csv')

# 提取所有回调样本
pullback_samples = df[df['Is_Pullback'] == 1].copy()

# 提取所有反弹样本
bounce_samples = df[df['Is_Bounce'] == 1].copy()

# 提取所有反转样本
reversal_samples = df[
    (df['Bullish_Reversal'] == 1) | (df['Bearish_Reversal'] == 1)
].copy()

print(f"回调样本数: {len(pullback_samples)}")
print(f"反弹样本数: {len(bounce_samples)}")
print(f"反转样本数: {len(reversal_samples)}")
```

### 2. 构建特征矩阵

```python
# 选择作为自变量的特征
feature_columns = [
    # 形态特征
    'Pullback_Depth',           # 回调深度
    'Pullback_Days',            # 回调天数
    'Bounce_Height',            # 反弹高度
    'Bounce_Days',              # 反弹天数
    'Reversal_Strength',        # 反转强度
    'Days_Since_Reversal',      # 距反转天数
    
    # 趋势特征
    'Trend',                    # 当前趋势
    'Drawdown_From_High',       # 回撤幅度
    'Rally_From_Low',           # 反弹幅度
    
    # 成功率特征
    'Pullback_Success_Rate',    # 回调成功率
    'Bounce_Success_Rate',      # 反弹成功率
    
    # 频率特征
    'Pullback_Frequency',       # 回调频率
    'Bounce_Frequency',         # 反弹频率
    
    # 配合技术指标
    'RSI', 'MACD', 'ATR', 'Volume'
]

# 构建特征矩阵 X（自变量）
X = df[feature_columns].copy()

# 构建目标变量 y（因变量）
# 例如：预测未来5天的收益率
df['Future_Return_5D'] = df['Close'].pct_change(5).shift(-5)
y = df['Future_Return_5D']

# 删除缺失值
mask = ~(X.isna().any(axis=1) | y.isna())
X_clean = X[mask]
y_clean = y[mask]

print(f"特征矩阵形状: {X_clean.shape}")
print(f"目标变量形状: {y_clean.shape}")
```

### 3. 统计分析示例

#### 3.1 相关性分析

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 计算形态特征与收益的相关性
correlation_matrix = pd.concat([X_clean, y_clean], axis=1).corr()

# 提取与目标变量的相关性
target_corr = correlation_matrix['Future_Return_5D'].sort_values(ascending=False)

print("各特征与未来收益的相关性:")
print(target_corr)

# 可视化
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('形态特征相关性矩阵')
plt.tight_layout()
plt.savefig('results/pattern_correlation.png')
```

#### 3.2 回归分析

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.6f}")

# 查看特征重要性
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\n特征重要性（回归系数）:")
print(feature_importance)
```

#### 3.3 分组比较分析

```python
# 比较不同形态的后续表现
pattern_performance = []

# 回调形态的后续表现
pullback_data = df[df['Is_Pullback'] == 1]
pullback_return = pullback_data['Future_Return_5D'].mean()

# 反弹形态的后续表现
bounce_data = df[df['Is_Bounce'] == 1]
bounce_return = bounce_data['Future_Return_5D'].mean()

# 牛市反转的后续表现
bull_reversal = df[df['Bullish_Reversal'] == 1]
bull_return = bull_reversal['Future_Return_5D'].mean()

# 熊市反转的后续表现
bear_reversal = df[df['Bearish_Reversal'] == 1]
bear_return = bear_reversal['Future_Return_5D'].mean()

print("各形态后续5日平均收益:")
print(f"  回调: {pullback_return:.2%}")
print(f"  反弹: {bounce_return:.2%}")
print(f"  牛市反转: {bull_return:.2%}")
print(f"  熊市反转: {bear_return:.2%}")
```

#### 3.4 机器学习分类

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 创建分类目标：未来是否上涨
y_class = (y_clean > 0).astype(int)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_class, test_size=0.2, random_state=42
)

# 训练随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 预测和评估
y_pred = rf.predict(X_test)
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 特征重要性
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n特征重要性排名:")
print(importance_df.head(10))
```

---

## 📊 实际应用场景

### 场景1: 回调买入策略评估

```python
# 找出所有回调后恢复的情况
pullback_recovery = df[
    (df['Is_Pullback'] == 1) & 
    (df['Pullback_Success'] == 1)
]

# 分析回调深度与成功率的关系
depth_groups = pd.cut(pullback_recovery['Pullback_Depth'], bins=5)
success_by_depth = pullback_recovery.groupby(depth_groups)['Future_Return_5D'].agg([
    'count', 'mean', 'std'
])

print("不同回调深度的后续表现:")
print(success_by_depth)
```

### 场景2: 反转信号可靠性分析

```python
# 分析反转信号的准确性
reversal_signals = df[
    (df['Bullish_Reversal'] == 1) | (df['Bearish_Reversal'] == 1)
].copy()

# 计算反转后实际趋势是否改变
reversal_signals['Actual_Reversal'] = (
    reversal_signals['Reversal_Strength'].abs() > 0.05
).astype(int)

# 准确率
accuracy = reversal_signals['Actual_Reversal'].mean()
print(f"反转信号准确率: {accuracy:.2%}")

# 按RSI分组分析
rsi_groups = pd.cut(reversal_signals['RSI'], bins=[0, 30, 70, 100])
accuracy_by_rsi = reversal_signals.groupby(rsi_groups)['Actual_Reversal'].mean()

print("\n不同RSI区间的反转准确率:")
print(accuracy_by_rsi)
```

### 场景3: 形态组合策略

```python
# 寻找高质量的买入信号：回调 + 超卖 + 上升趋势
buy_signals = df[
    (df['Is_Pullback'] == 1) &           # 正在回调
    (df['RSI'] < 40) &                   # RSI超卖
    (df['Trend'] == 1) &                 # 上升趋势
    (df['Pullback_Success_Rate'] > 0.6)  # 历史成功率高
]

print(f"找到 {len(buy_signals)} 个高质量买入信号")
print(f"平均后续收益: {buy_signals['Future_Return_5D'].mean():.2%}")
```

---

## 📝 完整工作流程

### 1. 数据准备
```bash
python stock_data_downloader.py
```

### 2. 形态识别与分析
```bash
python pattern_analysis_demo.py
```

### 3. 统计分析
```python
# 加载提取的形态数据
pattern_samples = pd.read_csv('results/pattern_samples_full.csv')
features = pd.read_csv('results/pattern_features.csv')

# 进行你的统计分析...
```

---

## 🎨 可视化示例

```python
import matplotlib.pyplot as plt

# 绘制价格图并标注形态
fig, ax = plt.subplots(figsize=(15, 8))

# 绘制价格
ax.plot(df.index, df['Close'], label='收盘价', color='black')

# 标注牛市反转
bull_idx = df[df['Bullish_Reversal'] == 1].index
ax.scatter(bull_idx, df.loc[bull_idx, 'Close'], 
          color='green', marker='^', s=100, label='牛市反转')

# 标注熊市反转
bear_idx = df[df['Bearish_Reversal'] == 1].index
ax.scatter(bear_idx, df.loc[bear_idx, 'Close'], 
          color='red', marker='v', s=100, label='熊市反转')

# 标注回调区域
ax.fill_between(df.index, df['Close'].min(), df['Close'].max(),
               where=df['Is_Pullback']==1, alpha=0.2, color='orange', label='回调')

# 标注反弹区域
ax.fill_between(df.index, df['Close'].min(), df['Close'].max(),
               where=df['Is_Bounce']==1, alpha=0.2, color='cyan', label='反弹')

ax.set_title('技术形态标注')
ax.set_xlabel('日期')
ax.set_ylabel('价格')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

---

## ⚙️ 参数调整

你可以根据需要调整形态识别的参数：

```python
from utils.pattern_recognition import identify_pullback, identify_bounce

# 自定义回调参数
df = identify_pullback(
    df, 
    trend_window=20,         # 趋势判断窗口：20日
    pullback_threshold=0.03  # 回调阈值：3%（更敏感）
)

# 自定义反弹参数
df = identify_bounce(
    df,
    trend_window=30,         # 趋势判断窗口：30日（更长期）
    bounce_threshold=0.08    # 反弹阈值：8%（更保守）
)
```

---

## 💡 最佳实践

1. **数据质量**: 确保有足够的历史数据（建议至少250个交易日）
2. **参数优化**: 根据不同股票特性调整识别阈值
3. **特征组合**: 将形态特征与传统技术指标结合使用
4. **回测验证**: 在使用形态信号前进行充分的历史回测
5. **样本平衡**: 注意不同形态样本数量的平衡性

---

## 🔧 故障排除

### 问题1: 识别不到形态
```python
# 检查数据长度
print(f"数据点数: {len(df)}")  # 应该 >= 100

# 检查价格波动
print(f"价格标准差: {df['Close'].std()}")  # 不应该为0
```

### 问题2: 特征全为0
```python
# 降低识别阈值
df = identify_pullback(df, pullback_threshold=0.02)  # 从5%降到2%
```

### 问题3: 导入错误
```python
# 确保模块路径正确
import sys
sys.path.append('.')
from utils.pattern_recognition import add_pattern_features
```

---

## 📚 相关文档

- `utils/pattern_recognition.py` - 形态识别核心代码
- `pattern_analysis_demo.py` - 完整分析示例
- `STOCK_FILTER_CONFIG.md` - 股票筛选配置

---

## 🎯 总结

通过本指南，你可以：
1. ✅ 自动识别反转、回调、反弹等技术形态
2. ✅ 将形态转化为可量化的特征
3. ✅ 使用这些特征进行统计分析
4. ✅ 构建基于形态的交易策略
5. ✅ 评估形态信号的有效性

这些形态特征可以直接作为**机器学习模型的输入特征**，或用于**传统统计分析**，帮助你更好地理解和预测股票价格走势！
