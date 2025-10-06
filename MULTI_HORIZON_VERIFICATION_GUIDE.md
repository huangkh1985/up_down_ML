# 多时间窗口验证系统使用指南

## 📖 系统概述

多时间窗口验证系统通过**同时预测1天、3天、5天、10天**的股票走势，实现预测结果的互相验证，大幅提高预测可靠性。

### 🎯 核心理念

```
单一时间窗口预测 → 可能有偏差
多时间窗口互相验证 → 提高准确性和置信度
```

### ✨ 系统特点

1. **多时间窗口并行预测**
   - 1天预测：短期趋势判断
   - 3天预测：中短期趋势
   - 5天预测：中期趋势（标准周）
   - 10天预测：中长期趋势

2. **双重预测方法**
   - MA20状态预测：判断价格相对20日均线的位置
   - 形态信号预测：识别反转、回调、反弹等技术形态

3. **智能综合决策**
   - 信号一致性分析
   - 置信度加权计算
   - 多级决策建议

## 🚀 快速开始

### 步骤1: 训练多时间窗口模型

```bash
python multi_horizon_prediction_system.py
```

**功能：**
- 训练1天、3天、5天、10天四个预测模型
- 自动进行模型评估
- 保存所有模型到 `models/multi_horizon_models.pkl`
- 生成演示预测结果

**输出示例：**
```
训练1天预测模型
  样本数: 5000, 信号比例: 12.50%
  性能指标:
    准确率: 78.50%
    精确率: 72.30%
    召回率: 68.50%
  ✅ 1天模型训练完成

训练3天预测模型
  样本数: 4800, 信号比例: 11.80%
  ...
```

### 步骤2: 实盘预测（综合版）

```bash
python stock_multi_horizon_live_prediction.py
```

**功能：**
- 获取实时股票数据
- 同时进行MA20状态和形态信号预测
- 多时间窗口互相验证
- 生成综合决策建议
- 可视化预测结果

**交互示例：**
```
请输入要预测的股票代码（多个用逗号分隔）:
示例: 600519,000001,002415
股票代码（回车使用默认）: 600519

多时间窗口综合预测 - 600519
================================================================================

1️⃣ MA20状态多时间窗口预测（价格相对MA20）
  🟢 1天后: 强势(≥MA20) (置信度:75.3%)
  🟢 3天后: 强势(≥MA20) (置信度:72.1%)
  🟡 5天后: 弱势(<MA20) (置信度:55.8%)
  🔴 10天后: 弱势(<MA20) (置信度:68.2%)

2️⃣ 形态信号多时间窗口预测（反转/回调/反弹）
  🟢 1天后: 有信号 (置信度:82.5%)
  🟢 3天后: 有信号 (置信度:78.3%)
  🟡 5天后: 无信号 (置信度:58.1%)
  🔴 10天后: 无信号 (置信度:71.5%)

3️⃣ 综合决策

🟢 1天预测: 强烈 | 强烈建议
   MA20: 强势(≥MA20) | 形态: 有信号
   一致性: 2/2 | 置信: 78.9%

🟢 3天预测: 推荐 | 建议关注
   MA20: 强势(≥MA20) | 形态: 有信号
   一致性: 2/2 | 置信: 75.2%

🟡 5天预测: 观望 | 暂不建议
   MA20: 弱势(<MA20) | 形态: 无信号
   一致性: 0/2 | 置信: 56.9%
```

## 📊 可视化输出

### 1. 多时间窗口预测图（multi_horizon_prediction_*.png）

包含四个子图：
- **信号概率对比**：各时间窗口的预测概率
- **置信度对比**：各时间窗口的置信度
- **预测结果汇总**：综合决策摘要
- **置信度雷达图**：多维度置信度展示

### 2. 综合预测图（comprehensive_multi_horizon_*.png）

包含：
- **价格走势图**：近60日价格和MA20
- **MA20状态预测**：各时间窗口MA20预测
- **形态信号预测**：各时间窗口形态预测
- **综合置信度**：多方法综合置信度
- **决策摘要**：详细决策建议

## 💡 预测结果解读

### 决策级别说明

| 级别 | 信号一致性 | 含义 | 建议行动 |
|------|----------|------|---------|
| 🟢 强烈 | 100% | 所有窗口信号一致 | 强烈建议交易 |
| 🟢 推荐 | ≥75% | 多数窗口信号一致 | 建议交易 |
| 🟡 谨慎 | ≥50% | 半数窗口有信号 | 可以考虑 |
| 🟡 观望 | <50% | 少数窗口有信号 | 暂不建议 |
| 🔴 避免 | 0% | 所有窗口无信号 | 不建议交易 |

### 置信度解释

- **极高 (>90%)**：模型非常确定
- **高 (75-90%)**：模型比较确定
- **中 (60-75%)**：模型有一定把握
- **低 (<60%)**：模型不太确定

### 时间窗口权重建议

根据投资风格选择关注的时间窗口：

| 投资风格 | 推荐关注窗口 | 理由 |
|---------|------------|------|
| 超短线 | 1天、3天 | 捕捉短期波动 |
| 短线 | 3天、5天 | 平衡短期和中期 |
| 中线 | 5天、10天 | 关注中期趋势 |
| 综合 | 全部窗口 | 多维度验证 |

## 🎯 实战应用策略

### 策略1: 全面一致性（保守型）

```python
# 决策条件：所有时间窗口都预测为"有信号"
if all_signals_positive and avg_confidence > 0.75:
    action = "强烈建议交易"
    position = "重仓"
```

**适用场景：**
- 风险厌恶型投资者
- 追求高胜率
- 可以接受错过一些机会

### 策略2: 多数决（平衡型）

```python
# 决策条件：75%以上窗口预测为"有信号"
if signal_consistency >= 0.75 and avg_confidence > 0.65:
    action = "建议交易"
    position = "中等仓位"
```

**适用场景：**
- 平衡型投资者
- 追求胜率和机会的平衡
- 最推荐的策略

### 策略3: 短期优先（激进型）

```python
# 决策条件：短期窗口（1天、3天）预测为"有信号"
if short_term_signals_positive and short_term_confidence > 0.70:
    action = "可以交易"
    position = "小仓位试探"
```

**适用场景：**
- 激进型投资者
- 短线交易为主
- 追求更多交易机会

### 策略4: 分层决策（专业型）

```python
# 根据不同时间窗口采取不同行动
if signal_1day and confidence_1day > 0.80:
    action_short = "开仓"
    
if signal_3day and confidence_3day > 0.75:
    action_mid = "加仓"
    
if signal_5day and confidence_5day > 0.70:
    action_long = "持有"
```

**适用场景：**
- 专业投资者
- 多策略组合
- 精细化仓位管理

## ⚙️ 高级配置

### 自定义时间窗口

在 `multi_horizon_prediction_system.py` 中修改：

```python
predictor = MultiHorizonPredictor(
    horizons=[1, 2, 3, 5, 8, 13],  # 自定义窗口，可以使用斐波那契数列
    lookback_days=60  # 回看天数
)
```

### 调整决策阈值

在 `IntegratedMultiHorizonPredictor.make_comprehensive_decision()` 中调整：

```python
# 更保守：提高一致性要求
if signal_count >= total_signals * 0.8:  # 原来是0.5
    level = "强烈"

# 更激进：降低一致性要求
if signal_count >= total_signals * 0.3:  # 原来是0.5
    level = "推荐"
```

### 特征选择

修改 `prepare_data_for_horizon()` 中的特征列表：

```python
feature_columns = [
    'Close', 'Volume', 'TurnoverRate',
    'MA5', 'MA10', 'MA20', 'MA50',  # 均线系列
    'RSI', 'MACD', 'ATR',           # 技术指标
    'Volatility', 'Momentum',       # 波动指标
    # 添加自定义特征
    'CustomIndicator1',
    'CustomIndicator2'
]
```

## 📈 性能优化建议

### 1. 增加训练数据

```python
# 下载更多股票数据
china_stocks = get_billboard_stocks(
    start_date='2023-01-01',  # 延长时间范围
    end_date='2025-09-30',
    min_frequency=1  # 降低频率要求，获取更多股票
)
```

### 2. 调整模型参数

```python
model = RandomForestClassifier(
    n_estimators=500,      # 增加树的数量（原300）
    max_depth=15,          # 增加树的深度（原12）
    min_samples_split=5,   # 减少分裂要求（原10）
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

### 3. 使用集成模型

```python
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb

# 创建集成模型
rf = RandomForestClassifier(...)
xgb_model = xgb.XGBClassifier(...)
lgb_model = lgb.LGBMClassifier(...)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb_model), ('lgb', lgb_model)],
    voting='soft'
)
```

## 🔍 故障排除

### 问题1: 模型文件不存在

**错误：** `⚠️ 未找到多时间窗口模型`

**解决：** 先运行训练脚本
```bash
python multi_horizon_prediction_system.py
```

### 问题2: 数据不足

**错误：** `⚠️ 数据不足60天`

**解决：** 确保股票有足够的历史数据
```python
# 检查数据长度
print(f"数据天数: {len(stock_data)}")

# 如果不足，调整lookback_days
predictor = MultiHorizonPredictor(lookback_days=30)  # 减少到30天
```

### 问题3: 预测结果都是无信号

**原因：** 模型过于保守或数据质量问题

**解决：**
1. 检查类别平衡：
```python
print(f"信号比例: {y.sum()/len(y):.2%}")
# 如果比例<5%，说明信号太少
```

2. 调整SMOTE参数：
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(
    sampling_strategy=0.5,  # 目标少数类比例
    k_neighbors=3  # 减少邻居数
)
```

### 问题4: 不同窗口预测差异大

**现象：** 1天预测强烈建议，10天预测不建议

**解释：** 这是正常现象，说明：
- 短期和长期趋势不一致
- 短期可能有交易机会，但长期趋势不明朗

**建议：**
- 短线交易可以关注短期窗口
- 但要注意及时止盈，因为长期趋势不支持

## 📚 扩展阅读

### 相关文档
- `TIME_ALIGNMENT_EXPLANATION.md` - 时间对齐详细说明
- `FORECAST_HORIZON_COMPARISON.md` - 预测窗口对比分析
- `PATTERN_FEATURES_GUIDE.md` - 形态特征指南

### 进一步优化方向

1. **引入更多预测方法**
   - 深度学习模型（LSTM、Transformer）
   - 基本面数据
   - 市场情绪指标

2. **动态权重分配**
   - 根据历史准确率动态调整各窗口权重
   - 自适应决策阈值

3. **风险管理集成**
   - 止损止盈策略
   - 仓位管理建议
   - 风险评分系统

4. **回测验证**
   - 历史数据回测
   - 实盘跟踪
   - 策略迭代优化

## ⚠️ 重要提示

1. **模型预测仅供参考**
   - 历史数据不代表未来表现
   - 需结合基本面分析
   - 注意市场环境变化

2. **风险管理至关重要**
   - 设置止损位
   - 控制仓位规模
   - 分散投资

3. **持续监控和优化**
   - 定期重新训练模型
   - 跟踪预测准确率
   - 根据反馈调整策略

---

**版本：** V1.0  
**更新日期：** 2025-10-05  
**作者：** AI股票分析系统
