# 完整模型重训练指南

## 🎯 系统架构

本系统包含**两套独立的预测模型**：

### 1️⃣ MA20状态预测模型
- **预测目标**: 未来N天收盘价是否≥MA20
- **模型数量**: 4个独立的XGBoost模型（1天、3天、5天、10天）
- **训练脚本**: `train_ma20_multi_horizon.py`
- **模型文件**: `models/ma20_multi_horizon_models.pkl`
- **使用场景**: 判断股票相对MA20的强弱势态

### 2️⃣ 形态反转信号预测模型
- **预测目标**: 未来N天是否出现形态反转信号
- **模型数量**: 4个独立的RandomForest模型（1天、3天、5天、10天）
- **训练脚本**: `multi_horizon_prediction_system.py`
- **模型文件**: `models/multi_horizon_models.pkl`
- **使用场景**: 识别看涨/看跌反转形态

---

## 🚀 完整重训练流程

### 方法1: 一键完整训练（推荐）⭐

```bash
retrain_all_models_complete.bat
```

**这将自动完成：**
1. ✅ 备份所有现有模型
2. ✅ 下载最新股票数据
3. ✅ 训练MA20多时间窗口模型（4个）
4. ✅ 训练形态识别模型（4个）
5. ✅ 验证所有8个模型
6. ✅ 生成训练报告

**预计耗时：** 3-5分钟

---

### 方法2: 分别训练

#### 选项A: 仅训练MA20模型

```bash
# 1. 备份
mkdir models\backup_%date:~0,4%%date:~5,2%%date:~8,2%
copy models\*.pkl models\backup_%date:~0,4%%date:~5,2%%date:~8,2%\

# 2. 下载数据
python stock_data_downloader.py

# 3. 训练MA20模型
python train_ma20_multi_horizon.py

# 4. 验证
python validate_models.py
```

**耗时：** 约2分钟

#### 选项B: 仅训练形态识别模型

```bash
# 1. 备份
mkdir models\backup_%date:~0,4%%date:~5,2%%date:~8,2%
copy models\*.pkl models\backup_%date:~0,4%%date:~5,2%%date:~8,2%\

# 2. 确保有最新数据
python stock_data_downloader.py

# 3. 训练形态识别模型
python multi_horizon_prediction_system.py

# 4. 验证
python validate_all_models.py
```

**耗时：** 约2-3分钟

---

## 📊 两套模型的详细对比

| 特性 | MA20模型 | 形态识别模型 |
|-----|----------|-------------|
| **预测内容** | 收盘价 vs MA20 | 形态反转信号 |
| **算法** | XGBoost | RandomForest |
| **特征数** | 37个技术指标 | ~75个（含形态特征） |
| **标签定义** | Close[t+N] ≥ MA20[t+N] | Any_Reversal[t+N] = 1 |
| **训练时间** | ~1-2分钟 | ~1-2分钟 |
| **模型文件** | ma20_multi_horizon_models.pkl | multi_horizon_models.pkl |
| **在Streamlit中** | "MA20状态预测"标签页 | "形态信号预测"标签页 |

---

## 🔍 训练成功标志

### MA20模型（预期准确率）

| 时间窗口 | 最低要求 | 当前基准 | 状态 |
|---------|---------|---------|------|
| 1天 | ≥75% | 89.58% | ✓ 优秀 |
| 3天 | ≥70% | 83.49% | ✓ 优秀 |
| 5天 | ≥65% | 79.58% | ✓ 优秀 |
| 10天 | ≥60% | 77.98% | ✓ 优秀 |

### 形态识别模型（预期准确率）

| 时间窗口 | 最低要求 | 典型范围 |
|---------|---------|---------|
| 1天 | ≥60% | 65-75% |
| 3天 | ≥58% | 62-72% |
| 5天 | ≥55% | 60-70% |
| 10天 | ≥52% | 58-68% |

**注意：** 形态识别模型的准确率通常低于MA20模型，这是正常的，因为形态反转是更稀有的事件。

---

## 📁 完整的文件清单

### 数据文件
```
data/
  ├── all_stock_data.pkl          # 原始股票数据（必需）
  ├── processed_features.pkl      # 处理后的特征（可选）
  └── processed_targets.csv       # 目标变量（可选）
```

### 模型文件
```
models/
  ├── ma20_multi_horizon_models.pkl      # ⭐ MA20模型（4个）
  ├── multi_horizon_models.pkl           # 📈 形态识别模型（4个）
  ├── trained_model.pkl                  # 旧版MA20单一模型（备份）
  ├── feature_list.pkl                   # 旧版特征列表（备份）
  └── backup_YYYYMMDD/                   # 历史备份文件夹
```

### 训练脚本
```
train_ma20_multi_horizon.py           # MA20模型训练
multi_horizon_prediction_system.py    # 形态识别模型训练
stock_data_downloader.py              # 数据下载
```

### 验证脚本
```
validate_models.py                    # 验证MA20模型
validate_all_models.py                # 验证所有模型
```

### 自动化脚本
```
retrain_all_models.bat                # 仅MA20模型
retrain_all_models_complete.bat       # ⭐ 所有模型（推荐）
rollback_models.bat                   # 回滚脚本
```

---

## 🔄 详细训练步骤说明

### 步骤1: 数据准备

```bash
python stock_data_downloader.py
```

**检查数据质量：**
```python
import pickle
with open('data/all_stock_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"股票数量: {len(data)}")
for code, df in list(data.items())[:3]:
    print(f"{code}: {df.index[-1]} - {len(df)}条")
    print(f"  列: {list(df.columns)}")
```

**必需的列：**
- Close, Open, High, Low, Volume
- MA5, MA10, MA20, MA50, MA100
- RSI, MACD, ATR, ADX
- 其他技术指标

### 步骤2: 训练MA20模型

```bash
python train_ma20_multi_horizon.py
```

**训练过程：**
```
[数据准备] 准备 1 天预测的训练数据...
  处理股票 002104... [OK] 获得 424 个样本
  ...
[完成] 1天数据准备完成:
   总样本数: 11078
   特征数量: 37
   强势样本: 5765 (52.0%)

[模型训练] 训练 1 天预测模型...
   训练集: 8862 样本
   测试集: 2216 样本
   
   [完成] 模型训练完成！
   [评估指标]:
      准确率:  0.8958
      F1分数:  0.8999
   
   [Top 10 重要特征]:
      Close_MA20_Diff    0.331292
      Close_MA20_Ratio   0.244448
      ...

# （3天、5天、10天模型类似过程）

[成功] 模型已保存到: models/ma20_multi_horizon_models.pkl
```

### 步骤3: 训练形态识别模型

```bash
python multi_horizon_prediction_system.py
```

**训练过程：**
```
多时间窗口预测系统
同时使用1天、3天、5天、10天预测进行互相验证

步骤1: 加载股票数据
[OK] 成功加载 28 只股票

步骤2: 创建多时间窗口预测器

步骤3: 训练多时间窗口模型
正在训练 1 天预测模型...
[OK] 训练完成 - 准确率: 0.6823, 精确率: 0.7012

正在训练 3 天预测模型...
[OK] 训练完成 - 准确率: 0.6541, 精确率: 0.6789

正在训练 5 天预测模型...
[OK] 训练完成 - 准确率: 0.6423, 精确率: 0.6612

正在训练 10 天预测模型...
[OK] 训练完成 - 准确率: 0.6201, 精确率: 0.6445

步骤4: 保存模型
[OK] 模型已保存到 models/multi_horizon_models.pkl
```

### 步骤4: 验证所有模型

```bash
python validate_all_models.py
```

**验证输出：**
```
验证数据文件
[成功] 数据文件加载成功
  总股票数: 28

验证 MA20 多时间窗口模型
[成功] 模型文件加载成功
  包含的时间窗口: [1, 3, 5, 10]

[1天模型]
  准确率: 0.8958
  [成功] 准确率符合要求
  [成功] 预测功能正常

# （其他模型类似）

验证形态识别模型
[成功] 模型文件加载成功
  包含的时间窗口: [1, 3, 5, 10]

总体验证结果
数据文件:     ✓ 通过
MA20模型:     ✓ 通过
形态识别模型: ✓ 通过

✓ 核心模型验证通过，可以使用Streamlit应用！
```

---

## 🎨 在Streamlit中的展示

训练完成后，在Streamlit应用中会有两个独立的预测标签页：

### 标签页1: MA20状态预测

```
MA20预测结果

时间窗口  概率     预测  信号          置信度    模型来源
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1天      0.8953   1     强势(≥MA20)   89.5%    独立ML模型-1天
3天      0.7531   1     强势(≥MA20)   75.3%    独立ML模型-3天
5天      0.4789   0     弱势(低于MA20) 52.1%    独立ML模型-5天
10天     0.4112   0     弱势(低于MA20) 58.9%    独立ML模型-10天
```

### 标签页2: 形态信号预测

```
形态反转信号预测

时间窗口  概率     信号         置信度
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1天      0.6823   看涨反转     68.2%
3天      0.5541   看涨反转     55.4%
5天      0.4789   无明显信号   52.1%
10天     0.4312   无明显信号   56.9%
```

---

## ⚠️ 常见问题

### Q1: 两个模型可以独立训练吗？
**A:** 可以！它们完全独立，可以分别训练。

### Q2: 必须两个模型都训练吗？
**A:** 不一定。MA20模型是核心，形态识别模型是辅助。如果只需要MA20预测，只训练MA20模型即可。

### Q3: 形态识别模型准确率为什么较低？
**A:** 这是正常的。形态反转是稀有事件，比价格状态更难预测。60-70%的准确率已经很好。

### Q4: 训练时间很长怎么办？
**A:** 
- MA20模型：1-2分钟（正常）
- 形态识别模型：1-3分钟（正常）
- 如果超过5分钟，检查数据量或电脑性能

### Q5: 如何知道应该重训练哪个模型？
**A:** 
- 如果MA20预测不准确 → 重训练MA20模型
- 如果形态信号不准确 → 重训练形态识别模型
- 建议每月同时重训练两个模型

---

## 📅 维护计划

### 每月任务 ✅
```bash
# 运行完整训练（推荐）
retrain_all_models_complete.bat
```

### 每季度任务 ✅
- [ ] 全面评估两套模型的表现
- [ ] 对比历史训练结果
- [ ] 考虑调整特征或超参数
- [ ] 清理旧备份（保留最近3个月）

---

## 🔙 回滚方案

如果新模型表现不佳：

```bash
rollback_models.bat
```

这会恢复两套模型：
- `ma20_multi_horizon_models.pkl`
- `multi_horizon_models.pkl`

---

## 📚 相关文档

- `MODEL_RETRAINING_GUIDE.md` - MA20模型重训练指南
- `MA20_MULTI_MODEL_UPGRADE.md` - MA20多模型升级说明
- `QUICK_RETRAIN_REFERENCE.md` - 快速参考卡片
- `MA20_DUAL_METHOD_GUIDE.md` - 双方法使用指南

---

**最后更新：** 2025-10-05
**文档版本：** 2.0.0 - 完整双模型系统

