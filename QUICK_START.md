# 快速开始指南

## ✅ 当前系统状态（2025-10-05）

### 已完成步骤
- ✅ **步骤1**: 数据下载 - 29只股票，9.35 MB
- ✅ **步骤2**: 特征工程 - 10,406个样本，42个特征
- ✅ **步骤3**: MA20模型 - 准确率83.62%
- ✅ **步骤4**: 多时间窗口模型 - 4个时间窗口(1,3,5,10天)

### 模型性能
| 时间窗口 | 准确率 | 精确率 | 召回率 |
|---------|--------|--------|--------|
| 1天 | 80.32% | 3.04% | 6.50% |
| 3天 | 80.95% | 2.83% | 5.69% |
| 5天 | 77.40% | 3.14% | 8.20% |
| 10天 | 78.85% | 3.45% | 8.55% |

---

## 🚀 现在可以做什么？

### 方案A: 实盘预测（推荐）

```bash
python stock_multi_horizon_live_prediction.py
```

**输入示例：**
```
股票代码（回车使用默认）: 600519,000001,002415
```

**输出：** 
- 多时间窗口预测（1天、3天、5天、10天）
- MA20状态预测
- 形态信号预测
- 综合决策建议
- 可视化图表

---

### 方案B: 查看已生成的预测结果

```bash
# Windows
explorer results\

# 查看特定图表
start results\multi_horizon_prediction_002104.png
```

**可用图表：**
- `binary_classification_results.png` - MA20分类结果
- `multi_horizon_prediction_*.png` - 多时间窗口预测
- `chinese_display_test.png` - 中文显示测试

---

### 方案C: 检查系统状态

```bash
python check_system_status.py
```

查看：
- 依赖包安装情况
- 数据文件状态
- 模型训练情况
- 下一步建议

---

## 📊 预测结果解读

### 决策级别

| 符号 | 级别 | 含义 | 建议 |
|------|------|------|------|
| [+] | 强烈 | 所有窗口一致 | 强烈建议交易 |
| [+] | 推荐 | ≥75%窗口一致 | 建议交易 |
| [*] | 谨慎 | ≥50%窗口一致 | 可以考虑 |
| [*] | 观望 | <50%窗口一致 | 暂不建议 |
| [-] | 避免 | 所有窗口无信号 | 不建议交易 |

### 示例输出解读

```
[+] 1天预测: 强烈 | 强烈建议
    信号一致性: 2/2        # MA20和形态信号都为正
    平均置信度: 78.5%       # 高置信度
    MA20预测: 强势(≥MA20)
    形态预测: 有信号
```

**含义：** 短期（1天）强烈看涨，两个预测方法都给出积极信号，置信度高。

---

## 🔄 如果需要更新数据

### 完整更新流程

```bash
# 1. 重新下载数据（5-10分钟）
python stock_data_downloader.py

# 2. 重新特征工程（15-30分钟）
python stock_feature_engineering.py

# 3. 重新训练MA20模型（3-5分钟）
python stock_statistical_analysis.py

# 4. 重新训练多时间窗口模型（10-20分钟）
python multi_horizon_prediction_system.py
```

### 快速更新（仅数据）

```bash
# 只重新下载数据
python stock_data_downloader.py

# 然后直接预测（使用现有模型）
python stock_multi_horizon_live_prediction.py
```

**注意：** 实盘预测会实时下载最新数据，无需手动更新。

---

## 💡 常用命令速查

| 命令 | 用途 | 耗时 |
|------|------|------|
| `python check_system_status.py` | 检查系统状态 | <1分钟 |
| `python stock_multi_horizon_live_prediction.py` | 实盘预测 | <1分钟 |
| `python test_chinese_display.py` | 测试中文显示 | <1分钟 |
| `python stock_data_downloader.py` | 重新下载数据 | 5-10分钟 |
| `python multi_horizon_prediction_system.py` | 重新训练模型 | 10-20分钟 |

---

## 🆘 常见问题

### Q: 图表中文显示乱码？
```bash
python test_chinese_display.py
# 查看 results/chinese_display_test.png 验证
```

### Q: 预测失败？
1. 检查网络连接
2. 确认股票代码正确（6位数字）
3. 检查模型文件是否存在：
   ```bash
   python check_system_status.py
   ```

### Q: 模型版本警告？
```
InconsistentVersionWarning: Trying to unpickle estimator...
```
**解决：** 可忽略，或重新训练模型：
```bash
python stock_statistical_analysis.py
python multi_horizon_prediction_system.py
```

---

## 📚 详细文档

- `COMPLETE_RUNNING_GUIDE.md` - 完整运行指南（必读）
- `MULTI_HORIZON_VERIFICATION_GUIDE.md` - 多时间窗口详解
- `TIME_ALIGNMENT_EXPLANATION.md` - 时间对齐说明
- `PATTERN_FEATURES_GUIDE.md` - 形态特征说明

---

## 🎯 下一步推荐

### 初次使用者
1. ✅ 已完成所有准备工作
2. ▶️ **运行实盘预测** → `python stock_multi_horizon_live_prediction.py`
3. 📊 查看预测结果和图表
4. 📖 阅读 `COMPLETE_RUNNING_GUIDE.md` 了解原理

### 进阶使用者
1. 调整模型参数优化性能
2. 增加更多股票数据
3. 自定义预测时间窗口
4. 开发自动交易策略

---

**版本：** V1.0  
**更新日期：** 2025-10-05  
**系统状态：** ✅ 就绪，可以开始预测！
