# 🚀 模型重训练快速参考

## 一键重训练（推荐）

```bash
# Windows
retrain_all_models.bat

# 或手动执行
python stock_data_downloader.py
python train_ma20_multi_horizon.py
python validate_models.py
```

**时间：** 约2-3分钟

---

## 步骤速查表

| 步骤 | 命令 | 耗时 | 输出文件 |
|-----|------|------|---------|
| 1️⃣ 备份 | `mkdir models\backup_日期` | 1秒 | 备份文件夹 |
| 2️⃣ 下载数据 | `python stock_data_downloader.py` | 30秒 | `data/all_stock_data.pkl` |
| 3️⃣ 训练模型 | `python train_ma20_multi_horizon.py` | 60秒 | `models/ma20_multi_horizon_models.pkl` |
| 4️⃣ 验证 | `python validate_models.py` | 5秒 | 验证报告 |
| 5️⃣ 部署 | `streamlit run app_streamlit.py` | 即时 | Web应用 |

---

## 常用命令

### 验证模型
```bash
python validate_models.py
```

### 回滚模型
```bash
# Windows
rollback_models.bat

# 或手动
copy models\backup_20251005\*.pkl models\
```

### 检查数据
```python
import pickle
with open('data/all_stock_data.pkl', 'rb') as f:
    data = pickle.load(f)
print(f"股票数: {len(data)}")
print(f"最新日期: {list(data.values())[0].index[-1]}")
```

---

## 检查清单

### 训练前 ✅
- [ ] 已备份旧模型
- [ ] 数据已更新到最新交易日
- [ ] 磁盘空间充足（>500MB）
- [ ] Python环境正常

### 训练后 ✅
- [ ] 4个模型都训练成功
- [ ] 准确率在合理范围（见下表）
- [ ] 验证脚本通过
- [ ] 预测功能正常

### 部署前 ✅
- [ ] 测试至少3个股票
- [ ] 预测结果合理
- [ ] 响应速度正常
- [ ] 日志无异常

---

## 准确率标准

| 时间窗口 | 最低要求 | 良好 | 优秀 |
|---------|---------|------|------|
| 1天 | ≥75% | ≥80% | ≥85% |
| 3天 | ≥70% | ≥75% | ≥80% |
| 5天 | ≥65% | ≥72% | ≥78% |
| 10天 | ≥60% | ≥68% | ≥75% |

**当前基准（2025-10-05）：**
- 1天: 89.58% ⭐
- 3天: 83.49% ⭐
- 5天: 79.58% ⭐
- 10天: 77.98% ⭐

---

## 故障速查

| 问题 | 解决方案 |
|-----|---------|
| 训练时间过长 | 正常现象，等待完成 |
| 准确率下降 | 检查数据质量，增加样本 |
| 内存不足 | 减少训练样本或关闭其他程序 |
| 模型加载失败 | 回滚到备份版本 |
| 预测结果异常 | 验证模型，检查特征 |

---

## 文件位置

### 数据文件
- `data/all_stock_data.pkl` - 原始股票数据
- `data/processed_features.pkl` - 处理后的特征（可选）

### 模型文件
- `models/ma20_multi_horizon_models.pkl` - ⭐ MA20多时间窗口模型（主要）
- `models/multi_horizon_models.pkl` - 形态识别模型
- `models/trained_model.pkl` - MA20单一模型（备份）
- `models/backup_*/` - 历史备份

### 脚本文件
- `train_ma20_multi_horizon.py` - 训练脚本
- `validate_models.py` - 验证脚本
- `retrain_all_models.bat` - 一键重训练
- `rollback_models.bat` - 回滚脚本

---

## 定期维护

### 每月任务
- [ ] 下载最新数据
- [ ] 重新训练模型
- [ ] 对比准确率
- [ ] 清理旧备份（保留最近3个月）

### 每季度任务
- [ ] 全面评估模型表现
- [ ] 考虑调整特征
- [ ] 优化超参数
- [ ] 更新文档

---

## 快速测试

运行完整测试：
```bash
# 1. 训练
python train_ma20_multi_horizon.py

# 2. 验证
python validate_models.py

# 3. 启动应用
streamlit run app_streamlit.py

# 4. 在浏览器中测试
#    - 输入股票代码：002104
#    - 选择"仅机器学习模型"
#    - 点击"开始分析"
#    - 检查4个时间窗口的预测差异
```

---

## 联系与支持

- 详细文档: `MODEL_RETRAINING_GUIDE.md`
- 升级说明: `MA20_MULTI_MODEL_UPGRADE.md`
- 使用指南: `MA20_DUAL_METHOD_GUIDE.md`

---

**快速参考版本：** 1.0  
**最后更新：** 2025-10-05

