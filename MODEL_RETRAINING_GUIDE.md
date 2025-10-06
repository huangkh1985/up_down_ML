# 模型重训练完整指南

## 📋 目录
1. [前置准备](#前置准备)
2. [数据更新](#数据更新)
3. [特征工程](#特征工程)
4. [模型训练](#模型训练)
5. [模型验证](#模型验证)
6. [模型部署](#模型部署)
7. [回滚方案](#回滚方案)

---

## 🎯 前置准备

### 检查环境
```bash
# 确认Python环境
python --version  # 应该是 3.8+

# 确认依赖包
pip list | findstr "xgboost scikit-learn pandas numpy"
```

### 备份现有模型
```bash
# 创建备份目录（使用当前日期）
mkdir models\backup_20251005

# 备份所有现有模型
copy models\*.pkl models\backup_20251005\
```

**重要文件备份清单：**
- `models/ma20_multi_horizon_models.pkl` - MA20多时间窗口模型（新版）
- `models/trained_model.pkl` - MA20单一模型（旧版）
- `models/multi_horizon_models.pkl` - 形态识别模型
- `models/feature_list.pkl` - 特征列表
- `models/model_info.pkl` - 模型信息

---

## 📥 步骤1: 数据更新

### 1.1 下载最新股票数据

```bash
python stock_data_downloader.py
```

**预期输出：**
```
正在下载股票数据...
✓ 002104: 450 条记录
✓ 002165: 448 条记录
...
成功保存到: data/all_stock_data.pkl
```

**检查数据：**
- 数据日期范围应该包含最新交易日
- 确认 `data/all_stock_data.pkl` 文件已更新
- 文件大小应该增大（如果有新数据）

### 1.2 验证数据质量

创建验证脚本 `check_data_quality.py`:

```python
import pickle
import pandas as pd

with open('data/all_stock_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

print(f"总股票数: {len(all_data)}")
print("\n最新数据日期:")
for code, df in list(all_data.items())[:5]:
    print(f"  {code}: {df.index[-1].strftime('%Y-%m-%d')} ({len(df)}条)")
```

运行验证：
```bash
python check_data_quality.py
```

---

## 🔧 步骤2: 特征工程（可选）

**注意：** 如果股票数据已包含完整的技术指标（MA20, RSI, MACD等），可以跳过此步骤。

### 2.1 计算技术指标

如果需要重新计算特征：

```bash
python stock_feature_engineering.py
```

**预期输出：**
```
正在提取特征...
✓ 处理完成 28 只股票
保存到: data/processed_features.pkl
```

### 2.2 验证特征

```python
# 快速检查特征
import pickle
with open('data/all_stock_data.pkl', 'rb') as f:
    data = pickle.load(f)
    
# 检查第一只股票的特征列
first_stock = list(data.values())[0]
print("可用特征:", first_stock.columns.tolist())

# 必需的特征
required_features = ['Close', 'MA20', 'RSI', 'MACD', 'ATR', 'Volume']
missing = [f for f in required_features if f not in first_stock.columns]
if missing:
    print(f"⚠️ 缺少特征: {missing}")
else:
    print("✓ 所有必需特征都存在")
```

---

## 🤖 步骤3: 模型训练

### 3.1 训练MA20多时间窗口模型（推荐）

```bash
python train_ma20_multi_horizon.py
```

**训练时间：** 约1-3分钟

**预期输出：**
```
================================================================================
MA20多时间窗口模型训练系统
================================================================================
训练目标: 为1天、3天、5天、10天分别训练独立的MA20预测模型
================================================================================

[数据加载] 加载股票数据...
[成功] 成功加载 28 只股票的数据

################################################################################
#  训练 1 天预测模型
################################################################################

[数据准备] 准备 1 天预测的训练数据...
  处理股票 002104... [OK] 获得 424 个样本 (强势: 237, 弱势: 187)
  ...

[完成] 1天数据准备完成:
   总样本数: 11078
   特征数量: 37
   强势样本: 5765 (52.0%)
   弱势样本: 5313 (48.0%)

[模型训练] 训练 1 天预测模型...
   训练集: 8862 样本
   测试集: 2216 样本
   开始训练...

   [完成] 模型训练完成！
   [评估指标]:
      准确率 (Accuracy):  0.8958
      精确率 (Precision): 0.8995
      召回率 (Recall):    0.9003
      F1分数 (F1-Score):  0.8999

   [Top 10 重要特征]:
      Close_MA20_Diff                0.331292
      Close_MA20_Ratio               0.244448
      ...

# （3天、5天、10天模型训练类似输出）

================================================================================
[保存] 保存模型...
[成功] 模型已保存到: models/ma20_multi_horizon_models.pkl

[训练总结]:
================================================================================

[1天预测模型]:
   训练样本数: 11078
   特征数量: 37
   准确率: 0.8958
   F1分数: 0.8999
   训练时间: 2025-10-05 20:50:33

[3天预测模型]:
   训练样本数: 11022
   准确率: 0.8349
   F1分数: 0.8408

[5天预测模型]:
   训练样本数: 10966
   准确率: 0.7958
   F1分数: 0.8025

[10天预测模型]:
   训练样本数: 10826
   准确率: 0.7798
   F1分数: 0.7864

[成功] 训练完成！
================================================================================
```

**训练成功标志：**
- ✅ 所有4个模型都成功训练
- ✅ 准确率在合理范围内（1天>85%, 3天>80%, 5天>75%, 10天>70%）
- ✅ 模型文件已保存

### 3.2 训练形态识别模型（可选）

如果需要更新形态识别模型：

```bash
python multi_horizon_prediction_system.py
```

**注意：** 这会训练形态反转信号的预测模型，与MA20预测是独立的。

---

## ✅ 步骤4: 模型验证

### 4.1 快速验证脚本

创建 `validate_models.py`:

```python
"""
模型验证脚本
检查新训练的模型是否可用
"""
import pickle
import numpy as np

def validate_ma20_models():
    """验证MA20多时间窗口模型"""
    print("="*80)
    print("验证 MA20 多时间窗口模型")
    print("="*80)
    
    try:
        with open('models/ma20_multi_horizon_models.pkl', 'rb') as f:
            models = pickle.load(f)
        
        print(f"\n✓ 模型文件加载成功")
        print(f"  包含的时间窗口: {list(models.keys())}")
        
        for horizon, model_info in models.items():
            print(f"\n[{horizon}天模型]")
            print(f"  模型类型: {type(model_info['model']).__name__}")
            print(f"  特征数量: {len(model_info['feature_list'])}")
            print(f"  训练样本数: {model_info['train_samples']}")
            print(f"  准确率: {model_info['metrics']['accuracy']:.4f}")
            print(f"  F1分数: {model_info['metrics']['f1_score']:.4f}")
            print(f"  训练时间: {model_info['train_date']}")
            
            # 测试预测功能
            model = model_info['model']
            n_features = len(model_info['feature_list'])
            dummy_input = np.random.randn(1, n_features)
            
            try:
                pred = model.predict(dummy_input)
                prob = model.predict_proba(dummy_input)
                print(f"  ✓ 预测功能正常")
            except Exception as e:
                print(f"  ✗ 预测功能异常: {e}")
                return False
        
        print("\n" + "="*80)
        print("✓ MA20模型验证通过")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n✗ 验证失败: {e}")
        return False

def validate_pattern_models():
    """验证形态识别模型"""
    print("\n" + "="*80)
    print("验证形态识别模型")
    print("="*80)
    
    try:
        with open('models/multi_horizon_models.pkl', 'rb') as f:
            data = pickle.load(f)
        
        models = data['models']
        print(f"\n✓ 模型文件加载成功")
        print(f"  包含的时间窗口: {list(models.keys())}")
        
        for horizon in models.keys():
            print(f"\n[{horizon}天模型]")
            print(f"  模型类型: {type(models[horizon]['model']).__name__}")
            print(f"  ✓ 加载成功")
        
        print("\n" + "="*80)
        print("✓ 形态模型验证通过")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n✗ 验证失败: {e}")
        return False

if __name__ == "__main__":
    ma20_ok = validate_ma20_models()
    pattern_ok = validate_pattern_models()
    
    print("\n" + "="*80)
    print("总体验证结果")
    print("="*80)
    print(f"MA20模型: {'✓ 通过' if ma20_ok else '✗ 失败'}")
    print(f"形态模型: {'✓ 通过' if pattern_ok else '✗ 失败'}")
    
    if ma20_ok and pattern_ok:
        print("\n✓ 所有模型验证通过，可以部署！")
    else:
        print("\n✗ 部分模型验证失败，请检查！")
```

**运行验证：**
```bash
python validate_models.py
```

### 4.2 实际预测测试

测试新模型在实际数据上的表现：

```bash
# 使用Streamlit应用测试
streamlit run app_streamlit.py
```

**测试步骤：**
1. 输入测试股票代码（如：002104）
2. 选择"仅机器学习模型"
3. 点击"开始分析"
4. 检查4个时间窗口的预测是否有合理差异

---

## 🚀 步骤5: 模型部署

### 5.1 确认模型文件

检查以下文件是否存在且为最新：

```bash
dir models\ma20_multi_horizon_models.pkl
```

**预期：** 文件修改时间应该是今天

### 5.2 重启Streamlit应用

如果Streamlit正在运行：

```bash
# 停止当前运行的Streamlit（Ctrl+C）
# 然后重新启动
streamlit run app_streamlit.py
```

**首次加载时：**
- Streamlit会自动加载新模型（使用 `@st.cache_resource` 缓存）
- 如果需要强制重新加载，点击页面右上角的"Clear cache"

### 5.3 生产环境部署检查清单

- [ ] 新模型文件已复制到生产环境
- [ ] 旧模型已备份
- [ ] 应用已重启
- [ ] 测试至少3个股票的预测
- [ ] 确认预测结果合理
- [ ] 监控应用性能（响应时间）

---

## 🔙 步骤6: 回滚方案

如果新模型表现不佳，可以快速回滚：

### 6.1 恢复旧模型

```bash
# 从备份恢复
copy models\backup_20251005\*.pkl models\
```

### 6.2 重启应用

```bash
# 重启Streamlit
streamlit run app_streamlit.py
```

### 6.3 清除缓存

在Streamlit界面中：
1. 点击右上角菜单（三个点）
2. 选择 "Clear cache"
3. 刷新页面

---

## 📊 训练监控和评估

### 关键指标对比

记录每次训练的结果，用于对比：

| 日期 | 1天准确率 | 3天准确率 | 5天准确率 | 10天准确率 | 样本数 |
|-----|----------|----------|----------|-----------|-------|
| 2025-10-05 | 89.58% | 83.49% | 79.58% | 77.98% | 11,078 |
| 2025-10-15 | ? | ? | ? | ? | ? |

### 预期准确率范围

| 时间窗口 | 最低可接受 | 良好 | 优秀 |
|---------|----------|-----|-----|
| 1天 | 75% | 80% | 85%+ |
| 3天 | 70% | 75% | 80%+ |
| 5天 | 65% | 72% | 78%+ |
| 10天 | 60% | 68% | 75%+ |

**如果准确率低于最低可接受值，需要：**
1. 检查数据质量
2. 检查特征完整性
3. 考虑增加训练样本
4. 调整模型超参数

---

## 🛠️ 故障排查

### 常见问题

#### 问题1: 训练时出现 "No module named 'tsfresh'"

**原因：** 缺少依赖包（但新版训练脚本已不需要）

**解决：**
```bash
# 如果使用旧版特征工程
pip install tsfresh
```

#### 问题2: 训练样本数为0

**原因：** 数据中缺少必需的列（MA20, Close等）

**解决：**
```python
# 检查数据列
import pickle
with open('data/all_stock_data.pkl', 'rb') as f:
    data = pickle.load(f)
print(list(data.values())[0].columns)
```

#### 问题3: 内存不足

**原因：** 训练数据过大

**解决：**
```python
# 在train_ma20_multi_horizon.py中减少样本
# 或者分批训练
```

#### 问题4: 模型准确率下降明显

**可能原因：**
1. 新数据质量问题
2. 市场环境变化
3. 样本分布不均衡

**解决：**
1. 检查数据质量
2. 增加更多历史数据
3. 调整训练参数

---

## 📅 定期重训练计划

### 建议频率

| 场景 | 重训练频率 |
|-----|----------|
| 生产环境 | 每月1次 |
| 市场波动大 | 每2周1次 |
| 测试开发 | 按需训练 |

### 自动化脚本

创建 `retrain_all_models.bat`:

```batch
@echo off
echo ========================================
echo 开始重训练所有模型
echo ========================================

echo.
echo [1/5] 备份现有模型...
mkdir models\backup_%date:~0,4%%date:~5,2%%date:~8,2%
copy models\*.pkl models\backup_%date:~0,4%%date:~5,2%%date:~8,2%\

echo.
echo [2/5] 下载最新数据...
python stock_data_downloader.py

echo.
echo [3/5] 训练MA20多时间窗口模型...
python train_ma20_multi_horizon.py

echo.
echo [4/5] 验证新模型...
python validate_models.py

echo.
echo [5/5] 完成！
echo ========================================
echo 重训练完成
echo ========================================
pause
```

**使用方法：**
```bash
retrain_all_models.bat
```

---

## 📚 相关文档

- `MA20_MULTI_MODEL_UPGRADE.md` - 多模型升级说明
- `MA20_DUAL_METHOD_GUIDE.md` - 双方法使用指南
- `train_ma20_multi_horizon.py` - 训练脚本源码
- `app_streamlit.py` - Streamlit应用源码

---

**最后更新：** 2025-10-05
**文档版本：** 1.0.0

