# 📊 部署模式说明

本系统支持两种部署模式，会根据环境自动选择最佳预测方案。

## 🖥️ 本地运行模式（完整版）

### 特点
- ✅ 使用完整的 ML 模型（77MB）
- ✅ 最高预测精度
- ✅ 所有功能完全可用

### 运行方式
```bash
streamlit run app_streamlit.py
```

### 模型文件
本地运行需要以下模型文件（已包含在仓库中）：
- `models/multi_horizon_models.pkl` (77MB) - 形态识别 ML 模型
- `models/ma20_multi_horizon_models.pkl` (2.8MB) - MA20 多窗口模型
- `models/ma10_multi_horizon_models.pkl` (2.8MB) - MA10 多窗口模型

### 系统要求
- Python 3.8+
- 至少 4GB RAM
- 安装所有依赖：`pip install -r requirements.txt`

---

## ☁️ 云端部署模式（Streamlit Cloud）

### 特点
- ✅ 轻量级，快速加载
- ✅ 使用规则分析作为形态识别备用方案
- ✅ MA20/MA10 预测完全可用

### 为什么使用规则备用？
Streamlit Cloud 对文件大小有限制，77MB 的形态识别模型无法加载。系统会自动：
1. 尝试加载 ML 模型
2. 如果失败，切换到基于规则的形态分析
3. 用户体验无缝，仍能获得预测结果

### 规则分析方法
使用多因子评分系统：
- **价格趋势** (40%): MA5/MA10 交叉判断
- **成交量** (30%): 成交量放大识别
- **价格动能** (30%): 价格变化率分析

### 访问地址
https://updownml-guwsvynbixtnosykzmgipc.streamlit.app/

---

## 🔄 智能切换机制

系统会自动检测运行环境：

```python
if self.pattern_models:  # ML 模型可用
    print("✅ 使用 ML 模型进行形态预测")
    # 使用机器学习模型
else:  # ML 模型不可用
    print("⚠️ ML 模型不可用，使用规则分析作为备用")
    # 使用规则分析
```

---

## 📈 预测结果对比

### ML 模型模式（本地）
```
形态预测: 有信号 (置信度: 78.5%, 方法: ML模型)
```

### 规则分析模式（云端）
```
形态预测: 有信号 (置信度: 72.0%, 方法: 规则分析)
```

---

## 🚀 推荐使用场景

### 本地运行（推荐）
- 日常交易分析
- 批量股票筛选
- 需要最高精度的预测

### 云端部署
- 随时随地访问
- 移动端使用
- 快速查看预测结果
- 分享给他人使用

---

## 💡 技术细节

### 模型加载优先级
1. **形态识别**: ML模型 → 规则分析
2. **MA20预测**: ML模型 → 规则分析（始终可用）
3. **MA10预测**: ML模型（本地） / 规则分析（云端）

### 文件大小对比
| 文件 | 大小 | 本地 | 云端 |
|------|------|------|------|
| multi_horizon_models.pkl | 77MB | ✅ | ⚠️ |
| ma20_multi_horizon_models.pkl | 2.8MB | ✅ | ✅ |
| ma10_multi_horizon_models.pkl | 2.8MB | ✅ | ✅ |

---

## 🔧 故障排除

### 本地运行时形态预测显示"规则分析"
**原因**: ML 模型文件未正确加载

**解决方案**:
```bash
# 1. 检查模型文件是否存在
ls -lh models/multi_horizon_models.pkl

# 2. 检查 Python 版本
python --version  # 需要 >= 3.8

# 3. 重新安装 scikit-learn
pip install scikit-learn==1.6.1

# 4. 如果问题依然存在，查看控制台错误信息
streamlit run app_streamlit.py
```

### 云端部署失败
查看 Streamlit Cloud 日志，确保：
- requirements.txt 中所有依赖都能安装
- 没有使用不兼容的 Python 版本
- 环境变量正确设置

---

## 📝 总结

- **本地运行** = 完整 ML 模型 = 最高精度 ⭐
- **云端部署** = 智能规则分析 = 随时访问 ☁️
- **自动切换** = 无缝体验 = 始终可用 🚀

两种模式各有优势，根据你的需求选择最合适的使用方式！

