# 📈 股票多时间窗口预测分析系统

> 基于机器学习的双模型股票预测系统，支持1天、3天、5天、10天多时间窗口预测

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 项目简介

本系统是一个完整的股票技术分析和预测平台，包含两套独立的机器学习模型：

### 🤖 双模型架构

| 模型 | 预测目标 | 算法 | 准确率 |
|-----|---------|------|--------|
| **MA20状态预测** | 未来N天收盘价是否≥MA20 | XGBoost | 80-90% |
| **形态反转信号** | 未来N天是否出现反转形态 | RandomForest | 60-70% |

每个模型包含**4个独立的子模型**，分别预测1天、3天、5天、10天的情况。

### ✨ 核心特性

- 🎨 **Streamlit Web界面** - 直观的交互式预测界面
- 🔄 **双方法对比** - 机器学习 vs 规则方法
- 📊 **多时间窗口** - 1天、3天、5天、10天独立预测
- 🎯 **高准确率** - MA20预测准确率达80-90%
- 🔧 **一键训练** - 自动化训练和验证流程
- 📈 **实时分析** - 支持实时股票数据获取和预测

---

## 🚀 快速开始

> **📦 本仓库已包含训练好的数据和模型（~148MB），开箱即用！**

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**注意：** 首次克隆需要下载约148MB，包含：
- ✅ 股票历史数据（36MB）
- ✅ 训练好的模型（110MB）

### 2. 安装依赖

```bash
pip install -r requirements_streamlit.txt
```

主要依赖：
- Streamlit >= 1.28.0
- scikit-learn >= 1.3.0
- XGBoost
- pandas, numpy, matplotlib

### 3. 直接运行 🎉

```bash
streamlit run app_streamlit.py
```

访问：http://localhost:8501

**无需下载数据，无需训练模型，立即可用！** ✨

---

### 📥 可选：更新数据和模型

如果需要最新数据和重新训练：

```bash
# 下载最新数据
python stock_data_downloader.py

# 重新训练模型
retrain_all_models_complete.bat
```

训练时间：约3-5分钟

---

## 📊 系统架构

```
┌─────────────────────────────────────────┐
│         股票预测分析系统                  │
└─────────────────────────────────────────┘
                  │
    ┌─────────────┴─────────────┐
    ▼                           ▼
┌──────────────┐          ┌──────────────┐
│ MA20预测模型  │          │ 形态识别模型  │
│ (XGBoost×4)  │          │(RandomForest×4)│
└──────────────┘          └──────────────┘
    │                           │
    └───────────┬───────────────┘
                ▼
        ┌───────────────┐
        │ Streamlit UI  │
        │ - MA20预测     │
        │ - 形态信号     │
        │ - 综合决策     │
        └───────────────┘
```

---

## 📁 项目结构

```
up_down_stock_analyst/
├── 📂 核心应用
│   ├── app_streamlit.py                    # ⭐ 主Web应用
│   ├── train_ma20_multi_horizon.py         # MA20模型训练
│   ├── multi_horizon_prediction_system.py  # 形态识别训练
│   └── stock_data_downloader.py            # 数据下载
│
├── 📂 工具模块
│   └── utils/                              # 技术指标、形态识别等
│
├── 📂 批处理脚本
│   ├── retrain_all_models_complete.bat     # 一键训练全部
│   ├── start_streamlit.bat                 # 启动应用
│   └── rollback_models.bat                 # 回滚模型
│
├── 📂 文档
│   ├── README.md                           # 本文档
│   ├── START_HERE.md                       # 快速开始
│   ├── COMPLETE_RETRAINING_GUIDE.md        # 训练指南
│   └── TWO_MODEL_SYSTEMS_COMPARISON.md     # 模型对比
│
└── 📂 数据目录（需自行生成）
    ├── data/                               # 股票数据
    ├── models/                             # 训练好的模型
    └── results/                            # 预测结果
```

---

## 🎨 功能展示

### MA20状态预测

```
时间窗口  ML预测      规则预测      模型来源
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1天      强势(89%)   强势(65%)    独立ML模型-1天
3天      强势(75%)   强势(63%)    独立ML模型-3天
5天      弱势(52%)   强势(61%)    独立ML模型-5天
10天     弱势(49%)   弱势(55%)    独立ML模型-10天
```

### 形态反转信号

```
时间窗口  信号类型      置信度
━━━━━━━━━━━━━━━━━━━━━━━━━
1天      看涨反转       68%
3天      看涨反转       55%
5天      无明显信号     48%
10天     无明显信号     43%
```

---

## 📚 详细文档

### 快速入门
- 📖 [START_HERE.md](START_HERE.md) - 从这里开始
- ⚡ [QUICK_START.md](QUICK_START.md) - 快速指南
- 🎥 [STREAMLIT_QUICK_START.md](STREAMLIT_QUICK_START.md) - Web应用使用

### 训练指南
- 🔧 [COMPLETE_RETRAINING_GUIDE.md](COMPLETE_RETRAINING_GUIDE.md) - 完整训练流程
- 📝 [QUICK_RETRAIN_REFERENCE.md](QUICK_RETRAIN_REFERENCE.md) - 快速参考
- 🔄 [MODEL_RETRAINING_GUIDE.md](MODEL_RETRAINING_GUIDE.md) - MA20模型训练

### 技术说明
- 🤖 [TWO_MODEL_SYSTEMS_COMPARISON.md](TWO_MODEL_SYSTEMS_COMPARISON.md) - 双模型对比
- 📊 [MA20_MULTI_MODEL_UPGRADE.md](MA20_MULTI_MODEL_UPGRADE.md) - MA20升级说明
- 📈 [PATTERN_RECOGNITION_EXPLAINED.md](PATTERN_RECOGNITION_EXPLAINED.md) - 形态识别

### 部署运维
- 🚀 [STREAMLIT_DEPLOYMENT_GUIDE.md](STREAMLIT_DEPLOYMENT_GUIDE.md) - 部署指南
- 🛠️ [PROJECT_CLEANUP_GUIDE.md](PROJECT_CLEANUP_GUIDE.md) - 项目清理

---

## 🎯 使用示例

### 命令行预测

```python
from app_streamlit import StreamlitPredictor

# 创建预测器
predictor = StreamlitPredictor()
predictor.load_models()

# 获取股票数据
stock_data = predictor.get_stock_data('002104')

# MA20预测
ma20_pred = predictor.predict_ma20(stock_data, method='ml')
print(ma20_pred)

# 形态信号预测
pattern_pred = predictor.predict_pattern(stock_data)
print(pattern_pred)
```

### Web界面使用

1. 启动应用：`streamlit run app_streamlit.py`
2. 输入股票代码：`002104`
3. 选择预测方法：`🔄 两种方法对比`
4. 点击"开始分析"
5. 查看预测结果

---

## 🔧 维护指南

### 定期更新（推荐每月）

```bash
# 一键更新所有模型
retrain_all_models_complete.bat
```

### 验证模型

```bash
python validate_all_models.py
```

### 回滚模型

```bash
rollback_models.bat
```

---

## 📊 模型性能

### MA20模型（当前基准）

| 时间窗口 | 准确率 | F1分数 | 训练样本 |
|---------|--------|--------|---------|
| 1天 | 89.58% | 89.99% | 11,078 |
| 3天 | 83.49% | 84.08% | 11,022 |
| 5天 | 79.58% | 80.25% | 10,966 |
| 10天 | 77.98% | 78.64% | 10,826 |

### 形态识别模型

| 时间窗口 | 典型准确率范围 |
|---------|---------------|
| 1天 | 65-75% |
| 3天 | 62-72% |
| 5天 | 60-70% |
| 10天 | 58-68% |

---

## ⚙️ 系统要求

### 硬件要求
- **CPU**: 双核及以上
- **内存**: 4GB以上（推荐8GB）
- **磁盘**: 1GB可用空间

### 软件要求
- **操作系统**: Windows 10/11, macOS, Linux
- **Python**: 3.8 或更高版本
- **浏览器**: Chrome, Firefox, Edge (用于Streamlit)

---

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📝 开发路线图

### v2.0（计划中）
- [ ] 增加更多股票市场支持（美股、港股）
- [ ] 集成更多机器学习算法（LightGBM, CatBoost）
- [ ] 实时推送和告警功能
- [ ] 移动端适配

### v1.5（近期）
- [ ] 模型自动重训练调度
- [ ] 更多技术指标和形态
- [ ] 回测系统
- [ ] API接口

---

## ⚠️ 免责声明

本系统仅用于技术研究和学习目的。股票投资有风险，预测结果仅供参考，不构成投资建议。请根据自己的判断做出投资决策。

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [Streamlit](https://streamlit.io/) - 优秀的Web框架
- [scikit-learn](https://scikit-learn.org/) - 机器学习库
- [XGBoost](https://xgboost.ai/) - 强大的梯度提升库
- [efinance](https://github.com/Micro-sheep/efinance) - 金融数据获取

---

## 📧 联系方式

- 问题反馈：[Issues](https://github.com/your-username/your-repo/issues)
- 功能建议：[Discussions](https://github.com/your-username/your-repo/discussions)

---

## ⭐ Star History

如果这个项目对你有帮助，请给个 Star ⭐！

---

**最后更新：** 2025-10-05  
**版本：** 1.0.0

