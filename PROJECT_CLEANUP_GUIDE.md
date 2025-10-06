# GitHub上传项目清理指南

## 📋 文件分类总览

### ✅ 必须保留（核心代码和文档）

#### 1. 核心Python脚本
```
app_streamlit.py                          # ⭐ Streamlit主应用
stock_data_downloader.py                  # 数据下载
stock_feature_engineering.py              # 特征工程
train_ma20_multi_horizon.py               # ⭐ MA20模型训练
multi_horizon_prediction_system.py        # ⭐ 形态识别训练
stock_multi_horizon_live_prediction.py    # 实时预测
stock_pattern_prediction_aligned.py       # 形态预测（对齐版）
stock_pattern_prediction.py               # 形态预测
stock_live_prediction.py                  # 实时预测
stock_statistical_analysis.py             # 统计分析
```

#### 2. 验证和工具脚本
```
validate_all_models.py                    # ⭐ 完整验证
validate_models.py                        # MA20验证
forecast_horizon_comparison.py            # 时间窗口对比
```

#### 3. 批处理脚本（Windows）
```
retrain_all_models_complete.bat          # ⭐ 完整训练
retrain_all_models.bat                    # MA20训练
rollback_models.bat                       # 回滚
start_streamlit.bat                       # 启动应用
```

#### 4. 配置文件
```
requirements_streamlit.txt                # ⭐ Python依赖
.gitignore                                # ⭐ Git忽略规则（新增）
```

#### 5. 工具模块
```
utils/                                    # ⭐ 整个utils目录
  ├── pattern_recognition.py              # 形态识别
  ├── technical_indicators.py             # 技术指标
  ├── matplotlib_config.py                # 图表配置
  └── ... (其他22个文件)
```

#### 6. 文档（核心）
```
README.md                                 # ⭐ 主README（需创建）
START_HERE.md                            # 快速开始
QUICK_START.md                           # 快速指南

# 训练相关
COMPLETE_RETRAINING_GUIDE.md             # ⭐ 完整训练指南
MODEL_RETRAINING_GUIDE.md                # MA20训练
QUICK_RETRAIN_REFERENCE.md               # 快速参考

# 模型说明
MA20_MULTI_MODEL_UPGRADE.md              # MA20升级说明
MA20_DUAL_METHOD_GUIDE.md                # 双方法指南
TWO_MODEL_SYSTEMS_COMPARISON.md          # ⭐ 双模型对比

# Streamlit相关
STREAMLIT_QUICK_START.md                 # Streamlit快速开始
STREAMLIT_FEATURES.md                    # 功能说明
STREAMLIT_DEPLOYMENT_GUIDE.md            # 部署指南

# 技术文档
PATTERN_RECOGNITION_EXPLAINED.md         # 形态识别解释
ML_MULTI_HORIZON_OPTIMIZATION.md         # ML优化说明
```

---

### ⚠️ 可选保留（演示和测试）

```
demo_pattern_visualization.py            # 形态可视化演示
pattern_analysis_demo.py                 # 形态分析演示
```

---

### ❌ 应该删除（临时/测试文件）

#### 1. 测试文件
```
test_chinese_display.py                  # 中文显示测试
test_pattern_recognition.py              # 形态识别测试
test_streamlit.py                        # Streamlit测试
test_pattern_result.csv                  # 测试结果
```

#### 2. 修复/清理脚本
```
clean_emojis.py                          # emoji清理（一次性）
fix_matplotlib_chinese.py                # 中文修复（一次性）
check_system_status.py                   # 系统检查（可选）
```

---

### 🚫 必须忽略（不上传GitHub）

#### 1. 数据文件（大文件）
```
data/
  ├── all_stock_data.pkl                 # ~50-100MB
  ├── processed_features.pkl             # ~30-50MB
  ├── processed_features.csv             # ~20-30MB
  ├── processed_targets.csv
  └── stock_*_data.csv                   # 28个CSV文件
```

#### 2. 模型文件（大文件）
```
models/
  ├── ma20_multi_horizon_models.pkl      # ~50-100MB
  ├── multi_horizon_models.pkl           # ~50-100MB
  ├── trained_model.pkl
  ├── feature_list.pkl
  ├── model_info.pkl
  ├── all_trained_models.pkl
  └── backup_*/                          # 备份文件夹
```

#### 3. 结果文件（可重新生成）
```
results/
  ├── *.png                              # 所有图片
  └── stock_*_results/                   # 结果文件夹
```

#### 4. Python缓存
```
__pycache__/
*.pyc
*.pyo
```

---

## 🗂️ GitHub仓库结构（推荐）

```
up_down_stock_analyst/
│
├── 📂 核心代码
│   ├── app_streamlit.py                  ⭐ 主应用
│   ├── train_ma20_multi_horizon.py       ⭐ MA20训练
│   ├── multi_horizon_prediction_system.py ⭐ 形态训练
│   ├── stock_data_downloader.py
│   ├── stock_feature_engineering.py
│   ├── stock_multi_horizon_live_prediction.py
│   ├── validate_all_models.py
│   └── validate_models.py
│
├── 📂 工具模块
│   └── utils/                            ⭐ 完整保留
│       ├── pattern_recognition.py
│       ├── technical_indicators.py
│       └── ... (22个文件)
│
├── 📂 脚本工具
│   ├── retrain_all_models_complete.bat   ⭐ 完整训练
│   ├── retrain_all_models.bat
│   ├── rollback_models.bat
│   └── start_streamlit.bat
│
├── 📂 文档
│   ├── README.md                         ⭐ 主文档（新建）
│   ├── START_HERE.md                     快速开始
│   ├── COMPLETE_RETRAINING_GUIDE.md      ⭐ 训练指南
│   ├── TWO_MODEL_SYSTEMS_COMPARISON.md   ⭐ 模型对比
│   ├── MA20_MULTI_MODEL_UPGRADE.md
│   ├── STREAMLIT_QUICK_START.md
│   └── ... (其他文档)
│
├── 📂 配置
│   ├── requirements_streamlit.txt        ⭐ 依赖
│   └── .gitignore                        ⭐ Git忽略
│
├── 📂 数据占位符（空文件夹）
│   ├── data/.gitkeep                     占位符
│   ├── models/.gitkeep                   占位符
│   └── results/.gitkeep                  占位符
│
└── 📂 演示（可选）
    ├── demo_pattern_visualization.py
    └── pattern_analysis_demo.py
```

---

## 🚀 清理步骤

### 步骤1: 删除临时文件

```bash
# 删除测试文件
del test_chinese_display.py
del test_pattern_recognition.py
del test_streamlit.py
del test_pattern_result.csv

# 删除一次性脚本
del clean_emojis.py
del fix_matplotlib_chinese.py
del check_system_status.py
```

### 步骤2: 创建.gitignore

已创建 `.gitignore` 文件（见上方）

### 步骤3: 创建空文件夹占位符

```bash
# 创建占位符文件
type nul > data\.gitkeep
type nul > models\.gitkeep
type nul > results\.gitkeep
```

### 步骤4: 创建主README

见下方的 `README.md` 模板

### 步骤5: 初始化Git

```bash
# 初始化Git仓库
git init

# 添加所有文件（.gitignore会自动排除大文件）
git add .

# 查看将要提交的文件
git status

# 确认没有大文件后提交
git commit -m "Initial commit: Stock prediction system with dual models"

# 关联远程仓库
git remote add origin https://github.com/your-username/your-repo-name.git

# 推送到GitHub
git push -u origin main
```

---

## 📄 推荐的README.md模板

见下一个文件...

---

## 📊 文件大小统计

| 类型 | 应该上传 | 文件大小 | 说明 |
|-----|---------|---------|-----|
| Python代码 (.py) | ✅ | ~500KB | 必须上传 |
| 文档 (.md) | ✅ | ~200KB | 必须上传 |
| 批处理 (.bat) | ✅ | ~5KB | 推荐上传 |
| 配置 (.txt, .yaml) | ✅ | ~2KB | 必须上传 |
| 数据 (.pkl, .csv) | ❌ | >200MB | 太大，不上传 |
| 模型 (.pkl) | ❌ | >200MB | 太大，不上传 |
| 结果 (.png) | ❌ | ~10MB | 可重新生成 |
| 缓存 (__pycache__) | ❌ | ~5MB | 自动生成 |

**上传后仓库大小：** 约1-2MB（非常合理）

---

## ⚠️ 重要提醒

### 1. 不要上传大文件
- GitHub有100MB单文件限制
- 仓库大小建议<1GB
- 数据和模型文件太大，不应上传

### 2. 添加数据获取说明
在README中说明：
```
用户需要自己运行 `python stock_data_downloader.py` 下载数据
用户需要自己运行 `retrain_all_models_complete.bat` 训练模型
```

### 3. 使用Git LFS（如果必须上传大文件）
```bash
# 安装Git LFS
git lfs install

# 追踪大文件
git lfs track "*.pkl"
git lfs track "*.csv"
```

但**不推荐**上传数据和模型文件！

---

## ✅ 检查清单

上传前检查：
- [ ] 已创建 `.gitignore`
- [ ] 已删除测试文件
- [ ] 已创建主 `README.md`
- [ ] 确认没有大文件（运行 `git status` 检查）
- [ ] 确认 `data/`, `models/`, `results/` 被忽略
- [ ] 文档完整且最新
- [ ] `requirements_streamlit.txt` 包含所有依赖

---

## 📚 相关命令

### 检查哪些文件会被上传
```bash
git add .
git status
```

### 查看文件大小
```bash
dir /s
```

### 如果不小心添加了大文件
```bash
git rm --cached data/*.pkl
git rm --cached models/*.pkl
```

### 强制排除已跟踪的文件
```bash
git rm -r --cached data/
git rm -r --cached models/
git rm -r --cached results/
git commit -m "Remove large files"
```

---

**最后更新：** 2025-10-05  
**版本：** 1.0.0

