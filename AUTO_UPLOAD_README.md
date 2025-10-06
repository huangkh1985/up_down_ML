# 🤖 自动上传脚本使用说明

## 📋 两个上传脚本对比

| 脚本 | 特点 | 适合场景 |
|-----|------|---------|
| **auto_upload_to_github.bat** ⭐ | 详细步骤，交互式引导 | 首次上传，需要详细提示 |
| **quick_upload.bat** | 快速上传，最少交互 | 熟悉流程，快速上传 |

---

## 🚀 方法1: 详细版（推荐首次使用）

### 使用步骤

#### 1. 准备工作

**在GitHub网站创建仓库：**
1. 访问 https://github.com/new
2. 填写仓库名称（如：`stock-prediction-system`）
3. 选择 Public（公开）或 Private（私有）
4. **不要** 勾选 "Initialize with README"
5. 点击 "Create repository"

**记下仓库地址：**
```
https://github.com/YOUR-USERNAME/YOUR-REPO.git
```

#### 2. 运行脚本

双击运行：
```
auto_upload_to_github.bat
```

#### 3. 跟随步骤

脚本会自动完成以下7个步骤：

```
[步骤 1/7] 清理临时文件
    ↓ 删除test_*.py等临时文件
    
[步骤 2/7] 检查Git仓库状态
    ↓ 初始化Git仓库
    
[步骤 3/7] 配置Git用户信息
    ↓ 设置用户名和邮箱（首次需要）
    
[步骤 4/7] 添加文件到Git
    ↓ 添加所有文件（包括数据和模型）
    ↓ 显示文件列表确认
    
[步骤 5/7] 提交到本地仓库
    ↓ 创建提交记录
    
[步骤 6/7] 配置远程GitHub仓库
    ↓ 输入GitHub仓库地址
    
[步骤 7/7] 推送到GitHub
    ↓ 上传约148MB（5-10分钟）
```

#### 4. 完成

看到成功消息后，访问您的GitHub仓库查看！

---

## ⚡ 方法2: 快速版（熟悉流程后使用）

### 使用步骤

#### 1. 准备GitHub仓库（同上）

#### 2. 运行脚本

双击运行：
```
quick_upload.bat
```

#### 3. 输入仓库地址

```
输入GitHub仓库地址: https://github.com/YOUR-USERNAME/YOUR-REPO.git
```

#### 4. 等待完成

脚本自动完成所有步骤（约5-10分钟）

---

## 🔑 GitHub登录方式

### 使用Personal Access Token（推荐）

GitHub已经不再支持密码验证，需要使用Token：

#### 创建Token

1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 填写Token名称（如：`stock-upload`）
4. 选择过期时间（建议：No expiration）
5. 勾选权限：**✅ repo**（完整的仓库权限）
6. 点击 "Generate token"
7. **复制Token并保存**（只显示一次！）

#### 使用Token推送

当Git要求输入密码时：
```
Username: your-github-username
Password: [粘贴您的Token]
```

**注意：** 粘贴Token时不会显示任何字符，直接粘贴后按回车即可。

---

## 📊 上传进度说明

### 上传阶段

```
[开始] 正在推送到GitHub...

Enumerating objects: 120, done.
Counting objects: 100% (120/120), done.
Delta compression using up to 8 threads
Compressing objects: 100% (95/95), done.
Writing objects:  42% (50/120), 50.00 MiB | 5.00 MiB/s  ← 正在上传
```

### 预计时间

| 网速 | 上传时间 |
|-----|---------|
| 10 Mbps | ~15分钟 |
| 50 Mbps | ~5分钟 |
| 100 Mbps | ~3分钟 |

**提示：** 第一次推送最慢，后续更新会很快。

---

## ⚠️ 常见问题

### Q1: 推送失败，提示需要登录

**解决：**
1. 检查是否在GitHub创建了仓库
2. 使用Personal Access Token而不是密码
3. 确认Token有repo权限

### Q2: 推送很慢或卡住

**解决：**
1. 这是正常的（148MB需要时间）
2. 保持网络连接
3. 不要关闭窗口
4. 可以看到进度百分比

### Q3: fatal: remote origin already exists

**解决：**
```bash
git remote remove origin
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main
```

### Q4: Everything up-to-date

**说明：** 文件已经上传过了，无需重复上传。

### Q5: 提示文件太大

**检查：**
- 应该没有单个文件超过100MB
- 如果有，需要使用Git LFS
- 运行 `check_file_sizes.py` 检查

---

## 🔄 后续更新代码

上传成功后，如果修改了代码想要更新：

### 方法1: 使用Git命令

```bash
git add .
git commit -m "Update: 描述你的更改"
git push
```

### 方法2: 创建更新脚本

创建 `update_to_github.bat`:
```batch
@echo off
git add .
git commit -m "Update code"
git push
echo [完成] 代码已更新
pause
```

---

## 📋 上传前检查清单

使用自动脚本前确认：

- [ ] 已在GitHub创建新仓库
- [ ] 已获取仓库地址
- [ ] Git已安装（运行 `git --version` 检查）
- [ ] 已准备好GitHub Token（如果需要登录）
- [ ] 网络连接稳定
- [ ] 了解上传大小（~148MB）和时间（5-10分钟）

---

## 🎯 脚本执行流程图

```
开始
  │
  ├─→ 检查Git是否安装
  │     └─→ 未安装 → 提示安装 → 退出
  │
  ├─→ 清理临时文件
  │     └─→ 删除test_*, __pycache__等
  │
  ├─→ Git初始化
  │     └─→ git init
  │
  ├─→ 配置用户信息
  │     └─→ git config user.name/email
  │
  ├─→ 添加文件
  │     └─→ git add .
  │
  ├─→ 提交
  │     └─→ git commit -m "..."
  │
  ├─→ 配置远程仓库
  │     └─→ 用户输入仓库地址
  │     └─→ git remote add origin
  │
  ├─→ 推送到GitHub
  │     └─→ git push -u origin main
  │     └─→ 需要5-10分钟
  │
  └─→ 完成
        └─→ 显示仓库链接
```

---

## 💡 最佳实践

### 1. 首次上传

使用 **auto_upload_to_github.bat**
- 有详细提示
- 每步确认
- 不容易出错

### 2. 熟练后

使用 **quick_upload.bat**
- 快速上传
- 最少交互
- 节省时间

### 3. 团队协作

上传后：
1. 添加README徽章
2. 设置仓库描述
3. 添加Topics标签
4. 创建Release版本

---

## 📚 相关文档

- `READY_TO_UPLOAD.md` - 手动上传指南
- `GITHUB_UPLOAD_WITH_DATA.md` - 详细步骤说明
- `README.md` - 项目说明

---

## 🎉 预期结果

上传成功后，您的GitHub仓库将包含：

```
your-repo/
├── 📄 代码文件 (~30个.py)
├── 📁 data/ (36MB)
│   ├── all_stock_data.pkl
│   ├── stock_*.csv (28个)
│   └── ...
├── 📁 models/ (110MB)
│   ├── ma20_multi_horizon_models.pkl
│   ├── multi_horizon_models.pkl
│   └── ...
├── 📁 utils/ (23个工具模块)
├── 📄 文档 (~20个.md)
└── 📄 配置文件

总大小: ~148MB
```

**用户体验：**
```bash
git clone https://github.com/you/repo.git
pip install -r requirements_streamlit.txt
streamlit run app_streamlit.py
# ✨ 立即可用！
```

---

**创建时间：** 2025-10-05  
**支持系统：** Windows 10/11  
**上传大小：** ~148MB  
**预计时间：** 5-10分钟

