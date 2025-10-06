# 📱 股票预测系统 - Streamlit Web版

## 🎯 项目简介

这是一个基于机器学习的股票多时间窗口预测系统，现已支持Web界面访问！

### ✨ 核心功能

1. **MA20状态预测** - 预测股价相对20日均线的位置
2. **形态信号预测** - 识别反转、回调、反弹等技术形态
3. **多时间窗口验证** - 同时预测1天、3天、5天、10天
4. **综合决策建议** - 基于多信号互相验证的决策

### 📱 支持平台

- ✅ PC浏览器（Chrome、Edge、Firefox）
- ✅ 手机浏览器（iOS Safari、Android Chrome）
- ✅ 平板设备
- ✅ 局域网内任意设备

---

## 🚀 快速开始

### 方法1：一键启动（推荐）

#### Windows
```bash
双击运行: start_streamlit.bat
```

#### Linux/Mac
```bash
chmod +x start_streamlit.sh
./start_streamlit.sh
```

### 方法2：命令行启动

```bash
# 1. 安装依赖
pip install -r requirements_streamlit.txt

# 2. 启动应用（支持手机访问）
streamlit run app_streamlit.py --server.address 0.0.0.0

# 或只在本机访问
streamlit run app_streamlit.py
```

---

## 📱 手机访问方法

### 步骤1：启动应用

在电脑上运行启动脚本或命令。

### 步骤2：获取访问地址

启动后会显示：
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501  ← 这是手机访问地址
```

### 步骤3：手机访问

1. 确保手机和电脑连接同一WiFi
2. 在手机浏览器中打开上面的Network URL
3. 开始使用！

### 技巧：添加到主屏幕

**iPhone:**
- Safari → 分享 → 添加到主屏幕
- 就像原生App一样！

**Android:**
- Chrome → 菜单(⋮) → 添加到主屏幕

---

## 📂 文件说明

### 核心文件

| 文件 | 说明 |
|------|------|
| `app_streamlit.py` | Streamlit Web应用主程序 |
| `stock_multi_horizon_live_prediction.py` | 命令行版预测系统 |
| `multi_horizon_prediction_system.py` | 多时间窗口预测核心 |

### 配置文件

| 文件 | 说明 |
|------|------|
| `requirements_streamlit.txt` | Web应用依赖包 |
| `start_streamlit.bat` | Windows启动脚本 |
| `.streamlit/config.toml` | Streamlit配置（可选） |

### 文档

| 文件 | 说明 |
|------|------|
| `STREAMLIT_QUICK_START.md` | 快速开始指南（3分钟上手）|
| `STREAMLIT_DEPLOYMENT_GUIDE.md` | 完整部署指南（包括云服务器）|
| `PATTERN_RECOGNITION_EXPLAINED.md` | 形态识别详细说明 |
| `COMPLETE_RUNNING_GUIDE.md` | 完整运行指南 |

---

## 🎨 功能界面

### 1. 主页面

- 📈 股票代码输入
- 📌 常用股票快速选择
- 🚀 一键预测按钮

### 2. 预测结果页

#### Tab1: 综合决策
- 显示1/3/5/10天的综合建议
- 绿色✅：强烈建议
- 黄色⚠️：建议关注
- 红色🚫：暂不建议

#### Tab2: MA20预测
- 数据表格显示
- 柱状图可视化
- 置信度显示

#### Tab3: 形态信号
- 信号概率
- 图表展示

#### Tab4: 价格走势
- 近60日K线图
- MA20均线
- 交互式图表

### 3. 侧边栏

- 🔍 系统状态检查
- 📖 使用说明
- 📜 预测历史记录

---

## 🔧 系统要求

### 硬件要求

- **最低配置：** 2核CPU，4GB内存
- **推荐配置：** 4核CPU，8GB内存

### 软件要求

- **Python：** 3.8+
- **操作系统：** Windows 10/11, Linux, macOS
- **浏览器：** Chrome 90+, Safari 14+, Edge 90+

### 网络要求

- **本地访问：** 无需网络
- **手机访问：** 同一局域网WiFi
- **远程访问：** 公网IP或内网穿透

---

## ⚙️ 高级配置

### 自定义端口

```bash
streamlit run app_streamlit.py --server.port 8888
```

### 配置文件

创建 `.streamlit/config.toml`：

```toml
[server]
port = 8501
address = "0.0.0.0"
maxUploadSize = 200

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"

[browser]
gatherUsageStats = false
```

### 添加密码保护

在 `.streamlit/secrets.toml` 中：

```toml
password = "your_password_here"
```

然后在代码中启用密码验证功能。

---

## 🌐 外网访问方案

### 方案1：Ngrok内网穿透（最简单）

```bash
# 1. 注册并下载 https://ngrok.com/
# 2. 启动应用
streamlit run app_streamlit.py

# 3. 新开终端，启动ngrok
ngrok http 8501

# 4. 使用生成的公网地址访问
# 例如: https://xxxx-xxx-xxx.ngrok.io
```

**优点：**
- ✅ 配置简单
- ✅ 支持HTTPS
- ✅ 免费版可用

**缺点：**
- ⚠️ 免费版URL每次重启会变
- ⚠️ 有流量限制

### 方案2：云服务器部署（推荐生产）

详见 `STREAMLIT_DEPLOYMENT_GUIDE.md`

**优点：**
- ✅ 固定域名
- ✅ 24小时在线
- ✅ 性能稳定
- ✅ 可配置SSL证书

---

## 📊 使用示例

### 示例1：预测贵州茅台

1. 启动应用
2. 输入股票代码：`600519`
3. 点击"开始预测"
4. 查看多时间窗口结果

### 示例2：批量对比

1. 点击"快速选择"按钮
2. 依次选择多只股票
3. 对比不同股票的预测结果
4. 查看预测历史

---

## 🐛 常见问题

### Q1: 启动后无法访问

**A:** 检查防火墙设置

```powershell
# Windows添加防火墙规则
netsh advfirewall firewall add rule name="Streamlit" dir=in action=allow protocol=TCP localport=8501
```

### Q2: 手机无法连接

**A:** 确保：
1. 手机和电脑在同一WiFi
2. 使用Network URL（不是localhost）
3. 电脑防火墙已开放端口

### Q3: 数据加载慢

**A:** 
- 首次加载会较慢（需要下载数据）
- 后续会使用缓存（1小时）
- 可以减少缓存时间或清除缓存

### Q4: 中文显示乱码

**A:** 
- 确保安装了中文字体
- 检查浏览器编码设置为UTF-8

---

## 📈 性能优化

### 1. 启用缓存

已在代码中实现：
```python
@st.cache_data(ttl=3600)  # 缓存1小时
def get_stock_data(stock_code):
    # ...
```

### 2. 减少数据量

只加载必要的日期范围：
```python
beg='20240101'  # 只加载今年的数据
```

### 3. 使用更快的服务器

部署到云服务器可获得更好的性能。

---

## 🔒 安全建议

### 生产环境

1. **启用HTTPS** - 使用SSL证书
2. **添加密码** - 保护数据安全
3. **限制IP** - 只允许特定IP访问
4. **定期备份** - 备份模型和数据
5. **监控日志** - 及时发现异常

### 开发环境

- ✅ 只在局域网内访问
- ✅ 不要暴露敏感信息
- ✅ 定期更新依赖包

---

## 📞 技术支持

### 文档

- [Streamlit快速开始](STREAMLIT_QUICK_START.md)
- [完整部署指南](STREAMLIT_DEPLOYMENT_GUIDE.md)
- [形态识别说明](PATTERN_RECOGNITION_EXPLAINED.md)

### 官方资源

- Streamlit文档: https://docs.streamlit.io/
- 社区论坛: https://discuss.streamlit.io/

---

## 🎉 开始使用

### 30秒快速启动

```bash
# 1. 安装依赖
pip install streamlit efinance pandas numpy matplotlib scikit-learn

# 2. 启动应用
streamlit run app_streamlit.py --server.address 0.0.0.0

# 3. 在浏览器打开显示的URL
```

### 推荐流程

1. **本地测试** - 先在PC浏览器测试
2. **局域网访问** - 确认手机能访问
3. **外网穿透** - 使用ngrok测试远程访问
4. **云服务器** - 最终部署到云端

---

## 📝 更新日志

### v1.0 (2025-10-05)

- ✅ 完整的Streamlit Web界面
- ✅ 支持移动端访问
- ✅ 多时间窗口预测
- ✅ 实时数据获取
- ✅ 图表可视化
- ✅ 预测历史记录

---

## ⚠️ 免责声明

**本系统仅供学习和研究使用，不构成任何投资建议！**

- 预测结果仅供参考
- 投资有风险，入市需谨慎
- 请根据自身情况做出投资决策
- 使用本系统造成的任何损失，开发者不承担责任

---

## 🎯 立即开始

```bash
# Windows用户
start_streamlit.bat

# Linux/Mac用户
./start_streamlit.sh

# 或手动启动
streamlit run app_streamlit.py --server.address 0.0.0.0
```

**访问地址会显示在终端中！📱**

---

## 💡 提示

- 💾 建议先运行数据下载和模型训练脚本
- 📱 添加到手机主屏幕可获得更好体验
- 🌐 使用ngrok可以随时随地访问
- 🔐 生产环境建议配置密码保护

**祝你使用愉快！📈🚀**
