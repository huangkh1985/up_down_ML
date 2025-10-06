# 📱 Streamlit Web应用部署指南

## 🎯 概述

本指南将帮助你将股票预测系统部署为Web应用，支持：
- ✅ PC浏览器访问
- ✅ 手机浏览器访问
- ✅ 局域网内其他设备访问
- ✅ 云服务器公网访问（可选）

---

## 🚀 快速启动

### 方法1: 使用启动脚本（推荐）

#### Windows系统
```bash
# 双击运行
start_streamlit.bat

# 或在命令行运行
start_streamlit.bat
```

#### Linux/Mac系统
```bash
chmod +x start_streamlit.sh
./start_streamlit.sh
```

### 方法2: 手动启动

```bash
# 1. 安装依赖
pip install -r requirements_streamlit.txt

# 2. 启动应用
streamlit run app_streamlit.py --server.port 8501 --server.address 0.0.0.0
```

---

## 📱 访问方式

### 1. 本机访问

启动后，在浏览器中打开：
```
http://localhost:8501
```

### 2. 手机访问（同一局域网）

#### 步骤1: 获取电脑IP地址

**Windows:**
```powershell
ipconfig
# 找到 IPv4 地址，例如: 192.168.1.100
```

**Linux/Mac:**
```bash
ifconfig
# 或
ip addr show
```

#### 步骤2: 在手机浏览器访问

在手机浏览器中打开：
```
http://你的电脑IP:8501
# 例如: http://192.168.1.100:8501
```

#### 注意事项：
- ⚠️ 手机和电脑必须连接到同一个WiFi网络
- ⚠️ 确保电脑防火墙允许8501端口访问

### 3. 外网访问（高级）

如果需要从外网访问，有以下几种方案：

#### 方案A: 使用内网穿透（推荐新手）

1. **使用Ngrok**
```bash
# 安装ngrok
# 访问 https://ngrok.com/ 注册并下载

# 启动内网穿透
ngrok http 8501

# 会生成一个公网地址，例如: https://xxxx.ngrok.io
```

2. **使用FRP**
```bash
# 需要有公网服务器
# 配置frp客户端和服务端
```

#### 方案B: 云服务器部署（推荐生产环境）

详见后续"云服务器部署"章节。

---

## 🔧 配置说明

### 端口配置

修改端口号：
```bash
streamlit run app_streamlit.py --server.port 你的端口号
```

### 性能优化

修改缓存设置（在`app_streamlit.py`中）：
```python
@st.cache_data(ttl=3600)  # 缓存1小时
def get_stock_data(_self, stock_code):
    # 数据获取函数
```

### 主题配置

创建`.streamlit/config.toml`文件：
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
address = "0.0.0.0"
maxUploadSize = 200
```

---

## ☁️ 云服务器部署

### 准备工作

1. **购买云服务器**
   - 阿里云、腾讯云、华为云等
   - 推荐配置：2核4G内存，Ubuntu 20.04

2. **连接服务器**
```bash
ssh root@你的服务器IP
```

### 部署步骤

#### 1. 安装依赖
```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装Python 3.9+
sudo apt install python3.9 python3-pip -y

# 安装git
sudo apt install git -y
```

#### 2. 上传代码

**方法A: 使用git**
```bash
cd /opt
git clone 你的代码仓库
cd 项目目录
```

**方法B: 使用SCP上传**
```bash
# 在本地电脑运行
scp -r D:\user_data\Deeplearn\up_down_stock_analyst root@你的服务器IP:/opt/
```

#### 3. 安装Python依赖
```bash
pip3 install -r requirements_streamlit.txt
```

#### 4. 配置防火墙
```bash
# 开放8501端口
sudo ufw allow 8501
sudo ufw reload
```

#### 5. 使用Screen保持后台运行
```bash
# 安装screen
sudo apt install screen -y

# 创建新会话
screen -S streamlit

# 启动应用
streamlit run app_streamlit.py --server.port 8501 --server.address 0.0.0.0

# 按 Ctrl+A+D 退出会话（应用继续运行）

# 重新连接会话
screen -r streamlit
```

#### 6. 配置Nginx反向代理（可选）

```bash
# 安装Nginx
sudo apt install nginx -y

# 创建配置文件
sudo nano /etc/nginx/sites-available/streamlit
```

配置内容：
```nginx
server {
    listen 80;
    server_name 你的域名或IP;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

启用配置：
```bash
sudo ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### 7. 配置SSL证书（推荐）

```bash
# 安装certbot
sudo apt install certbot python3-certbot-nginx -y

# 获取证书
sudo certbot --nginx -d 你的域名
```

---

## 🔒 安全配置

### 1. 添加密码保护

创建`.streamlit/secrets.toml`：
```toml
password = "你的密码"
```

在`app_streamlit.py`中添加：
```python
import streamlit as st

def check_password():
    """密码验证"""
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "请输入密码", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "请输入密码", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("密码错误")
        return False
    else:
        return True

# 在main函数开始处添加
if not check_password():
    st.stop()
```

### 2. 限制访问IP（Nginx）

```nginx
location / {
    allow 你的IP;
    deny all;
    # ... 其他配置
}
```

### 3. 配置HTTPS

使用Let's Encrypt免费证书（见上文SSL配置）

---

## 📊 监控和维护

### 查看日志

```bash
# 查看streamlit日志
tail -f ~/.streamlit/logs/streamlit.log

# 查看Nginx日志
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### 自动重启（systemd）

创建服务文件：
```bash
sudo nano /etc/systemd/system/streamlit.service
```

内容：
```ini
[Unit]
Description=Streamlit Stock Prediction
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/up_down_stock_analyst
ExecStart=/usr/local/bin/streamlit run app_streamlit.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable streamlit
sudo systemctl start streamlit
sudo systemctl status streamlit
```

---

## 🐛 常见问题

### 1. 端口被占用

**问题：** `OSError: [Errno 98] Address already in use`

**解决：**
```bash
# 查找占用端口的进程
netstat -tulpn | grep 8501

# 或使用lsof
lsof -i :8501

# 杀死进程
kill -9 进程ID
```

### 2. 防火墙阻止访问

**Windows:**
```powershell
# 添加防火墙规则
netsh advfirewall firewall add rule name="Streamlit" dir=in action=allow protocol=TCP localport=8501
```

**Linux:**
```bash
sudo ufw allow 8501
sudo ufw reload
```

### 3. 数据不刷新

清除缓存：
```bash
streamlit cache clear
```

或在应用中按 `C` 键清除缓存。

### 4. 内存不足

优化缓存设置：
```python
@st.cache_data(ttl=1800, max_entries=10)  # 减少缓存数量
```

### 5. 中文显示乱码

确保服务器安装了中文字体：
```bash
# Ubuntu/Debian
sudo apt install fonts-wqy-zenhei fonts-wqy-microhei

# CentOS/RHEL
sudo yum install wqy-zenhei-fonts wqy-microhei-fonts
```

---

## 📈 性能优化

### 1. 使用缓存

```python
@st.cache_data(ttl=3600)  # 缓存1小时
def expensive_computation():
    # 耗时操作
    pass
```

### 2. 异步加载

```python
import asyncio

async def load_data():
    # 异步加载数据
    pass
```

### 3. 使用数据库

将频繁查询的数据存入Redis或SQLite：
```python
import redis

r = redis.Redis(host='localhost', port=6379, db=0)

@st.cache_data(ttl=600)
def get_cached_prediction(stock_code):
    cached = r.get(stock_code)
    if cached:
        return pickle.loads(cached)
    
    # 计算预测
    result = make_prediction(stock_code)
    r.setex(stock_code, 600, pickle.dumps(result))
    return result
```

---

## 🎨 界面定制

### 自定义CSS

```python
st.markdown("""
<style>
    /* 自定义样式 */
    .stButton button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
    }
    
    /* 移动端适配 */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.2rem;
        }
    }
</style>
""", unsafe_allow_html=True)
```

### 添加Logo

```python
st.sidebar.image("logo.png", width=200)
```

---

## 📱 移动端优化建议

1. **响应式布局**
   - 使用`st.columns()`自适应布局
   - 使用`use_container_width=True`

2. **简化界面**
   - 移动端显示核心信息
   - 使用`st.tabs()`组织内容

3. **减少加载时间**
   - 优化图片大小
   - 使用懒加载

4. **触摸友好**
   - 增大按钮尺寸
   - 增加间距

---

## 🔄 更新和备份

### 更新应用

```bash
# 拉取最新代码
git pull

# 重启服务
sudo systemctl restart streamlit
```

### 数据备份

```bash
# 创建备份脚本
#!/bin/bash
tar -czf backup_$(date +%Y%m%d).tar.gz data/ models/

# 添加定时任务
crontab -e
# 每天凌晨2点备份
0 2 * * * /path/to/backup.sh
```

---

## 📞 技术支持

### 官方文档
- Streamlit: https://docs.streamlit.io/
- Nginx: https://nginx.org/en/docs/

### 社区支持
- Streamlit论坛: https://discuss.streamlit.io/
- GitHub Issues: 项目仓库

---

## 🎉 总结

### 本地开发
```bash
streamlit run app_streamlit.py
```

### 局域网访问
```bash
streamlit run app_streamlit.py --server.address 0.0.0.0
```

### 生产部署
```bash
# 使用systemd + Nginx + SSL
sudo systemctl start streamlit
```

### 外网访问
```bash
# 使用ngrok临时穿透
ngrok http 8501

# 或使用云服务器 + 域名 + SSL证书
```

---

**🎯 现在就开始体验你的Web应用吧！**

```bash
# 一键启动
start_streamlit.bat
```

访问地址会自动显示在命令行中 🚀
