@echo off
echo ========================================
echo 启动股票预测Streamlit应用
echo ========================================
echo.

REM 检查是否安装了streamlit
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [!] 未检测到streamlit，正在安装...
    pip install -r requirements_streamlit.txt
)

echo [OK] 正在启动应用...
echo.
echo ========================================
echo 访问地址:
echo   本地访问: http://localhost:8501
echo   手机访问: http://你的电脑IP:8501
echo ========================================
echo.
echo 按 Ctrl+C 停止服务
echo.

REM 启动streamlit
streamlit run app_streamlit.py --server.port 8501 --server.address 0.0.0.0

pause
