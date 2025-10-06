@echo off
:: 快速上传脚本 - 最少交互
chcp 65001 >nul
cls
echo.
echo ============================================================
echo     GitHub 快速上传（包含数据+模型 ~148MB）
echo ============================================================
echo.

:: 获取GitHub仓库地址
set /p repo_url="输入GitHub仓库地址: "
if "%repo_url%"=="" (
    echo [错误] 未输入仓库地址！
    pause
    exit /b 1
)

echo.
echo [1/4] 清理临时文件...
del test_*.py 2>nul
del test_*.csv 2>nul
del clean_*.py 2>nul
del fix_*.py 2>nul
del check_*.py 2>nul
rmdir /s /q __pycache__ 2>nul
rmdir /s /q utils\__pycache__ 2>nul

echo [2/4] Git初始化...
if exist ".git" rmdir /s /q ".git"
git init
git branch -M main

echo [3/4] 添加文件并提交...
git add .
git commit -m "Initial commit: Stock prediction system with data and models"

echo [4/4] 推送到GitHub（约5-10分钟）...
git remote add origin %repo_url%
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ============================================================
    echo [成功] 上传完成！
    echo 访问: %repo_url:~0,-4%
    echo ============================================================
) else (
    echo.
    echo [失败] 推送失败，请检查：
    echo   1. GitHub仓库是否已创建
    echo   2. 是否需要登录（用户名 + Personal Access Token）
    echo   3. 网络连接是否正常
)

echo.
pause

