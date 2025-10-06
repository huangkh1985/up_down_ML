@echo off
chcp 65001 >nul
echo.
echo ================================================================================
echo                      模型回滚脚本
echo ================================================================================
echo.

:: 查找最新的备份目录
echo [1/3] 查找备份...
echo -------------------------------------------------------------------------------

if not exist "models\backup_*" (
    echo [错误] 没有找到备份文件夹
    echo        请确保之前已经创建过备份
    pause
    exit /b 1
)

:: 列出所有备份
echo 可用的备份:
echo.
dir /B /AD models\backup_* 2>nul
echo.

:: 让用户选择备份日期
set /p backup_date="请输入要恢复的备份日期 (格式: 20251005): "

if not exist "models\backup_%backup_date%" (
    echo [错误] 找不到备份文件夹: models\backup_%backup_date%
    pause
    exit /b 1
)

echo.
echo [2/3] 恢复模型...
echo -------------------------------------------------------------------------------
echo 从 models\backup_%backup_date%\ 恢复...
copy "models\backup_%backup_date%\*.pkl" "models\" /Y
if %errorlevel%==0 (
    echo [成功] 模型已恢复
) else (
    echo [错误] 恢复失败
    pause
    exit /b 1
)

echo.
echo [3/3] 完成
echo ===============================================================================
echo.
echo [成功] 模型已回滚到备份: %backup_date%
echo.
echo 下一步:
echo   1. 重启Streamlit应用
echo   2. 在应用中点击 "Clear cache" 清除缓存
echo   3. 测试预测功能
echo.
echo ===============================================================================
echo.
pause

