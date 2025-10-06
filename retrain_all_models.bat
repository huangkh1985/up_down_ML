@echo off
chcp 65001 >nul
echo.
echo ================================================================================
echo          MA10+MA20多时间窗口模型 - 一键重训练系统
echo ================================================================================
echo.

:: 获取当前日期（格式：YYYYMMDD）
set today=%date:~0,4%%date:~5,2%%date:~8,2%

echo [步骤 1/6] 备份现有模型...
echo -------------------------------------------------------------------------------
if not exist "models\backup_%today%" mkdir "models\backup_%today%"
copy "models\*.pkl" "models\backup_%today%\" >nul 2>&1
if %errorlevel%==0 (
    echo [成功] 模型已备份到: models\backup_%today%\
) else (
    echo [警告] 备份失败或没有旧模型
)
echo.

echo [步骤 2/6] 下载最新股票数据...
echo -------------------------------------------------------------------------------
python stock_data_downloader.py
if %errorlevel% neq 0 (
    echo [错误] 数据下载失败！
    pause
    exit /b 1
)
echo.

echo [步骤 3/6] 训练MA20多时间窗口模型...
echo -------------------------------------------------------------------------------
python train_ma20_multi_horizon.py
if %errorlevel% neq 0 (
    echo [错误] MA20模型训练失败！
    echo [提示] 如需回滚，运行: copy models\backup_%today%\*.pkl models\
    pause
    exit /b 1
)
echo.

echo [步骤 4/6] 训练MA10多时间窗口模型...
echo -------------------------------------------------------------------------------
python train_ma10_multi_horizon.py
if %errorlevel% neq 0 (
    echo [错误] MA10模型训练失败！
    echo [提示] 如需回滚，运行: copy models\backup_%today%\*.pkl models\
    pause
    exit /b 1
)
echo.

echo [步骤 5/6] 验证新训练的模型...
echo -------------------------------------------------------------------------------
if exist "validate_models.py" (
    python validate_models.py
) else (
    echo [跳过] 验证脚本不存在，请手动验证
)
echo.

echo [步骤 6/6] 完成总结...
echo ===============================================================================
echo.
echo [成功] 所有模型已重新训练完成！
echo.
echo 新模型位置:
echo   - models\ma20_multi_horizon_models.pkl  (MA20多时间窗口模型 - 4个独立模型)
echo   - models\ma10_multi_horizon_models.pkl  (MA10多时间窗口模型 - 4个独立模型)
echo.
echo 旧模型备份:
echo   - models\backup_%today%\                (如需回滚请从这里恢复)
echo.
echo 支持的MA策略:
echo   - 动态选择: 1-3天用MA10，5-10天用MA20 (推荐)
echo   - 统一MA10: 所有预测窗口使用MA10 (短期敏感)
echo   - 统一MA20: 所有预测窗口使用MA20 (中期稳定)
echo.
echo 下一步操作:
echo   1. 运行 streamlit run app_streamlit.py
echo   2. 在侧边栏选择MA周期策略
echo   3. 测试几个股票的预测效果
echo   4. 如果效果不佳，运行回滚脚本
echo.
echo ===============================================================================
echo.
pause

