@echo off
chcp 65001 >nul
echo.
echo ================================================================================
echo              GitHub上传前项目清理脚本
echo ================================================================================
echo.
echo 本脚本将删除不需要的临时文件和测试文件
echo 不会删除data/models目录（这些会被.gitignore自动忽略）
echo.

set /p confirm="确认执行清理？(Y/N): "
if /i not "%confirm%"=="Y" (
    echo 取消清理
    pause
    exit /b 0
)

echo.
echo [1/4] 删除测试文件...
echo -------------------------------------------------------------------------------
if exist "test_chinese_display.py" (
    del "test_chinese_display.py"
    echo [删除] test_chinese_display.py
)
if exist "test_pattern_recognition.py" (
    del "test_pattern_recognition.py"
    echo [删除] test_pattern_recognition.py
)
if exist "test_streamlit.py" (
    del "test_streamlit.py"
    echo [删除] test_streamlit.py
)
if exist "test_pattern_result.csv" (
    del "test_pattern_result.csv"
    echo [删除] test_pattern_result.csv
)

echo.
echo [2/4] 删除一次性清理脚本...
echo -------------------------------------------------------------------------------
if exist "clean_emojis.py" (
    del "clean_emojis.py"
    echo [删除] clean_emojis.py
)
if exist "fix_matplotlib_chinese.py" (
    del "fix_matplotlib_chinese.py"
    echo [删除] fix_matplotlib_chinese.py
)
if exist "check_system_status.py" (
    del "check_system_status.py"
    echo [删除] check_system_status.py
)

echo.
echo [3/4] 创建占位符文件...
echo -------------------------------------------------------------------------------
if not exist "data\.gitkeep" (
    type nul > "data\.gitkeep"
    echo [创建] data\.gitkeep
)
if not exist "models\.gitkeep" (
    type nul > "models\.gitkeep"
    echo [创建] models\.gitkeep
)
if not exist "results\.gitkeep" (
    type nul > "results\.gitkeep"
    echo [创建] results\.gitkeep
)

echo.
echo [4/4] 清理Python缓存...
echo -------------------------------------------------------------------------------
if exist "__pycache__" (
    rmdir /s /q "__pycache__"
    echo [删除] __pycache__/
)
if exist "utils\__pycache__" (
    rmdir /s /q "utils\__pycache__"
    echo [删除] utils\__pycache__/
)

echo.
echo ================================================================================
echo [成功] 清理完成！
echo ================================================================================
echo.
echo 已删除的文件:
echo   - 测试文件 (test_*.py, test_*.csv)
echo   - 一次性脚本 (clean_*.py, fix_*.py)
echo   - Python缓存 (__pycache__)
echo.
echo 已创建的占位符:
echo   - data/.gitkeep
echo   - models/.gitkeep
echo   - results/.gitkeep
echo.
echo 不会上传到GitHub的文件（已在.gitignore中）:
echo   - data/*.pkl, data/*.csv （数据文件）
echo   - models/*.pkl （模型文件）
echo   - results/*.png （结果图片）
echo.
echo 下一步:
echo   1. 检查 .gitignore 文件
echo   2. 运行: git init
echo   3. 运行: git add .
echo   4. 运行: git status （确认没有大文件）
echo   5. 运行: git commit -m "Initial commit"
echo.
echo ================================================================================
echo.
pause

