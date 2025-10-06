@echo off
chcp 65001 >nul
echo.
echo ================================================================================
echo              GitHub 自动上传脚本
echo              包含数据和模型（约148MB）
echo ================================================================================
echo.

:: 检查Git是否安装
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到Git！
    echo        请先安装Git: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo [成功] Git已安装
git --version
echo.

:: 步骤1: 清理临时文件
echo ================================================================================
echo [步骤 1/7] 清理临时文件
echo ================================================================================
echo.

if exist "test_chinese_display.py" del "test_chinese_display.py"
if exist "test_pattern_recognition.py" del "test_pattern_recognition.py"
if exist "test_streamlit.py" del "test_streamlit.py"
if exist "test_pattern_result.csv" del "test_pattern_result.csv"
if exist "clean_emojis.py" del "clean_emojis.py"
if exist "fix_matplotlib_chinese.py" del "fix_matplotlib_chinese.py"
if exist "check_system_status.py" del "check_system_status.py"

if exist "__pycache__" rmdir /s /q "__pycache__"
if exist "utils\__pycache__" rmdir /s /q "utils\__pycache__"

echo [完成] 临时文件已清理
echo.
pause

:: 步骤2: 检查是否已初始化Git
echo ================================================================================
echo [步骤 2/7] 检查Git仓库状态
echo ================================================================================
echo.

if exist ".git" (
    echo [提示] 检测到已有Git仓库
    echo.
    set /p reinit="是否重新初始化？(会删除原有Git历史) Y/N: "
    if /i "%reinit%"=="Y" (
        rmdir /s /q ".git"
        echo [完成] 已删除原有Git仓库
        git init
        echo [完成] Git仓库已重新初始化
    ) else (
        echo [跳过] 保留现有Git仓库
    )
) else (
    git init
    echo [完成] Git仓库已初始化
)
echo.
pause

:: 步骤3: 配置Git用户信息（如果未配置）
echo ================================================================================
echo [步骤 3/7] 配置Git用户信息
echo ================================================================================
echo.

git config user.name >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] 需要配置Git用户信息
    set /p username="请输入您的用户名: "
    set /p email="请输入您的邮箱: "
    git config --global user.name "!username!"
    git config --global user.email "!email!"
    echo [完成] Git用户信息已配置
) else (
    echo [已配置] 用户名: 
    git config user.name
    echo           邮箱: 
    git config user.email
)
echo.
pause

:: 步骤4: 添加文件
echo ================================================================================
echo [步骤 4/7] 添加文件到Git
echo ================================================================================
echo.
echo [提示] 正在添加所有文件（包含数据和模型）...
echo        这可能需要30-60秒...
echo.

git add .

echo [完成] 文件已添加
echo.
echo 查看将要提交的文件:
echo -------------------------------------------------------------------------------
git status -s | findstr /C:"data/" /C:"models/" /C:".py" /C:".md"
echo -------------------------------------------------------------------------------
echo.

set /p continue="继续提交？(Y/N): "
if /i not "%continue%"=="Y" (
    echo [取消] 已取消上传
    pause
    exit /b 0
)
echo.

:: 步骤5: 提交到本地仓库
echo ================================================================================
echo [步骤 5/7] 提交到本地仓库
echo ================================================================================
echo.

set commit_msg=Initial commit: Stock prediction system with data and models

set /p custom_msg="提交信息（直接回车使用默认）: "
if not "%custom_msg%"=="" set commit_msg=%custom_msg%

git commit -m "%commit_msg%"

if %errorlevel% neq 0 (
    echo [错误] 提交失败！
    pause
    exit /b 1
)

echo.
echo [完成] 已提交到本地仓库
echo.
pause

:: 步骤6: 配置远程仓库
echo ================================================================================
echo [步骤 6/7] 配置远程GitHub仓库
echo ================================================================================
echo.
echo [重要] 请先在GitHub网站创建新仓库！
echo        访问: https://github.com/new
echo.
echo        创建仓库时：
echo        1. 填写仓库名称（如：stock-prediction-system）
echo        2. 选择Public（公开）或Private（私有）
echo        3. 不要勾选 "Initialize with README"
echo        4. 点击 "Create repository"
echo.
pause
echo.

set /p repo_url="请输入GitHub仓库地址（如：https://github.com/username/repo.git）: "

if "%repo_url%"=="" (
    echo [错误] 未输入仓库地址！
    pause
    exit /b 1
)

:: 检查是否已有远程仓库
git remote -v | findstr "origin" >nul 2>&1
if %errorlevel% equ 0 (
    echo [提示] 检测到已有远程仓库
    git remote remove origin
    echo [完成] 已删除旧的远程仓库配置
)

git remote add origin %repo_url%
git branch -M main

echo.
echo [完成] 远程仓库已配置
echo.
pause

:: 步骤7: 推送到GitHub
echo ================================================================================
echo [步骤 7/7] 推送到GitHub
echo ================================================================================
echo.
echo [重要提示]
echo -------------------------------------------------------------------------------
echo 即将上传约148MB数据到GitHub
echo 预计需要5-10分钟（取决于网速）
echo.
echo 上传内容:
echo   - 代码文件 (~2MB)
echo   - 数据文件 (~36MB)
echo   - 模型文件 (~110MB)
echo -------------------------------------------------------------------------------
echo.

set /p final_confirm="确认开始上传？(Y/N): "
if /i not "%final_confirm%"=="Y" (
    echo [取消] 已取消上传
    pause
    exit /b 0
)

echo.
echo [开始] 正在推送到GitHub...
echo        请保持网络连接稳定，不要中断...
echo.

git push -u origin main

if %errorlevel% neq 0 (
    echo.
    echo [错误] 推送失败！
    echo.
    echo 常见问题：
    echo   1. 需要登录GitHub账号（请按提示输入用户名和密码/Token）
    echo   2. 仓库地址错误（请检查是否创建了GitHub仓库）
    echo   3. 网络连接问题（请检查网络）
    echo   4. 需要Personal Access Token（密码验证已废弃）
    echo.
    echo 如何获取Personal Access Token：
    echo   1. 访问 https://github.com/settings/tokens
    echo   2. 点击 "Generate new token (classic)"
    echo   3. 勾选 "repo" 权限
    echo   4. 生成Token后复制保存
    echo   5. 推送时用Token代替密码
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [成功] 上传完成！
echo ================================================================================
echo.
echo 仓库地址: %repo_url%
echo 浏览地址: %repo_url:~0,-4%
echo.
echo 后续步骤:
echo   1. 访问您的GitHub仓库查看文件
echo   2. 编辑仓库描述和README
echo   3. 添加Topics标签（如：python, machine-learning, stock-prediction）
echo   4. 分享给其他人使用
echo.
echo 其他用户使用方式:
echo   git clone %repo_url%
echo   cd [仓库目录]
echo   pip install -r requirements_streamlit.txt
echo   streamlit run app_streamlit.py
echo.
echo ================================================================================
echo.
pause

