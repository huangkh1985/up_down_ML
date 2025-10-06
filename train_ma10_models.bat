@echo off
chcp 65001 >nul
echo ========================================
echo MA10多时间窗口模型训练系统
echo ========================================
echo.
echo 本脚本将训练MA10版本的机器学习模型
echo 训练完成后，可在Streamlit应用中使用MA10策略
echo.
echo 按任意键开始训练...
pause >nul

python train_ma10_multi_horizon.py

echo.
echo ========================================
echo 训练完成！
echo ========================================
echo.
echo 模型文件已保存到: models\ma10_multi_horizon_models.pkl
echo.
echo 下一步：
echo 1. 运行 start_streamlit.bat 启动Web应用
echo 2. 在应用中选择"统一MA10"策略即可使用新训练的模型
echo.
pause

