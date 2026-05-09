@echo off
echo 正在启动疾病诊断系统后端服务...
echo.
echo 请确保已安装Python依赖: pip install -r requirements.txt
echo 请确保已配置.env文件（参考config_example.txt）
echo.
python app.py
pause

