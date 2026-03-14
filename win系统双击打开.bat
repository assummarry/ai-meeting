@echo off
chcp 65001 >nul
echo =============================================
echo 🏗️ 多智能体白皮书工坊 v4 启动器 (Windows)
echo =============================================
echo.

:: 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python 未安装或未加入 PATH！
    echo 请先去 python.org 下载安装 Python 3.10+ 并勾选 "Add to PATH"
    pause
    exit /b 1
)

:: 创建虚拟环境（首次运行）
if not exist venv (
    echo 🔧 首次启动，正在创建虚拟环境 venv...
    python -m venv venv
    echo ✅ 虚拟环境创建完成！
)

:: 激活虚拟环境
call venv\Scripts\activate.bat

:: 安装/更新依赖（包含 langgraph）
echo 📦 正在安装依赖（streamlit + openai + langgraph）...
pip install -r requirements.txt --upgrade --quiet

echo.
echo 🚀 启动 Streamlit v4（workshop.py）...
echo.
streamlit run workshop.py

echo.
echo =============================================
echo ✅ 已退出，按任意键关闭窗口
echo =============================================
pause