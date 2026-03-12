@echo off
chcp 65001 >nul
title AI 虚拟董事会启动器

echo ========================================
echo   🚀 正在初始化 AI 虚拟董事会...
echo ========================================

:: 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未检测到 Python，请先安装 Python 3.8 或以上版本，并勾选 "Add to PATH"。
    pause
    exit /b
)

:: 检查并创建虚拟环境
if not exist "venv" (
    echo 📦 首次运行，正在创建独立的 Python 虚拟环境 (这可能需要一小会儿)...
    python -m venv venv
)

:: 激活虚拟环境并静默安装依赖
echo 📥 正在检查核心组件和依赖...
call venv\Scripts\activate
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q

echo 🟢 环境就绪！正在拉起浏览器...
echo ========================================
streamlit run app.py

pause