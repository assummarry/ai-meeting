#!/bin/bash

# 确保脚本在当前目录执行
cd "$(dirname "$0")"

echo "========================================"
echo "  🚀 正在初始化 AI 虚拟董事会..."
echo "========================================"

# 检查 Python3 是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ 未检测到 Python3，请先在 Mac 上安装 Python 环境。"
    exit 1
fi

# 检查并创建虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 首次运行，正在创建独立的 Python 虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境并静默安装依赖
echo "📥 正在检查核心组件和依赖..."
source venv/bin/activate
pip3 install --upgrade pip -q
pip3 install -r requirements.txt -q

echo "🟢 环境就绪！正在拉起浏览器..."
echo "========================================"
streamlit run app.py