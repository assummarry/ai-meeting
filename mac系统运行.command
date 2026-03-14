#!/bin/bash
echo "============================================="
echo "🏗️ 多智能体白皮书工坊 v4 启动器 (Mac)"
echo "============================================="

echo "当前 Python: $(python --version 2>/dev/null || python3 --version)"

# 如果 venv 不存在就创建（用 pyenv 当前 python）
if [ ! -d "venv" ]; then
    echo "🔧 首次启动，正在创建虚拟环境 venv..."
    python -m venv venv || python3 -m venv venv
    echo "✅ 虚拟环境创建完成！"
fi

# 激活虚拟环境
source venv/bin/activate

# 安装/更新依赖（包含 langgraph）
echo "📦 正在安装依赖（streamlit + openai + langgraph）..."
pip install -r requirements.txt --upgrade --quiet

echo ""
echo "🚀 启动 workshop.py（请稍等，浏览器会自动打开）..."
echo ""
streamlit run workshop.py

echo ""
echo "============================================="
echo "✅ 程序已退出"
read -n 1 -s -r -p "按任意键关闭窗口..."