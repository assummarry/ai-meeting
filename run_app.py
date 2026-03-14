import os
import sys
from streamlit.web import cli as stcli

if __name__ == "__main__":
    # exe 运行时自动切换到正确目录
    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 启动 workshop.py（你的原入口）
    sys.argv = [
        "streamlit", "run", "workshop.py",
        "--server.headless=true",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
        "--global.developmentMode=false"
    ]
    stcli.main()