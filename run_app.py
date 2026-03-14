import os
import sys
from streamlit.web import cli as stcli

if __name__ == "__main__":
    # 判断是否被 PyInstaller 打包
    if getattr(sys, 'frozen', False):
        # --onefile 模式下的临时解压目录
        base_path = sys._MEIPASS
    else:
        # 开发环境下的当前目录
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    # 拼接真正的入口文件绝对路径
    script_path = os.path.join(base_path, "workshop.py")
    
    # 切换工作目录至解压目录，确保内部相对路径资源读取正常
    os.chdir(base_path)
    
    # 启动 workshop.py
    sys.argv = [
        "streamlit", "run", script_path,
        "--server.headless=true",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
        "--global.developmentMode=false"
    ]
    sys.exit(stcli.main())