from flask import Flask
import os
from dotenv import load_dotenv
from mbti_browser_gradio import web_ui

# 初始化 Flask 應用
app = Flask(__name__)

# 載入環境變數
load_dotenv()

@app.route('/')
def index():
    """主路由：啟動 Gradio 介面"""
    try:
        
        required_env_vars = [
            "HUGGINGFACE_API_KEY",
            "HUGGINGFACE_MODEL",
            "EMBEDDING_MODEL_NAME"
        ]
        
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"缺少必要的環境變數: {', '.join(missing_vars)}")
        
        required_dirs = [
        "mbti_data",
        "mbti_data/csv",
        "mbti_data/vectors"
        ]
        
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
        # 啟動 Gradio 介面
        web_ui.launch(
            server_name="0.0.0.0",
            server_port=int(os.getenv("PORT", 7860)),
            share=False,
            debug=False
        )
        
        return "MBTI 分析工具已啟動"
    except Exception as e:
        return f"啟動失敗: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000), host='0.0.0.0')