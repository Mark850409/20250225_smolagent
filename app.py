import os
from dotenv import load_dotenv
from mbti_browser_gradio import web_ui

def main():
    """主程序入口"""
    try:
        # 載入環境變數
        load_dotenv()
        
        # 檢查必要的目錄
        required_dirs = [
            "mbti_data",
            "mbti_data/csv",
            "mbti_data/vectors"
        ]
        
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
        
        # 檢查環境變數
        required_env_vars = [
            "HUGGINGFACE_API_KEY",
            "HUGGINGFACE_MODEL",
            "EMBEDDING_MODEL_NAME"
        ]
        
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"缺少必要的環境變數: {', '.join(missing_vars)}")
        
        # 啟動 Gradio 介面
        print("正在啟動 MBTI 分析工具...")
        web_ui.launch(
            server_name="0.0.0.0",  # 允許外部訪問
            server_port=7860,       # 指定端口
            share=True,             # 生成公開連結
            debug=True             # 啟用調試模式
        )
        
    except Exception as e:
        print(f"啟動失敗: {str(e)}")
        raise

if __name__ == "__main__":
    main() 