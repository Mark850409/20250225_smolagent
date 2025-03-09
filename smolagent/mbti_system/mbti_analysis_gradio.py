from mbti_analysis_scraper import (scrape_mbti_data, save_and_analyze_mbti, 
                         create_vector_store, query_mbti_data)
import gradio as gr
import pandas as pd
import os
from datetime import datetime

def scrape_and_save():
    """爬取並保存 MBTI 數據"""
    # 爬取數據
    data = scrape_mbti_data()
    if "error" in data:
        return f"錯誤：{data['error']}"
    
    # 保存數據
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mbti_analysis_{timestamp}.csv"
    result = save_and_analyze_mbti(data, filename)
    
    return result

def create_vectors(csv_file):
    """從 CSV 檔案創建向量存儲"""
    if csv_file is None:
        return "請先上傳 CSV 檔案"
    
    result = create_vector_store(csv_file.name)
    return result

def query_data(query_text):
    """查詢 MBTI 數據"""
    if not query_text:
        return "請輸入查詢內容"
    
    result = query_mbti_data(query_text)
    return result

def list_csv_files():
    """列出所有可用的 CSV 檔案"""
    csv_dir = "mbti_data/csv"
    if not os.path.exists(csv_dir):
        return []
    return [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]

def view_csv_content(csv_path):
    """查看 CSV 檔案內容"""
    if not csv_path:
        return "請選擇 CSV 檔案"
    
    try:
        df = pd.read_csv(csv_path)
        return df.to_string()
    except Exception as e:
        return f"無法讀取檔案：{str(e)}"

# 創建 Gradio 介面
with gr.Blocks(title="MBTI 性格分析工具", theme=gr.themes.Soft()) as web_ui:
    gr.Markdown("""
    # MBTI 性格分析工具
    這是一個基於 Python 的 MBTI（Myers-Briggs Type Indicator）性格分析工具，可以爬取、分析和查詢 MBTI 相關數據。
    """)
    
    with gr.Tabs():
        # 數據爬取標籤
        with gr.Tab("數據爬取"):
            gr.Markdown("### 爬取 MBTI 數據")
            scrape_button = gr.Button("開始爬取")
            scrape_output = gr.Textbox(label="爬取結果", lines=10)
            scrape_button.click(scrape_and_save, outputs=scrape_output)
        
        # 數據查看標籤
        with gr.Tab("數據查看"):
            gr.Markdown("### 查看已保存的數據")
            csv_dropdown = gr.Dropdown(
                choices=list_csv_files(),
                label="選擇 CSV 檔案",
                interactive=True
            )
            refresh_button = gr.Button("刷新檔案列表")
            view_output = gr.Textbox(label="檔案內容", lines=15)
            
            refresh_button.click(
                lambda: gr.update(choices=list_csv_files()), 
                outputs=csv_dropdown
            )
            csv_dropdown.change(view_csv_content, csv_dropdown, view_output)
        
        # 向量存儲標籤
        with gr.Tab("向量存儲"):
            gr.Markdown("### 創建向量存儲")
            file_input = gr.File(label="上傳 CSV 檔案")
            create_button = gr.Button("創建向量存儲")
            create_output = gr.Textbox(label="創建結果")
            create_button.click(create_vectors, file_input, create_output)
        
        # 數據查詢標籤
        with gr.Tab("數據查詢"):
            gr.Markdown("### 查詢 MBTI 數據")
            query_input = gr.Textbox(
                label="輸入查詢內容",
                placeholder="例如：INTJ 的職業傾向是什麼？"
            )
            query_button = gr.Button("執行查詢")
            query_output = gr.Textbox(label="查詢結果", lines=10)
            query_button.click(query_data, query_input, query_output)

# 啟動 Web 介面
if __name__ == "__main__":
    web_ui.launch(share=True) 