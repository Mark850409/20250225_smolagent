from recommendation_analysis import analyze_recommendation_results, agent
import gradio as gr
import pandas as pd
import os
from datetime import datetime
import json
from dotenv import load_dotenv
from typing import Generator, Union

# 載入環境變數
load_dotenv()

def process_recommendation_data(csv_path: str, progress: str = "") -> Generator[str, None, None]:
    """處理推薦系統分析的主要流程，支援即時更新進度。

    Args:
        csv_path (str): CSV 檔案路徑
        progress (str, optional): 目前的進度文字. Defaults to "".

    Yields:
        str: 更新的進度文字
    """
    progress_text = progress
    progress_text += "開始推薦系統效能分析任務...\n\n"
    yield progress_text
    
    try:
        # 讀取實驗結果
        df = pd.read_csv(csv_path)
        
        if df.empty:
            progress_text += "警告：CSV 檔案是空的\n"
            yield progress_text
            return progress_text
            
        progress_text += "成功讀取數據：\n"
        progress_text += str(df) + "\n\n"
        yield progress_text
        
        # 讀取實驗設定
        settings_path = 'experiment_settings.json'
        settings = {}
        if os.path.exists(settings_path):
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                progress_text += "成功讀取實驗設定\n\n"
        else:
            progress_text += "警告：找不到實驗設定檔案\n\n"
        yield progress_text
        
        # 轉換實驗結果為分析所需格式
        analysis_data = {
            'experiment_settings': settings,
            'experiment_results': {
                'rating_only': {},
                'sentiment_only': {},
                'hybrid': {}
            }
        }
        
        # 安全地獲取數據
        metrics = ['RMSE', 'MAE', 'precision', 'recall', 'ndcg', 'f1']
        methods = ['rating_only', 'sentiment_only', 'hybrid']
        
        progress_text += "正在處理數據...\n"
        yield progress_text
        
        for method in methods:
            for metric in metrics:
                try:
                    value = df.loc[df['metric'].str.lower() == metric.lower(), method].iloc[0]
                    analysis_data['experiment_results'][method][metric.lower()] = float(value)
                except (IndexError, KeyError) as e:
                    progress_text += f"警告：無法獲取 {method} 的 {metric} 指標\n"
                    analysis_data['experiment_results'][method][metric.lower()] = 0.0
                yield progress_text
        
        # 準備可序列化的數據
        serializable_data = {
            'experiment_settings': settings,
            'experiment_results': {
                method: {
                    metric: float(value)
                    for metric, value in metrics_dict.items()
                }
                for method, metrics_dict in analysis_data['experiment_results'].items()
            }
        }
        
        # 使用工具函數進行初步分析
        progress_text += "\n正在進行初步分析...\n"
        yield progress_text
        
        initial_analysis = analyze_recommendation_results(serializable_data)
        progress_text += "\n初步分析完成。\n"
        yield progress_text
        
        # 使用 agent 進行深入分析
        progress_text += "\n正在使用 AI 進行深入分析...\n"
        yield progress_text
        
        analysis = agent.run(f"""
請簡要分析以下推薦系統實驗結果（請控制在 500 字以內）：

{initial_analysis}

重點分析：
1. 最佳推薦方法的選擇
2. 關鍵指標的表現
3. 具體改進建議

請使用繁體中文回應。
""")
        
        # 格式化輸出
        progress_text += "\n" + "="*80 + "\n"
        progress_text += " "*30 + "推薦系統效能分析報告" + " "*30 + "\n"
        progress_text += "="*80 + "\n\n"
        
        progress_text += "=== 初步分析 ===\n"
        progress_text += initial_analysis + "\n\n"
        
        progress_text += "=== 深入分析 ===\n"
        progress_text += str(analysis) + "\n"
        
        progress_text += "\n" + "="*80 + "\n"
        progress_text += " "*30 + "報告結束" + " "*30 + "\n"
        progress_text += "="*80 + "\n"
        
        # 儲存分析報告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_report_{timestamp}.json"
        filepath = os.path.join("recommendation_analysis", filename)
        
        os.makedirs("recommendation_analysis", exist_ok=True)
        
        report_data = {
            "timestamp": timestamp,
            "metrics": {
                "experiment_results": serializable_data['experiment_results']
            },
            "initial_analysis": initial_analysis,
            "final_analysis": analysis
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        progress_text += f"\n\n分析報告已儲存至: {filepath}"
        yield progress_text
        
    except Exception as e:
        progress_text += f"\n分析過程發生錯誤: {str(e)}\n"
        yield progress_text

def list_analysis_reports():
    """列出所有已生成的分析報告"""
    report_dir = "recommendation_analysis"
    if not os.path.exists(report_dir):
        return []
    files = [f for f in os.listdir(report_dir) if f.endswith('.json')]
    return sorted(files, reverse=True)  # 最新的檔案排在最前面

def view_report_content(report_name):
    """查看分析報告內容"""
    if not report_name:
        return "請選擇一個報告檔案"
    try:
        report_path = os.path.join("recommendation_analysis", report_name)
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        output = "=== 推薦系統效能分析報告 ===\n\n"
        output += f"生成時間：{report_data['timestamp']}\n\n"
        output += "=== 初步分析 ===\n"
        output += report_data['initial_analysis'] + "\n\n"
        output += "=== 深入分析 ===\n"
        output += str(report_data['final_analysis'])
        
        return output
    except Exception as e:
        return f"讀取報告錯誤：{str(e)}"

# 創建 Gradio 介面
with gr.Blocks(title="推薦系統效能分析工具", theme=gr.themes.Soft()) as web_ui:
    gr.Markdown("""
    # 推薦系統效能分析工具
    這是一個基於 AI 的推薦系統效能分析工具，可以分析和比較不同推薦方法的效能指標。
    """)
    
    with gr.Tabs():
        # 效能分析標籤
        with gr.Tab("效能分析"):
            gr.Markdown("### 選擇要分析的實驗結果檔案")
            
            with gr.Column():
                file_input = gr.File(
                    label="上傳 CSV 檔案",
                    file_types=[".csv"]
                )
                process_button = gr.Button("開始分析")
                progress_output = gr.Textbox(
                    label="分析進度和結果",
                    lines=30,
                    value=""
                )
                
                process_button.click(
                    fn=process_recommendation_data,
                    inputs=[file_input],
                    outputs=progress_output,
                    show_progress=True
                )
        
        # 報告查看標籤
        with gr.Tab("報告查看"):
            gr.Markdown("### 查看已生成的分析報告")
            with gr.Row():
                files = list_analysis_reports()
                report_dropdown = gr.Dropdown(
                    choices=files,
                    value=files[0] if files else None,
                    label="選擇報告檔案",
                    interactive=True
                )
                refresh_button = gr.Button("刷新報告列表")
            
            report_output = gr.Textbox(
                label="報告內容",
                lines=30,
                value=view_report_content(files[0]) if files else None
            )
            
            def update_report_list():
                files = list_analysis_reports()
                return gr.update(choices=files, value=files[0] if files else None)
            
            refresh_button.click(
                fn=update_report_list,
                outputs=report_dropdown
            )
            report_dropdown.change(
                fn=view_report_content,
                inputs=report_dropdown,
                outputs=report_output
            )

# 啟動 Web 介面
if __name__ == "__main__":
    web_ui.launch(share=True) 