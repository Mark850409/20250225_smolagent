from mbti_browser_scraper import (browse_webpage, extract_mbti_info, save_data, 
                                create_vector_store, query_mbti_data, agent)
import gradio as gr
import pandas as pd
import os
from datetime import datetime
from concurrent.futures import TimeoutError

def process_mbti_data(urls: str, progress: str = "") -> str:
    """處理 MBTI 數據的主要流程，支援即時更新進度"""
    
    # 初始化進度文本
    progress_text = progress
    
    # 分割並清理 URL
    target_urls = [url.strip() for url in urls.split('\n') if url.strip()]
    
    if not target_urls:
        return "請輸入至少一個網址"
    
    progress_text = "開始 MBTI 數據收集和分析任務...\n\n"
    
    all_data = {
        "personality_type": [],
        "occupation": [],
        "description": [],
        "source_url": [],
        "extracted_date": []
    }
    
    # 處理每個 URL
    for url in target_urls:
        progress_text += f"\n正在處理網站: {url}\n"
        try:
            # 1. 獲取網頁內容
            progress_text += "正在使用 AI 助手訪問網頁...\n"
            content = browse_webpage(url)
            
            if isinstance(content, str) and not content.startswith("錯誤："):
                progress_text += "成功獲取網頁內容\n"
                
                # 2. 提取 MBTI 信息
                progress_text += "正在分析網頁內容...\n"
                data = extract_mbti_info(content)
                
                # 3. 合併數據
                if data and len(data["personality_type"]) > 0:
                    for key in all_data:
                        all_data[key].extend(data[key])
                        if key == "source_url":
                            all_data[key][-len(data[key]):] = [url] * len(data[key])
                
                progress_text += f"從該網站提取了 {len(data['personality_type'])} 條記錄\n"
                yield progress_text  # 即時更新進度
            else:
                progress_text += f"無法獲取網頁內容: {content}\n"
                yield progress_text
            
        except Exception as e:
            progress_text += f"處理網站時出錯: {str(e)}\n"
            yield progress_text
            continue
    
    # 保存和分析數據
    if any(len(v) > 0 for v in all_data.values()):
        # 保存數據
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mbti_browser_{timestamp}.csv"
        result = save_data(all_data)
        progress_text += "\n最終結果:\n" + result + "\n"
        yield progress_text
        
        # 建立向量存儲
        progress_text += "\n正在建立向量存儲...\n"
        csv_path = os.path.join("mbti_data/csv", filename)
        vector_result = create_vector_store(csv_path)
        progress_text += vector_result + "\n"
        yield progress_text
        
        # 使用 agent 進行深入分析
        progress_text += "\n正在使用 AI 助手進行深入分析...\n"
        try:
            progress_text += "\n正在生成最終分析報告...\n"
            yield progress_text
            
            # 讀取最新的 CSV 檔案數據
            csv_files = list_csv_files()
            if not csv_files:
                progress_text += "\n錯誤：找不到 CSV 檔案\n"
                yield progress_text
                return progress_text
            
            latest_csv = os.path.join("mbti_data/csv", csv_files[0])
            df = pd.read_csv(latest_csv)
            
            # 準備分析數據
            analysis_prompt = f"""
請使用繁體中文分析以下統計數據，並生成結構化的分析報告：

=== MBTI 數據分析報告 ===

一、基本統計
{'-'*30}
總記錄數：{len(df)} 筆
MBTI類型：{len(df['personality_type'].unique())} 種
資料來源：{len(df['source_url'].unique())} 個

二、類型分布
{'-'*30}
| MBTI類型 | 數量 | 百分比 |
|----------|------|--------|
{df['personality_type'].value_counts().apply(lambda x: f"{x} | {x/len(df)*100:.1f}%").to_string()}

三、職業分布
{'-'*30}
| 職業類別 | 人數 | 比例 |
|----------|------|------|
{df['occupation'].value_counts().apply(lambda x: f"{x} | {x/len(df)*100:.1f}%").to_string()}

請根據以上數據進行分析：

1. 類型分布特徵
   - 主要類型分布
   - 特殊分布現象
   - 分布原因分析

2. 職業傾向分析
   - 主要職業分布
   - 類型職業關聯
   - 跨類型共同點

3. 綜合建議
   - 重要發現
   - 實務應用
   - 未來展望

請以繁體中文撰寫分析報告，確保內容清晰易讀。
"""
            
            final_analysis = agent.run(analysis_prompt)
            
            # 修改輸出格式部分
            progress_text += "\n" + "="*80 + "\n"  # 加寬分隔線
            progress_text += " "*30 + "MBTI 數據分析報告" + " "*30 + "\n"  # 置中標題
            progress_text += "="*80 + "\n\n"  # 加寬分隔線

            # 格式化 final_analysis 的輸出
            formatted_analysis = final_analysis.replace("\\n", "\n")  # 確保換行正確
            formatted_analysis = formatted_analysis.replace("|", "│")  # 使用更清晰的表格分隔線
            formatted_analysis = formatted_analysis.replace("-"*30, "─"*50)  # 使用更長的分隔線
            formatted_analysis = formatted_analysis.replace("---", "───")  # 使用更清晰的分隔線

            # 添加表格框線
            lines = formatted_analysis.split("\n")
            formatted_lines = []
            in_table = False
            for line in lines:
                if "│" in line:  # 表格行
                    if not in_table:
                        formatted_lines.append("┌" + "─"*(len(line)-2) + "┐")
                        in_table = True
                    formatted_lines.append(line)
                else:  # 非表格行
                    if in_table:
                        formatted_lines.append("└" + "─"*(len(line)-2) + "┘")
                        in_table = False
                    formatted_lines.append(line)

            progress_text += "\n".join(formatted_lines)
            progress_text += "\n" + "="*80 + "\n"  # 加寬分隔線
            progress_text += " "*30 + "報告結束" + " "*30 + "\n"  # 置中結束標題
            progress_text += "="*80 + "\n"  # 加寬分隔線

            yield progress_text
            
        except Exception as e:
            progress_text += f"\n分析過程出錯：{str(e)}\n"
            yield progress_text
        
    else:
        progress_text += "\n錯誤: 沒有收集到任何數據"
    
    return progress_text

def list_csv_files():
    """列出所有可用的 CSV 檔案"""
    csv_dir = "mbti_data/csv"
    if not os.path.exists(csv_dir):
        return []
    files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    return sorted(files, reverse=True)  # 最新的檔案排在最前面

def view_csv_content(csv_name):
    """查看 CSV 檔案內容"""
    if not csv_name:
        return None
    try:
        csv_path = os.path.join("mbti_data/csv", csv_name)
        df = pd.read_csv(csv_path)
        # 只保留需要顯示的欄位，並重新命名
        df = df[['personality_type', 'occupation', 'description', 'source_url', 'extracted_date']]
        df.columns = ['MBTI類型', '職業', '描述', '來源網址', '擷取時間']
        return df
    except Exception as e:
        print(f"讀取檔案錯誤：{str(e)}")
        return None

def custom_query(query_text):
    """執行自定義查詢"""
    if not query_text:
        return "請輸入查詢內容"
    return query_mbti_data(query_text)

# 創建 Gradio 介面
with gr.Blocks(title="MBTI 性格分析工具", theme=gr.themes.Soft()) as web_ui:
    gr.Markdown("""
    # MBTI 性格分析工具
    這是一個基於 AI 的 MBTI（Myers-Briggs Type Indicator）性格分析工具，可以爬取、分析和查詢 MBTI 相關數據。
    """)
    
    with gr.Tabs():
        # 數據處理標籤
        with gr.Tab("數據處理"):
            gr.Markdown("### 輸入要分析的網址（每行一個）")
            default_urls = """https://www.cosmopolitan.com/tw/horoscopes/spiritual-healing/g62945060/mbti-1119/
https://www.cosmopolitan.com/tw/horoscopes/spiritual-healing/g46433226/mbti-16-2024/
https://tw.imyfone.com/ai-tips/16-personalities-interpretation/"""
            
            with gr.Column():
                urls_input = gr.Textbox(
                    lines=5,
                    value=default_urls,
                    label="目標網址"
                )
                process_button = gr.Button("開始處理")
                progress_output = gr.Textbox(
                    label="處理進度和結果",
                    lines=30,
                    value=""
                )
                
                process_button.click(
                    fn=process_mbti_data,
                    inputs=urls_input,
                    outputs=progress_output,
                    show_progress=True
                )
        
        # 數據查看標籤
        with gr.Tab("數據查看"):
            gr.Markdown("### 查看已保存的數據")
            with gr.Row():
                files = list_csv_files()
                csv_dropdown = gr.Dropdown(
                    choices=files,
                    value=files[0] if files else None,
                    label="選擇 CSV 檔案",
                    interactive=True
                )
                refresh_button = gr.Button("刷新檔案列表")
            
            # 使用簡化版的 DataFrame 組件
            view_output = gr.DataFrame(
                interactive=False,
                wrap=True,
                row_count=(10, "dynamic"),  # 顯示10行，可以滾動
                value=view_csv_content(files[0]) if files else None  # 添加初始值
            )
            
            # 更新檔案列表的處理函數
            def update_file_list():
                files = list_csv_files()
                return gr.update(choices=files, value=files[0] if files else None)
            
            refresh_button.click(
                fn=update_file_list,
                outputs=csv_dropdown
            )
            csv_dropdown.change(
                fn=view_csv_content,
                inputs=csv_dropdown,
                outputs=view_output
            )
        
        # 自定義查詢標籤
        with gr.Tab("自定義查詢"):
            gr.Markdown("### 查詢 MBTI 數據")
            query_input = gr.Textbox(
                label="輸入查詢內容",
                placeholder="例如：INTJ 的職業傾向是什麼？"
            )
            query_button = gr.Button("執行查詢")
            query_output = gr.Textbox(label="查詢結果", lines=15)
            query_button.click(custom_query, inputs=query_input, outputs=query_output)

# 啟動 Web 介面
if __name__ == "__main__":
    web_ui.launch(share=True) 