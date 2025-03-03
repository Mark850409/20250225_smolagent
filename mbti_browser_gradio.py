from mbti_browser_scraper import (browse_webpage, extract_mbti_info, save_data, 
                                HfApiModel,create_vector_store, query_mbti_data, agent)
import gradio as gr
import pandas as pd
import os
from datetime import datetime
from concurrent.futures import TimeoutError
from tavily import TavilyClient
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import threading


# 載入環境變數
load_dotenv()
tavily_api_key = os.getenv('TAVILY_API_KEY')

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
            
            # 準備分析提示詞
            analysis_prompt = f"""
請根據以下 MBTI 數據進行深入分析，並以繁體中文回覆。請注意分析應包含具體數據支持：

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

請根據以上統計數據進行分析：

1. 類型分布特徵
   - 主要類型分布
   - 特殊分布現象
   - 分布原因分析

2. 職業傾向分析
   - 主要職業分布
   - 類型職業關聯
   - 跨類型共同點

3. 深入觀察
   - 性格特質與職業選擇的關係
   - 各類型在不同職業領域的表現
   - 職業發展潛力分析

4. 綜合建議
   - 重要發現
   - 實務應用
   - 未來展望
   - 研究限制

請提供詳細的分析報告，確保：
1. 數據論述具體且有支持證據
2. 分析角度多元且深入
3. 建議實用且具可行性
4. 文字表達清晰易讀
5. 結論具有實務參考價值
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

def list_vector_stores():
    """列出所有可用的向量存儲"""
    vector_dir = "mbti_data/vectors"
    if not os.path.exists(vector_dir):
        return []
    stores = [d for d in os.listdir(vector_dir) 
              if os.path.isdir(os.path.join(vector_dir, d))]
    return sorted(stores, reverse=True)  # 最新的排在最前面

def custom_query(query_text: str, store_name: str = None) -> str:
    """執行自定義查詢
    
    Args:
        query_text: 查詢文本
        store_name: 指定要查詢的向量存儲名稱
    """
    if not query_text:
        return "請輸入查詢內容"
    return query_mbti_data(query_text, store_name)

def is_english(text: str) -> bool:
    """檢查文本是否主要為英文
    
    Args:
        text: 要檢查的文本
        
    Returns:
        bool: 如果文本主要為英文則返回 True
    """
    # 移除空格和標點符號
    text = ''.join(c for c in text if c.isalpha() or c.isspace())
    words = text.split()
    if not words:
        return False
    
    # 檢查每個單詞是否為英文（包含 MBTI 相關術語）
    english_words = 0
    mbti_terms = {'INTJ', 'INTP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'ENFJ', 'ENFP',
                  'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ', 'ISTP', 'ISFP', 'ESTP', 'ESFP',
                  'MBTI', 'Myers', 'Briggs'}
    
    for word in words:
        if word.upper() in mbti_terms or all(c.isascii() and c.isalpha() for c in word):
            english_words += 1
    
    return english_words / len(words) > 0.5  # 如果超過 50% 是英文單詞，則視為英文

def batch_translate(texts: List[str], prompt_template: str, batch_size: int = 5) -> List[str]:
    """批量翻譯文本
    
    Args:
        texts: 要翻譯的文本列表
        prompt_template: 翻譯提示詞模板
        batch_size: 每批次處理的文本數量
        
    Returns:
        List[str]: 翻譯後的文本列表
    """
    translated_texts = []
    
    def translate_batch(batch: List[str]) -> str:
        # 將多個文本組合成一個批次
        combined_text = "\n---\n".join(batch)
        prompt = prompt_template.format(text=combined_text)
        
        # 使用 agent 進行翻譯
        result = agent.run(prompt).strip()
        return result
    
    # 將文本分批
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    # 使用線程池進行並行翻譯
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for batch in batches:
            future = executor.submit(translate_batch, batch)
            futures.append(future)
        
        # 獲取所有翻譯結果
        for future in futures:
            result = future.result()
            # 分割結果為單獨的翻譯
            translations = [t.strip() for t in result.split("\n---\n")]
            translated_texts.extend(translations)
    
    return translated_texts

def is_valid_mbti_content(text: str) -> bool:
    """檢查內容是否為有效的 MBTI 相關內容"""
    # 檢查是否包含 MBTI 相關關鍵詞（放寬條件）
    mbti_keywords = [
        'MBTI', 'Myers-Briggs', 'personality type', 'cognitive functions',
        'introvert', 'extrovert', 'intuitive', 'sensing', 'thinking', 'feeling',
        'judging', 'perceiving', 'personality', 'career', 'workplace',
        '性格', '人格', '職業', '工作', '領導', '管理', '團隊'  # 添加中文關鍵詞
    ]
    
    # 只要包含任一關鍵詞就視為相關內容
    if any(keyword.lower() in text.lower() for keyword in mbti_keywords):
        return True
    
    # 檢查是否包含 MBTI 類型代碼
    mbti_types = ['INTJ', 'INTP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'ENFJ', 'ENFP',
                  'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ', 'ISTP', 'ISFP', 'ESTP', 'ESFP']
    if any(mbti_type in text for mbti_type in mbti_types):
        return True
    
    return False

def tavily_search_mbti(prompt: str, progress: str = "") -> str:
    """使用 Tavily 搜索並處理 MBTI 相關數據"""
    progress_text = progress
    progress_text += f"開始處理提示詞：{prompt}\n\n"
    yield progress_text

    try:
        # 檢查是否需要翻譯
        if not is_english(prompt):
            progress_text += "檢測到中文提示詞，正在翻譯為英文...\n"
            yield progress_text
            
            translation_prompt = f"""
請將以下中文提示詞翻譯成英文，保持專業性和準確性：
{prompt}

注意：
1. 保留 MBTI 相關的專業術語
2. 翻譯需要自然流暢
3. 只需要輸出英文翻譯結果，不需要其他說明
"""
            english_prompt = agent.run(translation_prompt).strip()
            progress_text += f"翻譯結果：{english_prompt}\n\n"
            yield progress_text
        else:
            progress_text += "檢測到英文提示詞，無需翻譯\n\n"
            yield progress_text
            english_prompt = prompt

        # 構建更精確的搜索查詢
        search_query = f"MBTI personality type career analysis: {english_prompt}"
        
        # 初始化 Tavily 客戶端
        tavily_client = TavilyClient(api_key=tavily_api_key)
        
        progress_text += "正在使用 Tavily 搜索相關網頁...\n"
        yield progress_text
        
        search_response = tavily_client.search(
            query=search_query,
            search_depth="advanced",
            topic="general",
            max_results=10,  # 增加搜索結果數量
            include_answer=True,
            search_filter={
                "domains_to_include": [
                    "personalitypage.com",
                    "16personalities.com",
                    "truity.com",
                    "psychologyjunkie.com",
                    "personalitygrowth.com",
                    "getmarlee.com",
                    "indeed.com",
                    "linkedin.com",
                    "medium.com",
                    "forbes.com",
                    "psychologytoday.com"
                ]
            }
        )
        
        # 初始化總數據存儲
        all_data = {
            "personality_type": [],
            "occupation": [],
            "description": [],
            "source_url": [],
            "extracted_date": []
        }
        
        # 處理搜索結果
        processed_urls = set()
        
        if isinstance(search_response, dict) and 'results' in search_response:
            total_results = len(search_response['results'])
            progress_text += f"\n找到 {total_results} 個相關網頁\n"
            yield progress_text
            
            valid_results = 0
            for result in search_response['results']:
                url = result.get('url', '')
                content = result.get('content', '')
                
                # 跳過已處理的 URL
                if url in processed_urls:
                    continue
                
                # 使用更寬鬆的內容驗證
                if not is_valid_mbti_content(content):
                    progress_text += f"跳過不相關內容: {url}\n"
                    yield progress_text
                    continue
                
                progress_text += f"\n正在處理網頁: {url}\n"
                yield progress_text
                
                try:
                    # 提取 MBTI 信息
                    data = extract_mbti_info(content)
                    
                    # 驗證提取的數據
                    if data and len(data["personality_type"]) > 0:
                        # 去除重複數據
                        for key in data:
                            if key in ["personality_type", "occupation", "description"]:
                                # 使用集合去重
                                unique_items = []
                                seen = set()
                                for item in data[key]:
                                    item_lower = item.lower()
                                    if item_lower not in seen:
                                        seen.add(item_lower)
                                        unique_items.append(item)
                                data[key] = unique_items
                        
                        # 合併數據到總數據集
                        for key in all_data:
                            if key == "source_url":
                                all_data[key].extend([url] * len(data["personality_type"]))
                            elif key == "extracted_date":
                                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                all_data[key].extend([current_time] * len(data["personality_type"]))
                            else:
                                all_data[key].extend(data[key])
                        
                        progress_text += f"成功提取 {len(data['personality_type'])} 條有效記錄\n"
                        yield progress_text
                    
                    processed_urls.add(url)
                    
                except Exception as e:
                    progress_text += f"處理網頁時出錯: {str(e)}\n"
                    yield progress_text
                    continue
        
        # 在所有數據收集完成後進行翻譯
        progress_text += "\n開始批量翻譯所有數據...\n"
        yield progress_text
        
        # 準備翻譯後的數據結構
        translated_data = {
            "personality_type": all_data["personality_type"],  # MBTI 類型保持不變
            "occupation": [],
            "description": [],
            "source_url": all_data["source_url"],            # URL 保持不變
            "extracted_date": all_data["extracted_date"]      # 日期保持不變
        }
        
        # 翻譯職業
        if all_data["occupation"]:
            progress_text += "\n" + "-"*30 + "\n"
            progress_text += "【職業翻譯結果】\n"
            progress_text += f"共 {len(all_data['occupation'])} 筆\n"
            progress_text += "-"*30 + "\n"
            
            occupations_text = "\n---\n".join(all_data["occupation"])
            occupation_prompt = f"""請將以下職業列表翻譯成中文，保持專業性和準確性。
每個職業之間使用 "---" 分隔：

{occupations_text}

注意：
1. 保持專業術語的準確性
2. 每個翻譯結果單獨一行
3. 只需要輸出中文翻譯結果，不需要其他說明
4. 使用 "---" 分隔不同職業的翻譯
"""

            # 使用 agent 進行翻譯
            response = agent.run(occupation_prompt)
            occupations_zh = response.strip().split("\n---\n")
            
            translated_data["occupation"] = occupations_zh
            for en, zh in zip(all_data["occupation"], occupations_zh):
                progress_text += f"▪ {en}\n  → {zh}\n"
            yield progress_text
        
        # 翻譯描述
        if all_data["description"]:
            progress_text += "\n正在批量翻譯描述文本...\n"
            yield progress_text
            
            description_prompt = """
請將以下 MBTI 相關描述翻譯成中文，保持專業性和準確性。
每個描述之間使用 "---" 分隔：

{text}

注意：
1. 保留 MBTI 相關的專業術語
2. 保持描述的準確性和流暢性
3. 只需要輸出中文翻譯結果，不需要其他說明
4. 使用 "---" 分隔不同描述的翻譯
"""
            
            descriptions_zh = batch_translate(
                all_data["description"],
                description_prompt,
                batch_size=3
            )
            
            translated_data["description"] = descriptions_zh
            progress_text += f"\n描述翻譯結果（共 {len(descriptions_zh)} 筆）：\n"
            for i, (en, zh) in enumerate(zip(all_data["description"], descriptions_zh), 1):
                progress_text += f"\n=== 第 {i} 筆翻譯 ===\n"
                progress_text += f"原文：{en}\n"
                progress_text += f"翻譯：{zh}\n"
            yield progress_text
        
        # 確保所有列表長度一致
        max_length = max(len(v) for v in translated_data.values())
        for key in translated_data:
            if len(translated_data[key]) < max_length:
                # 使用最後一個值填充
                last_value = translated_data[key][-1] if translated_data[key] else ""
                translated_data[key].extend([last_value] * (max_length - len(translated_data[key])))
        
        # 確保目錄存在
        csv_dir = "mbti_data/csv"
        os.makedirs(csv_dir, exist_ok=True)
        
        # 使用翻譯後的數據保存為 CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mbti_browser_{timestamp}.csv"
        csv_path = os.path.join(csv_dir, filename)
        
        # 保存數據
        result = save_data(translated_data)
        progress_text += "\n最終結果:\n" + result + "\n"
        yield progress_text
        
        # 建立向量存儲
        vector_result = create_vector_store(csv_path)
        progress_text += vector_result + "\n"
        yield progress_text
        
        # 使用 agent 進行深入分析
        progress_text += "\n正在使用 AI 助手進行深入分析...\n"
        try:
            # 確保文件存在
            if os.path.exists(csv_path):
                # 讀取 CSV 數據
                df = pd.read_csv(csv_path)
                
                # 準備分析提示詞
                analysis_prompt = f"""
請根據以下 MBTI 數據進行深入分析，並以繁體中文回覆：

=== MBTI 數據分析報告 ===

一、基本統計
{'-'*30}
總記錄數：{len(df)} 筆
MBTI類型：{len(df['personality_type'].unique())} 種
資料來源：{len(df['source_url'].unique())} 個

二、類型分布
{'-'*30}
{df['personality_type'].value_counts().to_string()}

三、職業分布
{'-'*30}
{df['occupation'].value_counts().to_string()}

請提供詳細的分析報告，包括：
1. 類型分布特徵和趨勢
2. 職業選擇與性格類型的關聯
3. 重要發現和建議
"""
                
                final_analysis = agent.run(analysis_prompt)
                progress_text += "\n" + "="*50 + "\n"
                progress_text += final_analysis
                progress_text += "\n" + "="*50 + "\n"
                yield progress_text
            else:
                progress_text += f"\n錯誤：找不到文件 {csv_path}\n"
                yield progress_text
        
        except Exception as e:
            progress_text += f"\n分析過程出錯：{str(e)}\n"
            yield progress_text
    
    except Exception as e:
        progress_text += f"\n搜索過程出錯：{str(e)}\n"
        yield progress_text

# 創建 Gradio 介面
with gr.Blocks(title="MBTI 性格分析工具", theme=gr.themes.Soft()) as web_ui:
    gr.Markdown("""
    # MBTI 性格分析工具
    這是一個基於 AI 的 MBTI（Myers-Briggs Type Indicator）性格分析工具，可以爬取、分析和查詢 MBTI 相關數據。
    """)
    
    with gr.Tabs():
                # 添加 Tavily 搜索標籤
        with gr.Tab("智能提示詞搜尋"):
            gr.Markdown("""### MBTI 智能提示詞搜尋
            輸入你想了解的 MBTI 相關主題，系統將自動搜索並分析相關資訊。
            
            提示：
            1. MBTI 類型相關：
               - INTJ 職業發展與工作滿意度分析
               - ENFP 人際關係與溝通模式探討
               - ISTJ 性格優勢和發展建議
            2. MBTI 類型比較：
               - INTJ 如何在職場發揮領導力
               - ENFP 如何維持長期穩定的人際關係
               - ISTJ 如何改善團隊溝通效率
            3. 深入探討：
               - MBTI 在職場中的應用
               - MBTI 在人際關係中的作用
               - MBTI 在個人成長中的角色
            """)
            
            with gr.Column():
                search_input = gr.Textbox(
                    lines=2,
                    placeholder="例如：'INTJ career choices and workplace behavior'",
                    label="提示詞"
                )
                search_button = gr.Button("開始提示詞搜尋")
                search_output = gr.Textbox(
                    label="提示詞搜尋進度和結果",
                    lines=30,
                    value=""
                )
                
                search_button.click(
                    fn=tavily_search_mbti,
                    inputs=[search_input, search_output],
                    outputs=search_output,
                    show_progress=True
                )
        # 數據處理標籤
        with gr.Tab("網址數據分析處理"):
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
        with gr.Tab("RAG生成式檢索查詢"):
            gr.Markdown("""### 查詢 MBTI 數據
            選擇要查詢的知識庫並輸入查詢內容。知識庫按時間排序,最新的在最前面。
            """)
            
            with gr.Row():
                # 添加向量存儲選擇下拉框
                stores = list_vector_stores()
                vector_store_dropdown = gr.Dropdown(
                    choices=stores,
                    value=stores[0] if stores else None,
                    label="選擇知識庫",
                    interactive=True
                )
                refresh_stores_button = gr.Button("刷新知識庫列表")
            
            query_input = gr.Textbox(
                label="輸入查詢內容",
                placeholder="例如：INTJ 的職業傾向是什麼？"
            )
            query_button = gr.Button("執行查詢")
            query_output = gr.Textbox(label="查詢結果", lines=15)
            
            # 更新知識庫列表的處理函數
            def update_store_list():
                stores = list_vector_stores()
                return gr.update(choices=stores, value=stores[0] if stores else None)
            
            refresh_stores_button.click(
                fn=update_store_list,
                outputs=vector_store_dropdown
            )
            
            # 修改查詢按鈕的點擊事件
            query_button.click(
                fn=custom_query,
                inputs=[query_input, vector_store_dropdown],
                outputs=query_output
            )
        

# 啟動 Web 介面
if __name__ == "__main__":
    web_ui.launch(share=True) 