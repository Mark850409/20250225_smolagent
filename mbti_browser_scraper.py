from smolagents import ToolCallingAgent, HfApiModel, tool
from playwright.sync_api import sync_playwright
from llm_model.models import ModelFactory
import pandas as pd
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import json
from opencc import OpenCC
import faiss
import numpy as np

# 載入環境變數
load_dotenv()

# 初始化模型工廠
model_factory = ModelFactory()

# 初始化 HuggingFace 模型和 embeddings
hf_provider = ModelFactory.create_provider(
    "huggingface", 
    os.getenv('HUGGINGFACE_MODEL')
)

# 初始化 embeddings (使用專門的嵌入模型)
embeddings = ModelFactory.create_provider(
    "huggingface", 
    os.getenv('EMBEDDING_MODEL_NAME')
).embeddings

# 初始化繁簡轉換器
cc = OpenCC('s2t')  # 簡體轉繁體

@tool
def browse_webpage(url: str) -> str:
    """瀏覽網頁並返回內容

    Args:
        url: 要訪問的網頁網址

    Returns:
        str: 網頁內容
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            print(f"正在訪問: {url}")
            page.goto(url)
            page.wait_for_selector('body', timeout=30000)
            
            # 等待動態內容加載
            time.sleep(5)
            
            # 獲取所有文本內容
            text_content = page.evaluate('''() => {
                const elements = document.querySelectorAll('p, h1, h2, h3, h4, h5, article, section');
                return Array.from(elements).map(el => el.textContent).join('\\n');
            }''')
            
            # 轉換為繁體中文
            text_content = cc.convert(text_content)
            
            browser.close()
            return text_content
        except Exception as e:
            browser.close()
            error_msg = f"錯誤：{str(e)}"
            print(error_msg)
            return error_msg

@tool
def extract_mbti_info(content: str) -> dict:
    """從文本中提取 MBTI 相關信息

    Args:
        content: 網頁文本內容

    Returns:
        dict: 提取的 MBTI 數據
    """
    mbti_data = {
        "personality_type": [],
        "occupation": [],
        "description": [],
        "source_url": [],
        "extracted_date": []
    }

    # 處理所有 MBTI 類型
    mbti_types = [
        "INTJ", "ENTJ", "INTP", "ENTP",
        "INFJ", "ENFJ", "INFP", "ENFP",
        "ISTJ", "ESTJ", "ISFJ", "ESFJ",
        "ISTP", "ESTP", "ISFP", "ESFP"
    ]
    
    # 擴充職業關鍵詞列表
    occupations = [
        "工程師", "建築師", "策略規劃師", "分析師", "研究員", "科學家",
        "顧問", "專案經理", "資訊長", "技術總監", "系統架構師",
        "投資人", "企業家", "管理者", "規劃師", "設計師",
        "醫生", "教師", "心理諮詢師", "藝術家", "作家", "編輯",
        "行銷人員", "業務經理", "人資主管", "財務分析師"
    ]
    
    # 將內容按段落分割，並過濾空行
    paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
    
    # 定義多個關鍵詞組合來識別相關段落
    keywords = [
        "性格特質", "人格特質", "個性特點", "性格類型",
        "職業", "工作", "事業", "專業", "職場",
        "特色", "優勢", "適合", "傾向"
    ]
    
    # 為每個 MBTI 類型處理內容
    for mbti_type in mbti_types:
        current_paragraph = ""
        type_keywords = [
            mbti_type,
            f"{mbti_type}型",
            f"{mbti_type}人格",
            f"{mbti_type}性格"
        ]
        
        for para in paragraphs:
            # 檢查段落是否包含當前 MBTI 類型或相關關鍵詞
            is_relevant = (
                any(type_kw in para for type_kw in type_keywords) or
                any(keyword in para for keyword in keywords)
            )
            
            if is_relevant:
                current_paragraph = current_paragraph + " " + para if current_paragraph else para
            else:
                if current_paragraph:
                    # 處理累積的段落
                    if any(type_kw in current_paragraph for type_kw in type_keywords):
                        description = cc.convert(current_paragraph.strip())
                        
                        # 尋找職業關鍵詞
                        found_occupations = []
                        for occupation in occupations:
                            if occupation in description:
                                found_occupations.append(occupation)
                        
                        # 如果沒找到特定職業，設為"其他"
                        if not found_occupations:
                            found_occupations = ["其他"]
                        
                        # 為每個找到的職業添加一條記錄
                        for occupation in found_occupations:
                            mbti_data["personality_type"].append(mbti_type)
                            mbti_data["occupation"].append(occupation)
                            mbti_data["description"].append(description)
                            mbti_data["source_url"].append("current_url")
                            mbti_data["extracted_date"].append(
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                    
                    current_paragraph = ""
        
        # 處理最後一個段落
        if current_paragraph and any(type_kw in current_paragraph for type_kw in type_keywords):
            description = cc.convert(current_paragraph.strip())
            found_occupations = [occ for occ in occupations if occ in description] or ["其他"]
            for occupation in found_occupations:
                mbti_data["personality_type"].append(mbti_type)
                mbti_data["occupation"].append(occupation)
                mbti_data["description"].append(description)
                mbti_data["source_url"].append("current_url")
                mbti_data["extracted_date"].append(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
    
    print(f"找到 {len(mbti_data['description'])} 條相關描述")
    for mbti_type in mbti_types:
        type_count = mbti_data["personality_type"].count(mbti_type)
        if type_count > 0:
            print(f"{mbti_type}: {type_count} 條記錄")
    
    return mbti_data

@tool
def save_data(data: dict) -> str:
    """將數據保存為 CSV 文件

    Args:
        data: 要保存的 MBTI 數據字典

    Returns:
        str: 保存結果信息
    """
    try:
        if not data.get("personality_type") or len(data["personality_type"]) == 0:
            return "沒有找到可用的 MBTI 數據"
        
        os.makedirs("mbti_data/csv", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mbti_browser_{timestamp}.csv"
        filepath = os.path.join("mbti_data/csv", filename)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        stats = {
            "總記錄數": len(df),
            "MBTI類型分布": df["personality_type"].value_counts().to_dict(),
            "職業分布": df["occupation"].value_counts().to_dict()
        }
        
        return f"數據已保存至 {filepath}\n\n統計信息：\n{json.dumps(stats, ensure_ascii=False, indent=2)}"
    
    except Exception as e:
        return f"保存失敗：{str(e)}"

@tool
def create_vector_store(csv_path: str = None) -> str:
    """建立 MBTI 數據的向量存儲

    Args:
        csv_path: CSV 檔案路徑，如果為 None 則使用最新的 CSV 檔案

    Returns:
        str: 建立結果信息
    """
    try:
        # 如果沒有指定 CSV 檔案，尋找最新的檔案
        if csv_path is None:
            csv_dir = "mbti_data/csv"
            if not os.path.exists(csv_dir):
                return "錯誤：找不到 CSV 目錄"
            
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
            if not csv_files:
                return "錯誤：找不到 CSV 檔案"
            
            latest_csv = max(csv_files)
            csv_path = os.path.join(csv_dir, latest_csv)
        
        # 檢查檔案是否存在
        if not os.path.exists(csv_path):
            return f"錯誤：找不到檔案 {csv_path}"

        # 讀取 CSV 檔案
        df = pd.read_csv(csv_path)
        
        # 建立向量存儲目錄
        vector_dir = os.path.join("mbti_data", "vectors")
        os.makedirs(vector_dir, exist_ok=True)
        
        # 準備文本和向量
        texts = []
        all_vectors = []
        metadatas = []
        
        # 處理每一行數據
        for _, row in df.iterrows():
            # 組合文本
            text = (f"Personality Type: {row['personality_type']}\n"
                   f"Occupation: {row['occupation']}\n"
                   f"Description: {row['description']}\n")
            texts.append(text)
            
            # 獲取文本的向量表示
            vector = embeddings.embed_query(text)
            all_vectors.append(vector)
            
            # 準備元數據
            metadatas.append({
                "personality_type": row['personality_type'],
                "occupation": row['occupation']
            })

        # 轉換為 numpy 數組
        vectors = np.array(all_vectors, dtype=np.float32)
        
        # 建立 FAISS 索引
        dimension = len(vectors[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)

        # 保存數據
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        store_name = f"mbti_vectors_{timestamp}"
        store_dir = os.path.join(vector_dir, store_name)
        os.makedirs(store_dir, exist_ok=True)

        # 保存 FAISS 索引
        index_path = os.path.join(store_dir, "index.faiss")
        faiss.write_index(index, index_path)

        # 保存文本和元數據
        data = {
            "texts": texts,
            "metadatas": metadatas
        }
        data_path = os.path.join(store_dir, "data.json")
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return f"向量存儲已成功建立於 {store_dir}"

    except Exception as e:
        return f"向量存儲建立失敗: {str(e)}\n錯誤類型: {type(e).__name__}"

@tool
def query_mbti_data(query: str) -> str:
    """
    查詢 MBTI 向量數據庫。

    Args:
        query: 查詢文本

    Returns:
        查詢結果
    """
    try:
        vector_dir = "mbti_data/vectors"
        if not os.path.exists(vector_dir):
            return "錯誤：找不到向量存儲目錄"

        # 尋找最新的向量存儲目錄
        vector_stores = [d for d in os.listdir(vector_dir) 
                        if os.path.isdir(os.path.join(vector_dir, d))]
        
        if not vector_stores:
            return "錯誤：找不到向量存儲"

        # 使用最新的向量存儲
        latest_store = max(vector_stores)
        store_dir = os.path.join(vector_dir, latest_store)

        # 載入 FAISS 索引
        index_path = os.path.join(store_dir, "index.faiss")
        index = faiss.read_index(index_path)

        # 載入文本和元數據
        data_path = os.path.join(store_dir, "data.json")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 獲取查詢向量
        query_vector = embeddings.embed_query(query)
        query_vector = np.array([query_vector], dtype=np.float32)

        # 執行查詢
        k = 3  # 返回前3個最相關的結果
        distances, indices = index.search(query_vector, k)
        
        if len(indices[0]) == 0:
            return "未找到相關結果"

        # 格式化結果
        formatted_results = []
        for i, idx in enumerate(indices[0]):
            distance = distances[0][i]
            text = data["texts"][int(idx)]
            metadata = data["metadatas"][int(idx)]
            formatted_results.append(
                f"相關度: {1/(1+distance):.2f}\n"
                f"文本: {text}\n"
                f"類型: {metadata['personality_type']}\n"
                f"職業: {metadata['occupation']}"
            )

        return "\n\n---\n\n".join(formatted_results)

    except Exception as e:
        return f"查詢失敗: {str(e)}\n錯誤類型: {type(e).__name__}"

# 初始化模型和代理
messages = [
    {
        "role": "system",
        "content": """
你是一個專業的 MBTI 性格分析專家，負責分析和解釋 MBTI 相關數據。
請注意以下要求：
1. 所有回應必須使用繁體中文
2. 分析結果需要包含：
   - 數據收集概況
   - 各 MBTI 類型的分布情況
   - 職業傾向分析
   - 文化背景分析
   - 年齡分布分析
3. 提供具體的數據支持，包含百分比
4. 對特殊趨勢提出專業見解
5. 建議使用條列式呈現重要發現
"""
    }
]

# 初始化模型
model = HfApiModel(
    model_id=os.getenv('HUGGINGFACE_MODEL'),
    messages=messages
)

# 初始化代理
agent = ToolCallingAgent(
    tools=[browse_webpage, extract_mbti_info, save_data, create_vector_store, query_mbti_data],
    model=model
)

# 主程序
if __name__ == "__main__":
    print("開始 MBTI 數據收集和分析任務...")
    
    # 目標網站列表
    target_urls = [
        "https://www.cosmopolitan.com/tw/horoscopes/spiritual-healing/g62945060/mbti-1119/",
        "https://www.cosmopolitan.com/tw/horoscopes/spiritual-healing/g46433226/mbti-16-2024/",
        "https://tw.imyfone.com/ai-tips/16-personalities-interpretation/"
    ]
    
    all_data = {
        "personality_type": [],
        "occupation": [],
        "description": [],
        "source_url": [],
        "extracted_date": []
    }
    
    for url in target_urls:
        print(f"\n正在處理網站: {url}")
        try:
            # 1. 獲取網頁內容
            content = browse_webpage(url)
            
            if isinstance(content, str) and not content.startswith("錯誤："):
                print("成功獲取網頁內容")
                
                # 2. 提取 MBTI 信息
                data = extract_mbti_info(content)
                
                # 3. 合併數據
                if data and len(data["personality_type"]) > 0:
                    for key in all_data:
                        all_data[key].extend(data[key])
                        if key == "source_url":
                            all_data[key][-len(data[key]):] = [url] * len(data[key])
                
                print(f"從該網站提取了 {len(data['personality_type'])} 條記錄")
            else:
                print(f"無法獲取網頁內容: {content}\n")
            
        except Exception as e:
            print(f"處理網站時出錯: {str(e)}")
            continue
    
    # 保存和分析數據
    if any(len(v) > 0 for v in all_data.values()):
        # 保存數據
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mbti_browser_{timestamp}.csv"
        result = save_data(all_data)
        print("\n最終結果:")
        print(result)
        
        # 建立向量存儲
        print("\n正在建立向量存儲...")
        csv_path = os.path.join("mbti_data/csv", filename)
        vector_result = create_vector_store(csv_path)
        print(vector_result)
        
        # 使用 agent 進行深入分析
        print("\n正在使用 AI 助手進行深入分析...")
        analysis_prompt = """
請根據收集到的 MBTI 數據進行深入分析，並以繁體中文回覆。請使用以下表格格式輸出:

1. 數據收集概況
| 指標 | 數值 |
|------|------|
| 總記錄數量 | |
| 涵蓋MBTI類型數 | |
| 資料來源分布 | |

2. 各 MBTI 類型分析
| MBTI類型 | 數量 | 主要特徵 | 特殊發現 |
|----------|------|----------|----------|
| INTJ | | | |
| ENTJ | | | |
[...其他類型...]

3. 職業傾向分析
| MBTI類型 | 最常見職業 | 職業分布特點 |
|----------|------------|--------------|
| INTJ | | |
| ENTJ | | |
[...其他類型...]

跨類型共同職業趨勢:
- 
-
-

4. 綜合觀察
| 觀察面向 | 發現與建議 |
|----------|------------|
| 有趣發現 | |
| 特殊相關性 | |
| 建議與洞見 | |

請提供詳細的分析數據，並確保表格內容完整。
"""
        
        # 執行分析查詢
        print("\n開始進行數據分析...")
        queries = [
            "各 MBTI 類型的職業傾向是什麼？",
            "不同 MBTI 類型的特質分布如何？",
            "各 MBTI 類型的主要特徵是什麼？"
        ]
        
        analysis_results = []
        for query in queries:
            print(f"\n分析問題：{query}")
            result = query_mbti_data(query)
            analysis_results.append(f"問題：{query}\n回答：{result}\n")
        
        # 使用 agent 生成最終報告
        print("\n正在生成最終分析報告...")
        final_analysis = agent.run(
            analysis_prompt + "\n\n參考以下查詢結果：\n" + 
            "\n---\n".join(analysis_results)
        )
        
        print("\n=== MBTI 數據分析報告 ===")
        print(final_analysis)
        print("\n=== 報告結束 ===")
        
    else:
        print("\n錯誤: 沒有收集到任何數據") 