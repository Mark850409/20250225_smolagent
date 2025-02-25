from smolagents import ToolCallingAgent, HfApiModel, tool
import pandas as pd
from datetime import datetime
import numpy as np
from serpapi import GoogleSearch
import time
import os
import json
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()
serp_api_key = os.getenv('SERP_API_KEY')
model_id = os.getenv('MODEL_ID')

def scrape_mbti_data(state: str = "TW", num_results: int = 20) -> dict:
    """從 Google 搜索爬取 MBTI 相關數據"""
    try:
        if not serp_api_key:
            return {"error": "未設置 SERP API 金鑰"}

        mbti_data = {
            "personality_type": [],
            "occupation": [],
            "description": [],
            "source_url": [],
            "extracted_date": []
        }
        
        base_params = {
            "api_key": serp_api_key,
            "engine": "google",
            "gl": state,
            "hl": state,
            "num": num_results
        }

        mbti_types = ["INTJ", "ENTJ", "INTP", "ENTP", "INFJ", "ENFJ", "INFP", "ENFP",
                     "ISTJ", "ESTJ", "ISFJ", "ESFJ", "ISTP", "ESTP", "ISFP", "ESFP"]
        
        success_count = 0
        error_count = 0
        error_messages = []

        for mbti_type in mbti_types:
            try:
                params = base_params.copy()
                params["q"] = f"{mbti_type} personality career preferences research"
                
                search = GoogleSearch(params)
                results = search.get_dict()
                
                if "error" in results:
                    error_count += 1
                    error_messages.append(f"{mbti_type}: {results['error']}")
                    continue
                
                if "organic_results" in results and results["organic_results"]:
                    result = results["organic_results"][0]
                    snippet = result.get("snippet", "")
                    url = result.get("link", "")
                    
                    common_occupations = ["Engineer", "Manager", "Teacher", "Developer", 
                                        "Scientist", "Artist", "Writer", "Consultant"]
                    occupation = next((job for job in common_occupations 
                                     if job.lower() in snippet.lower()), "Other")
                    
                    mbti_data["personality_type"].append(mbti_type)
                    mbti_data["occupation"].append(occupation)
                    mbti_data["description"].append(snippet[:200] + "...")
                    mbti_data["source_url"].append(url)
                    mbti_data["extracted_date"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    success_count += 1
                else:
                    error_count += 1
                    error_messages.append(f"{mbti_type}: 未找到搜索結果")
                
                time.sleep(2)  # 避免請求過於頻繁

            except Exception as e:
                error_count += 1
                error_messages.append(f"{mbti_type}: {str(e)}")
                time.sleep(5)  # 發生錯誤時等待更長時間
                continue

        # 生成執行報告
        report = {
            "total_types": len(mbti_types),
            "success_count": success_count,
            "error_count": error_count,
            "error_messages": error_messages
        }

        if success_count == 0:
            return {
                "error": f"數據爬取完全失敗\n執行報告：{json.dumps(report, ensure_ascii=False, indent=2)}"
            }
        
        if error_count > 0:
            print(f"\n執行報告：\n{json.dumps(report, ensure_ascii=False, indent=2)}")

        return mbti_data

    except Exception as e:
        return {
            "error": f"嚴重錯誤: {str(e)}\n"
                    f"錯誤類型: {type(e).__name__}\n"
                    f"可能原因：\n"
                    f"1. API 金鑰無效或已過期\n"
                    f"2. 超過 API 請求限制\n"
                    f"3. 網路連接問題\n"
                    f"4. 服務暫時不可用"
        }

def save_data(data: dict) -> tuple:
    """將數據保存為 CSV 文件並返回統計信息"""
    try:
        if not data.get("personality_type") or len(data["personality_type"]) == 0:
            return None, "沒有找到可用的 MBTI 數據"
        
        os.makedirs("mbti_data/csv", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mbti_analysis_{timestamp}.csv"
        filepath = os.path.join("mbti_data/csv", filename)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        stats = {
            "總樣本數": len(df),
            "類型分布": df["personality_type"].value_counts().to_dict(),
            "職業分布": df["occupation"].value_counts().to_dict()
        }
        
        return stats, filepath
    
    except Exception as e:
        return None, f"保存失敗：{str(e)}"

def create_vector_store(csv_file: str) -> str:
    """將 CSV 數據轉換為向量存儲"""
    try:
        base_dir = "mbti_data"
        vector_dir = os.path.join(base_dir, "vectors")
        os.makedirs(vector_dir, exist_ok=True)

        df = pd.read_csv(csv_file)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        store_name = f"mbti_vectors_{timestamp}"
        store_dir = os.path.join(vector_dir, store_name)
        os.makedirs(store_dir, exist_ok=True)

        data = {
            "texts": df.to_dict('records'),
            "metadatas": df[['personality_type', 'occupation']].to_dict('records')
        }
        
        data_path = os.path.join(store_dir, "data.json")
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return store_dir

    except Exception as e:
        return f"數據保存失敗: {str(e)}"

@tool
def analyze_mbti_stats(stats: dict) -> str:
    """分析 MBTI 統計數據並生成報告
    
    Args:
        stats: 包含 MBTI 統計數據的字典
        
    Returns:
        str: 分析報告文本
    """
    return f"""
=== MBTI 數據分析報告 ===

一、基本統計
總樣本數：{stats['總樣本數']} 筆

二、類型分布
{'-'*50}
| MBTI類型 | 數量 | 百分比 |
|----------|------|--------|
{chr(10).join(f"| {k:<8} | {v:>4} | {v/stats['總樣本數']*100:>5.1f}% |" for k, v in stats['類型分布'].items())}

三、職業分布
{'-'*50}
| 職業類別 | 人數 | 比例 |
|----------|------|------|
{chr(10).join(f"| {k:<8} | {v:>4} | {v/stats['總樣本數']*100:>5.1f}% |" for k, v in stats['職業分布'].items())}

請分析以上數據並提供見解。
"""

if __name__ == "__main__":
    print("開始收集 MBTI 數據...")
    data = scrape_mbti_data()
    
    print("\n保存數據並進行統計...")
    stats, filepath = save_data(data)
    if stats is None:
        print(filepath)  # 顯示錯誤信息
        exit(1)
        
    print("\n建立向量存儲...")
    store_dir = create_vector_store(filepath)
    
    print("\n使用 AI 進行深入分析...")
    # 初始化 AI 模型
    messages = [{
        "role": "system",
        "content": """你是一個專業的 MBTI 性格分析專家。請注意：
1. 使用繁體中文回應
2. 提供具體的數據支持
3. 分析需要客觀且專業"""
    }]
    
    model = HfApiModel(model_id=model_id, messages=messages)
    agent = ToolCallingAgent(tools=[analyze_mbti_stats], model=model)
    
    analysis = agent.run("請分析 MBTI 數據並生成報告")
    print(analysis) 