from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import Dict, Optional, List, Union
import pandas as pd
import json
import os
from datetime import datetime
from recommendation_analysis import analyze_recommendation_results, agent, get_deep_analysis
from mbti_browser_scraper import browse_webpage, extract_mbti_info, save_data, create_vector_store, query_mbti_data
from mbti_analysis_scraper import analyze_mbti_stats
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from opencc import OpenCC
from smolagents import ToolCallingAgent, HfApiModel

# 載入環境變數
load_dotenv()

# 初始化繁簡轉換器
cc = OpenCC('s2t')

# 創建 FastAPI 應用
app = FastAPI(
    title="MBTI 性格分析系統 API",
    description="提供 MBTI 性格分析的 RESTful API 服務",
    version="1.0.0"
)

# 定義 API 標籤及其描述
tags_metadata = [
    {
        "name": "MBTI資料分析報告",
        "description": "生成和查看 MBTI 數據分析報告",
    },
    {
        "name": "MBTI資料爬取",
        "description": "從網頁爬取 MBTI 相關數據",
    },
    {
        "name": "MBTI資料存儲",
        "description": "管理 MBTI 數據的存儲和查詢",
    }
]

# 定義資料模型
class ExperimentSettings(BaseModel):
    data_settings: Dict
    dataset_info: Dict
    model_settings: Dict

class AnalysisResult(BaseModel):
    timestamp: str
    metrics: Dict
    initial_analysis: str
    final_analysis: str

class AnalysisResponse(BaseModel):
    status: str
    message: str
    data: Optional[AnalysisResult]

# 新增 MBTI 相關的資料模型
class MBTIData(BaseModel):
    personality_type: List[str]
    occupation: List[str]
    description: List[str]
    source_url: List[str]
    extracted_date: List[str]

class MBTIResponse(BaseModel):
    status: str
    message: str
    data: Optional[MBTIData]

class URLInput(BaseModel):
    urls: List[HttpUrl]

# 修改分析相關的資料模型
class MBTIStats(BaseModel):
    total_samples: int
    type_distribution: Dict[str, int]
    occupation_distribution: Dict[str, int]

class MBTIAnalysisResult(BaseModel):
    timestamp: str
    stats: MBTIStats
    analysis_report: str

class MBTIAnalysisResponse(BaseModel):
    status: str
    message: str
    data: Optional[MBTIAnalysisResult]


@app.get("/reports/", response_model=List[str], tags=["MBTI資料分析報告"])
async def list_reports():
    """
    列出所有已生成的分析報告
    """
    try:
        report_dir = "recommendation_analysis"
        if not os.path.exists(report_dir):
            return []
        files = [f for f in os.listdir(report_dir) if f.endswith('.json')]
        return sorted(files, reverse=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports/{report_name}", response_model=AnalysisResult, tags=["MBTI資料分析報告"])
async def get_report(report_name: str):
    """
    獲取指定報告的內容
    
    - **report_name**: 報告檔案名稱
    """
    try:
        report_path = os.path.join("recommendation_analysis", report_name)
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="找不到報告")

        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
            return AnalysisResult(**report_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def async_browse_webpage(url: str) -> str:
    """非同步版本的網頁瀏覽函數"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            print(f"正在訪問: {url}")
            await page.goto(url)
            await page.wait_for_selector('body', timeout=30000)
            
            # 等待動態內容加載
            await page.wait_for_timeout(5000)
            
            # 獲取所有文本內容
            text_content = await page.evaluate('''() => {
                const elements = document.querySelectorAll('p, h1, h2, h3, h4, h5, article, section');
                return Array.from(elements).map(el => el.textContent).join('\\n');
            }''')
            
            await browser.close()
            return cc.convert(text_content)  # 轉換為繁體中文
            
        except Exception as e:
            await browser.close()
            error_msg = f"錯誤：{str(e)}"
            print(error_msg)
            return error_msg

@app.post("/mbti/scrape/", response_model=MBTIResponse, tags=["MBTI資料爬取"])
async def scrape_mbti_data(url_input: URLInput):
    """
    從指定的網址抓取 MBTI 相關資料
    
    - **urls**: 要抓取的網址列表
    """
    try:
        all_data = {
            "personality_type": [],
            "occupation": [],
            "description": [],
            "source_url": [],
            "extracted_date": []
        }
        
        for url in url_input.urls:
            # 使用非同步版本的網頁瀏覽函數
            content = await async_browse_webpage(str(url))
            
            if isinstance(content, str) and not content.startswith("錯誤："):
                # 提取 MBTI 信息
                data = extract_mbti_info(content)
                
                # 合併數據
                if data and len(data["personality_type"]) > 0:
                    for key in all_data:
                        all_data[key].extend(data[key])
                        if key == "source_url":
                            all_data[key][-len(data[key]):] = [str(url)] * len(data[key])
            else:
                print(f"無法獲取網頁內容: {content}")
        
        # 儲存數據
        if any(len(v) > 0 for v in all_data.values()):
            save_result = save_data(all_data)
            
            # 建立向量存儲
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join("mbti_data/csv", f"mbti_browser_{timestamp}.csv")
            
            return MBTIResponse(
                status="success",
                message=f"成功抓取資料。{save_result}",
                data=MBTIData(**all_data)
            )
        else:
            return MBTIResponse(
                status="warning",
                message="未找到任何 MBTI 資料",
                data=None
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mbti/data/", response_model=List[str], tags=["MBTI資料存儲"])
async def list_mbti_data():
    """
    列出所有已抓取的 MBTI 資料檔案
    """
    try:
        data_dir = "mbti_data/csv"
        if not os.path.exists(data_dir):
            return []
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        return sorted(files, reverse=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mbti/data/{file_name}", tags=["MBTI資料存儲"])
async def get_mbti_data(file_name: str):
    """
    獲取指定 MBTI 資料檔案的內容
    
    - **file_name**: CSV 檔案名稱
    """
    try:
        file_path = os.path.join("mbti_data/csv", file_name)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="找不到檔案")
            
        df = pd.read_csv(file_path)
        return df.to_dict(orient='records')
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mbti/analyze/", response_model=MBTIAnalysisResponse, tags=["MBTI資料分析報告"])
async def analyze_mbti_data(file: UploadFile = File(...)):
    """
    分析 MBTI 數據檔案
    
    - **file**: CSV 格式的 MBTI 數據檔案
    """
    try:
        # 讀取 CSV 檔案內容
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV 檔案是空的")

        # 計算統計數據
        stats = {
            "總樣本數": len(df),
            "類型分布": df["personality_type"].value_counts().to_dict(),
            "職業分布": df["occupation"].value_counts().to_dict()
        }

        # 生成分析報告
        analysis_report = f"""
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
"""

        # 使用 agent 進行深入分析
        messages = [{
            "role": "system",
            "content": """你是一個專業的 MBTI 性格分析專家。請注意：
1. 使用繁體中文回應
2. 提供具體的數據支持
3. 分析需要客觀且專業"""
        }]
        
        model = HfApiModel(model_id=os.getenv('HUGGINGFACE_MODEL'))
        agent = ToolCallingAgent(tools=[analyze_mbti_stats], model=model)
        
        final_analysis = agent.run(f"請根據以下數據進行深入分析：\n\n{analysis_report}")

        # 準備結果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = MBTIAnalysisResult(
            timestamp=timestamp,
            stats=MBTIStats(
                total_samples=stats['總樣本數'],
                type_distribution=stats['類型分布'],
                occupation_distribution=stats['職業分布']
            ),
            analysis_report=final_analysis
        )

        # 儲存報告
        os.makedirs("mbti_analysis", exist_ok=True)
        filename = f"mbti_analysis_report_{timestamp}.json"
        filepath = os.path.join("mbti_analysis", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result.dict(), f, ensure_ascii=False, indent=2)

        return MBTIAnalysisResponse(
            status="success",
            message="分析完成",
            data=result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mbti/vectors/", response_model=List[str], tags=["MBTI資料存儲"])
async def list_vector_stores():
    """
    列出所有可用的向量存儲
    """
    try:
        vector_dir = "mbti_data/vectors"
        if not os.path.exists(vector_dir):
            return []
        stores = [d for d in os.listdir(vector_dir) 
                 if os.path.isdir(os.path.join(vector_dir, d))]
        return sorted(stores, reverse=True)  # 最新的排在最前面
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class QueryInput(BaseModel):
    query: str
    store_name: Optional[str] = None

class QueryResponse(BaseModel):
    status: str
    message: str
    results: str

@app.post("/mbti/query/", response_model=QueryResponse, tags=["MBTI資料存儲"])
async def query_vector_data(query_input: QueryInput):
    """
    查詢 MBTI 向量數據庫
    
    - **query**: 查詢文本
    - **store_name**: 指定要查詢的向量存儲名稱（可選）
    """
    try:
        # 解包參數
        query = query_input.query
        store_name = query_input.store_name
        
        # 如果 store_name 為 None，query_mbti_data 會使用最新的存儲
        if store_name:
            # 檢查存儲是否存在
            vector_dir = "mbti_data/vectors"
            if not os.path.exists(os.path.join(vector_dir, store_name)):
                raise HTTPException(
                    status_code=404,
                    detail=f"找不到向量存儲: {store_name}"
                )
        
        # 執行查詢，使用關鍵字參數
        results = query_mbti_data(query=query, store_name=store_name)
        
        return QueryResponse(
            status="success",
            message="查詢完成",
            results=results
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"查詢失敗: {str(e)}"
        )

@app.post("/mbti/create_vector_store/", tags=["MBTI資料存儲"])
async def create_new_vector_store(file: UploadFile = File(...)):
    """
    從 CSV 檔案建立新的向量存儲
    
    - **file**: CSV 格式的 MBTI 數據檔案
    """
    try:
        # 儲存上傳的檔案
        csv_dir = "mbti_data/csv"
        os.makedirs(csv_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"mbti_browser_{timestamp}.csv"
        csv_path = os.path.join(csv_dir, csv_filename)
        
        # 寫入檔案
        with open(csv_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
            
        # 建立向量存儲
        result = create_vector_store(csv_path)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "向量存儲建立完成",
                "details": result
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"建立向量存儲失敗: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500) 