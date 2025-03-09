from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.wsgi import WSGIMiddleware
import os
from dotenv import load_dotenv
from mbti_browser_gradio import web_ui
from pydantic import BaseModel, HttpUrl
from typing import Dict, Optional, List, Union, Any
import pandas as pd
from datetime import datetime
from recommendation_analysis import analyze_recommendation_results, agent, get_deep_analysis
from mbti_browser_scraper import browse_webpage, extract_mbti_info, save_data, create_vector_store, query_mbti_data
from mbti_analysis_scraper import analyze_mbti_stats
import gradio as gr
import json

# 初始化 FastAPI 應用
app = FastAPI(
    title='MBTI 性格分析系統 API',
    version='1.0.0',
    docs_url='/api/docs',
    openapi_url='/api/openapi.json'
)

# 載入環境變數
load_dotenv()

# 檢查環境
def check_environment():
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
        "mbti_data/vectors",
        "recommendation_analysis"
    ]
    
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)

# 初始化環境
check_environment()

# 定義請求/響應模型
class URLInput(BaseModel):
    urls: List[HttpUrl]

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

class QueryInput(BaseModel):
    query: str
    store_name: Optional[str] = None

class AnalysisInput(BaseModel):
    text: str
    analysis_type: str = "general"

class RecommendationInput(BaseModel):
    mbti_type: str
    career_goals: List[str]
    interests: List[str]
    skills: List[str]

# API 路由
@app.get('/api/mbti/reports/', tags=['MBTI 分析'])
async def list_reports():
    """列出所有已生成的分析報告"""
    try:
        report_dir = "recommendation_analysis"
        if not os.path.exists(report_dir):
            return []
        files = [f for f in os.listdir(report_dir) if f.endswith('.json')]
        return sorted(files, reverse=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/mbti/scrape/', tags=['MBTI 分析'])
async def scrape_mbti_data(body: URLInput):
    """從指定的網址抓取 MBTI 相關資料"""
    try:
        all_data = {
            "personality_type": [],
            "occupation": [],
            "description": [],
            "source_url": [],
            "extracted_date": []
        }
        
        for url in body.urls:
            content = browse_webpage(str(url))
            if content:
                data = extract_mbti_info(content)
                if data:
                    for key in all_data:
                        all_data[key].extend(data[key])
        
        if any(len(v) > 0 for v in all_data.values()):
            save_result = save_data(all_data)
            return {
                "status": "success",
                "message": f"成功抓取資料。{save_result}",
                "data": all_data
            }
        
        return {
            "status": "warning",
            "message": "未找到任何 MBTI 資料",
            "data": None
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/mbti/data/', tags=['MBTI 分析'])
async def list_mbti_data():
    """列出所有已抓取的 MBTI 資料檔案"""
    try:
        data_dir = "mbti_data/csv"
        if not os.path.exists(data_dir):
            return []
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        return sorted(files, reverse=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/mbti/query/', tags=['MBTI 分析'])
async def query_vector_data(body: QueryInput):
    """查詢 MBTI 向量數據"""
    try:
        results = query_mbti_data(query=body.query, store_name=body.store_name)
        return {
            "status": "success",
            "message": "查詢完成",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/mbti/analyze/', tags=['MBTI 分析'])
async def analyze_text(body: AnalysisInput):
    """分析文本內容"""
    try:
        result = analyze_mbti_stats(body.text, analysis_type=body.analysis_type)
        return {
            "status": "success",
            "message": "分析完成",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/mbti/recommend/', tags=['MBTI 分析'])
async def get_recommendations(body: RecommendationInput):
    """獲取 MBTI 職業推薦"""
    try:
        # 生成檔案名稱
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recommendation_{timestamp}.json"
        filepath = os.path.join("recommendation_analysis", filename)

        # 生成推薦結果
        results = analyze_recommendation_results(
            mbti_type=body.mbti_type,
            career_goals=body.career_goals,
            interests=body.interests,
            skills=body.skills
        )

        # 保存結果
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 進行深度分析
        deep_analysis = get_deep_analysis(results)

        return {
            "status": "success",
            "message": "推薦生成完成",
            "filename": filename,
            "results": results,
            "deep_analysis": deep_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/mbti/report/{filename}', tags=['MBTI 分析'])
async def get_report(filename: str):
    """獲取特定的分析報告"""
    try:
        filepath = os.path.join("recommendation_analysis", filename)
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="報告不存在")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            "status": "success",
            "message": "報告讀取成功",
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", tags=['系統'])
async def root():
    """主路由：重定向到 Gradio 介面"""
    return RedirectResponse(url="/gradio")

# 創建 Gradio 介面
interface = gr.Blocks()
with interface:
    web_ui.render()  # 假設 web_ui 有 render 方法來渲染介面

# 掛載 Gradio 介面
app = gr.mount_gradio_app(app, interface, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))