from smolagents import ToolCallingAgent, HfApiModel, tool
from scholarly import scholarly
import arxiv
from typing import List, Dict
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# 載入環境變數
load_dotenv()

@tool
def search_papers(query: str, max_results: int = 5) -> List[Dict]:
    """搜尋學術論文
    
    Args:
        query: 搜尋關鍵字
        max_results: 最大結果數量
        
    Returns:
        List[Dict]: 論文資訊列表
    """
    papers = []
    
    # 搜尋 Google Scholar
    try:
        print("正在搜尋 Google Scholar...")
        search_query = scholarly.search_pubs(query)
        for i in range(max_results):
            try:
                paper = next(search_query)
                papers.append({
                    "title": paper.get("bib", {}).get("title", ""),
                    "authors": paper.get("bib", {}).get("author", []),
                    "year": paper.get("bib", {}).get("year", ""),
                    "abstract": paper.get("bib", {}).get("abstract", ""),
                    "url": paper.get("pub_url", ""),
                    "source": "Google Scholar"
                })
            except StopIteration:
                break
    except Exception as e:
        print(f"Google Scholar 搜尋出錯: {str(e)}")

    # 搜尋 arXiv
    try:
        print("正在搜尋 arXiv...")
        arxiv_search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        for paper in arxiv_search.results():
            papers.append({
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "year": paper.published.year,
                "abstract": paper.summary,
                "url": paper.pdf_url,
                "source": "arXiv"
            })
    except Exception as e:
        print(f"arXiv 搜尋出錯: {str(e)}")

    return papers

@tool
def analyze_paper_for_code(paper_info: Dict) -> str:
    """分析論文內容，尋找和提取程式碼相關的描述
    
    Args:
        paper_info: 論文資訊
        
    Returns:
        str: 分析結果
    """
    # 準備提示詞
    prompt = f"""
請分析以下論文資訊，找出與推薦系統實作相關的關鍵描述，並提供可能的 Python 實作程式碼：

標題：{paper_info['title']}
作者：{', '.join(paper_info['authors'])}
年份：{paper_info['year']}
摘要：{paper_info['abstract']}

請提供：
1. 論文中描述的推薦系統核心演算法
2. 相關的 Python 實作程式碼
3. 程式碼的說明和使用方法

注意：
- 程式碼應該實用且可執行
- 使用現代的 Python 函式庫（如 numpy, pandas, scikit-learn 等）
- 包含必要的註解
- 考慮程式碼的效能和可讀性
"""
    return prompt

def main():
    # 初始化 AI 模型
    model = HfApiModel(
        model_id=os.getenv('HUGGINGFACE_MODEL')
    )
    
    # 初始化代理
    agent = ToolCallingAgent(
        tools=[search_papers, analyze_paper_for_code],
        model=model
    )
    
    print("=== 推薦系統論文程式碼提取工具 ===")
    
    # 搜尋論文
    search_query = "recommendation system implementation python code"
    papers = search_papers(search_query)
    
    print(f"\n找到 {len(papers)} 篇相關論文")
    
    # 建立輸出目錄
    output_dir = "recommendation_code_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 分析每篇論文
    for i, paper in enumerate(papers, 1):
        print(f"\n正在分析第 {i} 篇論文：{paper['title']}")
        
        try:
            # 使用 AI 代理分析論文
            analysis = agent.run(analyze_paper_for_code(paper))
            
            # 儲存分析結果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"paper_analysis_{timestamp}_{i}.json"
            filepath = os.path.join(output_dir, filename)
            
            result = {
                "paper_info": paper,
                "analysis": analysis
            }
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"分析結果已儲存至：{filepath}")
            
            # 印出分析結果
            print("\n=== 分析結果 ===")
            print(analysis)
            print("="*50)
            
        except Exception as e:
            print(f"分析過程出錯：{str(e)}")
            continue

if __name__ == "__main__":
    main() 