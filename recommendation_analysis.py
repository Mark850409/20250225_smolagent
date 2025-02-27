from smolagents import ToolCallingAgent, HfApiModel, tool
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import traceback

# 載入環境變數
load_dotenv()
model_id = os.getenv('HUGGINGFACE_MODEL')

# 初始化系統訊息
messages = [{
    "role": "system",
    "content": """你是一個專業的推薦系統分析專家。請注意：
1. 使用繁體中文回應
2. 提供具體的數據支持
3. 分析需要客觀且專業
4. 重點關注：
   - 不同實驗類型的優劣比較
   - 模型效能的穩定性
   - 具體的改進建議
   - 分析各指標（RMSE、MAE、精確率、召回率、NDCG、F1）的意義和影響"""
}]

@tool
def analyze_recommendation_results(results: dict) -> str:
    """分析推薦系統實驗結果。

    Args:
        results: 實驗結果數據，包含實驗設定和三種推薦方法的評估指標

    Returns:
        str: 分析報告
    """
    try:
        exp_results = results.get('experiment_results', {})
        settings = results.get('experiment_settings', {})
        
        report = ["=== 推薦系統效能分析報告 ===\n"]
        
        # 添加實驗設定資訊（如果有的話）
        if settings:
            data_settings = settings.get('data_settings', {})
            dataset_info = settings.get('dataset_info', {})
            
            if data_settings or dataset_info:
                report.append("零、實驗設定")
                report.append("-" * 50)
                
                if data_settings:
                    report.append(f"• 訓練集/測試集比例：{data_settings.get('train_ratio', 80)}/"
                                f"{data_settings.get('test_ratio', 20)}")
                    report.append(f"• 評分閾值：{data_settings.get('rating_threshold', 3.5)}")
                    report.append(f"• 情感分析閾值：{data_settings.get('sentiment_threshold', 0.5)}")
                
                if dataset_info:
                    report.append(f"• 店家數量：{dataset_info.get('total_items', 'N/A')}")
                    report.append(f"• 評論數量：{dataset_info.get('total_reviews', 'N/A')}")
                report.append("")
        
        # 添加實驗比較表格
        report.append("一、各推薦方法效能比較")
        report.append("-" * 50)
        report.append("| 方法 | RMSE | MAE | 精確率 | 召回率 | NDCG | F1 |")
        report.append("|------|------|-----|--------|--------|------|-----|")
        
        # 儲存每個方法的指標，用於後續分析
        method_metrics = {}
        
        for method, metrics in exp_results.items():
            method_name = {
                'rating_only': '純評分',
                'sentiment_only': '純情感',
                'hybrid': '混合'
            }.get(method, method)
            
            method_metrics[method] = metrics
            
            report.append(
                f"| {method_name} | {metrics['rmse']:.2f} | {metrics['mae']:.2f} | "
                f"{metrics['precision']:.2f} | {metrics['recall']:.2f} | "
                f"{metrics['ndcg']:.2f} | {metrics['f1']:.2f} |"
            )
        
        # 綜合分析
        report.append("\n二、綜合效能分析")
        report.append("-" * 50)
        
        # 分析預測準確性
        report.append("1. 預測準確性分析：")
        for method, metrics in method_metrics.items():
            method_name = {'rating_only': '純評分', 'sentiment_only': '純情感', 'hybrid': '混合'}.get(method)
            report.append(f"   - {method_name}方法：")
            report.append(f"     RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}")
            report.append(f"     評估：{'優秀' if metrics['rmse'] < 1.0 else '良好' if metrics['rmse'] < 1.5 else '需改進'}")
        
        # 分析推薦效果
        report.append("\n2. 推薦效果分析：")
        for method, metrics in method_metrics.items():
            method_name = {'rating_only': '純評分', 'sentiment_only': '純情感', 'hybrid': '混合'}.get(method)
            report.append(f"   - {method_name}方法：")
            report.append(f"     精確率: {metrics['precision']:.2f}, 召回率: {metrics['recall']:.2f}")
            report.append(f"     NDCG: {metrics['ndcg']:.2f}, F1: {metrics['f1']:.2f}")
            
            # 分析優缺點
            strengths = []
            weaknesses = []
            
            if metrics['precision'] > 0.7: strengths.append("高精確率")
            if metrics['recall'] > 0.7: strengths.append("高召回率")
            if metrics['ndcg'] > 0.7: strengths.append("良好的排序品質")
            
            if metrics['precision'] < 0.6: weaknesses.append("精確率較低")
            if metrics['recall'] < 0.6: weaknesses.append("召回率較低")
            if metrics['ndcg'] < 0.6: weaknesses.append("排序品質需改進")
            
            report.append(f"     優點：{', '.join(strengths) if strengths else '無明顯優勢'}")
            report.append(f"     缺點：{', '.join(weaknesses) if weaknesses else '無明顯缺點'}")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"分析過程發生錯誤: {str(e)}"

# 初始化模型和代理
model = HfApiModel(
    model_id=model_id, 
    messages=messages,
    temperature=0.7,
    max_tokens=1000,
    timeout=120
)
agent = ToolCallingAgent(
    tools=[analyze_recommendation_results], 
    model=model
)

def get_deep_analysis(initial_analysis: str) -> str:
    """執行深入分析，包含錯誤處理和備用方案"""
    try:
        analysis_prompt = f"""
請簡要分析以下推薦系統實驗結果（請控制在 500 字以內）：

{initial_analysis}

重點分析：
1. 最佳推薦方法的選擇
2. 關鍵指標的表現
3. 具體改進建議

請使用繁體中文回應。
"""
        # 嘗試使用 agent 進行分析
        analysis = agent.run(analysis_prompt)
        
        if isinstance(analysis, str) and len(analysis.strip()) > 0:
            return analysis
            
        # 如果 agent 分析失敗，返回一個基於初步分析的簡單總結
        return f"""
=== 深入分析 ===

基於上述初步分析結果，我們可以得出以下結論：

1. 最佳推薦方法：
   根據綜合指標表現，選擇表現最優異的方法作為主要推薦策略。

2. 效能評估：
   - 準確性指標（RMSE、MAE）顯示預測的精確程度
   - 排序指標（NDCG）反映推薦內容的相關性
   - 檢索指標（精確率、召回率）表明系統的實用性

3. 改進建議：
   - 持續優化模型參數
   - 考慮融合多種方法的優點
   - 根據實際應用場景調整策略

請參考初步分析中的具體數據作為依據。
"""
        
    except Exception as e:
        print(f"深入分析過程發生錯誤: {str(e)}")
        return initial_analysis

def main():
    """主程式：分析推薦系統實驗結果"""
    try:
        # 讀取實驗結果
        df = pd.read_csv('mbti_data/csv/a.csv')
        
        # 讀取實驗設定
        settings_path = 'recommendation_setting/experiment_settings.json'
        settings = {}
        if os.path.exists(settings_path):
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
        
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
        
        for method in methods:
            for metric in metrics:
                try:
                    value = df.loc[df['metric'].str.lower() == metric.lower(), method].iloc[0]
                    analysis_data['experiment_results'][method][metric.lower()] = float(value)
                except (IndexError, KeyError) as e:
                    print(f"警告：無法獲取 {method} 的 {metric} 指標")
                    analysis_data['experiment_results'][method][metric.lower()] = 0.0
        
        # 使用 agent 進行深入分析
        print("\n使用 AI 進行深入分析...")
        
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
        
        # 使用 analyze_recommendation_results 工具函數先進行分析
        initial_analysis = analyze_recommendation_results(serializable_data)
        
        # 使用新的深入分析函數
        analysis = get_deep_analysis(initial_analysis)
        
        # 儲存分析報告
        print("\n儲存分析報告...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_report_{timestamp}.json"
        filepath = os.path.join("recommendation_analysis", filename)
        
        os.makedirs("recommendation_analysis", exist_ok=True)
        
        # 準備要儲存的數據
        report_data = {
            "timestamp": timestamp,
            "metrics": {
                "experiment_results": serializable_data['experiment_results']
            },
            "initial_analysis": initial_analysis,
            "final_analysis": analysis
        }
        
        # 儲存為 JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n分析報告已儲存至: {filepath}")
        print("\n=== 初步分析 ===")
        print(initial_analysis)
        print("\n=== 深入分析 ===")
        print(analysis)
        
        return analysis_data
        
    except Exception as e:
        print(f"分析過程發生錯誤: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 