import os
import asyncio
from dataclasses import dataclass
from typing import List, Optional
import json
from datetime import datetime

# Third-party imports
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
import uvicorn

# Local module imports
from browser_use import Agent

load_dotenv()

# API 設定
app = FastAPI(
	title='瀏覽器自動化 API',
	version='1.0.0',
	description='提供瀏覽器自動化任務的 API 服務'
)

# API Models
class TaskRequest(BaseModel):
	task: str = Field(..., description='任務描述', example='請爬取 MBTI 相關資訊')
	api_key: str = Field(..., description='API 金鑰')
	model_type: str = Field(..., description='模型類型 (OpenAI 或 Gemini)', example='OpenAI')
	model: str = Field(..., description='模型名稱', example='gpt-4-turbo-preview')
	headless: bool = Field(True, description='是否使用無頭模式')

class TaskResponse(BaseModel):
	result: str = Field(..., description='執行結果')
	status: str = Field(default="success", description='執行狀態')
	timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description='執行時間')

class ErrorResponse(BaseModel):
	error: str = Field(..., description='錯誤訊息')

@dataclass
class ActionResult:
	is_done: bool
	extracted_content: Optional[str]
	error: Optional[str]
	include_in_memory: bool


@dataclass
class AgentHistoryList:
	all_results: List[ActionResult]
	all_model_outputs: List[dict]


def parse_agent_history(history_str: str) -> None:
	console = Console()

	# Split the content into sections based on ActionResult entries
	sections = history_str.split('ActionResult(')

	for i, section in enumerate(sections[1:], 1):  # Skip first empty section
		# Extract relevant information
		content = ''
		if 'extracted_content=' in section:
			content = section.split('extracted_content=')[1].split(',')[0].strip("'")

		if content:
			header = Text(f'步驟 {i}', style='bold blue')
			panel = Panel(content, title=header, border_style='blue')
			console.print(panel)
			console.print()


def format_agent_result(result: str) -> str:
	"""格式化代理執行結果"""
	try:
		# 檢查是否包含 ActionResult
		if "ActionResult" in result:
			formatted_parts = []
			steps = result.split("ActionResult")
			
			for i, step in enumerate(steps[1:], 1):
				# 提取關鍵信息
				if "extracted_content=" in step:
					content = step.split("extracted_content=")[1].split(",")[0].strip("'")
					if content:
						try:
							# 嘗試解析 JSON 內容
							if "MBTI" in content:
								data = json.loads(content)
								mbti_types = data.get("MBTI 人格類型", {}).get("personality_types", [])
								
								# 獲取當前時間
								current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
								
								formatted_lines = []
								for type_info in mbti_types:
									type_data = [
										f"步驟 {i}:",
										f"類型：{type_info.get('type', '')} ({type_info.get('name', '')})",
										f"描述：{type_info.get('description', '無描述')}",
										f"職業：{', '.join(type_info.get('occupations', ['無相關職業']))}",
										f"代表人物：{', '.join(type_info.get('representative_figures', ['無代表人物']))}",
										f"來源：https://www.managertoday.com.tw/articles/view/65508",
										f"爬取時間：{current_time}",
										"\n" + "-" * 40 + "\n"
									]
									formatted_lines.extend(type_data)
								
								formatted_content = "\n".join(formatted_lines)
								formatted_parts.append(formatted_content)
							else:
								# 一般文本內容處理
								formatted_parts.append(f"步驟 {i}:\n{content}\n" + "-" * 40 + "\n")
						except json.JSONDecodeError:
							# 如果不是 JSON 格式，直接處理文本
							formatted_parts.append(f"步驟 {i}:\n{content}\n" + "-" * 40 + "\n")
						except Exception as e:
							formatted_parts.append(f"步驟 {i}:\n處理資料時發生錯誤: {str(e)}\n" + "-" * 40 + "\n")
				
				# 提取錯誤信息
				if "error=" in step:
					error = step.split("error=")[1].split(",")[0].strip("'")
					if error and error != "None":
						formatted_parts.append(f"錯誤:\n{error}\n" + "-" * 40 + "\n")
			
			return "\n".join(formatted_parts)
		return result
	except Exception as e:
		return f"格式化輸出時發生錯誤: {str(e)}\n原始輸出:\n{result}"


async def run_browser_task(
	task: str,
	api_key: str,
	model_type: str,
	model: str,
	headless: bool = True,
) -> str:
	if not api_key.strip():
		return '請提供 API 金鑰'

	try:
		# 構建完整的任務提示詞
		prompt_template = f"""請使用繁體中文執行以下任務，並確保所有輸出內容都是繁體中文：

{task}

請嚴格按照以下 JSON 格式輸出結果（範例）：

{{
    "MBTI 人格類型": {{
        "personality_types": [
            {{
                "type": "ESTP",
                "name": "冒險家",
                "description": "活潑開朗、靈活應變、喜歡冒險",
                "occupations": ["銷售人員", "運動員", "創業家"],
                "representative_figures": ["黃仁勳", "馬克·貝佐斯"]
            }}
        ]
    }}
}}

注意事項：
1. 所有文字內容必須使用繁體中文，包括描述、職業和代表人物
2. 每個類型必須包含完整的資訊：類型代碼(type)、名稱(name)、描述(description)、適合職業(occupations)和代表人物(representative_figures)
3. 描述至少要包含三個主要特質
4. 職業至少要列出三個
5. 代表人物至少要有一個
6. 確保輸出的 JSON 格式完全正確
7. 資料必須基於網頁內容，不要編造或猜測

請開始執行任務，記得保持所有輸出都使用繁體中文。"""

		if model_type == "OpenAI":
			os.environ['OPENAI_API_KEY'] = api_key
			llm = ChatOpenAI(
				model=model,
				temperature=0.7
			)
		else:  # Gemini
			os.environ['GOOGLE_API_KEY'] = api_key
			llm = ChatGoogleGenerativeAI(
				model=model,
				google_api_key=api_key,
				temperature=0.7
			)

		# 創建 Agent 實例，只使用必要的參數
		agent = Agent(
			task=prompt_template,
			llm=llm
		)
		
		# 如果需要設置無頭模式，可以在創建後嘗試設置
		if hasattr(agent, 'browser'):
			agent.browser.headless = headless
		
		result = await agent.run()
		return format_agent_result(result)
	except Exception as e:
		return f'錯誤: {str(e)}'


def create_ui():
	with gr.Blocks(title='瀏覽器自動化任務介面', theme=gr.themes.Soft()) as interface:
		gr.Markdown('# 瀏覽器自動化任務系統')

		with gr.Row():
			with gr.Column():
				model_type = gr.Radio(
					choices=["OpenAI", "Gemini"],
					label="模型類型",
					value="OpenAI"
				)
				
				# 定義模型選項
				model_options = {
					"OpenAI": [
						"gpt-4-turbo-preview",
						"gpt-4",
						"gpt-3.5-turbo",
						"gpt-3.5-turbo-16k"
					],
					"Gemini": [
						"gemini-1.5-flash",
						"gemini-1.5-pro",
						"gemini-1.5-pro-vision"
					]
				}
				
				model_dropdown = gr.Dropdown(
					choices=model_options["OpenAI"],
					label="選擇模型",
					value="gpt-4-turbo-preview"
				)

				api_key = gr.Textbox(
					label='API 金鑰',
					placeholder='請輸入 API 金鑰...',
					type='password'
				)
				
				task = gr.Textbox(
					label='任務描述',
					placeholder='例如：搜尋下週從台北到東京的航班',
					lines=3,
				)
				
				headless = gr.Checkbox(label='無頭模式執行', value=True)
				submit_btn = gr.Button('執行任務')

			with gr.Column(scale=2):
				output = gr.Textbox(
					label='執行結果',
					lines=20,  # 增加顯示行數
					interactive=False,
					show_copy_button=True,  # 添加複製按鈕
					elem_classes="monospace"  # 使用等寬字體
				)

		def update_model_choices(provider):
			"""更新模型選項"""
			return gr.Dropdown(choices=model_options[provider], value=model_options[provider][0])

		# 當模型類型改變時，更新模型選項
		model_type.change(
			fn=update_model_choices,
			inputs=[model_type],
			outputs=[model_dropdown]
		)

		submit_btn.click(
			fn=lambda *args: asyncio.run(run_browser_task(
				args[0],  # task
				args[1],  # api_key
				args[2],  # model_type
				args[3],  # model
				args[4]   # headless
			)),
			inputs=[task, api_key, model_type, model_dropdown, headless],
			outputs=output,
		)

	return interface


# 創建 Gradio 介面
demo = create_ui()

# 將 Gradio 介面掛載到 FastAPI
app = gr.mount_gradio_app(app, demo, path="/gradio")

# API 端點
@app.post(
	'/api/browser-task',
	response_model=TaskResponse,
	responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
	tags=["browser"],
	summary="執行瀏覽器自動化任務",
	description="執行指定的瀏覽器自動化任務並返回結果"
)
async def api_browser_task(body: TaskRequest):
	"""執行瀏覽器自動化任務"""
	try:
		result = await run_browser_task(
			body.task,
			body.api_key,
			body.model_type,
			body.model,
			body.headless
		)
		
		# 確保結果是字符串
		if isinstance(result, (dict, list)):
			result = json.dumps(result, ensure_ascii=False)
		elif not isinstance(result, str):
			result = str(result)
			
		return TaskResponse(
			result=result,
			status="success",
			timestamp=datetime.now().isoformat()
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.get('/', tags=["system"])
async def root():
	"""重定向到 Gradio 介面"""
	return RedirectResponse(url='/gradio')

if __name__ == '__main__':
	# 使用 uvicorn 啟動服務器
	uvicorn.run(app, host="0.0.0.0", port=7860)