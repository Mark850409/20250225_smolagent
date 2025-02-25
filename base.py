from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
from huggingface_hub import login
from dotenv import load_dotenv
import os

# 載入環境變數
load_dotenv()

# 從環境變數獲取 API 金鑰
hf_api_key = os.getenv('HUGGINGFACE_API_KEY')

# 登入 Hugging Face
login(hf_api_key)

# 從環境變數獲取模型 ID
model_id = os.getenv('MODEL_ID')

# 創建一個使用 Hugging Face 模型的代理
model = HfApiModel(model_id=model_id)
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

# 讓 AI 助手執行任務
result = agent.run("請幫我找出全球最高的三座山，並列出它們的高度。")
print(result)
