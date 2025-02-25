from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import os
import json
import requests
from huggingface_hub import InferenceClient
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# 載入環境變數
load_dotenv()

class BaseModelProvider(ABC):
    """基礎模型供應商抽象類"""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """獲取模型名稱"""
        pass

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """生成回應"""
        pass

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """聊天對話"""
        pass

    def embed_query(self, text: str) -> List[float]:
        """將文本轉換為向量"""
        return []

class OpenAIProvider(BaseModelProvider):
    """OpenAI 模型供應商"""
    
    def __init__(self, model_name: str = "gpt-4-turbo-preview"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("未設置 OPENAI_API_KEY")
        self._model_name = model_name
        self.api_base = "https://api.openai.com/v1"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model_name,
            "messages": messages
        }
        
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenAI API 錯誤: {response.text}")

    def chat(self, messages: List[Dict[str, str]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": messages
        }
        
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenAI API 錯誤: {response.text}")

class GeminiProvider(BaseModelProvider):
    """Google Gemini 模型供應商"""
    
    def __init__(self, model_name: str = "gemini-pro"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("未設置 GOOGLE_API_KEY")
        self._model_name = model_name
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        response = self.model.generate_content(prompt)
        return response.text

    def chat(self, messages: List[Dict[str, str]]) -> str:
        chat = self.model.start_chat()
        for message in messages:
            if message["role"] != "system":
                chat.send_message(message["content"])
        return chat.last.text

class HuggingFaceProvider(BaseModelProvider):
    """HuggingFace 模型供應商"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("未設置 HUGGINGFACE_API_KEY")
        self._model_name = model_name
        self.client = InferenceClient(model=model_name, token=self.api_key)
        self._embedding_model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        response = self.client.text_generation(prompt, max_new_tokens=512)
        return response

    def chat(self, messages: List[Dict[str, str]]) -> str:
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                formatted_messages.append({"role": "system", "content": msg["content"]})
            else:
                formatted_messages.append({"role": msg["role"], "content": msg["content"]})
        
        response = self.client.text_generation(
            json.dumps(formatted_messages),
            max_new_tokens=512
        )
        return response

    def embed_query(self, text: str) -> List[float]:
        """將文本轉換為向量"""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return self._embedding_model.encode(text, convert_to_numpy=True)

class OllamaProvider(BaseModelProvider):
    """Ollama 模型供應商"""
    
    def __init__(self, model_name: str = "llama2"):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model_name,
            "prompt": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        }
        response = requests.post(url, json=data)
        return response.json()["response"]

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/api/chat"
        data = {
            "model": self.model_name,
            "messages": messages
        }
        response = requests.post(url, json=data)
        return response.json()["message"]["content"]

class ModelFactory:
    """模型工廠類"""
    
    @staticmethod
    def create_provider(provider_type: str, model_name: Optional[str] = None) -> BaseModelProvider:
        """創建模型供應商實例"""
        providers = {
            "openai": OpenAIProvider,
            "gemini": GeminiProvider,
            "huggingface": HuggingFaceProvider,
            "ollama": OllamaProvider
        }
        
        if provider_type not in providers:
            raise ValueError(f"不支援的供應商類型: {provider_type}")
            
        provider_class = providers[provider_type]
        return provider_class(model_name) if model_name else provider_class()

# 使用示例
if __name__ == "__main__":
    # 創建模型實例
    try:
        # OpenAI
        openai_provider = ModelFactory.create_provider("openai", "gpt-4-turbo-preview")
        
        # Gemini
        gemini_provider = ModelFactory.create_provider("gemini", "gemini-pro")
        
        # HuggingFace
        hf_provider = ModelFactory.create_provider("huggingface", "Qwen/Qwen2.5-Coder-32B-Instruct")
        
        # Ollama
        ollama_provider = ModelFactory.create_provider("ollama", "llama2")
        
        # 測試生成
        prompt = "請簡單介紹 MBTI 性格分析。"
        system_prompt = "你是一個專業的 MBTI 分析專家。"
        
        print("OpenAI 回應:", openai_provider.generate(prompt, system_prompt))
        print("Gemini 回應:", gemini_provider.generate(prompt, system_prompt))
        print("HuggingFace 回應:", hf_provider.generate(prompt, system_prompt))
        print("Ollama 回應:", ollama_provider.generate(prompt, system_prompt))
        
    except Exception as e:
        print(f"錯誤: {str(e)}") 