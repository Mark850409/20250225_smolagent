# 使用官方 Python 基礎映像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 複製專案檔案
COPY . .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 安裝 Playwright 依賴
RUN playwright install chromium
RUN playwright install-deps

# 設定環境變數
ENV PORT=8000
ENV HOST=0.0.0.0

# 暴露端口
EXPOSE 8000

# 啟動應用
CMD ["python", "app.py"] 