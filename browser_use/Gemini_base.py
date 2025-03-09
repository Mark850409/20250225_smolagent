from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Browser, BrowserConfig
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

async def main():
    agent = Agent(
        task="Go to https://www.google.com/travel/flights and book a flight from Gothenburg to London on 2025-03-01 to 2025-03-10.",
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    )
    result = await agent.run()
    print(result)
asyncio.run(main())