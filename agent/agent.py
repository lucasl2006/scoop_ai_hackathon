import asyncio
from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.chat import ChatBot
from spoon_ai.tools import ToolManager
from tools.predict_tool import CandlePredictionTool

class CandleAgent(ToolCallAgent):
    name: str = "candle_agent"
    description: str = "Agent that analyzes candle data and predicts market reactions"
    system_prompt: str = "You are an AI agent that predicts market reactions from candlestick patterns."

    available_tools: ToolManager = ToolManager([
        CandlePredictionTool(),
        # Optional: add tool to fetch candles dynamically via API
    ])

async def main():
    agent = CandleAgent(
        llm=ChatBot(
            llm_provider="openai",
            model_name="gpt-5.1"
        )
    )

    # Use uploaded CSV file for prediction
    candle_csv_path = "/mnt/data/9018a46c-c79d-4f8a-b0ff-ff7888f1ca1d.csv"
    response = await agent.run(f"Predict the market reaction using this candle CSV: {candle_csv_path}")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
