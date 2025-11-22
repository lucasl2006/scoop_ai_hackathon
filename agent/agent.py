from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.tools import ToolManager
from spoon_ai.chat import ChatBot

from tools.get_level2_data import GetLevel2DataTool
from tools.greeting_tool import GreetingTool


class MarketPredictionAgent(ToolCallAgent):
    name = "market_prediction_agent"

    system_prompt = """
    You are an AI agent running inside Spoon OS.
    You can access Level 2 market data and help analyze order book patterns.
    """

    available_tools = ToolManager([
        GetLevel2DataTool(),
        GreetingTool()
    ])


def build_agent():
    return MarketPredictionAgent(
        llm=ChatBot(
            llm_provider="openai",
            model_name="gpt-5.1"
        )
    )