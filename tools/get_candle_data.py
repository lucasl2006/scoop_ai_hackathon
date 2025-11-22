import requests
import pandas as pd
from spoon_ai.tools.base import BaseTool

class FetchCandlesTool(BaseTool):
    name: str = "fetch_candles"
    description: str = "Fetch candlestick data from Market Data API"

    parameters = {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "interval": {"type": "string"},
            "start": {"type": "string"},
            "end": {"type": "string"}
        },
        "required": ["symbol", "interval", "start", "end"]
    }

    async def execute(self, symbol: str, interval: str, start: str, end: str):
        # Placeholder API endpoint
        url = "https://api.marketdata.com/candles"

        params = {
            "symbol": symbol,
            "interval": interval,
            "start": start,
            "end": end,
            "apikey": "YOUR_API_KEY_HERE"
        }

        r = requests.get(url, params=params)
        r.raise_for_status()

        data = r.json()

        # Normalize expected format
        df = pd.DataFrame(data["candles"])
        csv_path = "/mnt/data/fetched_candles.csv"
        df.to_csv(csv_path, index=False)

        return {
            "status": "success",
            "csv_path": csv_path
        }
