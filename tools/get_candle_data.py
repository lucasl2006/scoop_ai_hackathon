import requests
import pandas as pd

def fetch_candle_data(symbol: str, interval: str = "1h", start: str = "2025-01-01", end: str = "2025-01-10") -> pd.DataFrame:
    url = "https://api.marketdata.com/candles"
    headers = {"Authorization": "Bearer YOUR_API_KEY"}  # replace with your Market Data free trial key
    params = {
        "symbol": symbol,
        "interval": interval,
        "start": start,
        "end": end
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data["candles"])
    return df

# Example usage:
# candle_df = fetch_candle_data("AAPL")
# print(candle_df.head())
