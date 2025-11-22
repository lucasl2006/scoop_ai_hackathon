import databento as db
from spoon_ai.tools.base import BaseTool

class GetLevel2DataTool(BaseTool):
    name = "get_level2_data"
    description = "Fetch Level 2 (MBP-10) order book data from databento"

    parameters = {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "start": {"type": "string"},
            "end": {"type": "string"}
        },
        "required": ["symbol"]
    }

    async def execute(self, symbol: str, start=None, end=None):
        client = db.Historical("YOUR_DATABENTO_API_KEY")

        data = client.timeseries.get_range(
            dataset="XNAS.ITCH",
            schema="mbp-10",
            symbols=[symbol],
            stype_in="parent",
            start=start,
            end=end,
        )

        df = data.to_df()

        return df[["ts_event", "price", "size", "side", "level"]].to_dict(orient="records")
