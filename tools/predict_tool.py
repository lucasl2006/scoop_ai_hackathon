from spoon_ai.tools.base import BaseTool
from tools.feature_builder import build_features_from_orderbook
import torch
from pytorch_model import OurModel # might face issues with file paths

class PredictTool(BaseTool):
    name = "predict_price_movement"
    description = "Predict next short-term price movement from Level-2 data"

    parameters = {
        "type": "object",
        "properties": {
            "l2_data": {"type": "array"}
        },
        "required": ["l2_data"]
    }

    def __init__(self):
        super().__init__()
        self.top_k = 5
        self.input_dim = self.top_k * 4 + 2
        self.model = OurModel(self.input_dim)
        self.model.eval()  # inference mode

    async def execute(self, l2_data):
        features = build_features_from_orderbook(l2_data, top_k=self.top_k)
        x = torch.tensor(features).unsqueeze(0)  # batch of 1
        with torch.no_grad():
            pred = self.model(x).item()
        return {"prediction": pred}
