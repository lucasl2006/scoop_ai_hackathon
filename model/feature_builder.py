import numpy as np
import pandas as pd

def compute_features(df: pd.DataFrame) -> np.ndarray:
    df = df.copy()
    # Basic OHLCV features
    features = df[["open", "high", "low", "close", "volume"]].values.astype(np.float32)

    # Derived features
    df["price_change"] = df["close"].diff().fillna(0)
    df["volatility"] = (df["high"] - df["low"]).fillna(0)
    df["ma_5"] = df["close"].rolling(5).mean().fillna(df["close"])

    derived_features = df[["price_change", "volatility", "ma_5"]].values.astype(np.float32)
    final_features = np.hstack([features, derived_features])
    return final_features

# Example:
# features = compute_features(candle_df)
# print(features.shape)
