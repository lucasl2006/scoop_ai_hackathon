import pandas as pd
import numpy as np

def compute_features(df: pd.DataFrame):
    """
    First column = rebound label.
    Remaining columns = candle data.
    """
    
    # Keep everything except the first column (label)
    feature_df = df.iloc[:, 1:]

    # Convert to float32 numpy array
    features = feature_df.values.astype(np.float32)

    return features
