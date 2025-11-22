import numpy as np
import pandas as pd

def build_features_from_orderbook(l2_data, top_k=5):
    df = pd.DataFrame(l2_data)
    
    # Separate bids and asks
    bids = df[df['side'] == 'bid'].sort_values(by='level').head(top_k)
    asks = df[df['side'] == 'ask'].sort_values(by='level').head(top_k)

    # Pad with zeros if less than top_k
    bids_prices = np.pad(bids['price'].to_numpy(), (0, top_k - len(bids)), 'constant')
    bids_sizes = np.pad(bids['size'].to_numpy(), (0, top_k - len(bids)), 'constant')
    asks_prices = np.pad(asks['price'].to_numpy(), (0, top_k - len(asks)), 'constant')
    asks_sizes = np.pad(asks['size'].to_numpy(), (0, top_k - len(asks)), 'constant')

    # Spread
    spread = asks_prices[0] - bids_prices[0] if len(bids) > 0 and len(asks) > 0 else 0

    # Imbalance
    imbalance = (bids_sizes.sum() - asks_sizes.sum()) / max(bids_sizes.sum() + asks_sizes.sum(), 1)

    # Combine into a feature vector
    features = np.concatenate([bids_prices, bids_sizes, asks_prices, asks_sizes, [spread, imbalance]])
    return features.astype(np.float32)
