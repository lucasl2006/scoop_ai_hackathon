import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model.new_feature_builder import compute_features
from model.pytorch_model import CandlePatternPredictor

# 1 Load your CSV with the rebound labels
def load_data(csv_path):

    df = pd.read_csv(csv_path)

    # Check rebound label
    #if "rebounded" not in df.columns:
    #    raise ValueError("CSV must contain a 'rebounded' column with 0/1 labels.")

    #shuffles rows
    df = df.sample(frac=1).reset_index(drop=True)

    # Compute feature matrix
    X = compute_features(df)

    # Target labels (0 or 1)
    y = df[:,0].values.astype(np.float32)

    return X, y


# 2 Training loop
def train_model(csv_path, batch_size = 32, lr = 1e-3, epochs = 20):

    # Load data
    X, y = load_data(csv_path)

    # Convert to tensors
    X_tensor = torch.tensor(X).float()
    y_tensor = torch.tensor(y).float().unsqueeze(1)  # shape: [N, 1]

    # Dataset + loader
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    input_dim = X.shape[1]
    model = CandlePatternPredictor(input_dim)

    # Loss + optimizer
    criterion = nn.BCELoss()  # binary cross entropy for 0/1 predictions
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in loader:

            # Forward
            preds = model(batch_X)
            loss = criterion(preds, batch_y)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Compute accuracy
            predicted_classes = (preds >= 0.5).float()
            correct += (predicted_classes == batch_y).sum().item()
            total += batch_y.size(0)

        accuracy = correct / total

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.3f}")

    # Save model weights
    save_path = "model/trained_candle_model.pth"
    torch.save(model.state_dict(), save_path)

    print(f"\nTraining complete! Model saved to: {save_path}")

    return model


# 3 Train if run as script

if __name__ == "__main__":

    csv_path = "data/bulls.csv"     # You can replace this with your path
    train_model(csv_path)
