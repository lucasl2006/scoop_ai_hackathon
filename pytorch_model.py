import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error 
#import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OurModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # output: next price movement (can be regression or classification)
        )

    def forward(self, x):
        return self.fc(x)
