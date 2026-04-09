import torch
import torch.nn as nn

from config import DROPOUT_RATE

class MatchPredictor(nn.Module):
    """
    MVP Binary Classifier for predicting CS2 match winners.
    Input: N tabular features
    Output: Probability Team A wins [0, 1]
    """
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)
