import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from processing.features import MODEL_FEATURES, TARGET_COL

class MatchDataset(Dataset):
    """
    PyTorch Dataset for CS2 match predictions.
    Separates features from identifiers and labels.
    """
    def __init__(self, df: pd.DataFrame, scaler=None, fit_scaler=False):
        # Filter to only actual features using the architecture-defined list
        self.X_raw = df[MODEL_FEATURES].values.astype(np.float32)
        self.y_raw = df[TARGET_COL].values.astype(np.float32)
        
        if fit_scaler and scaler is not None:
            self.X = scaler.fit_transform(self.X_raw)
        elif scaler is not None:
            self.X = scaler.transform(self.X_raw)
        else:
            self.X = self.X_raw
            
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y_raw, dtype=torch.float32).unsqueeze(1) # shape [N, 1]
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
