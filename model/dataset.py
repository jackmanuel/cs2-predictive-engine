import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MatchDataset(Dataset):
    """
    PyTorch Dataset for CS2 match predictions.
    Separates features from identifiers and labels.
    """
    def __init__(self, df: pd.DataFrame, scaler=None, fit_scaler=False):
        drop_cols = ["match_id", "date", "team_a_id", "team_b_id", "map_name", "label", "match_format", "score_a", "score_b"]
        # Filter to only actual features
        feature_cols = [c for c in df.columns if c not in drop_cols]
        
        self.X_raw = df[feature_cols].values.astype(np.float32)
        self.y_raw = df["label"].values.astype(np.float32)
        
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
