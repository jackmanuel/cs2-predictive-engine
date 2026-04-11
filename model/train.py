import os
import sys
import logging
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Ensure project root is in path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, PROCESSED_DIR, CHECKPOINT_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE, TRAIN_RATIO, VAL_RATIO
from model.dataset import MatchDataset
from model.net import MatchPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def save_training_state(df: pd.DataFrame):
    """Saves a snapshot of the raw data state at training time."""
    try:
        # Number of unique teams
        all_teams = set(df["team_a_id"].unique()) | set(df["team_b_id"].unique())
        num_teams = len(all_teams)
        
        # Match level stats
        # We group by match_id to count bo1, bo3, bo5
        match_formats = df.groupby("match_id")["match_format"].first().value_counts().to_dict()
        
        num_matches = df["match_id"].nunique()
        num_maps = len(df)
        
        # Additional stats
        total_rounds = (df["score_a"] + df["score_b"]).sum()
        avg_rounds = float(total_rounds / num_maps) if num_maps > 0 else 0
        top_maps = df["map_name"].value_counts().head(5).to_dict()
        
        state = {
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_teams": num_teams,
            "total_matches": num_matches,
            "total_maps": num_maps,
            "total_rounds": int(total_rounds),
            "avg_rounds_per_map": round(avg_rounds, 2),
            "formats": match_formats,
            "top_maps": top_maps,
            "date_range": {
                "start": str(df["date"].min()),
                "end": str(df["date"].max())
            }
        }
        
        out_path = DATA_DIR / "training_state.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
        logger.info(f"Training state statistics saved to {out_path}")
    except Exception as e:
        logger.error(f"Failed to save training state: {e}")

def train_model():
    data_path = PROCESSED_DIR / "features.parquet"
    if not data_path.exists():
        logger.error(f"Features file not found at {data_path}")
        return

    df = pd.read_parquet(data_path)
    # Ensure sorted temporally for our split to be valid
    df = df.sort_values("date").reset_index(drop=True)
    
    # Save training state info
    save_training_state(df)
    
    n = len(df)
    train_idx = int(n * TRAIN_RATIO)
    val_idx = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    train_df = df.iloc[:train_idx]
    val_df = df.iloc[train_idx:val_idx]
    test_df = df.iloc[val_idx:]
    
    logger.info(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Initialize scaler & Datasets
    scaler = StandardScaler()
    train_dataset = MatchDataset(train_df, scaler=scaler, fit_scaler=True)
    val_dataset = MatchDataset(val_df, scaler=scaler, fit_scaler=False)
    # Test dataset is held back for evaluation later
    
    # Save scaler for inference
    scaler_path = CHECKPOINT_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = train_dataset.X.shape[1]
    model = MatchPredictor(input_dim)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = CHECKPOINT_DIR / "best_mvp_model.pt"

    logger.info("Starting training loop...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * len(X_batch)
        val_loss /= len(val_dataset)
        
        logger.info(f"Epoch {epoch+1:03d}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    logger.info(f"Training complete. Best validation loss: {best_val_loss:.4f}. Model saved to {best_model_path}")

if __name__ == "__main__":
    train_model()
