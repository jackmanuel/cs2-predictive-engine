import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_DIR, CHECKPOINT_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE, TRAIN_RATIO, VAL_RATIO
from model.dataset import MatchDataset
from model.net import MatchPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_model():
    data_path = PROCESSED_DIR / "features.parquet"
    if not data_path.exists():
        logger.error(f"Features file not found at {data_path}")
        return

    df = pd.read_parquet(data_path)
    # Ensure sorted temporally for our split to be valid
    df = df.sort_values("date").reset_index(drop=True)
    
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
