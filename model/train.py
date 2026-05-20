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
import copy

# Ensure project root is in path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, PROCESSED_DIR, CHECKPOINT_DIR, BATCH_SIZE, EPOCHS,
                    LEARNING_RATE, EARLY_STOPPING_PATIENCE, TRAIN_RATIO, VAL_RATIO,
                    DROPOUT_RATE)
from processing.features import mirror_data, MODEL_FEATURES
from model.dataset import MatchDataset
from model.net import MatchPredictor
from model.veto_sim import MAP_POOL
from evaluation.shadow_ledger import register_model_version
from evaluation.metrics import compute_metrics

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
        map_counts = df["map_name"].value_counts()
        map_popularity = {
            map_name: int(map_counts.get(map_name, 0))
            for map_name in sorted(MAP_POOL, key=lambda name: map_counts.get(name, 0), reverse=True)
        }
        
        state = {
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_teams": num_teams,
            "total_matches": num_matches,
            "total_maps": num_maps,
            "total_rounds": int(total_rounds),
            "avg_rounds_per_map": round(avg_rounds, 2),
            "formats": match_formats,
            "map_popularity": map_popularity,
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
    
    # Apply data mirroring (augmentation) to make model order-robust
    logger.info("Applying data mirroring to training set...")
    train_df = mirror_data(train_df)
    
    logger.info(f"Split sizes (after mirroring) -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Initialize scaler & Datasets
    scaler = StandardScaler()
    train_dataset = MatchDataset(train_df, scaler=scaler, fit_scaler=True)
    val_dataset = MatchDataset(val_df, scaler=scaler, fit_scaler=False)
    test_dataset = MatchDataset(test_df, scaler=scaler, fit_scaler=False)
    
    # Save scaler for inference
    scaler_path = CHECKPOINT_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = train_dataset.X.shape[1]
    best_model_path = CHECKPOINT_DIR / "best_mvp_model.pt"

    NUM_SEEDS = 5
    best_overall_val_loss = float("inf")
    best_overall_state = None
    epochs_run_overall = 0
    test_metrics_list = []

    logger.info(f"Starting Ensemble Training over {NUM_SEEDS} random seeds to verify stability...")
    
    for i, seed in enumerate(range(1, NUM_SEEDS + 1)):
        logger.info(f"\n--- Training Seed {seed} ({i+1}/{NUM_SEEDS}) ---")
        torch.manual_seed(seed)
        
        model = MatchPredictor(input_dim)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_val_loss = float("inf")
        patience_counter = 0
        best_state_for_seed = None
        epochs_run = 0

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
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1:03d}/{EPOCHS} - Val Loss: {val_loss:.4f} - LR: {current_lr:.1e}")
            
            epochs_run = epoch + 1
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state_for_seed = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    break
        
        # Evaluate this seed on the completely unseen Test Set
        model.load_state_dict(best_state_for_seed)
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                all_preds.append(model(X_batch))
                all_labels.append(y_batch)
                
        test_preds = torch.cat(all_preds)
        test_labels = torch.cat(all_labels)
        metrics = compute_metrics(test_labels, test_preds)
        test_metrics_list.append(metrics)
        
        logger.info(f"Seed {seed} Test Results -> Brier: {metrics['brier_score']:.4f} | LogLoss: {metrics['log_loss']:.4f}")
        
        if best_val_loss < best_overall_val_loss:
            best_overall_val_loss = best_val_loss
            best_overall_state = best_state_for_seed
            epochs_run_overall = epochs_run

    # Ensemble summary
    avg_brier = sum(m["brier_score"] for m in test_metrics_list) / NUM_SEEDS
    avg_log_loss = sum(m["log_loss"] for m in test_metrics_list) / NUM_SEEDS
    logger.info(f"\n======================================")
    logger.info(f" ENSEMBLE TEST AVERAGES ({NUM_SEEDS} seeds)")
    logger.info(f" Brier Score: {avg_brier:.4f}")
    logger.info(f" Log Loss:    {avg_log_loss:.4f}")
    logger.info(f"======================================\n")

    # Save the absolute best model
    torch.save(best_overall_state, best_model_path)
    logger.info(f"Best model globally (val_loss: {best_overall_val_loss:.4f}) saved to {best_model_path}")

    # Register this training run in the model version registry
    training_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    hyperparams = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "dropout_rate": DROPOUT_RATE,
        "weight_decay": 1e-4,
        "lr_scheduler": "ReduceLROnPlateau(factor=0.5, patience=5)",
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
    }
    
    # Reuse the training state data we already computed
    state_path = DATA_DIR / "training_state.json"
    try:
        with open(state_path, "r") as f:
            data_stats = json.load(f)
    except Exception:
        data_stats = {}
    
    register_model_version(
        trained_at=training_timestamp,
        best_val_loss=best_overall_val_loss,
        epochs_run=epochs_run_overall,
        features=MODEL_FEATURES,
        hyperparams=hyperparams,
        data_stats=data_stats,
        weights_src=str(best_model_path),
        scaler_src=str(scaler_path),
        test_brier_score=avg_brier,
        test_log_loss=avg_log_loss,
    )

if __name__ == "__main__":
    train_model()
