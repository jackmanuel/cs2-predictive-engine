import logging
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_DIR, CHECKPOINT_DIR, TRAIN_RATIO, VAL_RATIO
from model.dataset import MatchDataset
from model.net import MatchPredictor
from evaluation.metrics import compute_metrics, print_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def backtest():
    """
    Evaluates the trained model on the temporally held-out test set.
    """
    data_path = PROCESSED_DIR / "features.parquet"
    model_path = CHECKPOINT_DIR / "best_mvp_model.pt"
    
    if not data_path.exists() or not model_path.exists():
        logger.error("Features or Model checkpoint not found.")
        return
        
    df = pd.read_parquet(data_path)
    df = df.sort_values("date").reset_index(drop=True)
    
    n = len(df)
    train_idx = int(n * TRAIN_RATIO)
    val_idx = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    # We need the training data just to fit the scaler 
    # (since we didn't serialize the scaler yet)
    train_df = df.iloc[:train_idx]
    test_df = df.iloc[val_idx:]
    
    logger.info(f"Backtesting on {len(test_df)} out-of-sample matches...")
    
    scaler = StandardScaler()
    train_dataset = MatchDataset(train_df, scaler=scaler, fit_scaler=True)
    test_dataset = MatchDataset(test_df, scaler=scaler, fit_scaler=False)
    
    model = MatchPredictor(input_dim=train_dataset.X.shape[1])
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except Exception as e:
        # Fallback for older PyTorch versions
        model.load_state_dict(torch.load(model_path))
        
    model.eval()
    
    with torch.no_grad():
        preds = model(test_dataset.X).numpy().flatten()
        y_true = test_dataset.y_raw
        
    metrics = compute_metrics(y_true, preds)
    print_report(metrics)
    
    # Mock EV simulation output
    logger.info("Ready for Phase 2: Generating Expected Value (EV) by integrating Odds APIs.")

if __name__ == "__main__":
    backtest()
