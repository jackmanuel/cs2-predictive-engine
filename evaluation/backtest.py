import logging
import torch
import joblib
import numpy as np
import pandas as pd

from config import PROCESSED_DIR, CHECKPOINT_DIR, TRAIN_RATIO, VAL_RATIO
from processing.features import MODEL_FEATURES, TARGET_COL
from model.net import MatchPredictor
from evaluation.metrics import compute_metrics, print_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def backtest():
    """
    Evaluates the trained model on the temporally held-out test set.
    Uses the serialized scaler and model checkpoint from the training pipeline.
    """
    data_path = PROCESSED_DIR / "features.parquet"
    model_path = CHECKPOINT_DIR / "best_mvp_model.pt"
    scaler_path = CHECKPOINT_DIR / "scaler.pkl"
    
    for p in [data_path, model_path, scaler_path]:
        if not p.exists():
            logger.error(f"Required file not found: {p}")
            return
        
    df = pd.read_parquet(data_path)
    df = df.sort_values("date").reset_index(drop=True)
    
    n = len(df)
    val_idx = int(n * (TRAIN_RATIO + VAL_RATIO))
    test_df = df.iloc[val_idx:]
    
    logger.info(f"Backtesting on {len(test_df)} out-of-sample maps...")
    
    # Load the serialized scaler from training (ensures identical transform)
    scaler = joblib.load(scaler_path)
    
    # Prepare test features using the same scaler as training
    X_test = scaler.transform(test_df[MODEL_FEATURES].values.astype(np.float32))
    y_true = test_df[TARGET_COL].values.astype(np.float32)
    
    # Load trained model
    model = MatchPredictor(input_dim=scaler.n_features_in_)
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except Exception:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X_tensor).numpy().flatten()
        
    metrics = compute_metrics(y_true, preds)
    print_report(metrics)

if __name__ == "__main__":
    backtest()
