import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

def compute_metrics(y_true, y_pred_prob):
    """
    Computes a comprehensive suite of evaluation metrics for a binary probability model.
    Accepts both torch tensors and numpy arrays.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_prob, torch.Tensor):
        y_pred_prob = y_pred_prob.detach().cpu().numpy()
        
    y_pred_prob = np.clip(y_pred_prob, 1e-7, 1 - 1e-7) # stability
    y_pred_class = (y_pred_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred_class)
    ll = log_loss(y_true, y_pred_prob)
    
    # Brier score is MSE of probabilities - excellent for calibration evaluation 
    brier = brier_score_loss(y_true, y_pred_prob)
    
    # Check if there's only one class (edge case in small batches)
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred_prob)
    else:
        auc = float("nan")
        
    return {
        "accuracy": acc,
        "log_loss": ll,
        "brier_score": brier,
        "roc_auc": auc
    }

def print_report(metrics_dict):
    """Pretty prints the metrics dict."""
    print("=== Evaluation Report ===")
    print(f"Accuracy:    {metrics_dict['accuracy']:.4f}")
    if not np.isnan(metrics_dict['roc_auc']):
         print(f"ROC AUC:     {metrics_dict['roc_auc']:.4f}")
    print(f"Log Loss:    {metrics_dict['log_loss']:.4f}")
    print(f"Brier Score: {metrics_dict['brier_score']:.4f}")
    print("=========================")

