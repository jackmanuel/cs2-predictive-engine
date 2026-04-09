import argparse
import logging
import torch
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from config import PROCESSED_DIR, CHECKPOINT_DIR, ROLLING_WINDOW_DAYS
from model.net import MatchPredictor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def get_recent_stats(history, current_date, days):
    """Helper to calculate rolling stats from a match history."""
    if not history:
        return {"matches": 0, "win_rate": 0.5}
    cutoff = current_date - pd.Timedelta(days=days)
    recent = [w for d, w in history if d >= cutoff]
    matches = len(recent)
    win_rate = sum(recent) / matches if matches > 0 else 0.5
    return {"matches": matches, "win_rate": win_rate}

def load_latest_state():
    """
    Simulates playing through the entire clean matches history to get the absolute 
    latest state (history and IDs) for all teams.
    """
    clean_path = PROCESSED_DIR / "clean_matches.parquet"
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean matches not found at {clean_path}")
        
    df = pd.read_parquet(clean_path).sort_values("date").reset_index(drop=True)
    
    # Store history for rolling calcs
    team_histories = {}
    team_name_to_id = {}
    h2h_stats = {}
    
    for _, row in df.iterrows():
        t_a = row["team_a_id"]
        t_b = row["team_b_id"]
        date = row["date"]
        
        team_name_to_id[row["team_a_name"].lower()] = t_a
        team_name_to_id[row["team_b_name"].lower()] = t_b
        
        if t_a not in team_histories: team_histories[t_a] = []
        if t_b not in team_histories: team_histories[t_b] = []
        
        label = 1 if row["winner_id"] == t_a else 0
        
        h2h_key = tuple(sorted([t_a, t_b]))
        if h2h_key not in h2h_stats:
            h2h_stats[h2h_key] = {h2h_key[0]: 0, h2h_key[1]: 0}
            
        if label == 1:
            team_histories[t_a].append((date, 1))
            team_histories[t_b].append((date, 0))
            h2h_stats[h2h_key][t_a] += 1
        else:
            team_histories[t_a].append((date, 0))
            team_histories[t_b].append((date, 1))
            h2h_stats[h2h_key][t_b] += 1
            
    return team_histories, team_name_to_id, h2h_stats

def predict_matchup(team_a_name: str, team_b_name: str):
    team_a_name = team_a_name.lower()
    team_b_name = team_b_name.lower()
    
    histories, name_map, h2h_stats = load_latest_state()
    
    if team_a_name not in name_map:
        logger.error(f"Team '{team_a_name}' not found in historical data.")
        return
    if team_b_name not in name_map:
        logger.error(f"Team '{team_b_name}' not found in historical data.")
        return
        
    t_a_id = name_map[team_a_name]
    t_b_id = name_map[team_b_name]
    now = pd.to_datetime(datetime.now(timezone.utc))
    
    # Calculate features as of present moment
    stats_a_90d = get_recent_stats(histories[t_a_id], now, 90)
    stats_b_90d = get_recent_stats(histories[t_b_id], now, 90)
    stats_a_30d = get_recent_stats(histories[t_a_id], now, ROLLING_WINDOW_DAYS)
    stats_b_30d = get_recent_stats(histories[t_b_id], now, ROLLING_WINDOW_DAYS)
    
    h2h_key = tuple(sorted([t_a_id, t_b_id]))
    h2h_a_wins = h2h_stats.get(h2h_key, {}).get(t_a_id, 0)
    h2h_b_wins = h2h_stats.get(h2h_key, {}).get(t_b_id, 0)
    
    # Assemble feature vector in precise order from features.py
    # Order: [team_a_matches_90d, team_b_matches_90d, team_a_win_rate_90d, team_b_win_rate_90d, 
    #         team_a_win_rate_30d, team_b_win_rate_30d, h2h_a_wins, h2h_b_wins]
    feature_vector = np.array([[
        stats_a_90d["matches"],
        stats_b_90d["matches"],
        stats_a_90d["win_rate"],
        stats_b_90d["win_rate"],
        stats_a_30d["win_rate"],
        stats_b_30d["win_rate"],
        h2h_a_wins,
        h2h_b_wins
    ]], dtype=np.float32)
    
    # Load scaler and transform
    scaler_path = CHECKPOINT_DIR / "scaler.pkl"
    if not scaler_path.exists():
        logger.error("Scaler not found. Please run training first.")
        return
        
    scaler = joblib.load(scaler_path)
    scaled_features = scaler.transform(feature_vector)
    tensor_features = torch.tensor(scaled_features, dtype=torch.float32)
    
    # Load model and predict
    model_path = CHECKPOINT_DIR / "best_mvp_model.pt"
    model = MatchPredictor(input_dim=tensor_features.shape[1])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        prob_a = model(tensor_features).item()
        
    print("\n" + "="*40)
    print(f" MATCHUP PREDICTION: {team_a_name.upper()} vs {team_b_name.upper()}")
    print("="*40)
    
    print(f"\n[Current Statistics - Last 90 days]")
    print(f"{team_a_name.upper()}: {stats_a_90d['matches']} matches played | {stats_a_90d['win_rate']*100:.1f}% win rate")
    print(f"{team_b_name.upper()}: {stats_b_90d['matches']} matches played | {stats_b_90d['win_rate']*100:.1f}% win rate")
    print(f"H2H Record: {team_a_name.upper()} {h2h_a_wins} - {h2h_b_wins} {team_b_name.upper()}")
    
    print(f"\n[Model Inference]")
    print(f"Probability {team_a_name.upper()} wins: {prob_a * 100:.2f}%")
    print(f"Probability {team_b_name.upper()} wins: {(1 - prob_a) * 100:.2f}%")
    print("="*40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a CS2 match outcome between two teams.")
    parser.add_argument("team_a", help="Name of Team A")
    parser.add_argument("team_b", help="Name of Team B")
    args = parser.parse_args()
    
    predict_matchup(args.team_a, args.team_b)
