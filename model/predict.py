import argparse
import logging
import torch
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Tuple

from config import PROCESSED_DIR, CHECKPOINT_DIR, ROLLING_WINDOW_DAYS
from model.net import MatchPredictor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def get_recent_stats(history, current_date, days):
    """Helper to calculate rolling stats from a history list."""
    if not history:
        return {"matches": 0, "win_rate": 0.5}
    cutoff = current_date - pd.Timedelta(days=days)
    recent = [w for d, w in history if d >= cutoff]
    matches = len(recent)
    win_rate = sum(recent) / matches if matches > 0 else 0.5
    return {"matches": matches, "win_rate": win_rate}

def load_latest_state():
    """
    Simulates playing through the entire clean maps history to get the absolute 
    latest state (general and map histories) for all teams.
    """
    clean_path = PROCESSED_DIR / "clean_maps.parquet"
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean maps not found at {clean_path}")
        
    df = pd.read_parquet(clean_path).sort_values("date").reset_index(drop=True)
    
    team_general_histories = {}
    team_map_histories = {}
    team_name_to_id = {}
    h2h_stats = {}
    
    for _, row in df.iterrows():
        t_a = row["team_a_id"]
        t_b = row["team_b_id"]
        map_name = row["map_name"]
        date = row["date"]
        
        team_name_to_id[row["team_a_name"].lower()] = t_a
        team_name_to_id[row["team_b_name"].lower()] = t_b
        
        if t_a not in team_general_histories: team_general_histories[t_a] = []
        if t_b not in team_general_histories: team_general_histories[t_b] = []
        
        if t_a not in team_map_histories: team_map_histories[t_a] = {}
        if t_b not in team_map_histories: team_map_histories[t_b] = {}
        
        if map_name not in team_map_histories[t_a]: team_map_histories[t_a][map_name] = []
        if map_name not in team_map_histories[t_b]: team_map_histories[t_b][map_name] = []
        
        label = 1 if row["winner_id"] == t_a else 0
        
        h2h_key = tuple(sorted([t_a, t_b]))
        if h2h_key not in h2h_stats:
            h2h_stats[h2h_key] = {h2h_key[0]: 0, h2h_key[1]: 0}
            
        team_general_histories[t_a].append((date, 1 if label == 1 else 0))
        team_general_histories[t_b].append((date, 1 if label == 0 else 0))
        team_map_histories[t_a][map_name].append((date, 1 if label == 1 else 0))
        team_map_histories[t_b][map_name].append((date, 1 if label == 0 else 0))
        
        if label == 1:
            h2h_stats[h2h_key][t_a] += 1
        else:
            h2h_stats[h2h_key][t_b] += 1
            
    return team_general_histories, team_map_histories, team_name_to_id, h2h_stats

def combine_probs(probs: List[float], bo: int) -> float:
    """
    Combines independent map probabilities into a series win probability for Team A.
    Uses basic probability rules for Bo1, Bo3, Bo5.
    """
    if bo == 1:
        return probs[0]
    
    if bo == 3:
        # A wins if they win: (M1 and M2) OR (M1 and not M2 and M3) OR (not M1 and M2 and M3)
        p1, p2, p3 = probs[0], probs[1], probs[2]
        return (p1 * p2) + (p1 * (1-p2) * p3) + ((1-p1) * p2 * p3)
        
    if bo == 5:
        # For Bo5, we can use a small binomial expansion or just calculate the paths.
        # Paths for Team A winning at least 3 maps:
        # 3-0: p1*p2*p3
        # 3-1: (p1*p2*(1-p3)*p4) + (p1*(1-p2)*p3*p4) + ((1-p1)*p2*p3*p4)
        # 3-2: ... and so on. 
        # Easier to simulate or use recursive pathing for Bo5.
        p = probs
        win_prob = 0
        # Recursive helper for paths
        def count_paths(idx, a_wins, b_wins, current_p):
            nonlocal win_prob
            if a_wins == 3:
                win_prob += current_p
                return
            if b_wins == 3 or idx == 5:
                return
            # A wins map idx
            count_paths(idx + 1, a_wins + 1, b_wins, current_p * p[idx])
            # B wins map idx
            count_paths(idx + 1, a_wins, b_wins + 1, current_p * (1 - p[idx]))
            
        count_paths(0, 0, 0, 1.0)
        return win_prob

    return probs[0]

def predict_matchup(team_a_name: str, team_b_name: str, maps: List[str]):
    team_a_name = team_a_name.lower()
    team_b_name = team_b_name.lower()
    
    gen_histories, map_histories, name_map, h2h_stats = load_latest_state()
    
    if team_a_name not in name_map:
        logger.error(f"Team '{team_a_name}' not found in historical data.")
        return
    if team_b_name not in name_map:
        logger.error(f"Team '{team_b_name}' not found in historical data.")
        return
        
    t_a_id = name_map[team_a_name]
    t_b_id = name_map[team_b_name]
    now = pd.to_datetime(datetime.now(timezone.utc))
    
    # Load model and scaler once
    scaler_path = CHECKPOINT_DIR / "scaler.pkl"
    model_path = CHECKPOINT_DIR / "best_mvp_model.pt"
    if not scaler_path.exists() or not model_path.exists():
        logger.error("Model or Scaler not found. Please run training first.")
        return
        
    scaler = joblib.load(scaler_path)
    # Get standard input dim from scaler
    input_dim = scaler.n_features_in_
    model = MatchPredictor(input_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    print("\n" + "="*60)
    print(f" SERIES PREDICTION: {team_a_name.upper()} vs {team_b_name.upper()}")
    print("="*60)

    map_probs = []
    
    for i, m_name in enumerate(maps):
        # Calculate features for this specific map
        g_a_90d = get_recent_stats(gen_histories[t_a_id], now, 90)
        g_b_90d = get_recent_stats(gen_histories[t_b_id], now, 90)
        g_a_30d = get_recent_stats(gen_histories[t_a_id], now, ROLLING_WINDOW_DAYS)
        g_b_30d = get_recent_stats(gen_histories[t_b_id], now, ROLLING_WINDOW_DAYS)
        
        m_a_90d = get_recent_stats(map_histories[t_a_id].get(m_name, []), now, 90)
        m_b_90d = get_recent_stats(map_histories[t_b_id].get(m_name, []), now, 90)
        
        h2h_key = tuple(sorted([t_a_id, t_b_id]))
        h2h_a = h2h_stats.get(h2h_key, {}).get(t_a_id, 0)
        h2h_b = h2h_stats.get(h2h_key, {}).get(t_b_id, 0)
        
        # We don't know who picked yet if it's a prediction, 
        # so we assume 0 for both unless specified? 
        # Or better: let's assume map 1 is A's pick, map 2 is B's pick, map 3 is neutral.
        is_a_picker = 1 if i == 0 and len(maps) > 1 else 0
        is_b_picker = 1 if i == 1 and len(maps) > 1 else 0

        # Assemble feature vector (order MUST match features.py compute_features)
        f_vec = np.array([[
            g_a_90d["matches"], g_b_90d["matches"], 
            g_a_90d["win_rate"], g_b_90d["win_rate"],
            g_a_30d["win_rate"], g_b_30d["win_rate"],
            m_a_90d["matches"], m_b_90d["matches"],
            m_a_90d["win_rate"], m_b_90d["win_rate"],
            is_a_picker, is_b_picker,
            h2h_a, h2h_b
        ]], dtype=np.float32)
        
        scaled_f = scaler.transform(f_vec)
        with torch.no_grad():
            p_a = model(torch.tensor(scaled_f, dtype=torch.float32)).item()
            map_probs.append(p_a)
        
        print(f"Map {i+1} ({m_name:10}): {team_a_name.upper()} {p_a*100:5.1f}% | {team_b_name.upper()} {(1-p_a)*100:5.1f}%")

    series_prob_a = combine_probs(map_probs, len(maps))
    
    print("-" * 60)
    print(f"OVERALL {team_a_name.upper()} WIN PROBABILITY: {series_prob_a*100:.2f}%")
    print(f"OVERALL {team_b_name.upper()} WIN PROBABILITY: {(1-series_prob_a)*100:.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a CS2 match outcome.")
    parser.add_argument("team_a", help="Name of Team A")
    parser.add_argument("team_b", help="Name of Team B")
    parser.add_argument("--maps", help="Comma-separated list of maps (e.g. Mirage,Ancient,Dust2)", default="Mirage")
    args = parser.parse_args()
    
    map_list = [m.strip() for m in args.maps.split(",")]
    predict_matchup(args.team_a, args.team_b, map_list)
