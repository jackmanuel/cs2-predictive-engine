import os
import sys
import argparse
import logging
import torch
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Tuple

# Ensure project root is in path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, DATA_DIR, CHECKPOINT_DIR, ROLLING_WINDOW_DAYS, DEFAULT_TEAM_RANK
from model.net import MatchPredictor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MAPPING_FILE = DATA_DIR / "team_mappings.json"

def load_mappings() -> dict:
    if MAPPING_FILE.exists():
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def normalize_name(name: str, mappings: dict) -> str:
    if not name: return ""
    name_strip = name.strip()
    if name_strip in mappings:
        return mappings[name_strip].upper().strip()
    return name_strip.upper()

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
    latest state (general and map histories, plus latest ranks) for all teams.
    """
    clean_path = PROCESSED_DIR / "clean_maps.parquet"
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean maps not found at {clean_path}. Please run clean.py first.")
        
    df = pd.read_parquet(clean_path).sort_values("date").reset_index(drop=True)
    
    team_general_histories = {}
    team_map_histories = {}
    team_latest_ranks = {} # team_id -> {"world": int, "vrs": int}
    h2h_stats = {}
    
    for _, row in df.iterrows():
        t_a = row["team_a_id"]
        t_b = row["team_b_id"]
        map_name = row["map_name"]
        date = row["date"]
        
        if t_a not in team_general_histories: team_general_histories[t_a] = []
        if t_b not in team_general_histories: team_general_histories[t_b] = []
        
        if t_a not in team_map_histories: team_map_histories[t_a] = {}
        if t_b not in team_map_histories: team_map_histories[t_b] = {}
        
        if map_name not in team_map_histories[t_a]: team_map_histories[t_a][map_name] = []
        if map_name not in team_map_histories[t_b]: team_map_histories[t_b][map_name] = []
        
        label = 1 if row["winner_id"] == t_a else 0
        
        h2h_key = tuple(sorted([str(t_a), str(t_b)]))
        if h2h_key not in h2h_stats:
            h2h_stats[h2h_key] = {h2h_key[0]: 0, h2h_key[1]: 0}
            
        team_general_histories[t_a].append((date, 1 if label == 1 else 0))
        team_general_histories[t_b].append((date, 1 if label == 0 else 0))
        team_map_histories[t_a][map_name].append((date, 1 if label == 1 else 0))
        team_map_histories[t_b][map_name].append((date, 1 if label == 0 else 0))
        
        team_latest_ranks[t_a] = {"world": row["team_a_world_rank"], "vrs": row["team_a_vrs_rank"]}
        team_latest_ranks[t_b] = {"world": row["team_b_world_rank"], "vrs": row["team_b_vrs_rank"]}
        
        if label == 1:
            h2h_stats[h2h_key][t_a] += 1
        else:
            h2h_stats[h2h_key][t_b] += 1
            
    return team_general_histories, team_map_histories, team_latest_ranks, h2h_stats

def combine_probs(probs: List[float], bo: int) -> float:
    """Combines map probs into series win prob."""
    if bo == 1: return probs[0]
    if bo == 3:
        p1 = probs[0]
        p2 = probs[1]
        p3 = probs[2] if len(probs) > 2 else 0.5
        return (p1 * p2) + (p1 * (1-p2) * p3) + ((1-p1) * p2 * p3)
    return probs[0]

def predict_matchup(team_a_raw: str, team_b_raw: str, maps: List[str], picker_override: str = "neutral"):
    mappings = load_mappings()
    team_a_norm = normalize_name(team_a_raw, mappings)
    team_b_norm = normalize_name(team_b_raw, mappings)
    
    t_a_id = team_a_norm
    t_b_id = team_b_norm
    
    gen_histories, map_histories, latest_ranks, h2h_stats = load_latest_state()
    
    # Load historical baseline
    hist_path = PROCESSED_DIR / "historical_pandascore_stats.parquet"
    hist_lookup = {}
    if hist_path.exists():
        hist_df = pd.read_parquet(hist_path)
        hist_lookup = hist_df.set_index('team_name').to_dict('index')
    
    # Load model and scaler
    scaler_path = CHECKPOINT_DIR / "scaler.pkl"
    model_path = CHECKPOINT_DIR / "best_mvp_model.pt"
    if not scaler_path.exists() or not model_path.exists():
        logger.error("Model or Scaler not found. Ensure training has completed.")
        return
        
    scaler = joblib.load(scaler_path)
    model = MatchPredictor(scaler.n_features_in_)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()

    now = pd.to_datetime(datetime.now(timezone.utc))
    
    print("\n" + "="*60)
    print(f" PREDICTION: {team_a_norm} vs {team_b_norm}")
    print("="*60)

    map_probs = []
    
    for i, m_name in enumerate(maps):
        # Calculate features (MATCHES features.py exactly)
        g_a_90d = get_recent_stats(gen_histories.get(t_a_id, []), now, 90)
        g_b_90d = get_recent_stats(gen_histories.get(t_b_id, []), now, 90)
        g_a_30d = get_recent_stats(gen_histories.get(t_a_id, []), now, ROLLING_WINDOW_DAYS)
        g_b_30d = get_recent_stats(gen_histories.get(t_b_id, []), now, ROLLING_WINDOW_DAYS)
        
        m_a_90d = get_recent_stats(map_histories.get(t_a_id, {}).get(m_name, []), now, 90)
        m_b_90d = get_recent_stats(map_histories.get(t_b_id, {}).get(m_name, []), now, 90)
        
        hist_a = hist_lookup.get(t_a_id, {"pcore_wr": 0.5, "pcore_matches": 0})
        hist_b = hist_lookup.get(t_b_id, {"pcore_wr": 0.5, "pcore_matches": 0})
        
        rank_a = latest_ranks.get(t_a_id, {"world": DEFAULT_TEAM_RANK, "vrs": DEFAULT_TEAM_RANK})
        rank_b = latest_ranks.get(t_b_id, {"world": DEFAULT_TEAM_RANK, "vrs": DEFAULT_TEAM_RANK})
        
        h2h_key = tuple(sorted([str(t_a_id), str(t_b_id)]))
        h2h_a = h2h_stats.get(h2h_key, {}).get(t_a_id, 0)
        h2h_b = h2h_stats.get(h2h_key, {}).get(t_b_id, 0)
        
        is_a_picker = 0
        is_b_picker = 0
        if picker_override.lower() in ["team_a", "a"]: is_a_picker = 1
        elif picker_override.lower() in ["team_b", "b"]: is_b_picker = 1
        elif len(maps) > 1:
            is_a_picker = 1 if i == 0 else 0
            is_b_picker = 1 if i == 1 else 0

        # Construct feature vector (Order MUST match features.py compute_features)
        f_vec = np.array([[
            rank_a["world"], rank_b["world"],
            rank_a["vrs"], rank_b["vrs"],
            hist_a["pcore_wr"], hist_b["pcore_wr"],
            hist_a["pcore_matches"], hist_b["pcore_matches"],
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
        
        print(f"[{m_name:10}] {team_a_norm}: {p_a*100:5.1f}% | {team_b_norm}: {(1-p_a)*100:5.1f}%")

    if len(maps) > 1:
        series_prob_a = combine_probs(map_probs, len(maps))
        print("-" * 60)
        print(f"SERIES WIN PROBABILITY: {team_a_norm} {series_prob_a*100:.1f}%")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a CS2 match outcome.")
    parser.add_argument("team_a", help="Name of Team A")
    parser.add_argument("team_b", help="Name of Team B")
    parser.add_argument("--maps", help="Comma-separated list of maps (e.g. Mirage,Ancient,Dust2)", default="Mirage")
    parser.add_argument("--picker", help="Who picked the map (team_a, team_b, or neutral)", default="neutral")
    args = parser.parse_args()
    
    map_list = [m.strip() for m in args.maps.split(",")]
    predict_matchup(args.team_a, args.team_b, map_list, args.picker)
