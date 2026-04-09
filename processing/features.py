import logging
import pandas as pd
import numpy as np
from pathlib import Path

from config import PROCESSED_DIR, MIN_MATCHES_THRESHOLD, ROLLING_WINDOW_DAYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_recent_stats(history, current_date, days):
    """Helper to calculate win rate and match count over a rolling window."""
    if not history:
        return {"matches": 0, "win_rate": 0.5}
    cutoff = current_date - pd.Timedelta(days=days)
    recent = [w for d, w in history if d >= cutoff]
    matches = len(recent)
    win_rate = sum(recent) / matches if matches > 0 else 0.5
    return {"matches": matches, "win_rate": win_rate}

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes map-level temporal features without data leakage.
    Tracks both general team form and map-specific form.
    """
    # Important: Ensure it is sorted by date!
    df = df.sort_values("date").reset_index(drop=True)
    
    features_list = []
    
    # State trackers
    # team_id -> [(datetime, int_win)] (general)
    team_general_history = {} 
    
    # team_id -> map_name -> [(datetime, int_win)]
    team_map_history = {}
    
    # (team_x, team_y) (sorted) -> x_wins, y_wins
    h2h_stats = {}
    
    def init_team(tid):
        if tid not in team_general_history:
            team_general_history[tid] = []
        if tid not in team_map_history:
            team_map_history[tid] = {}

    logger.info("Computing map-level features temporally...")
    
    for _, row in df.iterrows():
        t_a = row["team_a_id"]
        t_b = row["team_b_id"]
        map_name = row["map_name"]
        date = row["date"]
        
        init_team(t_a)
        init_team(t_b)
        
        # 1 means Team A won, 0 means Team B won
        label = 1 if row["winner_id"] == t_a else 0
        
        # Determine H2H prior (general H2H across all maps)
        h2h_key = tuple(sorted([t_a, t_b]))
        if h2h_key not in h2h_stats:
            h2h_stats[h2h_key] = {h2h_key[0]: 0, h2h_key[1]: 0}
            
        h2h_a_wins = h2h_stats[h2h_key][t_a]
        h2h_b_wins = h2h_stats[h2h_key][t_b]
        
        # Calculate FEATURES BEFORE THE MAP START
        
        # General Form (all maps)
        gen_a_90d = get_recent_stats(team_general_history[t_a], date, 90)
        gen_b_90d = get_recent_stats(team_general_history[t_b], date, 90)
        gen_a_30d = get_recent_stats(team_general_history[t_a], date, ROLLING_WINDOW_DAYS)
        gen_b_30d = get_recent_stats(team_general_history[t_b], date, ROLLING_WINDOW_DAYS)
        
        # Map-Specific Form
        hist_a_map = team_map_history[t_a].get(map_name, [])
        hist_b_map = team_map_history[t_b].get(map_name, [])
        
        map_a_90d = get_recent_stats(hist_a_map, date, 90)
        map_b_90d = get_recent_stats(hist_b_map, date, 90)
        
        feat_dict = {
            "match_id": row["match_id"],
            "map_name": map_name,
            "date": date,
            "team_a_id": t_a,
            "team_b_id": t_b,
            # General Features
            "team_a_gen_matches_90d": gen_a_90d["matches"],
            "team_b_gen_matches_90d": gen_b_90d["matches"],
            "team_a_gen_wr_90d": gen_a_90d["win_rate"],
            "team_b_gen_wr_90d": gen_b_90d["win_rate"],
            "team_a_gen_wr_30d": gen_a_30d["win_rate"],
            "team_b_gen_wr_30d": gen_b_30d["win_rate"],
            # Map-Specific Features
            "team_a_map_matches_90d": map_a_90d["matches"],
            "team_b_map_matches_90d": map_b_90d["matches"],
            "team_a_map_wr_90d": map_a_90d["win_rate"],
            "team_b_map_wr_90d": map_b_90d["win_rate"],
            # Picker Feature
            "team_a_is_picker": 1 if row["team_a_picked"] else 0,
            "team_b_is_picker": 1 if row["team_b_picked"] else 0,
            # H2H (General)
            "h2h_a_wins": h2h_a_wins,
            "h2h_b_wins": h2h_b_wins,
            "label": label
        }
        features_list.append(feat_dict)
        
        # UPDATE STATES AFTER THE MAP
        team_general_history[t_a].append((date, 1 if label == 1 else 0))
        team_general_history[t_b].append((date, 1 if label == 0 else 0))
        
        if map_name not in team_map_history[t_a]: team_map_history[t_a][map_name] = []
        if map_name not in team_map_history[t_b]: team_map_history[t_b][map_name] = []
        
        team_map_history[t_a][map_name].append((date, 1 if label == 1 else 0))
        team_map_history[t_b][map_name].append((date, 1 if label == 0 else 0))
        
        if label == 1:
            h2h_stats[h2h_key][t_a] += 1
        else:
            h2h_stats[h2h_key][t_b] += 1
            
    feature_df = pd.DataFrame(features_list)
    
    # Filtering: We still want pairs that have some history to avoid noise
    # We'll filter on general matches Played as map-specific history can be sparse
    initial_len = len(feature_df)
    feature_df = feature_df[
        (feature_df["team_a_gen_matches_90d"] >= MIN_MATCHES_THRESHOLD) & 
        (feature_df["team_b_gen_matches_90d"] >= MIN_MATCHES_THRESHOLD)
    ]
    
    logger.info(f"Filtered out {initial_len - len(feature_df)} maps due to low team experience.")
    
    return feature_df

def feature_pipeline():
    clean_path = PROCESSED_DIR / "clean_maps.parquet"
    if not clean_path.exists():
        logger.error(f"Could not find {clean_path}. Please run clean.py first.")
        return
        
    df = pd.read_parquet(clean_path)
    feature_df = compute_features(df)
    
    out_path = PROCESSED_DIR / "features.parquet"
    feature_df.to_parquet(out_path, index=False)
    logger.info(f"Map-level features saved to {out_path} with {len(feature_df)} rows.")

if __name__ == "__main__":
    feature_pipeline()
