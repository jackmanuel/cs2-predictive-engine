import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure project root is in path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, DATA_DIR, MIN_MATCHES_THRESHOLD, ROLLING_WINDOW_DAYS

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

def compute_features(df: pd.DataFrame, historical_stats: pd.DataFrame = None) -> pd.DataFrame:
    """
    Computes map-level temporal features without data leakage.
    Tracks both general team form and map-specific form.
    Integrates historical PandaScore stats and HLTV ranks.
    """
    # Ensure it is sorted by date!
    df = df.sort_values("date").reset_index(drop=True)
    
    features_list = []
    
    # State trackers
    team_general_history = {} # team_id -> [(datetime, int_win)]
    team_map_history = {}     # team_id -> map_name -> [(datetime, int_win)]
    h2h_stats = {}            # (team_x, team_y) (sorted) -> wins_x, wins_y
    
    def init_team(tid):
        if tid not in team_general_history:
            team_general_history[tid] = []
        if tid not in team_map_history:
            team_map_history[tid] = {}

    # Pre-processing historical stats for fast lookup
    hist_lookup = {}
    if historical_stats is not None:
        hist_lookup = historical_stats.set_index('team_name').to_dict('index')

    logger.info("Computing map-level features (including ranks) temporally...")
    
    for _, row in df.iterrows():
        t_a = row["team_a_id"]
        t_b = row["team_b_id"]
        map_name = row["map_name"]
        date = row["date"]
        
        init_team(t_a)
        init_team(t_b)
        
        # 1 means Team A won, 0 means Team B won
        label = 1 if row["winner_id"] == t_a else 0
        
        # Determine H2H prior
        h2h_key = tuple(sorted([str(t_a), str(t_b)]))
        if h2h_key not in h2h_stats:
            h2h_stats[h2h_key] = {h2h_key[0]: 0, h2h_key[1]: 0}
            
        h2h_a_wins = h2h_stats[h2h_key].get(t_a, 0)
        h2h_b_wins = h2h_stats[h2h_key].get(t_b, 0)
        
        # HLTV Recent Form (BEFORE the map start)
        gen_a_90d = get_recent_stats(team_general_history[t_a], date, 90)
        gen_b_90d = get_recent_stats(team_general_history[t_b], date, 90)
        gen_a_30d = get_recent_stats(team_general_history[t_a], date, ROLLING_WINDOW_DAYS)
        gen_b_30d = get_recent_stats(team_general_history[t_b], date, ROLLING_WINDOW_DAYS)
        
        # Map-Specific Form
        hist_a_map = team_map_history[t_a].get(map_name, [])
        hist_b_map = team_map_history[t_b].get(map_name, [])
        map_a_90d = get_recent_stats(hist_a_map, date, 90)
        map_b_90d = get_recent_stats(hist_b_map, date, 90)
        
        # PandaScore Long-Term Baseline (Static)
        pa_stats = hist_lookup.get(t_a, {"pcore_wr": 0.5, "pcore_matches": 0})
        pb_stats = hist_lookup.get(t_b, {"pcore_wr": 0.5, "pcore_matches": 0})
        
        feat_dict = {
            "match_id": row["match_id"],
            "map_name": map_name,
            "date": date,
            "team_a_id": t_a,
            "team_b_id": t_b,
            # NEW: Ranks (Direct from cleaned data)
            "team_a_world_rank": row["team_a_world_rank"],
            "team_b_world_rank": row["team_b_world_rank"],
            "team_a_vrs_rank": row["team_a_vrs_rank"],
            "team_b_vrs_rank": row["team_b_vrs_rank"],
            # PandaScore Tandem Features (Long-period baseline)
            "team_a_pcore_wr": pa_stats["pcore_wr"],
            "team_b_pcore_wr": pb_stats["pcore_wr"],
            "team_a_pcore_matches": pa_stats["pcore_matches"],
            "team_b_pcore_matches": pb_stats["pcore_matches"],
            # HLTV-based Recent Features
            "team_a_gen_matches_90d": gen_a_90d["matches"],
            "team_b_gen_matches_90d": gen_b_90d["matches"],
            "team_a_gen_wr_90d": gen_a_90d["win_rate"],
            "team_b_gen_wr_90d": gen_b_90d["win_rate"],
            "team_a_gen_wr_30d": gen_a_30d["win_rate"],
            "team_b_gen_wr_30d": gen_b_30d["win_rate"],
            "team_a_map_matches_90d": map_a_90d["matches"],
            "team_b_map_matches_90d": map_b_90d["matches"],
            "team_a_map_wr_90d": map_a_90d["win_rate"],
            "team_b_map_wr_90d": map_b_90d["win_rate"],
            # Picker Feature
            "team_a_is_picker": 1 if row["team_a_picked"] else 0,
            "team_b_is_picker": 1 if row["team_b_picked"] else 0,
            "h2h_a_wins": h2h_a_wins,
            "h2h_b_wins": h2h_b_wins,
            "match_format": row.get("match_format", "unknown"),
            "score_a": row.get("score_a", 0),
            "score_b": row.get("score_b", 0),
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
            
    return pd.DataFrame(features_list)

def feature_pipeline():
    clean_path = PROCESSED_DIR / "clean_maps.parquet"
    if not clean_path.exists():
        logger.error(f"Could not find {clean_path}. Please run clean.py first.")
        return
        
    df = pd.read_parquet(clean_path)
    
    # Load historical baseline
    hist_path = PROCESSED_DIR / "historical_pandascore_stats.parquet"
    hist_df = pd.read_parquet(hist_path) if hist_path.exists() else None
    
    feature_df = compute_features(df, historical_stats=hist_df)
    
    # Filtering (Reduced threshold for small testing datasets)
    initial_len = len(feature_df)
    effective_threshold = min(MIN_MATCHES_THRESHOLD, 1) 
    
    feature_df = feature_df[
        (feature_df["team_a_gen_matches_90d"] >= effective_threshold) & 
        (feature_df["team_b_gen_matches_90d"] >= effective_threshold)
    ]
    
    out_path = PROCESSED_DIR / "features.parquet"
    feature_df.to_parquet(out_path, index=False)
    logger.info(f"Map-level features (including Ranks) saved to {out_path} ({len(feature_df)} rows).")

if __name__ == "__main__":
    feature_pipeline()
