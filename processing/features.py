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

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes map-level temporal features with dimensionality reduction.
    Uses differentials and momentum (streaks) instead of absolute values.
    """
    # Ensure it is sorted by date!
    df = df.sort_values("date").reset_index(drop=True)
    
    # 1. Pre-calculate match-level streaks to avoid data leakage
    # A team's streak for a map is their win streak BEFORE that match started.
    matches_df = df.groupby("match_id").agg({
        "date": "min",
        "team_a_id": "first",
        "team_b_id": "first",
        "winner_id": lambda x: x.value_counts().index[0] # Series winner
    }).sort_values("date")
    
    streaks_before_match = {} # match_id -> {team_id: streak}
    current_streaks = {}      # team_id -> streak
    
    for m_id, row in matches_df.iterrows():
        t_a = row["team_a_id"]
        t_b = row["team_b_id"]
        winner = row["winner_id"]
        
        # Store what streaks were BEFORE this match
        streaks_before_match[m_id] = {
            t_a: current_streaks.get(t_a, 0),
            t_b: current_streaks.get(t_b, 0)
        }
        
        # Update current streaks (capped at 5)
        if winner == t_a:
            current_streaks[t_a] = min(current_streaks.get(t_a, 0) + 1, 5)
            current_streaks[t_b] = 0
        else:
            current_streaks[t_b] = min(current_streaks.get(t_b, 0) + 1, 5)
            current_streaks[t_a] = 0
            
    features_list = []
    
    # State trackers for form
    team_general_history = {} # team_id -> [(datetime, int_win)]
    h2h_stats = {}            # (team_x, team_y) (sorted) -> wins_x, wins_y
    
    def init_team(tid):
        if tid not in team_general_history:
            team_general_history[tid] = []

    logger.info("Computing reduced map-level features temporally...")
    
    for _, row in df.iterrows():
        t_a = row["team_a_id"]
        t_b = row["team_b_id"]
        m_id = row["match_id"]
        date = row["date"]
        
        init_team(t_a)
        init_team(t_b)
        
        # Determine H2H prior
        h2h_key = tuple(sorted([str(t_a), str(t_b)]))
        if h2h_key not in h2h_stats:
            h2h_stats[h2h_key] = {h2h_key[0]: 0, h2h_key[1]: 0}
            
        h2h_a_wins = h2h_stats[h2h_key].get(t_a, 0)
        h2h_b_wins = h2h_stats[h2h_key].get(t_b, 0)
        
        # HLTV Recent Form (BEFORE the map start)
        gen_a_30d = get_recent_stats(team_general_history[t_a], date, 30)
        gen_b_30d = get_recent_stats(team_general_history[t_b], date, 30)
        gen_a_7d = get_recent_stats(team_general_history[t_a], date, 7)
        gen_b_7d = get_recent_stats(team_general_history[t_b], date, 7)
        
        # Streaks (from pre-calc)
        s_a = streaks_before_match[m_id].get(t_a, 0)
        s_b = streaks_before_match[m_id].get(t_b, 0)
        
        # Differentials
        # Rank Diff: log(rank_b) - log(rank_a)
        # Positive value means Team A is favoured (higher-ranked/lower rank number)
        r_a = max(row["team_a_world_rank"], 1)
        r_b = max(row["team_b_world_rank"], 1)
        rank_diff = np.log(r_b) - np.log(r_a)
        
        wr_30d_diff = gen_a_30d["win_rate"] - gen_b_30d["win_rate"]
        wr_7d_diff = gen_a_7d["win_rate"] - gen_b_7d["win_rate"]
        
        feat_dict = {
            "match_id": m_id,
            "map_name": row["map_name"],
            "date": date,
            "team_a_id": t_a,
            "team_b_id": t_b,
            
            # Form Differentials
            "rank_diff": rank_diff,
            "win_rate_30d_diff": wr_30d_diff,
            "win_rate_7d_diff": wr_7d_diff,
            
            # Momentum Features (Capped at 5)
            "team_a_win_streak": s_a,
            "team_b_win_streak": s_b,
            
            # Orthogonal Features (Retained as they are)
            "team_a_is_picker": 1 if row["team_a_picked"] else 0,
            "team_b_is_picker": 1 if row["team_b_picked"] else 0,
            "h2h_a_wins": h2h_a_wins,
            "h2h_b_wins": h2h_b_wins,
            
            # Labels and metadata
            "match_format": row.get("match_format", "unknown"),
            "score_a": row.get("score_a", 0),
            "score_b": row.get("score_b", 0),
            "label": 1 if row["winner_id"] == t_a else 0,
            
            # Carry over match counts for filtering later (not used as features)
            "team_a_gen_matches_30d": gen_a_30d["matches"],
            "team_b_gen_matches_30d": gen_b_30d["matches"]
        }
        features_list.append(feat_dict)
        
        # UPDATE STATES AFTER THE MAP
        team_general_history[t_a].append((date, 1 if row["winner_id"] == t_a else 0))
        team_general_history[t_b].append((date, 1 if row["winner_id"] == t_b else 0))
        
        if row["winner_id"] == t_a:
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
    
    # Compute features with reduced dimensionality
    feature_df = compute_features(df)
    
    # Filtering (using 30d window instead of 90d as requested)
    initial_len = len(feature_df)
    effective_threshold = min(MIN_MATCHES_THRESHOLD, 1) 
    
    feature_df = feature_df[
        (feature_df["team_a_gen_matches_30d"] >= effective_threshold) & 
        (feature_df["team_b_gen_matches_30d"] >= effective_threshold)
    ]
    
    out_path = PROCESSED_DIR / "features.parquet"
    feature_df.to_parquet(out_path, index=False)
    logger.info(f"Reduced features saved to {out_path} ({len(feature_df)} rows).")

if __name__ == "__main__":
    feature_pipeline()
