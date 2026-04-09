import logging
import pandas as pd
import numpy as np
from pathlib import Path

from config import PROCESSED_DIR, MIN_MATCHES_THRESHOLD, ROLLING_WINDOW_DAYS

logger = logging.getLogger(__name__)

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes temporal features without data leakage.
    Simulates a walk-forward process row by row.
    """
    # Important: Ensure it is sorted by date!
    df = df.sort_values("date").reset_index(drop=True)
    
    features_list = []
    
    # State trackers
    # team_id -> {"wins": int, "matches": int, "history": [(datetime, int_win)]}
    team_stats = {} 
    
    # (team_x, team_y) (sorted) -> x_wins, y_wins
    h2h_stats = {}
    
    def get_team_state(idx):
        if idx not in team_stats:
            team_stats[idx] = {"wins": 0, "matches": 0, "history": []}
        return team_stats[idx]
        
    def get_recent_stats(history, current_date, days):
        if not history:
            return {"matches": 0, "win_rate": 0.5}
        cutoff = current_date - pd.Timedelta(days=days)
        recent = [w for d, w in history if d >= cutoff]
        matches = len(recent)
        win_rate = sum(recent) / matches if matches > 0 else 0.5
        return {"matches": matches, "win_rate": win_rate}

    logger.info("Computing features temporally...")
    
    for _, row in df.iterrows():
        t_a = row["team_a_id"]
        t_b = row["team_b_id"]
        date = row["date"]
        
        # Determine winner
        # 1 means Team A won, 0 means Team B won
        label = 1 if row["winner_id"] == t_a else 0
        
        # Get prior state
        state_a = get_team_state(t_a)
        state_b = get_team_state(t_b)
        
        # H2H prior
        h2h_key = tuple(sorted([t_a, t_b]))
        if h2h_key not in h2h_stats:
            h2h_stats[h2h_key] = {h2h_key[0]: 0, h2h_key[1]: 0}
            
        h2h_a_wins = h2h_stats[h2h_key][t_a]
        h2h_b_wins = h2h_stats[h2h_key][t_b]
        
        stats_a_90d = get_recent_stats(state_a["history"], date, 90)
        stats_b_90d = get_recent_stats(state_b["history"], date, 90)
        
        stats_a_30d = get_recent_stats(state_a["history"], date, ROLLING_WINDOW_DAYS)
        stats_b_30d = get_recent_stats(state_b["history"], date, ROLLING_WINDOW_DAYS)
        
        # Calculate features BEFORE the match
        feat_dict = {
            "match_id": row["match_id"],
            "date": date,
            "team_a_id": t_a,
            "team_b_id": t_b,
            "team_a_matches_90d": stats_a_90d["matches"],
            "team_b_matches_90d": stats_b_90d["matches"],
            "team_a_win_rate_90d": stats_a_90d["win_rate"],
            "team_b_win_rate_90d": stats_b_90d["win_rate"],
            "team_a_win_rate_30d": stats_a_30d["win_rate"],
            "team_b_win_rate_30d": stats_b_30d["win_rate"],
            "h2h_a_wins": h2h_a_wins,
            "h2h_b_wins": h2h_b_wins,
            "label": label
        }
        features_list.append(feat_dict)
        
        # Update states AFTER the match
        state_a["matches"] += 1
        state_b["matches"] += 1
        
        if label == 1:
            state_a["wins"] += 1
            state_a["history"].append((date, 1))
            state_b["history"].append((date, 0))
            h2h_stats[h2h_key][t_a] += 1
        else:
            state_b["wins"] += 1
            state_a["history"].append((date, 0))
            state_b["history"].append((date, 1))
            h2h_stats[h2h_key][t_b] += 1
            
    feature_df = pd.DataFrame(features_list)
    
    # Filter out matches where teams haven't played enough history
    # This prevents the initial noisy model predictions
    initial_len = len(feature_df)
    feature_df = feature_df[
        (feature_df["team_a_matches_90d"] >= MIN_MATCHES_THRESHOLD) & 
        (feature_df["team_b_matches_90d"] >= MIN_MATCHES_THRESHOLD)
    ]
    
    logger.info(f"Filtered out {initial_len - len(feature_df)} matches due to low experience.")
    
    return feature_df

def feature_pipeline():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    clean_path = PROCESSED_DIR / "clean_matches.parquet"
    if not clean_path.exists():
        logger.error(f"Could not find {clean_path}. Please run clean.py first.")
        return
        
    df = pd.read_parquet(clean_path)
    feature_df = compute_features(df)
    
    out_path = PROCESSED_DIR / "features.parquet"
    feature_df.to_parquet(out_path, index=False)
    logger.info(f"Features saved to {out_path} with {len(feature_df)} rows.")

if __name__ == "__main__":
    feature_pipeline()
