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
from processing.features import MODEL_FEATURES
import model.veto_sim as veto_sim

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
    
    outcomes = []
    for item in history:
        if item[0] < cutoff:
            continue
        # Support both (date, win_bool) and (date, score_self, score_opp)
        if len(item) == 3:
            outcomes.append(1 if item[1] > item[2] else 0)
        else:
            outcomes.append(item[1])
            
    matches = len(outcomes)
    win_rate = sum(outcomes) / matches if matches > 0 else 0.5
    return {"matches": matches, "win_rate": win_rate}

def get_dominance_metrics(history, current_date, days):
    """Calculates average win/loss margins over a rolling window."""
    if not history:
        return {"avg_win_margin": 0.0, "avg_loss_margin": 0.0}
        
    cutoff = current_date - pd.Timedelta(days=days)
    # history stores (date, score_self, score_opp)
    recent = [(s_self, s_opp) for d, s_self, s_opp in history if d >= cutoff]
    
    if not recent:
        return {"avg_win_margin": 0.0, "avg_loss_margin": 0.0}
    
    wins = [my - opp for my, opp in recent if my > opp]
    losses = [opp - my for my, opp in recent if opp > my]
    
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    
    return {"avg_win_margin": avg_win, "avg_loss_margin": avg_loss}

def load_latest_state():
    """
    Simulates playing through the entire clean maps history to get the absolute 
    latest state (form, ranks, and win streaks) for all teams.
    """
    clean_path = PROCESSED_DIR / "clean_maps.parquet"
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean maps not found at {clean_path}. Please run clean.py first.")
        
    df = pd.read_parquet(clean_path).sort_values("date").reset_index(drop=True)
    
    team_general_histories = {}
    team_map_histories = {}
    team_first_picks = {}
    team_total_series = {}
    team_latest_ranks = {} # team_id -> {"world": int}
    h2h_stats = {}
    current_streaks = {}   # team_id -> int
    match_picked_seen = {} # match_id -> set
    
    # 1. Pre-calculate streaks by processing matches chronologically
    matches_df = df.groupby("match_id").agg({
        "date": "min",
        "team_a_id": "first",
        "team_b_id": "first",
        "winner_id": lambda x: x.value_counts().index[0]
    }).sort_values("date")
    
    for _, row in matches_df.iterrows():
        t_a = row["team_a_id"]
        t_b = row["team_b_id"]
        winner = row["winner_id"]
        
        if winner == t_a:
            current_streaks[t_a] = min(current_streaks.get(t_a, 0) + 1, 5)
            current_streaks[t_b] = 0
        else:
            current_streaks[t_b] = min(current_streaks.get(t_b, 0) + 1, 5)
            current_streaks[t_a] = 0

    # 2. Process maps for history tracking
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
            
        score_a = row.get("score_a", 0)
        score_b = row.get("score_b", 0)
            
        team_general_histories[t_a].append((date, score_a, score_b))
        team_general_histories[t_b].append((date, score_b, score_a))
        team_map_histories[t_a][map_name].append((date, 1 if label == 1 else 0))
        team_map_histories[t_b][map_name].append((date, 1 if label == 0 else 0))
        
        team_latest_ranks[t_a] = {"world": row["team_a_world_rank"]}
        team_latest_ranks[t_b] = {"world": row["team_b_world_rank"]}
        
        # Track Picks for Comfort
        m_id = row["match_id"]
        if m_id not in match_picked_seen:
            match_picked_seen[m_id] = set()

        if row["team_a_picked"] and t_a not in match_picked_seen[m_id]:
            if t_a not in team_first_picks: team_first_picks[t_a] = {}
            if map_name not in team_first_picks[t_a]: team_first_picks[t_a][map_name] = []
            team_first_picks[t_a][map_name].append(date)
            if t_a not in team_total_series: team_total_series[t_a] = []
            team_total_series[t_a].append(date)
            match_picked_seen[m_id].add(t_a)

        if row["team_b_picked"] and t_b not in match_picked_seen[m_id]:
            if t_b not in team_first_picks: team_first_picks[t_b] = {}
            if map_name not in team_first_picks[t_b]: team_first_picks[t_b][map_name] = []
            team_first_picks[t_b][map_name].append(date)
            if t_b not in team_total_series: team_total_series[t_b] = []
            team_total_series[t_b].append(date)
            match_picked_seen[m_id].add(t_b)
        
        if label == 1:
            h2h_stats[h2h_key][t_a] += 1
        else:
            h2h_stats[h2h_key][t_b] += 1
            
    return team_general_histories, team_map_histories, team_latest_ranks, h2h_stats, current_streaks, team_first_picks, team_total_series

def combine_probs(probs: List[float], bo: int) -> float:
    """
    Combines map-level win probabilities into a series win probability.
    Uses dynamic programming to calculate the chance of reaching the 
    required number of wins (first-to-2 for Bo3, first-to-3 for Bo5).
    """
    if not probs:
        return 0.5
    if bo == 1:
        return probs[0]
    
    # Determine wins needed to clinche the series
    wins_needed = (bo // 2) + 1
    
    # Pad probs if the input list is shorter than the BO format
    # (e.g., simulating a Bo3 with only the first 2 maps known)
    full_probs = probs + [0.5] * (bo - len(probs))
    
    # dp[j] is the probability of winning exactly j maps
    dp = [0.0] * (bo + 1)
    dp[0] = 1.0
    
    for p in full_probs:
        new_dp = [0.0] * (bo + 1)
        for j in range(bo + 1):
            if dp[j] > 0:
                # Scenario: Team A wins this map
                if j + 1 <= bo:
                    new_dp[j+1] += dp[j] * p
                # Scenario: Team A loses this map
                new_dp[j] += dp[j] * (1 - p)
        dp = new_dp
        
    # Series win probability is the sum of winning 'wins_needed' or more maps
    return sum(dp[wins_needed:])

class PredictorContext:
    """
    Holds the loaded state, model, and scaler to allow for efficient batch 
    predictions without reloading data from disk repeatedly.
    """
    def __init__(self):
        self.mappings = load_mappings()
        self.gen_histories, self.map_histories, self.latest_ranks, self.h2h_stats, \
        self.latest_streaks, self.team_fpicks, self.team_tseries = load_latest_state()
        
        scaler_path = CHECKPOINT_DIR / "scaler.pkl"
        model_path = CHECKPOINT_DIR / "best_mvp_model.pt"
        if not scaler_path.exists() or not model_path.exists():
            raise FileNotFoundError("Model or Scaler not found. Ensure training has completed.")
            
        self.scaler = joblib.load(scaler_path)
        self.model = MatchPredictor(self.scaler.n_features_in_)
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
        self.model.eval()

def get_win_probabilities(ctx: PredictorContext, t_a_id: str, t_b_id: str, maps: List[str], picker_override: str = "neutral") -> List[float]:
    """
    Calculates map-level win probabilities for a given matchup and map set.
    """
    now = pd.to_datetime(datetime.now(timezone.utc))
    map_probs = []
    
    # Pre-calculate global 30d stats for both teams (used multiple times)
    g_a_30d = get_recent_stats(ctx.gen_histories.get(t_a_id, []), now, 30)
    g_b_30d = get_recent_stats(ctx.gen_histories.get(t_b_id, []), now, 30)

    for i, m_name in enumerate(maps):
        g_a_7d = get_recent_stats(ctx.gen_histories.get(t_a_id, []), now, 7)
        g_b_7d = get_recent_stats(ctx.gen_histories.get(t_b_id, []), now, 7)
        
        s_a = ctx.latest_streaks.get(t_a_id, 0)
        s_b = ctx.latest_streaks.get(t_b_id, 0)
        
        rank_a = ctx.latest_ranks.get(t_a_id, {"world": DEFAULT_TEAM_RANK})["world"]
        rank_b = ctx.latest_ranks.get(t_b_id, {"world": DEFAULT_TEAM_RANK})["world"]
        log_a = np.log(max(rank_a, 1))
        log_b = np.log(max(rank_b, 1))
        
        rank_diff = log_b - log_a
        avg_log_rank = (log_a + log_b) / 2
        
        wr_30d_diff = g_a_30d["win_rate"] - g_b_30d["win_rate"]
        wr_7d_diff = g_a_7d["win_rate"] - g_b_7d["win_rate"]
        
        h2h_key = tuple(sorted([str(t_a_id), str(t_b_id)]))
        h2h_a = ctx.h2h_stats.get(h2h_key, {}).get(t_a_id, 0)
        h2h_b = ctx.h2h_stats.get(h2h_key, {}).get(t_b_id, 0)
        
        is_a_picker = 0
        is_b_picker = 0
        if picker_override.lower() in ["team_a", "a"]: is_a_picker = 1
        elif picker_override.lower() in ["team_b", "b"]: is_b_picker = 1
        elif len(maps) == 3: # Bo3: A pick, B pick, Decider
            is_a_picker = 1 if i == 0 else 0
            is_b_picker = 1 if i == 1 else 0
        elif len(maps) == 5: # Bo5: A pick, B pick, A pick, B pick, Decider
            is_a_picker = 1 if i in [0, 2] else 0
            is_b_picker = 1 if i in [1, 3] else 0
        elif len(maps) == 1: # Bo1: Neutral
            pass
        else: # Default fallback
            is_a_picker = 1 if i == 0 else 0
            is_b_picker = 1 if i == 1 else 0

        # Map-Specific Win Rate (90d)
        m_a_90d = get_recent_stats(ctx.map_histories.get(t_a_id, {}).get(m_name, []), now, 90)
        m_b_90d = get_recent_stats(ctx.map_histories.get(t_b_id, {}).get(m_name, []), now, 90)
        map_wr_diff = m_a_90d["win_rate"] - m_b_90d["win_rate"]
        
        # Map Comfort (30d)
        def get_comfort(tid, m_n):
            cutoff = now - pd.Timedelta(days=30)
            picks = len([d for d in ctx.team_fpicks.get(tid, {}).get(m_n, []) if d >= cutoff])
            total = len([d for d in ctx.team_tseries.get(tid, []) if d >= cutoff])
            return picks / total if total > 0 else 0.0
            
        map_comfort_diff = get_comfort(t_a_id, m_name) - get_comfort(t_b_id, m_name)
        
        # Dominance & Resilience (30-day window)
        dom_a = get_dominance_metrics(ctx.gen_histories.get(t_a_id, []), now, 30)
        dom_b = get_dominance_metrics(ctx.gen_histories.get(t_b_id, []), now, 30)
        
        dominance_diff = dom_a["avg_win_margin"] - dom_b["avg_win_margin"]
        resilience_diff = dom_b["avg_loss_margin"] - dom_a["avg_loss_margin"]

        feat_vals = {
            "rank_diff": rank_diff,
            "win_rate_30d_diff": wr_30d_diff,
            "win_rate_7d_diff": wr_7d_diff,
            "team_a_win_streak": s_a,
            "team_b_win_streak": s_b,
            "picker_diff": is_a_picker - is_b_picker,
            "h2h_a_wins": h2h_a,
            "h2h_b_wins": h2h_b,
            "map_win_rate_diff": map_wr_diff,
            "map_comfort_diff": map_comfort_diff,
            "dominance_diff": dominance_diff,
            "resilience_diff": resilience_diff,
            "avg_log_rank": avg_log_rank
        }
        
        f_vec = np.array([[feat_vals[col] for col in MODEL_FEATURES]], dtype=np.float32)
        scaled_f = ctx.scaler.transform(f_vec)
        with torch.no_grad():
            p_a = ctx.model(torch.tensor(scaled_f, dtype=torch.float32)).item()
            map_probs.append(p_a)
            
    return map_probs

def calculate_expected_series_win(team_a_raw, team_b_raw, series_format="bo3", threshold=0.90, iters=10000, starts_veto=None, ctx=None):
    """
    Calculates the Expected Series Win Probability by mating the Veto Simulator
    and the Neural Network Predictor using the Law of Total Probability.
    """
    # 1. Initialize Predictor context (expensive state load)
    if ctx is None:
        ctx = PredictorContext()
        
    t_a_id = normalize_name(team_a_raw, ctx.mappings)
    t_b_id = normalize_name(team_b_raw, ctx.mappings)

    # 2. Run Veto simulation to get path probabilities
    # We need the stats format expected by veto_sim.run_simulations
    veto_df = veto_sim.load_data()
    stats_a = veto_sim.get_team_stats(t_a_id, veto_df)
    stats_b = veto_sim.get_team_stats(t_b_id, veto_df)
    
    sequence_counts, _ = veto_sim.run_simulations(
        stats_a, stats_b, iters=iters, series_format=series_format, starts_veto=starts_veto
    )
    
    # 3. Sort paths and truncate based on threshold
    sorted_paths = sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)
    selected_paths = []
    cumulative_count = 0
    
    for path_str, count in sorted_paths:
        selected_paths.append((path_str, count))
        cumulative_count += count
        if (cumulative_count / iters) >= threshold:
            break
            
    # 4. Normalize probabilities and calculate expected win prob
    expected_win_prob = 0.0
    bo = int(series_format.replace("bo", ""))
    
    for path_str, count in selected_paths:
        path_prob_norm = count / cumulative_count # Normalization: P(Path) / Sum(Selected Path Probs)
        maps = path_str.split(",")
        
        # Fetch map-level win probabilities from NN
        # For BO3/BO5, map order in path_str determines the picker
        map_probs = get_win_probabilities(ctx, t_a_id, t_b_id, maps)
        
        # Conditional series win probability: P(Win | Path)
        p_win_given_path = combine_probs(map_probs, bo)
        
        # Law of Total Probability contribution
        expected_win_prob += p_win_given_path * path_prob_norm
        
    return {
        "expected_win_prob": expected_win_prob,
        "sequence_counts": sequence_counts,
        "team_a_id": t_a_id,
        "team_b_id": t_b_id,
        "predictor_ctx": ctx
    }

def predict_matchup(team_raw_a: str, team_raw_b: str, maps: List[str], picker_override: str = "neutral"):
    try:
        ctx = PredictorContext()
    except Exception as e:
        logger.error(f"Error initializing predictor: {e}")
        return

    t_a_id = normalize_name(team_raw_a, ctx.mappings)
    t_b_id = normalize_name(team_raw_b, ctx.mappings)
    
    now = pd.to_datetime(datetime.now(timezone.utc))
    
    # ANSI escape codes for coloring
    RED = "\033[91m"
    RESET = "\033[0m"

    print("\n" + "="*60)
    print(f" PREDICTION: {t_a_id} vs {t_b_id}")
    print("="*60)

    # Global team sample size info
    g_a_30d = get_recent_stats(ctx.gen_histories.get(t_a_id, []), now, 30)
    g_b_30d = get_recent_stats(ctx.gen_histories.get(t_b_id, []), now, 30)
    
    for tid, stats in [(t_a_id, g_a_30d), (t_b_id, g_b_30d)]:
        count = stats["matches"]
        msg = f"{tid}: {count} maps in 30d window"
        if count < 10:
            print(f"{RED}{msg} (LOW SAMPLE SIZE){RESET}")
        else:
            print(msg)
    print("-" * 60)

    map_probs = get_win_probabilities(ctx, t_a_id, t_b_id, maps, picker_override)
    
    for i, m_name in enumerate(maps):
        p_a = map_probs[i]
        
        m_a_30d_matches = get_recent_stats(ctx.map_histories.get(t_a_id, {}).get(m_name, []), now, 30)["matches"]
        m_b_30d_matches = get_recent_stats(ctx.map_histories.get(t_b_id, {}).get(m_name, []), now, 30)["matches"]
        
        cnt_a_txt = f"({m_a_30d_matches} maps)"
        if m_a_30d_matches < 3: cnt_a_txt = f"{RED}{cnt_a_txt}{RESET}"
        
        cnt_b_txt = f"({m_b_30d_matches} maps)"
        if m_b_30d_matches < 3: cnt_b_txt = f"{RED}{cnt_b_txt}{RESET}"
        
        print(f"[{m_name:10}] {t_a_id}: {p_a*100:5.1f}% {cnt_a_txt} | {t_b_id}: {(1-p_a)*100:5.1f}% {cnt_b_txt}")

    if len(maps) > 1:
        bo = len(maps)
        # Handle cases where maps input might not perfectly match bo3/bo5 but we use map count as bo
        series_prob_a = combine_probs(map_probs, bo)
        print("-" * 60)
        print(f"SERIES WIN PROBABILITY: {t_a_id} {series_prob_a*100:.1f}%")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a CS2 match outcome.")
    parser.add_argument("team_a", help="Name of Team A")
    parser.add_argument("team_b", help="Name of Team B")
    parser.add_argument("--maps", help="Comma-separated list of maps", default="Mirage")
    parser.add_argument("--picker", help="Who picked (team_a, team_b, or neutral)", default="neutral")
    args = parser.parse_args()
    
    map_list = [m.strip() for m in args.maps.split(",")]
    predict_matchup(args.team_a, args.team_b, map_list, args.picker)
