import os
import sys
import argparse
import logging
import torch
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Tuple

# Ensure project root is in path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, CHECKPOINT_DIR, DEFAULT_TEAM_RANK, FORM_WINDOW_DAYS, FORM_WINDOW_DAYS_SHORT, FORM_WINDOW_DAYS_LONG, MAP_WINDOW_DAYS, DEFAULT_SOS_RANK, MC_ITERATIONS, MC_THRESHOLD
from model.net import MatchPredictor
from processing.features import MODEL_FEATURES, get_sos, get_recent_stats, get_dominance_metrics
import model.veto_sim as veto_sim

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def normalize_name(name: str) -> str:
    """Normalises a team name to a consistent uppercase format for matching."""
    if not name: return ""
    return name.strip().upper()



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
    team_sos_history = {}  # team_id -> [(datetime, log_rank_opp)]
    team_lan_history = {}  # team_id -> [(datetime, is_lan_int)]
    current_streaks = {}   # team_id -> int
    match_picked_seen = {} # match_id -> set
    
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
            current_streaks[t_a] = current_streaks.get(t_a, 0) + 1
            current_streaks[t_b] = 0
        else:
            current_streaks[t_b] = current_streaks.get(t_b, 0) + 1
            current_streaks[t_a] = 0

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
            
        if t_a not in team_sos_history: team_sos_history[t_a] = []
        if t_b not in team_sos_history: team_sos_history[t_b] = []
        if t_a not in team_lan_history: team_lan_history[t_a] = []
        if t_b not in team_lan_history: team_lan_history[t_b] = []
            
        r_a = row["team_a_world_rank"]
        r_b = row["team_b_world_rank"]

        score_a = row.get("score_a", 0)
        score_b = row.get("score_b", 0)

        team_general_histories[t_a].append((date, score_a, score_b, r_b))
        team_general_histories[t_b].append((date, score_b, score_a, r_a))
        team_sos_history[t_a].append((date, np.log(max(r_b, 1))))
        team_sos_history[t_b].append((date, np.log(max(r_a, 1))))
        
        is_lan = 1 if row.get("is_lan", False) else 0
        team_lan_history[t_a].append((date, is_lan))
        team_lan_history[t_b].append((date, is_lan))
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
            h2h_stats[h2h_key][str(t_a)] += 1
        else:
            h2h_stats[h2h_key][str(t_b)] += 1
            
    return team_general_histories, team_map_histories, team_latest_ranks, h2h_stats, current_streaks, team_first_picks, team_total_series, team_sos_history, team_lan_history

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
        self.gen_histories, self.map_histories, self.latest_ranks, self.h2h_stats, \
        self.latest_streaks, self.team_fpicks, self.team_tseries, self.sos_histories, \
        self.lan_histories = load_latest_state()
        
        scaler_path = CHECKPOINT_DIR / "scaler.pkl"
        model_path = CHECKPOINT_DIR / "best_mvp_model.pt"
        if not scaler_path.exists() or not model_path.exists():
            raise FileNotFoundError("Model or Scaler not found. Ensure training has completed.")
            
        self.scaler = joblib.load(scaler_path)
        self.model = MatchPredictor(self.scaler.n_features_in_)
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
        self.model.eval()

def get_win_probabilities(ctx: PredictorContext, t_a_id: str, t_b_id: str, maps: List[str], picker_override: str = "neutral", veto_starter: str = "a", is_lan: bool = False) -> List[float]:
    """
    Calculates map-level win probabilities for a given matchup and map set.
    """
    now = pd.to_datetime(datetime.now(timezone.utc))
    map_probs = []
    
    g_a_90d = get_recent_stats(ctx.gen_histories.get(t_a_id, []), now, FORM_WINDOW_DAYS_LONG)
    g_b_90d = get_recent_stats(ctx.gen_histories.get(t_b_id, []), now, FORM_WINDOW_DAYS_LONG)
    g_a_30d = get_recent_stats(ctx.gen_histories.get(t_a_id, []), now, FORM_WINDOW_DAYS)
    g_b_30d = get_recent_stats(ctx.gen_histories.get(t_b_id, []), now, FORM_WINDOW_DAYS)

    for i, m_name in enumerate(maps):
        g_a_7d = get_recent_stats(ctx.gen_histories.get(t_a_id, []), now, FORM_WINDOW_DAYS_SHORT)
        g_b_7d = get_recent_stats(ctx.gen_histories.get(t_b_id, []), now, FORM_WINDOW_DAYS_SHORT)
        
        s_a = ctx.latest_streaks.get(t_a_id, 0)
        s_b = ctx.latest_streaks.get(t_b_id, 0)
        
        rank_a = ctx.latest_ranks.get(t_a_id, {"world": DEFAULT_TEAM_RANK})["world"]
        rank_b = ctx.latest_ranks.get(t_b_id, {"world": DEFAULT_TEAM_RANK})["world"]
        log_a = np.log(max(rank_a, 1))
        log_b = np.log(max(rank_b, 1))
        
        rank_diff = log_b - log_a
        avg_log_rank = (log_a + log_b) / 2
        
        wr_90d_diff = g_a_90d["win_rate"] - g_b_90d["win_rate"]
        wr_30d_diff = g_a_30d["win_rate"] - g_b_30d["win_rate"]
        wr_7d_diff = g_a_7d["win_rate"] - g_b_7d["win_rate"]
        
        h2h_key = tuple(sorted([str(t_a_id), str(t_b_id)]))
        h2h_a = ctx.h2h_stats.get(h2h_key, {}).get(str(t_a_id), 0)
        h2h_b = ctx.h2h_stats.get(h2h_key, {}).get(str(t_b_id), 0)
        
        is_a_picker = 0
        is_b_picker = 0
        if picker_override.lower() in ["team_a", "a"]: is_a_picker = 1
        elif picker_override.lower() in ["team_b", "b"]: is_b_picker = 1
        elif len(maps) == 3: # Bo3: Veto starter picks map[0], other picks map[1], Decider
            if veto_starter == "a":
                is_a_picker = 1 if i == 0 else 0
                is_b_picker = 1 if i == 1 else 0
            else:  # Team B started the veto -> B picks first
                is_a_picker = 1 if i == 1 else 0
                is_b_picker = 1 if i == 0 else 0
        elif len(maps) == 5: # Bo5: Starter picks first pair, other picks second pair
            if veto_starter == "a":
                is_a_picker = 1 if i in [0, 2] else 0
                is_b_picker = 1 if i in [1, 3] else 0
            else:
                is_a_picker = 1 if i in [1, 3] else 0
                is_b_picker = 1 if i in [0, 2] else 0
        elif len(maps) == 1: # Bo1: Neutral (no picks, only bans)
            pass
        else: # Default fallback
            if veto_starter == "a":
                is_a_picker = 1 if i == 0 else 0
                is_b_picker = 1 if i == 1 else 0
            else:
                is_a_picker = 1 if i == 1 else 0
                is_b_picker = 1 if i == 0 else 0

        m_a_90d = get_recent_stats(ctx.map_histories.get(t_a_id, {}).get(m_name, []), now, MAP_WINDOW_DAYS)
        m_b_90d = get_recent_stats(ctx.map_histories.get(t_b_id, {}).get(m_name, []), now, MAP_WINDOW_DAYS)
        map_wr_diff = m_a_90d["win_rate"] - m_b_90d["win_rate"]
        
        def get_comfort(tid, m_n):
            cutoff = now - pd.Timedelta(days=FORM_WINDOW_DAYS)
            picks = len([d for d in ctx.team_fpicks.get(tid, {}).get(m_n, []) if d >= cutoff])
            total = len([d for d in ctx.team_tseries.get(tid, []) if d >= cutoff])
            return picks / total if total > 0 else 0.0
            
        map_comfort_diff = get_comfort(t_a_id, m_name) - get_comfort(t_b_id, m_name)
        
        dom_a = get_dominance_metrics(ctx.gen_histories.get(t_a_id, []), now, FORM_WINDOW_DAYS)
        dom_b = get_dominance_metrics(ctx.gen_histories.get(t_b_id, []), now, FORM_WINDOW_DAYS)
        
        dominance_diff = dom_a["avg_win_margin"] - dom_b["avg_win_margin"]
        resilience_diff = dom_b["avg_loss_margin"] - dom_a["avg_loss_margin"]
        
        sos_a_90d = get_sos(ctx.sos_histories.get(t_a_id, []), now, FORM_WINDOW_DAYS_LONG)
        sos_b_90d = get_sos(ctx.sos_histories.get(t_b_id, []), now, FORM_WINDOW_DAYS_LONG)
        sos_90d_diff = sos_b_90d - sos_a_90d

        sos_a = get_sos(ctx.sos_histories.get(t_a_id, []), now, FORM_WINDOW_DAYS)
        sos_b = get_sos(ctx.sos_histories.get(t_b_id, []), now, FORM_WINDOW_DAYS)
        # Positive = Team A faced harder opponents (more battle-tested)
        sos_diff = sos_b - sos_a
        
        # LAN Rate: percentage of recent matches played on LAN
        def get_lan_rate(history, current_date, days):
            if not history:
                return 0.0
            cutoff = current_date - pd.Timedelta(days=days)
            recent = [is_l for d, is_l in history if d >= cutoff]
            if not recent:
                return 0.0
            return sum(recent) / len(recent)
        
        lan_rate_a = get_lan_rate(ctx.lan_histories.get(t_a_id, []), now, FORM_WINDOW_DAYS)
        lan_rate_b = get_lan_rate(ctx.lan_histories.get(t_b_id, []), now, FORM_WINDOW_DAYS)
        lan_rate_diff = lan_rate_a - lan_rate_b

        feat_vals = {
            "rank_diff": rank_diff,
            "win_rate_90d_diff": wr_90d_diff,
            "win_rate_30d_diff": wr_30d_diff,
            "win_rate_7d_diff": wr_7d_diff,
            "team_a_win_streak": np.log1p(s_a),
            "team_b_win_streak": np.log1p(s_b),
            "picker_diff": is_a_picker - is_b_picker,
            "h2h_a_wins": h2h_a,
            "h2h_b_wins": h2h_b,
            "map_win_rate_diff": map_wr_diff,
            "map_comfort_diff": map_comfort_diff,
            "dominance_diff": dominance_diff,
            "resilience_diff": resilience_diff,
            "avg_log_rank": avg_log_rank,
            "sos_90d_diff": sos_90d_diff,
            "sos_diff": sos_diff,
            "lan_rate_diff": lan_rate_diff
        }
        
        f_vec = np.array([[feat_vals[col] for col in MODEL_FEATURES]], dtype=np.float32)
        scaled_f = ctx.scaler.transform(f_vec)
        with torch.no_grad():
            p_a = ctx.model(torch.tensor(scaled_f, dtype=torch.float32)).item()
            map_probs.append(p_a)
            
    return map_probs

def calculate_expected_series_win(team_a_raw, team_b_raw, series_format="bo3", threshold=MC_THRESHOLD, iters=MC_ITERATIONS, starts_veto=None, ctx=None):
    """
    Calculates the Expected Series Win Probability by mating the Veto Simulator
    and the Neural Network Predictor using the Law of Total Probability.
    Correctly tracks which team starts the veto to assign picker roles.
    """
    # 1. Initialize Predictor context (expensive state load)
    if ctx is None:
        ctx = PredictorContext()
        
    t_a_id = normalize_name(team_a_raw)
    t_b_id = normalize_name(team_b_raw)

    # 2. Prepare veto sim stats
    veto_df = veto_sim.load_data()
    stats_a = veto_sim.get_team_stats(t_a_id, veto_df)
    stats_b = veto_sim.get_team_stats(t_b_id, veto_df)
    
    bo = int(series_format.replace("bo", ""))
    
    # 3. Determine veto start and run simulations in batches.
    #    When starts_veto is unspecified (50/50), we split iterations into two
    #    batches — one where team A starts the veto, one where team B starts.
    #    This ensures picker_diff is correctly assigned for every veto path.
    veto_start = None
    if starts_veto:
        sv = starts_veto.lower()
        if sv in ["a", "team_a"]:
            veto_start = "a"
        elif sv in ["b", "team_b"]:
            veto_start = "b"
        else:
            # Handle team name input (e.g. --starts-veto "G2")
            sv_norm = normalize_name(starts_veto)
            if sv_norm == t_a_id:
                veto_start = "a"
            elif sv_norm == t_b_id:
                veto_start = "b"
    
    if veto_start:
        # Single batch: one team always starts
        seq, _ = veto_sim.run_simulations(
            stats_a, stats_b, iters=iters, series_format=series_format, starts_veto=veto_start
        )
        batches = [(seq, iters, veto_start)]
        sequence_counts = seq
    else:
        # 50/50 split: run half iterations for each starting team
        iters_a = iters // 2
        iters_b = iters - iters_a
        seq_a, _ = veto_sim.run_simulations(
            stats_a, stats_b, iters=iters_a, series_format=series_format, starts_veto="a"
        )
        seq_b, _ = veto_sim.run_simulations(
            stats_a, stats_b, iters=iters_b, series_format=series_format, starts_veto="b"
        )
        batches = [(seq_a, iters_a, "a"), (seq_b, iters_b, "b")]
        # Merge sequence counts for display
        sequence_counts = {}
        for sd, _, _ in batches:
            for k, v in sd.items():
                sequence_counts[k] = sequence_counts.get(k, 0) + v
    
    # 4. Calculate expected win probability using Law of Total Probability
    expected_win_prob = 0.0
    
    for seq_dict, batch_iters, starter in batches:
        batch_weight = batch_iters / iters
        
        # Sort paths and truncate based on threshold
        sorted_paths = sorted(seq_dict.items(), key=lambda x: x[1], reverse=True)
        selected_paths = []
        cumulative_count = 0
        
        for path_str, count in sorted_paths:
            selected_paths.append((path_str, count))
            cumulative_count += count
            if (cumulative_count / batch_iters) >= threshold:
                break
        
        for path_str, count in selected_paths:
            path_prob_norm = count / cumulative_count
            maps = path_str.split(",")
            
            # Fetch map-level win probabilities with picker assignment
            map_probs = get_win_probabilities(ctx, t_a_id, t_b_id, maps, veto_starter=starter)
            
            # Conditional series win probability: P(Win | Path)
            p_win_given_path = combine_probs(map_probs, bo)
            
            # Law of Total Probability contribution, weighted by batch share
            expected_win_prob += p_win_given_path * path_prob_norm * batch_weight
        
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

    t_a_id = normalize_name(team_raw_a)
    t_b_id = normalize_name(team_raw_b)
    
    now = pd.to_datetime(datetime.now(timezone.utc))
    
    # ANSI escape codes for coloring
    RED = "\033[91m"
    RESET = "\033[0m"

    print("\n" + "="*60)
    print(f" PREDICTION: {t_a_id} vs {t_b_id}")
    print("="*60)

    # Global team sample size info
    g_a_30d = get_recent_stats(ctx.gen_histories.get(t_a_id, []), now, FORM_WINDOW_DAYS)
    g_b_30d = get_recent_stats(ctx.gen_histories.get(t_b_id, []), now, FORM_WINDOW_DAYS)
    
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
        
        m_a_30d_matches = get_recent_stats(ctx.map_histories.get(t_a_id, {}).get(m_name, []), now, FORM_WINDOW_DAYS)["matches"]
        m_b_30d_matches = get_recent_stats(ctx.map_histories.get(t_b_id, {}).get(m_name, []), now, FORM_WINDOW_DAYS)["matches"]
        
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
